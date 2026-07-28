import torch
import torch.nn.functional as F
import triton
import triton.testing
from lightop import op
import lightop
from lightop.attention import mqa_logits
import pdb
import pandas as pd  # Added for Excel export
import os            # Added for file path helper
from typing import Tuple

def ceil_to_ue8m0(x: torch.Tensor):
    assert x.view(-1).amax().item() > 0
    return torch.pow(2.0, torch.ceil(torch.log2(x.abs())))

def per_custom_dims_cast_to_fp8(x: torch.Tensor, dims: Tuple, use_ue8m0: bool) -> Tuple[torch.Tensor, torch.Tensor]:
    excluded_dims = tuple([i for i in range(x.dim()) if i not in set(dims)])
    x_amax = x.abs().float().amax(dim=excluded_dims, keepdim=True).clamp(1e-4)
    sf = x_amax / 448.0
    sf = ceil_to_ue8m0(sf) if use_ue8m0 else sf
    x_scaled = (x * (1.0 / sf)).to(torch.float8_e4m3fn)
    return x_scaled, sf.squeeze()

def calc_diff(x: torch.Tensor, y: torch.Tensor):
    x, y = x.double(), y.double()
    denominator = (x * x + y * y).sum()
    sim = 2 * (x * y).sum() / denominator
    return 1 - sim

def ref_fp8_mqa_logits(q: torch.Tensor, kv: torch.Tensor, weights: torch.Tensor,
                       cu_seqlen_ks: torch.Tensor, cu_seqlen_ke: torch.Tensor, cost_only: bool = False):
    seq_len_kv = kv.shape[0]

    if cost_only:
        start = cu_seqlen_ks.clamp(min=0, max=seq_len_kv)
        end   = cu_seqlen_ke.clamp(min=0, max=seq_len_kv)
        count_ones_per_row = (end - start).clamp(min=0)
        return count_ones_per_row.sum()

    k = kv
    q = q.float()
    k = k.float()

    mask_lo = torch.arange(0, seq_len_kv, device='cuda')[None, :] >= cu_seqlen_ks[:, None]
    mask_hi = torch.arange(0, seq_len_kv, device='cuda')[None, :] < cu_seqlen_ke[:, None]
    mask = mask_lo & mask_hi

    score = torch.einsum('mhd,nd->hmn', q, k)
    logits = (score.relu() * weights.unsqueeze(-1).transpose(0, 1)).sum(dim=0)
    logits = logits.masked_fill(~mask, float('-inf'))

    cost = mask.sum()
    return logits, cost

def generate_cp_test_data(seq_len, seq_len_kv):
    assert seq_len_kv % seq_len == 0 and seq_len % 2 == 0
    chunk_size = seq_len // 2
    cp_size = seq_len_kv // seq_len
    # Select an arbitrary CP rank
    cp_id = cp_size // 3
    ks = torch.zeros(seq_len, dtype=torch.int, device='cuda')
    ke = torch.zeros(seq_len, dtype=torch.int,  device='cuda')
    for i in range(chunk_size):
        ke[i] = cp_id * chunk_size + i
        ke[i + chunk_size] = (cp_size * 2 - 1 - cp_id) * chunk_size + i
    return ks, ke


if __name__ == "__main__":
    torch.manual_seed(0)
    #汇编
    # q_seq_len_list =  [128,255,321,369,666,812,1024,4096]
    # kv_seq_len_list = [4096,8192,11111,16384,32768, 65536,131072]
    #cuda
    q_seq_len_list =  [16,32,64,127]
    kv_seq_len_list = [66,4096,8192,16384,32768, 65536,131072]

    head_num_list = [8,64]
    head_dim = 128
    test_speed = True
    use_random = True
    disable_cp = True
    is_fp8 = True  
    q_k_dtype = torch.bfloat16
    results_data = []
    write_csv = False
    print("Starting MQA Logits Benchmark...")
    print(f"FP8 Mode: {is_fp8}")
    print(f"Q Seq Lens: {q_seq_len_list}")
    print(f"KV Seq Lens: {kv_seq_len_list}")
    print(f"Head Num: {head_num_list}, Head Dim: {head_dim}")
    print("-" * 40)

    for q_seq_len in q_seq_len_list:
        for kv_seq_len in kv_seq_len_list:
            for head_num in head_num_list:
                print(f"Running test: Q={q_seq_len:3d}, KV={kv_seq_len:6d}...", end="", flush=True)
                if use_random:
                    Q = torch.randn(q_seq_len, head_num, head_dim, device='cuda', dtype=q_k_dtype)
                    K = torch.randn(kv_seq_len, head_dim, device='cuda', dtype=q_k_dtype)
                    weights = torch.randn(q_seq_len, head_num, device='cuda', dtype=torch.float32)
                else:
                    Q = torch.ones(q_seq_len, head_num, head_dim, dtype=q_k_dtype, device="cuda")
                    K = torch.ones(kv_seq_len, head_dim, dtype=q_k_dtype, device="cuda")
                    weights = torch.ones(q_seq_len, head_num, device='cuda', dtype=torch.float32)
                
                if disable_cp:
                    ks = torch.zeros(q_seq_len, dtype=torch.int, device='cuda')
                    ke = torch.arange(q_seq_len, dtype=torch.int, device='cuda') + (kv_seq_len - q_seq_len)
                else:
                    ks, ke = generate_cp_test_data(q_seq_len, kv_seq_len)
                
                if is_fp8:
                    q_in = Q.to(torch.float8_e4m3fn)
                    k_in, kv_scale = per_custom_dims_cast_to_fp8(K, (0, ), False)
                    scale_in = kv_scale
                else:
                    q_in = Q
                    k_in = K
                    scale_in = None
                    
                def run_lightop():
                    if is_fp8:
                        return mqa_logits(q_in, k_in, weights, ks, ke, scale_in)
                    else:
                        return mqa_logits(q_in, k_in, weights, ks, ke)
                
                if test_speed:
                    for _ in range(10):
                        _ = run_lightop()
                    
                    torch.cuda.synchronize()
                    
                    num_iterations = 100 
                    start_event = torch.cuda.Event(enable_timing=True)
                    end_event = torch.cuda.Event(enable_timing=True)
                    
                    start_event.record()
                    for _ in range(num_iterations):
                        _ = run_lightop()
                    end_event.record()
                    # pdb.set_trace()
                    torch.cuda.synchronize()
                    
                    total_time_ms = start_event.elapsed_time(end_event)
                    mean_time_ms = total_time_ms / num_iterations
                    elapsed_time_us = mean_time_ms * 1000
                    print(f" Avg Time: {elapsed_time_us:10.2f} us.")
                    #精度
                    ref_logits, ref_cost = ref_fp8_mqa_logits(Q, K, weights, ks, ke)
                    lightop_logits = run_lightop()
                    # pdb.set_trace()
                    ref_neginf_mask = (ref_logits == float('-inf'))
                    neginf_mask = (lightop_logits == float('-inf'))
                    # pdb.set_trace()
                    assert torch.equal(neginf_mask, ref_neginf_mask), "Masks differ!"

                    ref_logits = ref_logits.masked_fill(ref_neginf_mask, 0)
                    lightop_logits = lightop_logits.masked_fill(neginf_mask, 0)
                    
                    if is_fp8:
                        diff = calc_diff(lightop_logits, ref_logits)
                        assert diff < 1e-3, f"❌ FP8 Precision check failed! Diff: {diff:.4e}"
                        print(f"✅ FP8 Precision OK (Diff: {diff:.2e})")
                        error_val = diff.item()
                    else:
                        diff_max = (ref_logits - lightop_logits).abs().max()
                        assert torch.allclose(lightop_logits, ref_logits, rtol=1e-3, atol=1e-3), \
                            f"❌ BF16 Precision check failed! Max diff: {diff_max:.4e}"
                        print(f"✅ BF16 Precision OK (Max Err: {diff_max:.2e})")
                        error_val = diff_max.item()

                    results_data.append({
                        "Q_Seq_Len": q_seq_len,
                        "KV_Seq_Len": kv_seq_len,
                        "Head_Num": head_num,
                        "Head_Dim": head_dim,
                        "Is_FP8": is_fp8,
                        "Avg_Time_us": elapsed_time_us,
                        "Error_Val": error_val  # 此处根据 is_fp8 动态记录了 Diff 或 Max_Err
                    })
                else:
                    ref_logits, ref_cost = ref_fp8_mqa_logits(Q, K, weights, ks, ke)
                    lightop_logits = run_lightop()
                    # pdb.set_trace()
                    ref_neginf_mask = (ref_logits == float('-inf'))
                    neginf_mask = (lightop_logits == float('-inf'))
                    
                    assert torch.equal(neginf_mask, ref_neginf_mask), "Masks differ!"

                    ref_logits = ref_logits.masked_fill(ref_neginf_mask, 0)
                    lightop_logits = lightop_logits.masked_fill(neginf_mask, 0)
                    
                    if is_fp8:
                        diff = calc_diff(lightop_logits, ref_logits)
                        assert diff < 1e-3, f"❌ FP8 Precision check failed! Diff: {diff:.4e}"
                        print(f"✅ FP8 Precision OK (Diff: {diff:.2e})")
                    else:
                        diff_max = (ref_logits - lightop_logits).abs().max()
                        assert torch.allclose(lightop_logits, ref_logits, rtol=1e-3, atol=1e-3), \
                            f"❌ BF16 Precision check failed! Max diff: {diff_max:.4e}"
                        print(f"✅ BF16 Precision OK (Max Err: {diff_max:.2e})")


    if write_csv:
        df = pd.DataFrame(results_data)
        csv_path = 'mqa_logits_benchmark.csv' 
        
        try:
            df.to_csv(csv_path, index=False)  
            print("-" * 40)
            print(f"✅ Benchmark complete. Results saved to:")
            print(f"{os.path.abspath(csv_path)}")
        except Exception as e:
            print(f"Error saving to CSV: {e}")
            print("Displaying results in console instead:")
            print(df)
    else:
        print("No benchmark data was generated.")