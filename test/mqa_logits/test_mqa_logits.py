import torch
import torch.nn.functional as F
import triton
import triton.testing
from lightop import op
import lightop
import pdb
import pandas as pd  # Added for Excel export
import os            # Added for file path helper
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

    q_seq_len_list = [16,32,64,2048, 4096]
    kv_seq_len_list = [4096, 8192, 16384, 32768, 65536, 131072]
    head_num = 64
    head_dim = 128
    test_speed = True
    use_random = True
    disable_cp = True
    q_k_dtype = torch.float16
    results_data = []
    write_csv = test_speed
    print("Starting MQA Logits Benchmark...")
    print(f"Q Seq Lens: {q_seq_len_list}")
    print(f"KV Seq Lens: {kv_seq_len_list}")
    print(f"Head Num: {head_num}, Head Dim: {head_dim}")
    print("-" * 40)

    for q_seq_len in q_seq_len_list:
        for kv_seq_len in kv_seq_len_list:
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
        
            if test_speed:
                for _ in range(10):
                    _ = lightop.mqa_logits(Q, K, weights, ks, ke)
                
                torch.cuda.synchronize()
                
                num_iterations = 100 
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                
                start_event.record()
                for _ in range(num_iterations):
                    _ = lightop.mqa_logits(Q, K, weights, ks, ke)
                end_event.record()
                
                torch.cuda.synchronize()
                
                total_time_ms = start_event.elapsed_time(end_event)
                mean_time_ms = total_time_ms / num_iterations
                elapsed_time_us = mean_time_ms * 1000
                print(f" Avg Time: {elapsed_time_us:10.2f} us.")

                # # ---Run Correctness Check ---
                # ref_logits, ref_cost = ref_fp8_mqa_logits(Q, K, weights, ks, ke)
                # lightop_logits = lightop.mqa_logits(Q, K, weights, ks, ke)
                # # pdb.set_trace()
                # ref_neginf_mask = (ref_logits == float('-inf'))
                # neginf_mask = (lightop_logits == float('-inf'))
                # assert torch.equal(neginf_mask, ref_neginf_mask), "Masks differ!"

                # ref_logits = ref_logits.masked_fill(ref_neginf_mask, 0)
                # lightop_logits = lightop_logits.masked_fill(neginf_mask, 0)
                
                # diff_max = (ref_logits - lightop_logits).abs().max()
                
                # assert torch.allclose(lightop_logits, ref_logits, rtol=1e-3, atol=1e-3), \
                #      f"❌ Precision check failed! Max diff: {diff_max}"

                # print(f" Precision OK (Max Err: {diff_max:.2e})")
                
                results_data.append({
                    "Q Seq Len": q_seq_len,
                    "KV Seq Len": kv_seq_len,
                    "Head Num": head_num,
                    "Head Dim": head_dim,
                    "Time (us)": round(elapsed_time_us, 2)
                })

            else:
                ref_logits, ref_cost = ref_fp8_mqa_logits(Q, K, weights, ks, ke)
                lightop_logits = lightop.mqa_logits(Q, K, weights, ks, ke)
                
                ref_neginf_mask = (ref_logits == float('-inf'))
                neginf_mask = (lightop_logits == float('-inf'))
                
                assert torch.equal(neginf_mask, ref_neginf_mask), "Masks differ!"

                ref_logits = ref_logits.masked_fill(ref_neginf_mask, 0)
                lightop_logits = lightop_logits.masked_fill(neginf_mask, 0)
                
                diff_max = (ref_logits - lightop_logits).abs().max()
                
                assert torch.allclose(lightop_logits, ref_logits, rtol=1e-3, atol=1e-3), \
                     f"❌ Precision check failed! Max diff: {diff_max}"

                print(f" Precision OK (Max Err: {diff_max:.2e})")


    if write_csv:
        df = pd.DataFrame(results_data)
        excel_path = 'mqa_logits_benchmark.xlsx'
        
        try:
            df.to_excel(excel_path, index=False, sheet_name='BenchmarkResults')
            print("-" * 40)
            print(f"✅ Benchmark complete. Results saved to:")
            print(f"{os.path.abspath(excel_path)}")
        except Exception as e:
            print(f"Error saving to Excel: {e}")
            print("Displaying results in console instead:")
            print(df)
    else:
        print("No benchmark data was generated.")
