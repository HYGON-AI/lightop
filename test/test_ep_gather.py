import torch
import triton
import triton.language as tl
import time
import csv
import os
import pdb
# 尝试导入自定义算子
try:
    from lightop.moe import ep_gather
    # print("[INFO] Successfully imported lightop")
except ImportError:
    print("[ERROR] Failed to import lightop. Please ensure the CUDA extension is installed.")
    exit(1)

# -----------------------------------------------------------------------------
# Triton Implementation (保持不变)
# -----------------------------------------------------------------------------

@triton.jit
def apply_expert_map(expert_id, expert_map):
    if expert_id != -1:
        expert_id = tl.load(expert_map + expert_id).to(expert_id.dtype)
    return expert_id

@triton.jit
def _fwd_kernel_ep_gather(
    total_token_num,
    input_tensor,
    input_tensor_stride0,
    input_tensor_stride1,
    recv_topk_ids,
    recv_topk_ids_stride0,
    recv_topk_ids_stride1,
    recv_topk_weight,
    recv_topk_weight_stride0,
    recv_topk_weight_stride1,
    input_index,
    input_index_stride0,
    input_index_stride1,
    output_tensor,
    output_tensor_stride0,
    output_tensor_stride1,
    topk_num: tl.constexpr,
    expert_map,
    HAS_EXPERT_MAP: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    cur_block_int32 = tl.program_id(0)
    cur_block = cur_block_int32.to(tl.int64)

    start_cur_token_int32 = tl.program_id(1)

    grid_num = tl.num_programs(1)

    for cur_token_int32 in range(start_cur_token_int32, total_token_num, grid_num):
        cur_token = cur_token_int32.to(tl.int64)

        off_d = tl.arange(0, BLOCK_D)
        accumulator = tl.zeros([BLOCK_D], dtype=tl.float32)

        for topk_index_int32 in range(0, topk_num):
            topk_index = topk_index_int32.to(tl.int64)

            expert_id = tl.load(
                recv_topk_ids + cur_token * recv_topk_ids_stride0 + topk_index
            )
            if HAS_EXPERT_MAP:
                expert_id = apply_expert_map(expert_id, expert_map)

            if expert_id >= 0:
                source_token_index_int32 = tl.load(
                    input_index + cur_token * input_index_stride0 + topk_index
                )
                source_token_index = source_token_index_int32.to(tl.int64)

                acc_weight = tl.load(
                    recv_topk_weight + cur_token * recv_topk_weight_stride0 + topk_index
                )
                tmp = tl.load(
                    input_tensor
                    + source_token_index * input_tensor_stride0
                    + cur_block * BLOCK_D
                    + off_d
                )
                accumulator += tmp.to(tl.float32) * acc_weight

        tl.store(
            output_tensor
            + cur_token * output_tensor_stride0
            + cur_block * BLOCK_D
            + off_d,
            accumulator.to(output_tensor.dtype.element_ty),
        )



@torch.no_grad()
def triton_ep_gather(
    input_tensor: torch.Tensor,
    recv_topk_ids: torch.Tensor,
    recv_topk_weight: torch.Tensor,
    input_index: torch.Tensor,
    expert_map: torch.Tensor | None,
    output_tensor: torch.Tensor,
):
    num_warps = 2
    num_tokens = output_tensor.shape[0]
    hidden_size = input_tensor.shape[1]
    BLOCK_D = min(hidden_size, 1024)
    assert hidden_size % BLOCK_D == 0
    grid = (triton.cdiv(hidden_size, BLOCK_D), min(num_tokens, 1024))

    _fwd_kernel_ep_gather[grid](
        num_tokens,
        input_tensor,
        input_tensor.stride(0),
        input_tensor.stride(1),
        recv_topk_ids,
        recv_topk_ids.stride(0),
        recv_topk_ids.stride(1),
        recv_topk_weight,
        recv_topk_weight.stride(0),
        recv_topk_weight.stride(1),
        input_index,
        input_index.stride(0),
        input_index.stride(1),
        output_tensor,
        output_tensor.stride(0),
        output_tensor.stride(1),
        topk_num=recv_topk_ids.shape[1],
        expert_map=expert_map,
        HAS_EXPERT_MAP=expert_map is not None,
        num_warps=num_warps,
        BLOCK_D=BLOCK_D,
    )
    return




def run_single_benchmark(M, H, K, E, dtype, device, csv_writer):
    """运行单个 M 值的 Benchmark 并写入 CSV"""
    
    M_sum = M * K
    

    try:
        a = torch.randn((M_sum, H), dtype=dtype, device=device)
        topk_ids = torch.randint(0, E, (M, K), dtype=torch.int64, device=device)
        topk_weights = torch.randn((M, K), dtype=torch.float32, device=device)
        topk_weights = torch.softmax(topk_weights, dim=-1)
        inv_perm = torch.randperm(M_sum, dtype=torch.int32, device=device).reshape(M, K)
        expert_map = None 
        
        out_triton = torch.zeros((M, H), dtype=dtype, device=device)
        out_custom = torch.zeros((M, H), dtype=dtype, device=device)
    except torch.cuda.OutOfMemoryError:
        print(f"{M:<8} | OOM ERROR during allocation")
        return

    # -------------------------------------------------------------------------
    # 正确性验证 (只打印简略结果)
    # -------------------------------------------------------------------------
    triton_ep_gather(a, topk_ids, topk_weights, inv_perm, expert_map, out_triton)
    ep_gather(a, topk_ids, topk_weights, inv_perm, expert_map, out_custom)
    torch.cuda.synchronize()

    passed = False
    if torch.allclose(out_custom, out_triton, rtol=1e-2, atol=1e-2):
        passed = True
    else:
        max_diff = (out_triton - out_custom).abs().max().item()
    
    # pdb.set_trace()
    n_warmup = 10
    n_repeat = 100
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    # 1. Triton Benchmark
    for _ in range(n_warmup):
        triton_ep_gather(a, topk_ids, topk_weights, inv_perm, expert_map, out_triton)
    torch.cuda.synchronize()

    start_event.record()
    for _ in range(n_repeat):
        triton_ep_gather(a, topk_ids, topk_weights, inv_perm, expert_map, out_triton)
    end_event.record()
    torch.cuda.synchronize()
    triton_time_ms = start_event.elapsed_time(end_event) / n_repeat

    # 2. Custom Op Benchmark
    for _ in range(n_warmup):
        ep_gather(a, topk_ids, topk_weights, inv_perm, expert_map, out_custom)
    torch.cuda.synchronize()

    start_event.record()
    for _ in range(n_repeat):
        ep_gather(a, topk_ids, topk_weights, inv_perm, expert_map, out_custom)
    end_event.record()
    torch.cuda.synchronize()
    custom_time_ms = start_event.elapsed_time(end_event) / n_repeat

    # a + weight + topkid(int64)  + map(int32) + out   
    # 优化 a + mask(int8) + out 
    total_bytes = (M * K * H * 2) + (M * K * 8) + (M * K * 8) + (M * H * 2)
    total_gb = total_bytes / 1e9

    triton_bw = total_gb / (triton_time_ms / 1000.0)
    custom_bw = total_gb / (custom_time_ms / 1000.0)
    speedup = triton_time_ms / custom_time_ms

    # 打印表格行
    passed_str = '✅' if passed else '❌'
    print(f"{M:<8} | {passed_str:<8} | {triton_time_ms:<12.3f} | {triton_bw:<15.2f} | {custom_time_ms:<12.3f} | {custom_bw:<15.2f} | {speedup:<8.2f}")

    # 写入 CSV
    if csv_writer:
        csv_writer.writerow([
            M, H, K, E, 
            passed,
            f"{triton_time_ms:.4f}", 
            f"{custom_time_ms:.4f}", 
            f"{triton_bw:.4f}", 
            f"{custom_bw:.4f}", 
            f"{speedup:.4f}"
        ])

    # 清理显存
    del a, topk_ids, topk_weights, inv_perm, out_triton, out_custom
    torch.cuda.empty_cache()

def main():
    # 配置
    M_list = [256,512,1024,2048,4096,8192,16383,32768,65536]
    # M_list = [65536]
    H = 7168
    K = 8
    E = 256
    dtype = torch.bfloat16
    device = "cuda"
    csv_file = "ep_gather_benchmark_results.csv"

    print(f"==========================================================================================")
    print(f"Running Benchmark Sweep: H={H}, K={K}, E={E}, dtype={dtype}")
    print(f"Results will be saved to: {csv_file}")
    print(f"==========================================================================================")

    # 打开 CSV 文件准备写入
    with open(csv_file, mode='w', newline='') as f:
        writer = csv.writer(f)
        # 写入表头
        writer.writerow([
            "M (Tokens)", "Hidden_Dim", "TopK", "Num_Experts", 
            "Correctness",
            "Triton_Time_ms", "Custom_Time_ms", 
            "Triton_BW_GBs", "Custom_BW_GBs", 
            "Speedup (Custom vs Triton)"
        ])
        
        torch.manual_seed(42)
        
        # 打印控制台表头
        print(f"{'M':<8} | {'Correct':<8} | {'Triton (ms)':<12} | {'Tri BW (GB/s)':<15} | {'Custom (ms)':<12} | {'Cus BW (GB/s)':<15} | {'Speedup':<8}")
        print("-" * 100)

        for M in M_list:
            run_single_benchmark(M, H, K, E, dtype, device, writer)
            
    print(f"\nBenchmark finished. Data saved to {csv_file}")

if __name__ == "__main__":
    main()