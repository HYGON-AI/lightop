import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from typing import Optional
import sys
import time
import csv
import os
import gc

try:
    from lightop import op
    HAS_LIGHTOP = True
except ImportError:
    print("警告: 未找到 lightop 库。将仅运行 Triton 实现，无法运行 CUDA 内核。")
    HAS_LIGHTOP = False

# =========================================================================
#  Triton 量化内核
# =========================================================================

@triton.jit
def _per_token_quant_int8_kernel(
    x_ptr, xq_ptr, scale_ptr, stride_x, stride_xq, N, BLOCK: tl.constexpr, num_tokens_ptr
):
    row_id = tl.program_id(0)
    if num_tokens_ptr is not None:
        num_valid_tokens = tl.load(num_tokens_ptr)
        if row_id >= num_valid_tokens: return
    cols = tl.arange(0, BLOCK)
    mask = cols < N
    x = tl.load(x_ptr + row_id * stride_x + cols, mask=mask, other=0.0).to(tl.float32)
    # 计算最大绝对值
    absmax = tl.maximum(tl.max(tl.abs(x)), 1e-10)
    scale_x = absmax / 127 
    x_q = libdevice.nearbyint(x * (127 / absmax)).to(tl.int8)
    tl.store(xq_ptr + row_id * stride_xq + cols, x_q, mask=mask)
    tl.store(scale_ptr + row_id, scale_x)

def per_token_quant_int8_triton(x: torch.Tensor, x_q: Optional[torch.Tensor] = None, scales: Optional[torch.Tensor] = None):
    M, N = x.shape
    if x_q is None: x_q = torch.empty_like(x, device=x.device, dtype=torch.int8)
    if scales is None: scales = torch.empty((M, 1), device=x.device, dtype=torch.float32)
    
    BLOCK = triton.next_power_of_2(N)
    num_warps = min(max(BLOCK // 256, 1), 8)

    _per_token_quant_int8_kernel[(M, )](
        x, x_q, scales, 
        stride_x=x.stride(-2), stride_xq=x_q.stride(-2), 
        N=N, BLOCK=BLOCK, num_warps=num_warps, num_tokens_ptr=None
    )
    return x_q, scales

# =========================================================================
#  Triton Scatter 实现
# =========================================================================

@triton.jit
def apply_expert_map(expert_id, expert_map):
    if expert_id != -1:
        expert_id = tl.load(expert_map + expert_id).to(expert_id.dtype)
    return expert_id

@triton.jit
def round_up_256(x: int) -> int:
    return ((x + 256 - 1) // 256) * 256

@triton.jit
def _fwd_kernel_ep_scatter_1(
    num_recv_tokens_per_expert, expert_start_loc, m_indices,
    num_experts: tl.constexpr, BLOCK_E: tl.constexpr, BLOCK_EXPERT_NUM: tl.constexpr,
):
    cur_expert = tl.program_id(0)
    offset_cumsum = tl.arange(0, BLOCK_EXPERT_NUM)
    tokens_per_expert = tl.load(num_recv_tokens_per_expert + offset_cumsum, mask=offset_cumsum < num_experts, other=0)
    tokens_per_expert = round_up_256(tokens_per_expert)
    cumsum = tl.cumsum(tokens_per_expert) - tokens_per_expert
    
    tl.store(expert_start_loc + offset_cumsum, cumsum, mask=offset_cumsum < num_experts)
    tl.debug_barrier()
    
    cur_expert_start = tl.load(expert_start_loc + cur_expert)
    cur_expert_token_num = tl.load(num_recv_tokens_per_expert + cur_expert)
    m_indices_start_ptr = m_indices + cur_expert_start
    off_expert = tl.arange(0, BLOCK_E)
    
    for start_m in tl.range(0, cur_expert_token_num, BLOCK_E, num_stages=4):
        tl.store(m_indices_start_ptr + start_m + off_expert, cur_expert, mask=start_m + off_expert < cur_expert_token_num)

@triton.jit
def _fwd_kernel_ep_scatter_2(
    total_token_num, expert_start_loc, recv_x, recv_x_stride0, recv_x_stride1,
    recv_x_scale, recv_x_scale_stride0, recv_x_scale_stride1,
    recv_topk, recv_topk_stride0, recv_topk_stride1,
    output_tensor, output_tensor_stride0, output_tensor_stride1,
    output_tensor_scale, output_tensor_scale_stride0, output_tensor_scale_stride1,
    output_index, output_index_stride0, output_index_stride1,
    topk_num: tl.constexpr, expert_map, HAS_EXPERT_MAP: tl.constexpr,
    HIDDEN_SIZE: tl.constexpr, HIDDEN_SIZE_PAD: tl.constexpr,
    SCALE_HIDDEN_SIZE: tl.constexpr, SCALE_HIDDEN_SIZE_PAD: tl.constexpr,
):
    start_token_id = tl.program_id(0)
    grid_num = tl.num_programs(0)
    offset_in = tl.arange(0, HIDDEN_SIZE_PAD)
    mask = offset_in < HIDDEN_SIZE
    index_in_s = tl.arange(0, SCALE_HIDDEN_SIZE_PAD)
    mask_s = index_in_s < SCALE_HIDDEN_SIZE

    for token_id_int32 in range(start_token_id, total_token_num, grid_num):
        token_id = token_id_int32.to(tl.int64)
        
        to_copy = tl.load(recv_x + token_id * recv_x_stride0 + offset_in, mask=mask)
        to_copy_s = tl.load(recv_x_scale + token_id * recv_x_scale_stride0 + index_in_s * recv_x_scale_stride1, mask=mask_s)

        for topk_idx_int32 in tl.range(0, topk_num, 1, num_stages=4):
            topk_index = topk_idx_int32.to(tl.int64)
            expert_id = tl.load(recv_topk + token_id * recv_topk_stride0 + topk_index)

            if HAS_EXPERT_MAP: expert_id = apply_expert_map(expert_id, expert_map)
            
            if expert_id >= 0:
                dest_token_index_int32 = tl.atomic_add(expert_start_loc + expert_id, 1)
                dest_token_index = dest_token_index_int32.to(tl.int64)

                tl.store(output_index + token_id * output_index_stride0 + topk_index, dest_token_index_int32)
                
                output_tensor_ptr = output_tensor + dest_token_index * output_tensor_stride0
                output_tensor_scale_ptr = output_tensor_scale + dest_token_index * output_tensor_scale_stride0
                
                tl.store(output_tensor_ptr + offset_in, to_copy, mask=mask)
                tl.store(output_tensor_scale_ptr + index_in_s * output_tensor_scale_stride1, to_copy_s, mask=mask_s)

@torch.no_grad()
def ep_scatter(
    recv_x, recv_x_scale, recv_topk, num_recv_tokens_per_expert, expert_map, expert_start_loc,
    output_tensor, output_tensor_scale, m_indices, output_index, alignment=256
):
    num_experts = num_recv_tokens_per_expert.shape[0]
    hidden_size = recv_x.shape[1]
    
    grid = num_experts
    _fwd_kernel_ep_scatter_1[(grid,)](
        num_recv_tokens_per_expert, expert_start_loc, m_indices,
        num_experts=num_experts, num_warps=8, BLOCK_E=alignment, BLOCK_EXPERT_NUM=triton.next_power_of_2(num_experts)
    )

    grid = min(recv_topk.shape[0], 1024 * 8)
    _fwd_kernel_ep_scatter_2[(grid,)](
        recv_topk.shape[0], expert_start_loc, recv_x, recv_x.stride(0), recv_x.stride(1),
        recv_x_scale, recv_x_scale.stride(0), recv_x_scale.stride(1),
        recv_topk, recv_topk.stride(0), recv_topk.stride(1),
        output_tensor, output_tensor.stride(0), output_tensor.stride(1),
        output_tensor_scale, output_tensor_scale.stride(0), output_tensor_scale.stride(1),
        output_index, output_index.stride(0), output_index.stride(1),
        topk_num=recv_topk.shape[1], expert_map=expert_map, HAS_EXPERT_MAP=expert_map is not None,
        HIDDEN_SIZE=hidden_size, HIDDEN_SIZE_PAD=triton.next_power_of_2(hidden_size),
        SCALE_HIDDEN_SIZE=recv_x_scale.shape[-1], SCALE_HIDDEN_SIZE_PAD=triton.next_power_of_2(recv_x_scale.shape[-1]),
        num_warps=8
    )

def count_expert_num_tokens(topk_ids, num_experts):
    flat_ids = topk_ids.view(-1)
    return torch.bincount(flat_ids[flat_ids >= 0], minlength=num_experts).to(torch.int32)

# =========================================================================
#  基准测试逻辑 (Fused: Quant + Scatter)
# =========================================================================

def run_single_benchmark(M, H, K, E, ALIGNMENT, device):
    torch.cuda.empty_cache() 
    torch.manual_seed(42)

    # 1. 数据准备 (输入为 FP16/BF16)
    x = torch.randn((M, H), dtype=torch.bfloat16, device=device)
    topk_ids = torch.randint(0, E, (M, K), dtype=torch.int32, device=device)
    expert_map = None

    expert_counts = count_expert_num_tokens(topk_ids, E).to(device)
    # 对齐并计算总输出行数
    expert_counts_padded = (expert_counts + ALIGNMENT - 1) // ALIGNMENT * ALIGNMENT
    M_sum = expert_counts_padded.sum().item()
    
    # 预分配输出 Tensor
    aq_out = torch.empty((M_sum, H), device=device, dtype=torch.int8)
    aq_scale = torch.empty((M_sum, 1), device=device, dtype=torch.float32)
    m_indices = torch.full((M_sum,), -1, dtype=torch.int32, device=device)
    inv_perm = torch.empty((M, K), device=device, dtype=torch.int32)
    
    # --- 验证阶段 ---
    is_correct = False
    max_data_diff = -1.0

    if HAS_LIGHTOP:
        # A. 运行 CUDA Fused Kernel
        kernel_counters = expert_counts.clone()
        op.ep_fused_quant_scatter(
            x, topk_ids, expert_map, kernel_counters, 
            aq_out, aq_scale, m_indices, inv_perm, E, ALIGNMENT
        )
        torch.cuda.synchronize()

        # B. 运行 Triton Pipeline (Quant -> Scatter)
        ref_aq_out = torch.empty_like(aq_out)
        ref_aq_scale = torch.empty_like(aq_scale)
        ref_inv_perm = torch.empty_like(inv_perm)
        ref_counters = expert_counts.clone()
        ref_start_loc = torch.zeros((E,), dtype=torch.int32, device=device)
        
        # Step 1: Quant
        ref_q, ref_s = per_token_quant_int8_triton(x)
        # Step 2: Scatter
        ep_scatter(
            ref_q, ref_s, topk_ids, ref_counters, expert_map, ref_start_loc,
            ref_aq_out, ref_aq_scale, torch.full_like(m_indices, -1), ref_inv_perm, alignment=ALIGNMENT
        )
        torch.cuda.synchronize()

        # C. 数据对比 (使用切片防止大 Batch OOM)
        VERIFY_M = min(M, 4096)
        slice_ids = inv_perm[:VERIFY_M].long()
        slice_ref_ids = ref_inv_perm[:VERIFY_M].long()
        
        # 提取数据
        cuda_recon_data = aq_out[slice_ids] 
        cuda_recon_scale = aq_scale[slice_ids]
        ref_recon_data = ref_aq_out[slice_ref_ids]
        ref_recon_scale = ref_aq_scale[slice_ref_ids]
        
        valid_mask = (topk_ids[:VERIFY_M] != -1).unsqueeze(-1).float()
        
        # 对比
        data_diff = (cuda_recon_data.float() - ref_recon_data.float()).abs() * valid_mask
        scale_diff = (cuda_recon_scale - ref_recon_scale).abs() * valid_mask
        
        max_data_diff = data_diff.max().item()
        if max_data_diff == 0 and scale_diff.max().item() < 1e-4:
            is_correct = True
            
        del cuda_recon_data, ref_recon_data, data_diff, scale_diff, ref_aq_out, ref_aq_scale, ref_q, ref_s
        torch.cuda.empty_cache()

    # --- 性能测试 ---
    # 读: x(2B) + topk(4B)
    bytes_read = x.numel() * 2 + topk_ids.numel() * 4
    # 写: out(1B) + scale(4B) + indices(4B) + inv_perm(4B)
    total_valid_tokens = M * K
    bytes_written = total_valid_tokens * (H*1 + 4 + 4 + 4)
    total_gb = (bytes_read + bytes_written) / 1e9
    
    WARMUP, REP = 10, 100
    
    # 准备 Bench Buffer
    bench_aq_out = aq_out 
    bench_scale_out = aq_scale
    bench_m_indices = m_indices
    bench_inv_perm = inv_perm
    bench_start_loc = torch.empty((E), device=device, dtype=torch.int32)
    
    # 1. Triton Bench (Quant + Scatter)
    t_q_buffer = torch.empty_like(x, dtype=torch.int8)
    t_s_buffer = torch.empty((M, 1), device=device, dtype=torch.float32)

    def triton_step():
        per_token_quant_int8_triton(x, x_q=t_q_buffer, scales=t_s_buffer)
        bench_start_loc.zero_() 
        ep_scatter(
            t_q_buffer, t_s_buffer, topk_ids, expert_counts, expert_map, bench_start_loc,
            bench_aq_out, bench_scale_out, bench_m_indices, bench_inv_perm, alignment=ALIGNMENT
        )

    for _ in range(WARMUP): triton_step()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(REP): triton_step()
    end.record()
    torch.cuda.synchronize()
    t_triton = start.elapsed_time(end) / REP * 1000.0
    bw_triton = (total_gb * 1e6) / t_triton

    # 2. CUDA Fused Bench
    t_cuda, bw_cuda = 0.0, 0.0
    if HAS_LIGHTOP:
        counters_list = [expert_counts.clone() for _ in range(REP)]
        for _ in range(WARMUP):
             op.ep_fused_quant_scatter(x, topk_ids, expert_map, expert_counts.clone(), bench_aq_out, bench_scale_out, bench_m_indices, bench_inv_perm, E, ALIGNMENT)
        torch.cuda.synchronize()
        start.record()
        for i in range(REP):
            op.ep_fused_quant_scatter(x, topk_ids, expert_map, counters_list[i], bench_aq_out, bench_scale_out, bench_m_indices, bench_inv_perm, E, ALIGNMENT)
        end.record()
        torch.cuda.synchronize()
        t_cuda = start.elapsed_time(end) / REP * 1000.0
        bw_cuda = (total_gb * 1e6) / t_cuda if t_cuda > 0 else 0

    return {
        "M": M, "Is_Correct": is_correct, "Max_Data_Diff": max_data_diff,
        "Triton_Time_us": t_triton, "Triton_BW_GBs": bw_triton,
        "CUDA_Time_us": t_cuda, "CUDA_BW_GBs": bw_cuda, "Speedup": t_triton/t_cuda if t_cuda>0 else 0
    }

def run_suite():
    print("\n" + "="*85)
    print(" 🚀 MOE Fused Quantization & Scatter Benchmark Suite")
    print("="*85)
    device = torch.device("cuda")
    H, K, E, ALIGNMENT = 7168, 8, 256, 256
    
    # 指定测试的 M 值
    M_values = [256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536]
    
    csv_filename = "moe_fused_benchmark_results.csv"
    headers = [
        "M", "Correctness", "Max_Diff", 
        "Triton_Latency(us)", "Triton_BW(GB/s)", 
        "CUDA_Fused_Latency(us)", "CUDA_Fused_BW(GB/s)", 
        "Speedup(Triton/CUDA)"
    ]
    
    results = []

    print(f"{'M':<8} | {'Triton(us)':<12} | {'Triton(GB/s)':<14} | {'CUDA(us)':<12} | {'CUDA(GB/s)':<12} | {'Speedup':<8} | {'Status':<6}")
    print("-" * 85)

    for M in M_values:
        try:
            res = run_single_benchmark(M, H, K, E, ALIGNMENT, device)
            results.append(res)
            
            status = "PASS" if res["Is_Correct"] else "FAIL"
            print(f"{res['M']:<8} | {res['Triton_Time_us']:<12.2f} | {res['Triton_BW_GBs']:<14.2f} | {res['CUDA_Time_us']:<12.2f} | {res['CUDA_BW_GBs']:<12.2f} | {res['Speedup']:<8.2f} | {status:<6}")
        except RuntimeError as e:
            print(f"{M:<8} | ERROR: {e}")
            results.append({
                "M": M, "Is_Correct": False, "Max_Data_Diff": -1,
                "Triton_Time_us": 0, "Triton_BW_GBs": 0,
                "CUDA_Time_us": 0, "CUDA_BW_GBs": 0, "Speedup": 0
            })
            torch.cuda.empty_cache()

    # 写入 CSV
    print("-" * 85)
    print(f"正在写入结果到 {csv_filename} ...")
    with open(csv_filename, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for r in results:
            writer.writerow({
                "M": r["M"],
                "Correctness": "PASS" if r["Is_Correct"] else "FAIL",
                "Max_Diff": f"{r['Max_Data_Diff']:.4f}",
                "Triton_Latency(us)": f"{r['Triton_Time_us']:.3f}",
                "Triton_BW(GB/s)": f"{r['Triton_BW_GBs']:.3f}",
                "CUDA_Fused_Latency(us)": f"{r['CUDA_Time_us']:.3f}",
                "CUDA_Fused_BW(GB/s)": f"{r['CUDA_BW_GBs']:.3f}",
                "Speedup(Triton/CUDA)": f"{r['Speedup']:.2f}"
            })
    print("✅ 测试完成！")

if __name__ == "__main__":
    run_suite()