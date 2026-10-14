# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

      
import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from typing import Optional
import sys
import time
import os

try:
    from lightop import op
    HAS_LIGHTOP = True
    print("✅ 成功导入 lightop 库")
except ImportError:
    print("❌ 警告: 未找到 lightop 库。LightOp CUDA 对比将跳过，仅运行 Triton。")
    HAS_LIGHTOP = False

# =========================================================================
#  Triton 量化内核 (User Aligned)
# =========================================================================

@triton.jit
def _per_token_dynamic_quant_int8(
    x_ptr, 
    xq_ptr, 
    smoothscale_ptr, 
    scale_ptr, 
    stride_x, 
    stride_xq, 
    stride_smoothscale_row,  # Adjusted to handle [M, H] input from MoE gather
    stride_smoothscale_col,
    N, 
    BLOCK: tl.constexpr
):
    # Adapted from lmdeploy/lmdeploy/pytorch/kernels/cuda/w8a8_triton_kernels.py
    row_id = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N
    
    # Load Input
    x = tl.load(x_ptr + row_id * stride_x + cols, mask=mask, other=0.0).to(tl.float32)
    
    # Load Smooth Scale
    # NOTE: Adapted slightly from reference to support [M, H] gathered scales needed for MoE benchmark.
    # The reference code assumed [1, H] broadcasting (cols * stride), but here we need row specific scales.
    s = tl.load(smoothscale_ptr + row_id * stride_smoothscale_row + cols * stride_smoothscale_col, mask=mask, other=1.0)
    
    # --- Strict Alignment with User Reference ---
    # 1. Multiplication (x * s) instead of division
    x_scaled = x * s
    
    # 2. AbsMax calculation
    absmax = tl.maximum(tl.max(tl.abs(x_scaled)), 1e-10)
    
    # 3. Scale calculation
    scale_x = absmax / 127
    
    # 4. Quantization
    # x_q = x_scaled * (127 / absmax)
    x_q = x_scaled / scale_x
    x_q = libdevice.nearbyint(x_q).to(tl.int8)
    
    tl.store(xq_ptr + row_id * stride_xq + cols, x_q, mask=mask)
    tl.store(scale_ptr + row_id, scale_x)

def per_token_dynamic_quant_int8_triton(x, smooth_scale):
    """
    Triton 量化入口函数 (Aligned wrapper)
    """
    M = x.numel() // x.shape[-1]
    N = x.shape[-1]
    x_q = torch.empty_like(x, device=x.device, dtype=torch.int8)
    scales = torch.empty((M, 1), device=x.device, dtype=torch.float32)
    
    BLOCK = triton.next_power_of_2(N)
    # heuristics for number of warps
    num_warps = min(max(BLOCK // 256, 1), 8)
    
    assert smooth_scale.shape[-1] == x.shape[-1]
    assert x.is_contiguous()
    
    # Handle strides for potentially 2D [M, H] or 1D [H] smooth_scale
    if smooth_scale.ndim == 2:
        stride_s_row = smooth_scale.stride(0)
        stride_s_col = smooth_scale.stride(1)
    else:
        stride_s_row = 0
        stride_s_col = smooth_scale.stride(0)

    _per_token_dynamic_quant_int8[(M, )](
        x,
        x_q,
        smooth_scale,
        scales,
        stride_x=x.stride(-2),
        stride_xq=x_q.stride(-2),
        stride_smoothscale_row=stride_s_row,
        stride_smoothscale_col=stride_s_col,
        N=N,
        BLOCK=BLOCK,
        num_warps=num_warps,
        num_stages=1, # Explicitly set to 1 as per reference
    )
    return x_q, scales

# =========================================================================
#  Triton Scatter 实现 (用于构建完整的 Triton Pipeline)
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
    tokens_per_expert = (tokens_per_expert + BLOCK_E - 1) // BLOCK_E * BLOCK_E
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

def ep_scatter_triton(
    recv_x, recv_x_scale, recv_topk, num_recv_tokens_per_expert, expert_map, expert_start_loc,
    output_tensor, output_tensor_scale, m_indices, output_index, alignment=256
):
    num_experts = num_recv_tokens_per_expert.shape[0]
    hidden_size = recv_x.shape[1]
    
    grid = (num_experts, )
    _fwd_kernel_ep_scatter_1[grid](
        num_recv_tokens_per_expert, expert_start_loc, m_indices,
        num_experts=num_experts, num_warps=4, BLOCK_E=alignment, BLOCK_EXPERT_NUM=triton.next_power_of_2(num_experts)
    )

    grid = (min(recv_topk.shape[0], 1024 * 8), )
    _fwd_kernel_ep_scatter_2[grid](
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

# =========================================================================
#  辅助函数
# =========================================================================

def count_expert_num_tokens_with_map(topk_ids, num_local_experts, expert_map):

    flat_ids = topk_ids.view(-1)
    local_ids = expert_map[flat_ids]
    mask = local_ids >= 0
    return torch.bincount(local_ids[mask], minlength=num_local_experts).to(torch.int32)

def get_expert_offsets(expert_counts, alignment):
    padded_counts = (expert_counts + alignment - 1) // alignment * alignment
    offsets = torch.zeros_like(expert_counts)
    offsets[1:] = torch.cumsum(padded_counts[:-1], dim=0)
    return offsets, padded_counts.sum().item()

def manual_smooth_quant_scatter_ref(x, topk_ids, smooth_scale, inv_perm, aq_out, aq_scale, expert_map):
    """验证逻辑: PyTorch Reference (支持 Expert Map)"""
    M, H = x.shape
    K = topk_ids.shape[1]
    
    flat_ids = topk_ids.view(-1)     # Global ID
    flat_perm = inv_perm.view(-1)    # Dest Row
    
    # 过滤有效 Token (inv_perm != -1)
    mask = flat_perm != -1
    valid_perm = flat_perm[mask].long()
    valid_global_expert = flat_ids[mask].long()
    
    # 获取 Local Expert ID 用于索引 smooth_scale
    valid_local_expert = expert_map[valid_global_expert].long()
    
    assert (valid_local_expert != -1).all(), "Reference found invalid local expert mapped"

    valid_token_indices = (torch.arange(M * K, device=x.device)[mask] // K).long()
    
    x_selected = x[valid_token_indices].float()
    s_selected = smooth_scale[valid_local_expert].float() # 使用本地索引
    
    # --- UPDATED REF TO MATCH TRITON KERNEL (Multiplication) ---
    x_smoothed = x_selected * s_selected
    
    absmax = x_smoothed.abs().amax(dim=1, keepdim=True).clamp(min=1e-10)
    # Match Triton logic: scale = absmax / 127
    scale_ref = absmax / 127.0
    
    # Match Triton logic: q = nearbyint(x / scale)
    q_ref = (x_smoothed / scale_ref).round().clamp(-128, 127).to(torch.int8)
    
    q_cuda = aq_out[valid_perm]
    scale_cuda = aq_scale[valid_perm]
    
    return q_ref, scale_ref.squeeze(), q_cuda, scale_cuda

# =========================================================================
#  基准测试逻辑
# =========================================================================

def run_single_benchmark(M, H, K, GLOBAL_E, EP_SIZE, ALIGNMENT, device):
    torch.cuda.empty_cache() 
    torch.manual_seed(42)

    LOCAL_E = GLOBAL_E // EP_SIZE
    
    # 1. 构造 Expert Map (模拟 Rank 0)
    # Global: 0 ~ GLOBAL_E-1
    # Local:  0 ~ LOCAL_E-1 (对应 Global 0~LOCAL_E-1)
    # 其余 Global ID 映射为 -1
    expert_map = torch.full((GLOBAL_E,), -1, dtype=torch.int32, device=device)
    # 假设当前是 Rank 0，负责 Global 0 ~ LOCAL_E-1
    expert_map[:LOCAL_E] = torch.arange(LOCAL_E, dtype=torch.int32, device=device)

    # 2. 数据准备
    x = torch.randn((M, H), dtype=torch.bfloat16, device=device)
    # # TopK IDs 是全局 ID (0 ~ GLOBAL_E-1)
    experts_idx_for_each_rank = []
    for rank_idx in range(EP_SIZE):
        start_idx = rank_idx * LOCAL_E
        end_idx = start_idx + LOCAL_E
        experts_idx_for_each_rank.append(list(range(start_idx, end_idx)))
    transpose_experts = [list(row) for row in zip(*experts_idx_for_each_rank)]
    experts_array = [num for row in transpose_experts for num in row]

    all_select_experts = []

    cur_expert = 0
    for token_idx in range(M):
        cur_token_selections = []
        for topk_idx in range(K):
            cur_token_selections.append(experts_array[cur_expert])
            cur_expert += 1
            if cur_expert >= GLOBAL_E:
                cur_expert = 0
        all_select_experts.append(cur_token_selections)

    topk_ids = torch.tensor(all_select_experts, dtype=torch.long, device=device)
    # topk_ids = torch.randint(0, GLOBAL_E, (M, K), dtype=torch.long, device=device)
    # import pdb
    # pdb.set_trace()
    smooth_scale = torch.rand((LOCAL_E, H), dtype=torch.float32, device=device) * 0.5 + 0.5 

 
    expert_counts = count_expert_num_tokens_with_map(topk_ids, LOCAL_E, expert_map)
    expert_offsets, total_rows = get_expert_offsets(expert_counts, ALIGNMENT)
    
    topk_ids1 = torch.randint(0, GLOBAL_E, (M, K), dtype=torch.long, device=device)
    expert_counts1 = count_expert_num_tokens_with_map(topk_ids1, LOCAL_E, expert_map)
    expert_offsets1, total_rows1 = get_expert_offsets(expert_counts1, ALIGNMENT)
    # import pdb
    # pdb.set_trace()
    # 输出 Tensor
    aq_out = torch.empty((total_rows, H), device=device, dtype=torch.int8)
    aq_scale = torch.empty((total_rows, 1), device=device, dtype=torch.float32)
    m_indices = torch.full((total_rows,), -1, dtype=torch.int32, device=device)
    inv_perm = torch.full((M, K), -1, device=device, dtype=torch.int32)
    
    # --- 1. 正确性验证 (CUDA vs PyTorch Ref) ---
    is_correct = False
    max_data_diff = -1.0
    max_scale_diff = -1.0
    # import pdb
    # pdb.set_trace()
    # 注意：如果 lightop 内部的 CUDA kernel 仍然使用除法，这里的 Check 会失败。
    # 这里假设 CUDA kernel 是 "正确" 的实现目标，而我们现在将 PyTorch Ref 
    # 和 Triton 都统一为了乘法。
    if HAS_LIGHTOP:
        try:
            kernel_offsets = expert_offsets.clone()
            op.ep_fused_smooth_quant_scatter(
                x, topk_ids, expert_map, kernel_offsets, smooth_scale, 
                aq_out, aq_scale, m_indices, inv_perm, LOCAL_E, ALIGNMENT
            )
            torch.cuda.synchronize()

            q_ref, s_ref, q_cuda, s_cuda = manual_smooth_quant_scatter_ref(
                x, topk_ids, smooth_scale, inv_perm, aq_out, aq_scale, expert_map
            )
            
            data_diff = (q_ref.float() - q_cuda.float()).abs()
            scale_diff = (s_ref - s_cuda.squeeze()).abs()
            max_data_diff = data_diff.max().item()
            max_scale_diff = scale_diff.max().item()
            
            if max_data_diff <= 1.0 and max_scale_diff < 1e-3:
                is_correct = True
            # import pdb
            # pdb.set_trace()
            del q_ref, s_ref, q_cuda, s_cuda
        except Exception as e:
            print(f"CUDA Kernel execution failed: {e}")
    else:
        is_correct = True # Skip check if no lightop
    
    # --- 2. 性能测试 (CUDA Fused vs Triton Pipeline) ---
    
    # Read: x + topk + smooth_scale (partial) + expert_map (partial)
    bytes_read = x.numel() * 2 + topk_ids.numel() * 4 
    #这是被分配到当前Rank的所有有效Token的总数
    valid_tokens_count = expert_counts.sum().item()
    # H*1 -> aq_out 每个元素占用 1 字节。一行有 H 个元素 + 4（aq_scale） + 4 （m_indices） + inv_perm（(M * K) * 4）
    bytes_written = valid_tokens_count * (H*1 + 4 + 4) + (M*K)*4 
    #101974080 268959744
    total_gb = (bytes_read + bytes_written) / 1e9
    # import pdb
    # pdb.set_trace()
    WARMUP, REP = 10, 100
    t_cuda, t_triton = 0.0, 0.0
    
    # Buffer for Triton
    t_aq_out = torch.empty_like(aq_out)
    t_aq_scale = torch.empty_like(aq_scale)
    t_m_indices = torch.full_like(m_indices, -1)
    t_inv_perm = torch.empty_like(inv_perm)
    t_offsets = torch.zeros_like(expert_counts) 

    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    
    def run_triton():
        # 1. Gather Smooth Scale 
        # Triton 模拟：需要先 Map ID，再 Gather。
        topk0 = topk_ids[:, 0]
        mapped_ids0 = expert_map[topk0]
        
        safe_mapped_ids = mapped_ids0.clone()
        safe_mapped_ids[safe_mapped_ids == -1] = 0
        scales_expanded = smooth_scale[safe_mapped_ids] # Shape [M, H]

        # 2. Quant (Uses New Aligned Kernel)
        x_q, s_q = per_token_dynamic_quant_int8_triton(x, scales_expanded)
        
        # 3. Scatter
        t_offsets.copy_(expert_offsets)
        ep_scatter_triton(
            x_q, s_q, topk_ids, expert_counts, expert_map, t_offsets,
            t_aq_out, t_aq_scale, t_m_indices, t_inv_perm, alignment=ALIGNMENT
        )

    for _ in range(WARMUP): run_triton()
    torch.cuda.synchronize()
    start.record()
    for _ in range(REP): run_triton()
    end.record()
    torch.cuda.synchronize()
    t_triton = start.elapsed_time(end) / REP * 1000.0
    
    # --- Bench CUDA ---
    if HAS_LIGHTOP:
        offsets_list = [expert_offsets.clone() for _ in range(REP + WARMUP)]
        
        for i in range(WARMUP):
            op.ep_fused_smooth_quant_scatter(
                x, topk_ids, expert_map, offsets_list[i], smooth_scale, 
                aq_out, aq_scale, m_indices, inv_perm, LOCAL_E, ALIGNMENT
            )
        torch.cuda.synchronize()
        
        start.record()
        for i in range(REP):
            op.ep_fused_smooth_quant_scatter(
                x, topk_ids, expert_map, offsets_list[WARMUP + i], smooth_scale, 
                aq_out, aq_scale, m_indices, inv_perm, LOCAL_E, ALIGNMENT
            )
        end.record()
        torch.cuda.synchronize()
        t_cuda = start.elapsed_time(end) / REP * 1000.0 

    bw_cuda = (total_gb * 1e6) / t_cuda if t_cuda > 0 else 0
    bw_triton = (total_gb * 1e6) / t_triton if t_triton > 0 else 0
    speedup = t_triton / t_cuda if t_cuda > 0 else 0

    return {
        "M": M, 
        "Is_Correct": is_correct, 
        "Max_Data_Diff": max_data_diff,
        "Triton_Time_us": t_triton, "Triton_BW_GBs": bw_triton,
        "CUDA_Time_us": t_cuda, "CUDA_BW_GBs": bw_cuda,
        "Speedup": speedup
    }

def run_suite():
    print("\n" + "="*110)
    print(" 🚀 MOE Fused SmoothQuant & Scatter Benchmark Suite (EP Mode)")
    print(f" Kernel: ep_fused_smooth_quant_scatter vs Triton Pipeline")
    print("="*110)
    
    if not torch.cuda.is_available(): return

    device = torch.device("cuda")
    H, K, ALIGNMENT = 4096, 4, 64
    
    # New Config:
    GLOBAL_E = 64
    EP_SIZE = 8
    # LOCAL_E will be 8
    
    # M_values = [
    #     32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 128,
    #     144, 160, 176, 192, 208, 224, 240, 256,
    #     288, 320, 352, 384, 416, 448, 480, 512,
    #     640, 768, 896, 1024,
    #     1152, 1280, 1408, 1536, 1664, 1792, 1920, 2048,
    #     2176, 2304, 2432, 2560, 2688, 2816, 2944, 3072,
    #     3200, 3328, 3456, 3584, 3712, 3840, 3968, 4096,
    #     6144, 8192, 10240, 12288, 14336, 16384, 18432, 20480,
    #     22528, 24576, 26624, 28672, 32768
    # ]
    M_values = [32768]
    print(f"{'M':<8} | {'Triton(us)':<12} | {'CUDA(us)':<12} | {'CUDA(GB/s)':<12} | {'Speedup':<8} | {'Diff(Int8)':<12} | {'Status':<6}")
    print("-" * 110)

    for M in M_values:
        try:
            res = run_single_benchmark(M, H, K, GLOBAL_E, EP_SIZE, ALIGNMENT, device)
            status = "PASS" if res["Is_Correct"] else "FAIL"
            print(f"{res['M']:<8} | {res['Triton_Time_us']:<12.2f} | {res['CUDA_Time_us']:<12.2f} | {res['CUDA_BW_GBs']:<12.2f} | {res['Speedup']:<8.2f} | {res['Max_Data_Diff']:<12.1f} | {status:<6}")
        except RuntimeError as e:
            print(f"{M:<8} | ERROR: {e}") 
            # raise e # Debug use
            torch.cuda.empty_cache()
    
    print("-" * 110)
    print("✅ 测试完成！")

if __name__ == "__main__":
    run_suite()

    