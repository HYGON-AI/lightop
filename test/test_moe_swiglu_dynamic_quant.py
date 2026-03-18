# -*- coding: UTF-8 -*-
import torch
import time
import numpy as np

import lightop
from lightop import miopen_moe_swiglu_dynamic_quant
from lightop import moe_swiglu_dynamic_quant
# 既然参考实现已经用 torch.chunk 和 silu 替代了，fuse_silu_and_mul 就不强制需要了，但保留导入防报错
from lightop import fuse_silu_and_mul 


def smooth_per_token_dynamic_quant(
    hidden_states: torch.Tensor, 
    smooth_scale: torch.Tensor, 
    dst_torch_dtype=torch.int8
):
    """单 Expert 内部的 Smooth + 动态量化核心逻辑"""
    max_dtype_val = 1.0
    if dst_torch_dtype == torch.int8:
        max_dtype_val = 127.0
    elif dst_torch_dtype == torch.float8_e4m3fn:
        max_dtype_val = 448.0
    else:
        raise ValueError(f"dst_torch_dtype {dst_torch_dtype} is not supported")

    # [num_tokens, hidden_size]
    ori_shape = hidden_states.shape
    hidden_states = hidden_states.contiguous().view(ori_shape[0], -1).to(torch.float32)

    # [1, hidden_size]
    smooth_scale = smooth_scale.contiguous().view(1, -1)

    # [num_tokens, hidden_size]
    smoothed_input = torch.mul(hidden_states, smooth_scale)

    # [num_tokens, 1], 1 / max
    per_token_max = torch.max(smoothed_input.abs(), -1, keepdim=True)[0].reciprocal()

    # [num_tokens, 1], max_dtype_val / max
    per_token_scale = per_token_max * max_dtype_val

    # [num_tokens, hidden_size], quantized
    # quant_tokens_fp32 = torch.mul(smooth_scale, per_token_scale).clamp(-max_dtype_val, max_dtype_val)
    quant_tokens_fp32 = torch.mul(smoothed_input, per_token_scale).clamp(-max_dtype_val, max_dtype_val) #lj
    if dst_torch_dtype == torch.int8:
        quant_tokens_fp32 = quant_tokens_fp32.round()

    # float32 --> int8 / float8
    quant_tokens = quant_tokens_fp32.type(dst_torch_dtype).view(ori_shape)

    # max_dtype_val / max --> max / max_dtype_val
    per_token_scale = per_token_scale.reciprocal().view(ori_shape[0])

    return quant_tokens, per_token_scale


def ref_moe_swiglu_dynamic_quant(scatter_tokens, experts_smooth_scale, experts_token_count, experts_token_offset, dst_dtype=torch.int8):
    """
    MoE SwiGLU + Dynamic Quant 的纯 PyTorch 标准参考实现 (Baseline)
    按 Expert 切片分配 Token，解决广播维度报错问题。
    """
    leading, d2 = scatter_tokens.shape
    d = d2 // 2
    
    # 预分配输出 Tensor
    quant_tokens = torch.empty(leading, d, dtype=dst_dtype, device=scatter_tokens.device)
    per_token_scale = torch.empty((leading,), dtype=torch.float32, device=scatter_tokens.device)

    # 1. SwiGLU 激活计算
    x1, x2 = torch.chunk(scatter_tokens, 2, dim=-1)
    swiglu_tokens = torch.mul(torch.nn.functional.silu(x1), x2)

    # 2. Per-expert 动态量化
    num_experts = experts_smooth_scale.shape[0]
    for expert_idx in range(num_experts):
        count = experts_token_count[expert_idx].item()
        if count == 0:
            continue
            
        start = experts_token_offset[expert_idx].item()
        end = start + count

        # 对当前 Expert 的 Token 进行 Smooth 和量化
        q_toks, scales = smooth_per_token_dynamic_quant(
            swiglu_tokens[start:end], 
            experts_smooth_scale[expert_idx], 
            dst_torch_dtype=dst_dtype
        )
        
        # 写回总 Tensor
        quant_tokens[start:end] = q_toks
        per_token_scale[start:end] = scales

    return quant_tokens, per_token_scale


def get_moe_swiglu_data(hidden_size, num_tokens, world_size, rank, num_shared_experts,
                        dp_size, sp_size, num_experts, topk, ep_size, use_half=False):
    hidden_size = int(hidden_size)
    num_tokens = int(num_tokens)
    num_experts = int(num_experts)
    topk = int(topk)
    ep_size = int(ep_size)

    local_experts = num_experts // ep_size
    local_tokens = (num_tokens * topk) // ep_size

    # 输入数据：BF16
    scatter_tokens = torch.randn(local_tokens, hidden_size * 2, dtype=torch.bfloat16, device='cuda')
    # Smooth 权重：FP32 (每个专家一份)
    smooth = torch.randn(local_experts, hidden_size, dtype=torch.float32, device='cuda')

    base_count = local_tokens // local_experts
    remainder = local_tokens % local_experts
    counts = [base_count + 1] * remainder + [base_count] * (local_experts - remainder)
    experts_tokens_count = torch.tensor(counts, dtype=torch.int32, device='cuda')

    offsets = [0] * local_experts
    current_offset = 0
    for i in range(local_experts):
        offsets[i] = current_offset
        current_offset += counts[i]
    exports_tokens_start = torch.tensor(offsets, dtype=torch.int32, device='cuda')

    return scatter_tokens, smooth, experts_tokens_count, exports_tokens_start


def test_dual_performance(hidden_size, num_tokens, world_size, rank, num_shared_experts, dp_size, sp_size,
                                     num_experts, topk, ep_size, use_half=False):
    
    scatter_tokens, smooth, experts_tokens_count, exports_tokens_start = get_moe_swiglu_data(
        hidden_size, num_tokens, world_size, rank, num_shared_experts,
        dp_size, sp_size, num_experts, topk, ep_size, use_half
    )

    if scatter_tokens.size(0) == 0:
        return

    # --- 1. 正确性校验 ---
    # PyTorch 纯计算 Baseline (修正了维度问题)
    y_ref, s_ref = ref_moe_swiglu_dynamic_quant(
        scatter_tokens, smooth, experts_tokens_count, exports_tokens_start, torch.int8
    )

    # 你的自定义优化算子 Target
    y_target, s_target = moe_swiglu_dynamic_quant(
        scatter_tokens, smooth, experts_tokens_count, exports_tokens_start, 1.0
    )

    # 注意：s_target 可能需要 squeeze() 才能与 s_ref 形状完全对齐，视你底层 C++ 实现而定
    # 如果遇到 s_target 是 [N, 1] 报错，可以改成 s_target.view_as(s_ref)
    out_diff = torch.abs(y_ref.float() - y_target.float()).max().item()
    scale_diff = torch.abs(s_ref.view(-1) - s_target.view(-1)).max().item()

    # --- 2. 性能测试 ---
    num_iters = 10
    
    # Warmup
    for _ in range(3):
        miopen_moe_swiglu_dynamic_quant(scatter_tokens, smooth, experts_tokens_count, exports_tokens_start, 1.0)
        moe_swiglu_dynamic_quant(scatter_tokens, smooth, experts_tokens_count, exports_tokens_start, 1.0)
    torch.cuda.synchronize()

    # Benchmark MiOpen
    t1_start = time.time()
    for _ in range(num_iters):
        miopen_moe_swiglu_dynamic_quant(scatter_tokens, smooth, experts_tokens_count, exports_tokens_start, 1.0)
    torch.cuda.synchronize()
    avg_t1 = (time.time() - t1_start) * 1_000_000 / num_iters

    # Benchmark Custom
    t2_start = time.time()
    for _ in range(num_iters):
        moe_swiglu_dynamic_quant(scatter_tokens, smooth, experts_tokens_count, exports_tokens_start, 1.0)
    torch.cuda.synchronize()
    avg_t2 = (time.time() - t2_start) * 1_000_000 / num_iters

    # --- 3. 带宽、吞吐量与加速比计算 ---
    local_experts = num_experts // ep_size
    local_tokens = scatter_tokens.size(0)
    
    bytes_read = (local_tokens * hidden_size * 2 * 2) + (local_experts * hidden_size * 4) + (local_experts * 8)
    bytes_write = (local_tokens * hidden_size * 1) + (local_tokens * 4)
    total_bytes = bytes_read + bytes_write
    
    # 防止时间为0导致除零错误
    avg_t1_s = max(avg_t1 / 1_000_000, 1e-9)
    avg_t2_s = max(avg_t2 / 1_000_000, 1e-9)
    
    # 计算内存带宽 (GB/s)
    bw_ref = (total_bytes / 1e9) / avg_t1_s
    bw_target = (total_bytes / 1e9) / avg_t2_s

    # 计算 Token 吞吐量 (KTokens/s)
    thr_ref = (local_tokens / 1000) / avg_t1_s
    thr_target = (local_tokens / 1000) / avg_t2_s

    # 计算加速比 (Speedup)
    speedup = avg_t1 / max(avg_t2, 1e-9)

    # 判定
    status = "PASS" if out_diff <= 1 and scale_diff < 1e-4 else "FAIL"
    
    # 打印排版微调
    print(f"Tokens:{num_tokens:<5} | Local:{local_tokens:<5} | "
          f"MiOpen: {avg_t1:>6.2f}us, {bw_ref:>6.1f} GB/s | "
          f"Target: {avg_t2:>6.2f}us, {bw_target:>6.1f} GB/s | "
          f"Speedup: \033[1;32m{speedup:>4.2f}x\033[0m | "
          f"Diff:{out_diff:.0f}/{scale_diff:.1e} | [{status}]")


if __name__ == '__main__':
    NUM_TOKENS_LIST = [
        32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 128,
        144, 160, 176, 192, 208, 224, 240, 256,
        288, 320, 352, 384, 416, 448, 480, 512,
        640, 768, 896, 1024,
        1152, 1280, 1408, 1536, 1664, 1792, 1920, 2048,
        2176, 2304, 2432, 2560, 2688, 2816, 2944, 3072,
        3200, 3328, 3456, 3584, 3712, 3840, 3968, 4096,
        6144, 8192, 10240, 12288, 14336, 16384, 18432, 20480,
        22528, 24576, 26624, 28672, 32768, 
        # 65536, 131072
    ]
    
    # 把 hidden_size 变成一个列表，想测多少个就加多少个
    HIDDEN_SIZE_LIST = [1024]

    print(f"{'='*35} MoE SwiGLU 双算子性能与精度对齐测试 {'='*35}")
    
    config = {
        "world_size": 8,
        "rank": 0,
        "num_shared_experts": 0,
        "dp_size": 1,
        "sp_size": 1,
        "num_experts": 64,
        "topk": 4,
        "ep_size": 8,
    }

    # 外层循环遍历 hidden_size
    for hs in HIDDEN_SIZE_LIST:
        print(f"\n[{'#'*20} Testing hidden_size = {hs} {'#'*20}]")
        # 内层循环遍历 tokens
        for t in NUM_TOKENS_LIST:
            test_dual_performance(
                hidden_size=hs,  # 传入当前的 hidden_size
                num_tokens=t,
                world_size=config["world_size"],
                rank=config["rank"],
                num_shared_experts=config["num_shared_experts"],
                dp_size=config["dp_size"],
                sp_size=config["sp_size"],
                num_experts=config["num_experts"],
                topk=config["topk"],
                ep_size=config["ep_size"]
            )
