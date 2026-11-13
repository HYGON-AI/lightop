# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

import torch
import triton
from common.vllm_reference import vllm_ops
# [确保 lightop 已正确安装]
from lightop.activation import fuse_silu_mul_quant_ep, silu_and_mul_opt
from typing import Optional

# ------------------ 接口 ------------------
def fuse_silu_mul_quant_init(output: torch.Tensor, input: torch.Tensor):
    # Baseline: lightop silu_and_mul + vLLM int8 quant
    silu_and_mul_opt(output, input)
    ref_out, ref_scales, _ = vllm_ops.scaled_int8_quant(output)
    return ref_out, ref_scales

def fuse_silu_mul_quant_opt(input: torch.Tensor,
                            tokens_per_expert: Optional[torch.Tensor] = None): 
    # [修改] 调用 opt
    opt_out, opt_scales = fuse_silu_mul_quant_ep(input, tokens_per_expert) 
    return opt_out, opt_scales

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is required for this benchmark")
device_for_tokens = torch.device("cuda")

# ------------------ 数据 ------------------
FIXED_TOKENS_MAP = {
    "case_1": torch.tensor([ 1,  3,  1,  9, 20,  1,  1,  0,  1,  1,  1, 11,  0,  2,  4,  1], device=device_for_tokens, dtype=torch.int32),
    "case_2": torch.tensor([16, 15, 14, 16, 12, 18, 15, 15, 16, 14, 15, 17, 12, 18, 10, 10], device=device_for_tokens, dtype=torch.int32),
    "case_3": torch.tensor([30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30], device=device_for_tokens, dtype=torch.int32),
    "all_zeros": torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], device=device_for_tokens, dtype=torch.int32),
}

# ------------------ 正确性 ------------------
def check_correctness_once(tokens_per_expert: torch.Tensor, T_dim: int, H_dim: int, 
                           rtol=1e-2, atol=1e-2):
    device = torch.device("cuda")
    dtype = torch.bfloat16
    
    E_dim = len(tokens_per_expert) 
    max_tokens_needed = tokens_per_expert.max().item()
    
    # 如果 T_dim 小于最大需要的 token 数，无法存放数据，直接跳过
    if T_dim < max_tokens_needed:
        return 
    
    d_dim = H_dim // 2

    # 构造 Mask
    t_indices = torch.arange(T_dim, device=device, dtype=torch.int32).unsqueeze(0)
    mask = (t_indices < tokens_per_expert.unsqueeze(1).to(torch.int32)) # (E, T)

    # 初始化输入
    input_3d = torch.randn((E_dim, T_dim, H_dim), device=device, dtype=dtype)
    
    # [修改2] 将 Padding 区域清零 (Good Practice)
    input_3d[~mask.unsqueeze(-1).expand_as(input_3d)] = 0.0

    out_init_3d = torch.empty((E_dim, T_dim, d_dim), device=device, dtype=dtype)
    
    input_2d_ref = input_3d.view(-1, H_dim)
    out_init_2d_ref = out_init_3d.view(-1, d_dim)
    
    # --- Baseline 计算 ---
    out_init, scales_init = fuse_silu_mul_quant_init(out_init_2d_ref, input_2d_ref)
    
    out_init = out_init.view(E_dim, T_dim, d_dim)
    scales_init = scales_init.view(E_dim, T_dim, 1)

    mask_for_out = mask.unsqueeze(-1).expand(E_dim, T_dim, d_dim)
    mask_for_scales = mask.unsqueeze(-1)

    scales_init_valid = scales_init.masked_select(mask_for_scales)
    out_init_valid = out_init.masked_select(mask_for_out)
    
    common_error_msg = f"Shapes (E,T,d): ({E_dim}, {T_dim}, {d_dim})\nTokens: {tokens_per_expert.tolist()}"

    # --- 检查 EP opt ---
    out_opt, scales_opt = fuse_silu_mul_quant_opt(input_3d, tokens_per_expert)
    
    # 检查 Scale
    scales_opt_valid = scales_opt.masked_select(mask_for_scales)
    if not torch.allclose(scales_init_valid, scales_opt_valid, rtol=rtol, atol=atol):
        diff = (scales_init_valid - scales_opt_valid).abs()
        max_abs = diff.max().item() if diff.numel() > 0 else 0.0
        print(common_error_msg)
        print("scales_init (valid sample):", scales_init_valid.flatten()[:5])
        print("scales_opt (valid sample):", scales_opt_valid.flatten()[:5])
        raise AssertionError(f"[EP opt] scales_max_abs={max_abs:.4e}")

    # 检查 Output
    out_opt_valid = out_opt.masked_select(mask_for_out)

    if not torch.allclose(out_init_valid, out_opt_valid, rtol=rtol, atol=atol):
        diff = (out_init_valid - out_opt_valid).abs()
        max_abs = diff.max().item() if diff.numel() > 0 else 0.0
        print(common_error_msg)
        raise AssertionError(f"[EP opt] output_max_abs={max_abs:.4e}")

# 正确性验证
def run_correctness_suite():
    # 测试集配置
    T_dim_list = [8192]
    H_dim_list = [4096]
    
    print("Running correctness suite (EP opt vs Baseline)...")
    for T_dim in T_dim_list:
        for H_dim in H_dim_list:
            for case_name, tokens in FIXED_TOKENS_MAP.items():
                # print(f"Checking {case_name}...")
                check_correctness_once(tokens_per_expert=tokens, T_dim=T_dim, H_dim=H_dim)

    print("[Correctness] All tests passed.")
    print("---")

# ------------------ 性能 (只测 EP opt) ------------------
expert_list = [16]      
# numtokens_list = [8192] 
numtokens_list = [8192, 4096, 2048, 1024] 
hiddensizes_list = [4096]

configs = [(experts, num_tokens, hidden_sizes) for experts in expert_list for num_tokens in numtokens_list for hidden_sizes in hiddensizes_list]
test_cases_keys = sorted(list(FIXED_TOKENS_MAP.keys())) 

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["shape"],                        
        x_vals=[f"[{experts},{num_tokens},{hidden_sizes}]" for experts, num_tokens, hidden_sizes in configs], 
        line_arg="test_case", 
        line_vals=[f"opt_{key}" for key in test_cases_keys],
        line_names=[f"EP opt ({key})" for key in test_cases_keys], 
        styles=[("blue", "-"), ("red", "-"), ("green", "-"), ("orange", "-"), 
                ("purple", "-"), ("brown", "-"), ("pink", "-"), ("gray", "-"), ("cyan", "-")],
        ylabel="us",
        plot_name="fuse-silu-mul-quant-ep-opt-only",
        args={},
    )
)

def benchmark(shape, test_case, warmup=10):
    E_dim, num_tokens, hidden_sizes = map(int, shape.strip("[]").split(","))
    
    dtype = torch.bfloat16
    device = torch.device("cuda")

    # [修改] Benchmark 时保持输入 Padding 为 0 是个好习惯
    input_3d = torch.randn((E_dim, num_tokens, hidden_sizes), device=device, dtype=dtype)
    
    key = test_case.replace("opt_", "")
    
    if key not in FIXED_TOKENS_MAP:
         raise ValueError(f"Unknown test_case: {test_case} (key: {key})")
    
    tokens = FIXED_TOKENS_MAP[key]

    if E_dim != len(tokens):
          raise ValueError(f"Benchmark E_dim ({E_dim}) does not match token length ({len(tokens)})")
    
    f = lambda: fuse_silu_mul_quant_opt(input_3d, tokens)

    for _ in range(warmup):
        f()
    torch.cuda.synchronize()
    ms, min_ms, max_ms = triton.testing.do_bench(f, quantiles=[0.5, 0.2, 0.8])

    return 1000 * ms, 1000 * max_ms, 1000 * min_ms

# ------------------ 入口 ------------------
if __name__ == "__main__":
    run_correctness_suite()
    print(f"Running Performance Benchmarks (EP opt Only) for {len(FIXED_TOKENS_MAP)} cases...")
    benchmark.run(print_data=True, warmup=100)
