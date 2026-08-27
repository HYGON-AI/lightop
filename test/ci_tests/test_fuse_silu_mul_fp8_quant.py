# SPDX-License-Identifier: Apache-2.0
# vllm 缺失时 RMSNorm 等注解名不存在, 延迟求值避免模块导入即崩
from __future__ import annotations

import pickle as pkl
import time
import csv
from collections.abc import Iterable
from dataclasses import dataclass
from itertools import product
from typing import Callable, Optional

import torch
import pytest
import torch.utils.benchmark as TBenchmark
from torch.utils.benchmark import Measurement as TMeasurement
from tqdm import tqdm

try:
    import vllm._custom_ops as ops
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.model_executor.layers.layernorm import RMSNorm
    from lightop.norm import rms_norm_per_token_fp8_quant
    HAS_VLLM = True
except ImportError:
    HAS_VLLM = False

import triton
import triton.language as tl

# ------------------ fp8 torch ------------------
def per_token_quant_fp8_torch(x):
    target_dtype = torch.float8_e4m3fn
    finfo = torch.finfo(target_dtype)
    fp8_min = finfo.min
    fp8_max = finfo.max
    
    x_fp32 = x.to(torch.float32)
    x_abs = x_fp32.abs()
    max_val = x_abs.max(dim=-1, keepdim=True).values

    scales = max_val / fp8_max

    x_scaled = x / scales
    
    x_scaled = torch.clamp(x_scaled, min=fp8_min, max=fp8_max)
    x_q = x_scaled.to(target_dtype)

    return x_q, scales
# ------------------ fp8 torch ------------------

# ------------------ fp8 triton ------------------
@triton.jit
def _per_token_quant_fp8(
    x_ptr,
    xq_ptr,
    scale_ptr,
    stride_x,
    stride_xq,
    N,
    BLOCK: tl.constexpr,
    fp8_min,
    fp8_max
):
    row_id = tl.program_id(0)

    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(x_ptr + row_id * stride_x + cols, mask=mask,
                other=0.0).to(tl.float32)
    absmax = tl.maximum(tl.max(tl.abs(x)), 1e-10)
    scale_x = absmax / fp8_max
    x_q = tl.clamp(x / scale_x, min=fp8_min, max=fp8_max).to(xq_ptr.dtype.element_ty)
    tl.store(xq_ptr + row_id * stride_xq + cols, x_q, mask=mask)
    tl.store(scale_ptr + row_id, scale_x)


def per_token_quant_fp8(x):
    M = x.numel() // x.shape[-1]
    N = x.shape[-1]
    x_q = torch.empty_like(x, device=x.device, dtype=torch.float8_e4m3fn)
    scales = torch.empty(x.shape[:-1] + (1, ),
                         device=x.device,
                         dtype=torch.float32)
    BLOCK = triton.next_power_of_2(N)
    # heuristics for number of warps
    num_warps = min(max(BLOCK // 256, 1), 8)
    finfo = torch.finfo(x_q.dtype)
    fp8_min = finfo.min
    fp8_max = finfo.max
    #assert x.is_contiguous()
    _per_token_quant_fp8[(M, )](
        x,
        x_q,
        scales,
        stride_x=x.stride(-2),
        stride_xq=x_q.stride(-2),
        N=N,
        BLOCK=BLOCK,
        num_warps=num_warps,
        num_stages=1,
        fp8_min=fp8_min,
        fp8_max=fp8_max
    )
    return x_q, scales
# ------------------ fp8 triton ------------------

# ------------------ Bench Utility ------------------
def bench_fn_triton_stable(rms_norm_layer, x, residual, quant_dtype, fn):
    # 增加预热和重复次数以稳定结果
    warmup = 25
    rep = 100
    ms = triton.testing.do_bench(
        lambda: fn(rms_norm_layer, x, residual, quant_dtype),
        warmup=warmup,
        rep=rep
    )
    return ms

def bench_fn_multi(rms_norm_layer, x, residual, quant_dtype, fn, runs=3):
    results = []
    for _ in range(runs):
        ms = bench_fn_triton_stable(rms_norm_layer, x, residual, quant_dtype, fn)
        results.append(ms)
    results.sort()
    median_ms = results[len(results) // 2]
    return median_ms, results

@dataclass
class bench_params_t:
    num_tokens: int
    hidden_size: int
    add_residual: bool
    dtype: torch.dtype

    def description(self):
        return (f'N {self.num_tokens} '
                f'x D {self.hidden_size} '
                f'x R {self.add_residual} '
                f'x DT {self.dtype}')

def get_bench_params() -> list[bench_params_t]:
    # 修改这里的范围以匹配你的日志或需求
    NUM_TOKENS = [1024] 
    HIDDEN_SIZES = [4608, 5632]
    ADD_RESIDUAL = [True, False]
    DTYPES = [torch.bfloat16, torch.float16]

    combinations = product(NUM_TOKENS, HIDDEN_SIZES, ADD_RESIDUAL, DTYPES)
    bench_params = list(map(lambda x: \
        bench_params_t(x[0], x[1], x[2], x[3]), combinations))
    return bench_params

# ------------------ Implementations ------------------

# 修复了之前重复定义的问题，现在这是唯一的 unfused 实现
def unfused_fp8_impl(rms_norm_layer: RMSNorm, x: torch.Tensor,
                      residual: Optional[torch.Tensor],
                      quant_dtype: torch.dtype):
    # Norm
    torch_out = None
    if residual is None:
        torch_out = rms_norm_layer.forward_cuda(x, residual)
    else:
        torch_out, _ = rms_norm_layer.forward_cuda(x, residual)

    torch_out, _ = per_token_quant_fp8_torch(torch_out)
    # torch_out, _ = per_token_quant_fp8(torch_out)
    return torch_out


input_is_not_contiguous = True
def fused_fast_impl(
        rms_norm_layer: RMSNorm,  
        x: torch.Tensor,
        residual: Optional[torch.Tensor],
        quant_dtype: torch.dtype):
    import torch.nn.functional as F

    # 模拟非连续输入
    _padded = F.pad(x, (0, 20))
    y = _padded[:, :x.stride(-2)]
    if input_is_not_contiguous is True and residual is None:
        x = y
        
    out, _ = rms_norm_per_token_fp8_quant(x,
                                          rms_norm_layer.weight,
                                          1e-6,
                                          0,
                                          residual=residual, 
                                          update_input=True)
    return out

# ------------------ Accuracy Check ------------------
def check_accuracy(rms_norm_layer: RMSNorm, x: torch.Tensor, residual: Optional[torch.Tensor]):
    import copy
    residual_copy = copy.deepcopy(residual)
    rms_norm_layer_copy = copy.deepcopy(rms_norm_layer)
    x_copy = copy.deepcopy(x)
    
    with torch.no_grad():
        import torch.nn.functional as F

        # 准备 Fused 输入
        _padded = F.pad(x_copy, (0, 16))
        y = _padded[:, :x_copy.stride(-2)]
        if input_is_not_contiguous is True and residual_copy is None:
            x_copy = y

        # 运行 Fused Kernel
        test_out, test_scales = rms_norm_per_token_fp8_quant(
            x_copy,
            rms_norm_layer_copy.weight,
            1e-6,
            0,
            residual=residual_copy if residual_copy is not None else None,
            update_input=True,
        )
        
        # 运行 Unfused Reference (PyTorch)
        torch_out = None
        if residual is None:
            torch_out = rms_norm_layer.forward_cuda(x, residual)
        else:
            torch_out, _ = rms_norm_layer.forward_cuda(x, residual)
        
        ref_out, ref_scales = per_token_quant_fp8_torch(torch_out)
        # ref_out, ref_scales = per_token_quant_fp8(torch_out)

        # === 验证 Scales ===
        ref_scales_f = ref_scales.to(torch.float32)
        test_scales_f = test_scales.to(torch.float32)

        abs_diff_s = (ref_scales_f - test_scales_f).abs()
        max_diff_s = abs_diff_s.max().item()
        mean_diff_s = abs_diff_s.mean().item()
        denom_s = ref_scales_f.abs().max().item() or 1.0
        rel_err_s = max_diff_s / denom_s

        print(f"[scale精度验证] max_abs_diff={max_diff_s:.6e}, mean_abs_diff={mean_diff_s:.6e}, rel_err={rel_err_s:.6e}")
        assert rel_err_s < 1e-3, (
            f"scale 精度不达标: rel_err={rel_err_s:.6e} "
            f"(max_abs_diff={max_diff_s:.6e})"
        )

        # === 验证 Output ===
        ref_out_f = ref_out.to(torch.float32)
        test_out_f = test_out.to(torch.float32)

        abs_diff = (ref_out_f - test_out_f).abs()
        max_diff = abs_diff.max().item()
        mean_diff = abs_diff.mean().item()
        denom = ref_out_f.abs().max().item() or 1.0
        rel_err = max_diff / denom

        print(f"[out精度验证] max_abs_diff={max_diff:.6e}, mean_abs_diff={mean_diff:.6e}, rel_err={rel_err:.6e}")
        assert rel_err < 1e-2, (
            f"输出精度不达标: rel_err={rel_err:.6e} "
            f"(max_abs_diff={max_diff:.6e})"
        )

        return max_diff, mean_diff, rel_err

# ------------------ Main Bench Loop ------------------
def create_rms_norm(hidden_size: int, eps: float,
                    dtype: torch.dtype) -> RMSNorm:
    """Create vLLM RMSNorm inside the required config context."""
    with set_current_vllm_config(VllmConfig()):
        layer = RMSNorm(hidden_size, eps).to(dtype=dtype, device="cuda")
    return layer


def bench(params: bench_params_t, label: str, sub_label: str):
    layer = create_rms_norm(params.hidden_size, 1e-6, params.dtype)
    layer.weight.data.normal_(mean=1.0, std=0.1)

    scale = 1 / params.hidden_size
    x = torch.randn(params.num_tokens, params.hidden_size, dtype=params.dtype, device='cuda') * scale
    residual = (torch.randn_like(x) * scale) if params.add_residual else None

    # === 1. 精度验证 (会触发内部的打印) ===
    max_diff, mean_diff, rel_err = check_accuracy(layer, x, residual)

    results = []
    
    # === 2. 性能测试与打印 ===
    # 这里我们定义要测试的方法列表
    methods = [
        ("unfused_fp8_impl", unfused_fp8_impl),
        ("fused_fp8_fast_impl", fused_fast_impl)
    ]

    for name, fn in methods:
        ms, all_times = bench_fn_multi(layer, x, residual, torch.float8_e4m3fn, fn, runs=1)
        
        # === 复刻打印格式 ===
        print(f"[{label} | {sub_label} | {name}] avg={ms:.5f} ms")

        results.append({
            "Label": label,
            "SubLabel": sub_label,
            "NumTokens": params.num_tokens,
            "HiddenSize": params.hidden_size,
            "AddResidual": params.add_residual,
            "DType": str(params.dtype),
            "Impl": name,
            "AvgTime_ms": ms,
            "MaxAbsDiff": max_diff,
            "MeanAbsDiff": mean_diff,
            "RelError": rel_err,
        })

    return results

def main():
    torch.set_default_device('cuda')
    bench_params = get_bench_params()

    all_results = []
    # 使用 tqdm 进度条，并在内部进行测试
    for bp in tqdm(bench_params):
        all_results.extend(
            bench(bp, "rms-norm-fp8-quant", bp.description())
        )

    csv_filename = "rms_norm_dpt_quant_full.csv"
    fieldnames = [
        "Label", "SubLabel", "NumTokens", "HiddenSize", "AddResidual",
        "DType", "Impl", "AvgTime_ms", "MaxAbsDiff", "MeanAbsDiff",
        "RelError"
    ]
    with open(csv_filename, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_results)

    print(f"\n完整结果已保存到 {csv_filename}")

@pytest.mark.parametrize(
    "bp", get_bench_params(), ids=lambda bp: bp.description()
)
def test_fuse_silu_mul_fp8_quant(bp):
    if not HAS_VLLM:
        pytest.skip("vllm 不可用")
    if not torch.cuda.is_available():
        pytest.skip("需要 CUDA/ROCm 设备")
    bench(bp, "rms-norm-fp8-quant", bp.description())


if __name__ == '__main__':
    main()
