# SPDX-License-Identifier: Apache-2.0

import pickle as pkl
import time
from collections.abc import Iterable
from dataclasses import dataclass
from itertools import product
from typing import Callable, Optional

import torch
import triton
import triton.language as tl
import pandas as pd

import vllm._custom_ops as ops
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.layernorm import RMSNorm

# 引入您指定的 gemma 专用算子
from lightop.norm import gemma_rms_norm_per_token_fp8_quant


# =========================================================================
# 🚨 修复说明: 
# 这里必须设为 False。因为 C++ 侧的回退 Kernel (rms_norm_fp8_quant_loop) 中
# 错误地将 input 的 stride 用在了 contiguous 的 output 寻址上 (out[bid * stride + i])。
# 当此值为 True 时, input 被 pad 成 stride = hidden_size + 20, 会导致
# output 寻址越界触发 GPU VMFault 错误。
# =========================================================================
input_is_not_contiguous = False

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


@dataclass
class bench_params_t:
    num_tokens: int
    hidden_size: int
    add_residual: bool
    dtype: torch.dtype

def get_bench_params() -> list[bench_params_t]:
    ## Test Fixtures
    NUM_TOKENS = [2**x for x in range(11)]
    # 只测试 1024, 2048, 4096, 8192, 16384
    HIDDEN_SIZES = [1024, 2048, 4096, 8192, 16384]
    ADD_RESIDUAL = [False, True]
    DTYPES = [torch.float16, torch.bfloat16]

    combinations = product(NUM_TOKENS, HIDDEN_SIZES, ADD_RESIDUAL, DTYPES)
    bench_params = list(map(lambda x: \
        bench_params_t(x[0], x[1], x[2], x[3]), combinations))
    return bench_params


# Reference impls
def unfused_fp8_impl(rms_norm_layer: RMSNorm, x: torch.Tensor,
                     residual: Optional[torch.Tensor],
                     quant_dtype: torch.dtype):
    # Norm
    torch_out = None
    
    # 为了模拟 Gemma, 这里基准实现要将 w 临时 +1
    rms_norm_layer.weight.data.add_(1.0)
    
    if residual is None:
        torch_out = rms_norm_layer.forward_cuda(x, residual)
    else:
        torch_out, _ = rms_norm_layer.forward_cuda(x, residual)
        
    rms_norm_layer.weight.data.sub_(1.0) # 算完之后恢复原状

    # Quant
    torch_out, _ = per_token_quant_fp8(torch_out)


def fused_fast_impl(
        rms_norm_layer: RMSNorm,
        x: torch.Tensor,
        residual: Optional[torch.Tensor],
        quant_dtype: torch.dtype):
    import torch.nn.functional as F

    _padded = F.pad(x, (0, 20))
    y = _padded[:, :x.stride(-2)]
    if input_is_not_contiguous is True and residual is None:
        x = y
        
    # 调用 Gemma 专用版接口
    out, _ = gemma_rms_norm_per_token_fp8_quant(x,
                                                rms_norm_layer.weight,
                                                1e-6,
                                                0,
                                                residual=residual, update_input=True)


def measure_time(fn, *args, warmup=10, rep=100) -> float:
    """10轮预热，100轮取平均，返回微秒(us)"""
    for _ in range(warmup):
        fn(*args)
    
    torch.cuda.synchronize()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    start_event.record()
    for _ in range(rep):
        fn(*args)
    end_event.record()
    torch.cuda.synchronize()
    
    total_ms = start_event.elapsed_time(end_event)
    avg_us = (total_ms * 1000.0) / rep
    return avg_us


def check_accuracy(rms_norm_layer: RMSNorm, x: torch.Tensor, residual: Optional[torch.Tensor]):
    """比较 unfused_impl 和 fused_fast_impl (Gemma版) 输出精度并返回差异指标"""
    import copy
    residual_copy = copy.deepcopy(residual)
    rms_norm_layer_copy = copy.deepcopy(rms_norm_layer)
    x_copy = copy.deepcopy(x)
    
    with torch.no_grad():
        import torch.nn.functional as F

        _padded = F.pad(x_copy, (0, 16))
        y = _padded[:, :x_copy.stride(-2)]
        if input_is_not_contiguous is True and residual_copy is None:
            x_copy = y

        # 测试自定义 Gemma 量化算子
        test_out, test_scales = gemma_rms_norm_per_token_fp8_quant(
            x_copy,
            rms_norm_layer_copy.weight,
            1e-6,
            0,
            residual=residual_copy if residual_copy is not None else None,
            update_input=True,
        )
        
        # unfused fp8 baseline (使用模拟的 Gemma)
        torch_out = None
        rms_norm_layer.weight.data.add_(1.0)
        if residual is None:
            torch_out = rms_norm_layer.forward_cuda(x, residual)
        else:
            torch_out, _ = rms_norm_layer.forward_cuda(x, residual)
        rms_norm_layer.weight.data.sub_(1.0)
        
        # Quant
        ref_out, ref_scales = per_token_quant_fp8(torch_out)

        # 比较 Scales
        ref_scale_f = ref_scales.to(torch.float32)
        test_scale_f = test_scales.to(torch.float32)
        scale_abs_diff = (ref_scale_f - test_scale_f).abs()
        scale_max_diff = scale_abs_diff.max().item()
        scale_rel_err = scale_max_diff / (ref_scale_f.abs().max().item() or 1.0)

        # 比较 Outputs
        ref_out_f = ref_out.to(torch.float32)
        test_out_f = test_out.to(torch.float32)
        out_abs_diff = (ref_out_f - test_out_f).abs()
        out_max_diff = out_abs_diff.max().item()
        out_rel_err = out_max_diff / (ref_out_f.abs().max().item() or 1.0)

        return scale_max_diff, scale_rel_err, out_max_diff, out_rel_err


def create_rms_norm(hidden_size: int, eps: float,
                    dtype: torch.dtype) -> RMSNorm:
    """Create vLLM RMSNorm inside the required config context."""
    with set_current_vllm_config(VllmConfig()):
        layer = RMSNorm(hidden_size, eps).to(dtype=dtype, device="cuda")
    return layer


def bench(params: bench_params_t):
    layer = create_rms_norm(params.hidden_size, 1e-6, params.dtype)
    layer.weight.data.normal_(mean=1.0, std=0.1)

    scale = 1 / params.hidden_size
    x = torch.randn(params.num_tokens, params.hidden_size, dtype=params.dtype, device='cuda') * scale
    residual = (torch.randn_like(x) * scale) if params.add_residual else None

    # === 1. 先进行精度验证 ===
    scale_max_diff, scale_rel_err, out_max_diff, out_rel_err = check_accuracy(layer, x, residual)
    
    # 判定规则: 相对误差处于容忍阈值内 (FP8 e4m3 容忍度设为 0.1)
    status = "PASS" if (scale_rel_err < 0.1 and out_rel_err < 0.1) else "FAIL"

    # === 2. 性能测试 (预热10次，运行100次取均值) ===
    unfused_us = measure_time(unfused_fp8_impl, layer, x, residual, torch.float8_e4m3fn)
    fused_us   = measure_time(fused_fast_impl, layer, x, residual, torch.float8_e4m3fn)

    dtype_str = "fp16" if params.dtype == torch.float16 else "bf16"
    res_str = "Y" if params.add_residual else "N"

    # 简略形式表格打印
    print(f"{params.num_tokens:<6} | {params.hidden_size:<6} | {res_str:<4} | {dtype_str:<6} | "
          f"{scale_max_diff:<10.3e} | {out_max_diff:<10.3e} | {status:<6} | "
          f"{unfused_us:<12.2f} | {fused_us:<12.2f}")

    return {
        "NumTokens": params.num_tokens,
        "HiddenSize": params.hidden_size,
        "AddResidual": params.add_residual,
        "DType": str(params.dtype),
        "ScaleMaxDiff": scale_max_diff,
        "OutMaxDiff": out_max_diff,
        "Status": status,
        "UnfusedTime_us": unfused_us,
        "FusedTime_us": fused_us,
        "Speedup": unfused_us / fused_us if fused_us > 0 else 0.0
    }


def main():
    torch.set_default_device('cuda')
    bench_params = get_bench_params()

    # 打印表头
    print(f"{'N':<6} | {'D':<6} | {'Res':<4} | {'DType':<6} | "
          f"{'Scale Diff':<10} | {'Out Diff':<10} | {'Status':<6} | "
          f"{'Unfused (us)':<12} | {'Fused (us)':<12}")
    print("-" * 95)

    all_results = []
    for bp in bench_params:
        result_dict = bench(bp)
        all_results.append(result_dict)

    # 导出整合后直观的一张表，因为包含 Unfused 和 Fused 字段因此不需要再 pivot
    df = pd.DataFrame(all_results)
    df.to_excel("rms_norm_gemma_quant_benchmark.xlsx", index=False)
    print("\n完整精度与性能结果已保存到 rms_norm_gemma_quant_benchmark.xlsx")


if __name__ == '__main__':
    main()
