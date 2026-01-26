# SPDX-License-Identifier: Apache-2.0

import pickle as pkl
import time
from collections.abc import Iterable
from dataclasses import dataclass
from itertools import product
from typing import Callable, Optional

import torch
import torch.utils.benchmark as TBenchmark
from torch.utils.benchmark import Measurement as TMeasurement
from tqdm import tqdm

import vllm._custom_ops as ops
from vllm.model_executor.layers.layernorm import RMSNorm
from lightop import rms_norm_per_token_fp8_quant

import triton
import triton.language as tl
import pandas as pd

#input contiguous status

# ------------------ fp8 torch ------------------
def per_token_quant_fp8_torch(x):
    target_dtype = torch.float8_e4m3fn
    finfo = torch.finfo(target_dtype)
    fp8_min = finfo.min
    fp8_max = finfo.max

    print("fp8_max",fp8_max)
    print("x",x.dtype)
    x_fp32 = x.to(torch.float32)
    x_abs= x_fp32.abs()
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

input_is_not_contiguous=True
def bench_fn_triton_stable(rms_norm_layer, x, residual, quant_dtype, fn):

    # 增加预热次数，保证GPU进入稳定状态，JIT编译完成等
    warmup = 50

    # 增大重复次数，让统计更加稳定
    rep = 50

    ms = triton.testing.do_bench(
        lambda: fn(rms_norm_layer, x, residual, quant_dtype),
        warmup=warmup,
        rep=rep
    )

    return ms

def bench_fn_multi(rms_norm_layer, x, residual, quant_dtype, fn, runs=5):
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
    ## Test Fixtures
    NUM_TOKENS = [2**x for x in range(11)]
    HIDDEN_SIZES = list(range(512, 8129, 1024))
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
    if residual is None:
        torch_out = rms_norm_layer.forward_cuda(x, residual)
    else:
        torch_out, _ = rms_norm_layer.forward_cuda(x, residual)

    # Quant
    torch_out, _ = per_token_quant_fp8(torch_out)
    # torch_out, _ = per_token_quant_fp8_torch(torch_out)


def unfused_fp8_impl(rms_norm_layer: RMSNorm, x: torch.Tensor,
                     residual: Optional[torch.Tensor],
                     quant_dtype: torch.dtype):
    # Norm
    torch_out = None
    if residual is None:
        torch_out = rms_norm_layer.forward_cuda(x, residual)
    else:
        torch_out, _ = rms_norm_layer.forward_cuda(x, residual)

    # Quant
    # torch_out, _ = ops.scaled_fp8_quant(torch_out)


def fused_fast_impl(
        rms_norm_layer: RMSNorm,  # this stores the weights
        x: torch.Tensor,
        residual: Optional[torch.Tensor],
        quant_dtype: torch.dtype):
    import torch.nn.functional as F

    _padded = F.pad(x, (0, 20))
    y = _padded[:, :x.stride(-2)]
    if input_is_not_contiguous is True and residual is None:
        x = y
    out, _ = rms_norm_per_token_fp8_quant(x,
                                          rms_norm_layer.weight,
                                          1e-6,
                                          0,
                                          residual=residual,update_input=True)

# Bench functions
def bench_fn(rms_norm_layer: RMSNorm, x: torch.Tensor, residual: torch.Tensor,
             quant_dtype: torch.dtype, label: str, sub_label: str,
             fn: Callable, description: str) -> TMeasurement:

    min_run_time = 1

    globals = {
        "rms_norm_layer": rms_norm_layer,
        "x": x,
        "residual": residual,
        "quant_dtype": quant_dtype,
        "fn": fn,
    }
    return TBenchmark.Timer(
        stmt="fn(rms_norm_layer, x, residual, quant_dtype)",
        globals=globals,
        label=label,
        sub_label=sub_label,
        description=description,
    ).blocked_autorange(min_run_time=min_run_time)

def check_accuracy(rms_norm_layer: RMSNorm, x: torch.Tensor, residual: Optional[torch.Tensor]):
    """比较 fused_impl 和 fused_fast_impl 输出精度"""
    import copy
    residual_copy = copy.deepcopy(residual)
    rms_norm_layer_copy = copy.deepcopy(rms_norm_layer)
    x_copy = copy.deepcopy(x)
    xx_copy = copy.deepcopy(x)
    with torch.no_grad():
        import torch.nn.functional as F

        _padded = F.pad(x_copy, (0, 16))
        y = _padded[:, :x_copy.stride(-2)]
        if input_is_not_contiguous is True and residual_copy is None:
            x_copy = y

        test_out, test_scales = rms_norm_per_token_fp8_quant(
            x_copy,
            rms_norm_layer_copy.weight,
            1e-6,
            0,
            residual=residual_copy if residual_copy is not None else None,
            update_input=True,
        )
        
        #unfused int8
        torch_out = None
        if residual is None:
            torch_out = rms_norm_layer.forward_cuda(x, residual)
        else:
            torch_out, _ = rms_norm_layer.forward_cuda(x, residual)
        torch_out_copy=copy.deepcopy(torch_out)
        # Quant
        # ref_out, ref_scales = per_token_quant_fp8_torch(torch_out)
        ref_out, ref_scales = per_token_quant_fp8(torch_out)

        # 转为 float32 再比较
        ref_out_f = ref_scales.to(torch.float32)
        test_out_f = test_scales.to(torch.float32)

        abs_diff = (ref_out_f - test_out_f).abs()
        max_diff = abs_diff.max().item()
        mean_diff = abs_diff.mean().item()

        # 避免除零
        denom = ref_out_f.abs().max().item() or 1.0
        rel_err = max_diff / denom

        print(f"[scale精度验证] max_abs_diff={max_diff:.6e}, mean_abs_diff={mean_diff:.6e}, rel_err={rel_err:.6e}")


        # 转为 float32 再比较
        ref_out_f = ref_out.to(torch.float32)
        test_out_f = test_out.to(torch.float32)

        abs_diff = (ref_out_f - test_out_f).abs()
        max_diff = abs_diff.max().item()
        mean_diff = abs_diff.mean().item()

        # 避免除零
        denom = ref_out_f.abs().max().item() or 1.0
        rel_err = max_diff / denom

        print(f"[out精度验证] max_abs_diff={max_diff:.6e}, mean_abs_diff={mean_diff:.6e}, rel_err={rel_err:.6e}")
        
        return max_diff, mean_diff, rel_err


def bench(params: bench_params_t, label: str, sub_label: str):
    layer = RMSNorm(params.hidden_size, 1e-6).to(dtype=params.dtype, device="cuda")
    layer.weight.data.normal_(mean=1.0, std=0.1)

    scale = 1 / params.hidden_size
    x = torch.randn(params.num_tokens, params.hidden_size, dtype=params.dtype, device='cuda') * scale
    residual = (torch.randn_like(x) * scale) if params.add_residual else None

    # === 精度验证 ===
    max_diff, mean_diff, rel_err = check_accuracy(layer, x, residual)

    results = []
    for name, fn in [
        ("unfused_fp8_impl", unfused_fp8_impl),
        ("fused_fp8_fast_impl", fused_fast_impl)
    ]:
        ms, all_times = bench_fn_multi(layer, x, residual, torch.float8_e4m3fn, fn, runs=1)
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
            # === 新增: 把精度结果写进去 ===
            "MaxAbsDiff": max_diff,
            "MeanAbsDiff": mean_diff,
            "RelError": rel_err,
        })

    return results



# launch bench
# runner
def print_timers(timers: Iterable[TMeasurement]):
    compare = TBenchmark.Compare(timers)
    compare.print()


def main():
    torch.set_default_device('cuda')
    bench_params = get_bench_params()

    all_results = []
    for bp in tqdm(bench_params):
        all_results.extend(
            bench(bp, "rms-norm-fp8-quant", bp.description())
        )

    df = pd.DataFrame(all_results)

    # 保留精度字段，不再只 pivot 时间
    df.to_excel("rms_norm_dpt_quant_full.xlsx", index=False)
    print("完整结果已保存到 rms_norm_dpt_quant_full.xlsx")

    # 如果你还想要之前的透视表（只比较时间），可以单独再做一份：
    df_pivot = df.pivot_table(
        index=["SubLabel", "NumTokens", "HiddenSize", "AddResidual", "DType"],
        columns="Impl",
        values="AvgTime_ms"
    ).reset_index()
    df_pivot.columns.name = None
    
    # 这里已经移除了 fused_int8_impl，防止 KeyError
    df_pivot = df_pivot[[
        "SubLabel", "NumTokens", "HiddenSize", "AddResidual", "DType",
        "unfused_fp8_impl", "fused_fp8_fast_impl"
    ]]
    df_pivot.to_excel("rms_norm_dpt_quant_pivot.xlsx", index=False)
    print("透视表已保存到 rms_norm_dpt_quant_pivot.xlsx")



if __name__ == '__main__':
    main()