# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

import torch
from lightop import op
from typing import Optional
import csv
import pytest


def smooth_per_token_dynamic_quant(
    hidden_states: torch.Tensor, 
    smooth_scale: torch.Tensor, 
    dst_torch_dtype=torch.int8
):
    max_dtype_val = 1.0
    if dst_torch_dtype == torch.int8:
        max_dtype_val = 127.0
    elif dst_torch_dtype == torch.float8_e4m3fn:
        max_dtype_val = 448.0
    else:
        raise ValueError(f"dst_torch_dtype {dst_torch_dtype} is not supported")

    ori_shape = hidden_states.shape
    hidden_states = hidden_states.contiguous().view(ori_shape[0], -1).to(torch.float32)
    smooth_scale = smooth_scale.contiguous().view(1, -1).to(torch.float32)

    smoothed_input = torch.mul(hidden_states, smooth_scale)
    per_token_max = torch.max(smoothed_input.abs(), -1, keepdim=True)[0].clamp(min=1e-12).reciprocal()
    per_token_scale = per_token_max * max_dtype_val

    quant_tokens_fp32 = torch.mul(smoothed_input, per_token_scale).clamp(-max_dtype_val, max_dtype_val)
    if dst_torch_dtype == torch.int8:
        quant_tokens_fp32 = quant_tokens_fp32.round()

    quant_tokens = quant_tokens_fp32.type(dst_torch_dtype).view(ori_shape)
    per_token_scale = per_token_scale.reciprocal().view(ori_shape[0])

    return quant_tokens, per_token_scale


def int8_quant_init(input: torch.Tensor, weight: Optional[torch.Tensor], smoothscale: torch.Tensor, 
                    eps: float = 1e-5, residual: Optional[torch.Tensor] = None, 
                    after_norm: Optional[torch.Tensor] = None,
                    after_residual: Optional[torch.Tensor] = None,
                    update_input: bool = False,
                    update_res: bool = True):
    """
    使用 PyTorch 模拟完整的 RMS Norm + Smooth + Dynamic Quant 操作
    """
    # 1. 融合残差 (如果存在)
    if residual is not None:
        x_fp32 = input.to(torch.float32) + residual.to(torch.float32)
        if after_residual is not None:
            after_residual.copy_(x_fp32)
        elif update_res:
            residual.copy_(x_fp32)
    else:
        x_fp32 = input.to(torch.float32)

    # 2. RMS Norm
    variance = x_fp32.pow(2).mean(-1, keepdim=True)
    normed_input = x_fp32 * torch.rsqrt(variance + eps)
    if weight is not None:
        normed_input = normed_input * weight.to(torch.float32)
        
    # 3. After Norm 写入 (RMS + Smooth 的浮点结果)
    if after_norm is not None:
        smoothed_input = normed_input * smoothscale.to(torch.float32)
        after_norm.copy_(smoothed_input)
    elif update_input:
        smoothed_input = normed_input * smoothscale.to(torch.float32)
        input.copy_(smoothed_input)
        
    # 4. Smooth + Quantize
    ref_out, ref_scales = smooth_per_token_dynamic_quant(normed_input, smoothscale, dst_torch_dtype=torch.int8)
    return ref_out, ref_scales

def int8_quant_opt(input: torch.Tensor, weight: Optional[torch.Tensor], smoothscale: torch.Tensor, 
                   eps: float = 1e-5, residual: Optional[torch.Tensor] = None, 
                   after_norm: Optional[torch.Tensor] = None,
                   after_residual: Optional[torch.Tensor] = None,
                   update_input: bool = False,
                   update_res: bool = True):
    """
    使用 C++ 融合内核
    """
    opt_out = torch.empty_like(input, device=input.device, dtype=torch.int8)
    opt_scales = torch.empty(input.shape[:-1], device=input.device, dtype=torch.float32)
    
    op.rms_norm_smooth_per_token_dynamic_quant(
        opt_out, 
        input, 
        weight, 
        smoothscale, 
        opt_scales, 
        eps, 
        residual,    
        after_norm,  
        after_residual, 
        update_input,
        update_res
    )
    return opt_out, opt_scales

# ------------------ 指标计算函数 ------------------
def calculate_io_metrics(num_tokens, hidden_sizes, config: dict, input_dtype=torch.bfloat16, weight=None):
    input_elem_bytes = torch.finfo(input_dtype).bits // 8 # bf16: 2 bytes
    
    # 基础读取量: input(bf16) + smoothscale(fp32=4)
    read_byte = (num_tokens * hidden_sizes * input_elem_bytes) + (hidden_sizes * 4) 
    # + weight        
    if weight is not None:
        read_byte +=  hidden_sizes * 4        
    # 基础写入量: 量化激活值(int8=1 byte) + scale(fp32=4 bytes)
    write_byte = (num_tokens * hidden_sizes * 1) + (num_tokens * 4)
    
    # 残差读取
    if config["has_res"]:
        read_byte += (num_tokens * hidden_sizes * input_elem_bytes)
        
    # 各种写出组合
    if config["up_res"]:
        write_byte += (num_tokens * hidden_sizes * input_elem_bytes)
    if config["has_after_res"]:
        write_byte += (num_tokens * hidden_sizes * input_elem_bytes)
    if config["has_after_norm"]:
        write_byte += (num_tokens * hidden_sizes * input_elem_bytes)
    elif config["up_input"]:
        write_byte += (num_tokens * hidden_sizes * input_elem_bytes)
        
    io_bytes = read_byte + write_byte
    return read_byte, write_byte, io_bytes

# ------------------ 测试 ------------------
def run_and_benchmark(num_tokens: int, hidden_sizes: int, config: dict, warmup=10, rep=100):
    device = torch.device("cuda")
    dtype = torch.bfloat16
    eps = 1e-5
    
    # 解析配置标志
    has_res = config["has_res"]
    has_after_norm = config["has_after_norm"]
    has_after_res = config["has_after_res"]
    update_input = config["up_input"]
    update_res = config["up_res"]
    
    # 1. 准备数据
    input = torch.randn((num_tokens, hidden_sizes), device=device, dtype=dtype)
    weight = None
    smoothscale = torch.rand((hidden_sizes,), device=device, dtype=torch.float32) + 0.5 
    smoothscale = smoothscale.contiguous()
    
    residual_ref, residual_opt = None, None
    after_norm_ref, after_norm_opt = None, None
    after_residual_ref, after_residual_opt = None, None
    
    if has_res:
        residual = torch.randn((num_tokens, hidden_sizes), device=device, dtype=dtype)
        residual_ref = residual.clone()
        residual_opt = residual.clone()
        
    if has_after_norm:
        after_norm = torch.empty((num_tokens, hidden_sizes), device=device, dtype=dtype)
        after_norm_ref = after_norm.clone()
        after_norm_opt = after_norm.clone()
        
    if has_after_res:
        after_residual = torch.empty((num_tokens, hidden_sizes), device=device, dtype=dtype)
        after_residual_ref = after_residual.clone()
        after_residual_opt = after_residual.clone()
    
    input_ref = input.clone()
    input_opt = input.clone()

    # 用于校验 update_res=False 时，原残差未被污染
    orig_residual_clone = residual_opt.clone() if has_res else None

    # 2. 精度验证 (Correctness)
    passed = True
    error_msg = ""
    try:
        out_init, scales_init = int8_quant_init(
            input_ref, weight, smoothscale, eps, residual_ref, after_norm_ref, after_residual_ref, update_input, update_res
        )
        out_opt , scales_opt = int8_quant_opt(
            input_opt, weight, smoothscale, eps, residual_opt, after_norm_opt, after_residual_opt, update_input, update_res
        )

        if not torch.allclose(scales_init, scales_opt, rtol=1e-3, atol=1e-3):
            passed = False
            error_msg = f"Scales Max Diff: {(scales_init - scales_opt).abs().max().item():.4e}"

        if not torch.allclose(out_init, out_opt, rtol=0.0, atol=2.0): 
            passed = False
            error_msg = f"Out Max Diff: {(out_init.float() - out_opt.float()).abs().max().item():.4e}"

        if has_res and update_res and not torch.allclose(residual_ref, residual_opt, rtol=1e-2, atol=1e-2):
            passed = False
            error_msg = f"Residual Max Diff: {(residual_ref.float() - residual_opt.float()).abs().max().item():.4e}"

        if has_after_norm and not torch.allclose(after_norm_ref, after_norm_opt, rtol=1e-2, atol=1e-2):
            passed = False
            error_msg = f"AfterNorm Max Diff: {(after_norm_ref.float() - after_norm_opt.float()).abs().max().item():.4e}"

        if has_after_res:
            if not torch.allclose(after_residual_ref, after_residual_opt, rtol=1e-2, atol=1e-2):
                passed = False
                error_msg = f"AfterRes Max Diff: {(after_residual_ref.float() - after_residual_opt.float()).abs().max().item():.4e}"

     
        # 如果未开启 update_res 但传入了 residual，必须保证原有 residual 未被污染
        if has_res and not update_res and not has_after_res:
             if not torch.equal(residual_opt, orig_residual_clone):
                 passed = False
                 error_msg = "FATAL: Original residual was unexpectedly modified (update_res=False)!"

        # 如果开启了 update_input，校验 input 是否被正确更新
        if update_input and not has_after_norm:
            if not torch.allclose(input_ref, input_opt, rtol=1e-2, atol=1e-2):
                passed = False
                error_msg = f"UpdatedInput Max Diff: {(input_ref.float() - input_opt.float()).abs().max().item():.4e}"

    except Exception as e:
        passed = False
        error_msg = f"Exception: {str(e)}"

    if not passed:
        return False, error_msg, 0.0, 0.0, 0.0
    
    # 3. 性能测试 (Performance)
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    # --- 测试 Ref (PyTorch) ---
    for _ in range(warmup):
        int8_quant_init(input_ref, weight, smoothscale, eps, residual_ref, after_norm_ref, after_residual_ref, update_input, update_res)
    torch.cuda.synchronize()
    
    start_event.record()
    for _ in range(rep):
        int8_quant_init(input_ref, weight, smoothscale, eps, residual_ref, after_norm_ref, after_residual_ref, update_input, update_res)
    end_event.record()
    torch.cuda.synchronize()
    ref_us = (start_event.elapsed_time(end_event) * 1000.0) / rep
    
    # --- 测试 Opt (C++ 内核) ---
    for _ in range(warmup):
        int8_quant_opt(input_opt, weight, smoothscale, eps, residual_opt, after_norm_opt, after_residual_opt, update_input, update_res)
    torch.cuda.synchronize()
    
    start_event.record()
    for _ in range(rep):
        int8_quant_opt(input_opt, weight, smoothscale, eps, residual_opt, after_norm_opt, after_residual_opt, update_input, update_res)
    end_event.record()
    torch.cuda.synchronize()
    opt_us = (start_event.elapsed_time(end_event) * 1000.0) / rep

    # 计算理论内存带宽 (GB/s)
    _, _, io_bytes = calculate_io_metrics(num_tokens, hidden_sizes, config, input_dtype=dtype,weight=weight)
    io_gb = io_bytes / (1024 ** 3)
    bw_gbs = io_gb / (opt_us / 1e6) if opt_us > 0 else 0.0

    return True, "PASS", ref_us, opt_us, bw_gbs


NUM_TOKENS_LIST = [
    32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 128, 144, 160, 176, 192,
    208, 224, 240, 256, 288, 320, 352, 384, 416, 448, 480, 512, 640, 768, 896,
    1024, 1152, 1280, 1408, 1536, 1664, 1792, 1920, 2048, 2176, 2304, 2432,
    2560, 2688, 2816, 2944, 3072, 3200, 3328, 3456, 3584, 3712, 3840, 3968,
    4096, 6144, 8192, 10240, 12288, 14336, 16384, 18432, 20480, 22528, 24576,
    26624, 28672, 32768
]

HIDDEN_SIZES_LIST = [4096]

CONFIGS = [
    {"name": "Base",         "has_res": False, "has_after_norm": False, "has_after_res": False, "up_input": False, "up_res": False},
    {"name": "Res(AfterRes)","has_res": True,  "has_after_norm": False, "has_after_res": True,  "up_input": False, "up_res": False},
    {"name": "AfterNorm",    "has_res": False, "has_after_norm": True,  "has_after_res": False, "up_input": False, "up_res": False},
]


def run_suite():
    torch.manual_seed(23)

    print("======================================== 测试情形说明 (Legend) ========================================")
    print("  [Base]          : 仅执行基本 RMSNorm + SmoothQuant。 (无残差, 不写回状态)")
    print("  [UpInput]       : 原地写回 input (update_input=True)。")
    print("  [Res(In-place)] : 带残差计算，且原地更新 residual (update_res=True)。")
    print("  [Res(AfterRes)] : 带残差计算，不污染原残差，将结果输出至 after_residual (update_res=False)。")
    print("  [AfterNorm]     : 将 RMSNorm+Smooth 计算结果输出至 after_norm 显存。")
    print("  [AllFeatures]   : 同时启用 after_norm 和 after_residual 输出的复杂情景。")
    print("=======================================================================================================\n")

    print(f"| {'Shape (N, H)':<15} | {'Scenario':<14} | {'Correct':<8} | {'Ref (us)':<10} | {'Opt (us)':<10} | {'Speedup':<8} | {'BW (GB/s)':<9} |")
    print("-" * 105)

    # 存储结果的列表
    results_data = []

    for config in CONFIGS:
        for hidden_size in HIDDEN_SIZES_LIST:
            for num_tokens in NUM_TOKENS_LIST:
                
                shape_str = f"[{num_tokens}, {hidden_size}]"
                scen_str = config["name"]
                
                passed, msg, ref_us, opt_us, bw_gbs = run_and_benchmark(
                    num_tokens, hidden_size, config, warmup=10, rep=100
                )
                
                # 整理当前行的数据以便保存到 CSV
                row_data = {
                    "Shape": shape_str,
                    "Num Tokens": num_tokens,
                    "Hidden Size": hidden_size,
                    "Scenario": scen_str,
                    "Correct": "Yes" if passed else "No",
                    "Ref (us)": round(ref_us, 2) if passed else None,
                    "Opt (us)": round(opt_us, 2) if passed else None,
                    "Speedup": round(ref_us / opt_us, 2) if (passed and opt_us > 0) else None,
                    "BW (GB/s)": round(bw_gbs, 2) if passed else None,
                    "Error Message": "" if passed else msg
                }
                results_data.append(row_data)

                if passed:
                    speedup_str = f"{ref_us / opt_us:.2f}x" if opt_us > 0 else "N/A"
                    print(f"| {shape_str:<15} | {scen_str:<14} | {'✔':<8} | {ref_us:<10.1f} | {opt_us:<10.1f} | {speedup_str:<8} | {bw_gbs:<9.1f} |")
                else:
                    print(f"| {shape_str:<15} | {scen_str:<14} | {'✘':<8} | {msg[:35]:<43} |")
                    raise AssertionError(f"[{shape_str} | {scen_str}] {msg}")

        print("-" * 105)

    # 将结果保存为 CSV 文件
    csv_filename = "per_token_smooth_quant_int8_results.csv"
    fieldnames = [
        "Shape", "Num Tokens", "Hidden Size", "Scenario", "Correct",
        "Ref (us)", "Opt (us)", "Speedup", "BW (GB/s)", "Error Message"
    ]
    with open(csv_filename, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results_data)

    print(f"\n✅ 测试完毕！结果已成功保存到 {csv_filename}")


@pytest.mark.parametrize("num_tokens", NUM_TOKENS_LIST)
@pytest.mark.parametrize("hidden_size", HIDDEN_SIZES_LIST)
@pytest.mark.parametrize("config", CONFIGS, ids=lambda c: c["name"])
def test_per_token_smooth_quant_int8(num_tokens, hidden_size, config):
    if not torch.cuda.is_available():
        pytest.skip("需要 CUDA/ROCm 设备")
    passed, msg, _, _, _ = run_and_benchmark(
        num_tokens, hidden_size, config, warmup=10, rep=100
    )
    assert passed, msg


if __name__ == "__main__":
    run_suite()
