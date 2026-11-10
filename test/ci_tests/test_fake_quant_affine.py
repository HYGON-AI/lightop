# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

import torch
import time
import numpy as np
from lightop import op

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
WARMUP_ITER = 100
TEST_ITER = 1000
SEED = 42
# float32 can be near bit-exact; float16 needs looser bounds (ULP ~1e-3..1e-2).
RTOL = {torch.float32: 1e-5, torch.float16: 1e-2, torch.bfloat16: 1.6e-2}
ATOL = {torch.float32: 1e-8, torch.float16: 1e-2, torch.bfloat16: 1.6e-2}


def _tol(dtype):
    return RTOL.get(dtype, 1e-2), ATOL.get(dtype, 1e-2)

def get_kernel_time_ms(start_event, end_event):
    start_event.synchronize()
    end_event.synchronize()
    elapsed_time_ms = start_event.elapsed_time(end_event)
    return elapsed_time_ms

def warmup_operator_tensor(input_tensor, scale, zero_point, quant_min, quant_max):
    for _ in range(WARMUP_ITER):
        _ = op.fake_quantize_per_tensor_affine_custom(
            input_tensor, scale, zero_point, quant_min, quant_max
        )
        _ = torch.fake_quantize_per_tensor_affine(
            input_tensor, scale, zero_point, quant_min, quant_max
        )
    torch.cuda.synchronize()

def warmup_operator_channel(input_tensor, scale, zero_point, axis, quant_min, quant_max):
    for _ in range(WARMUP_ITER):
        _ = op.fake_quantize_per_channel_affine_custom(
            input_tensor, scale, zero_point, axis, quant_min, quant_max
        )
        _ = torch.fake_quantize_per_channel_affine(
            input_tensor, scale, zero_point, axis, quant_min, quant_max
        )
    torch.cuda.synchronize()

def validate_consistency_tensor():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    test_cases = [
        # (shape, scale, zero_point, quant_min, quant_max)
        ((1, 128), 0.01, 0, -128, 127),
        ((8, 256), 0.05, 10, -64, 63),
        ((32, 512), 0.1, -20, -128, 127),
        ((100, 1000), 0.001, 0, -128, 127),
        ((1, 4096), 0.02, 50, -128, 127),
        ((32, 2048, 4096), 0.001, 0, -128, 127),
    ]
    
    for idx, (shape, scale, zero_point, quant_min, quant_max) in enumerate(test_cases):
        input_tensor = torch.randn(shape, dtype=torch.float32, device=DEVICE)

        custom_out = op.fake_quantize_per_tensor_affine_custom(
            input_tensor, scale, zero_point, quant_min, quant_max
        )

        torch_out = torch.fake_quantize_per_tensor_affine(
            input_tensor, scale, zero_point, quant_min, quant_max
        )

        rtol, atol = _tol(input_tensor.dtype)
        is_close = torch.allclose(custom_out, torch_out, rtol=rtol, atol=atol)
        if is_close:
            print(f"用例 {idx+1}/{len(test_cases)}: ✅ 一致 (shape={shape})")
        else:
            diff = torch.abs(custom_out - torch_out)
            max_diff_idx = torch.argmax(diff).item()
            max_diff = diff.flatten()[max_diff_idx].item()
            print(f"用例 {idx+1}/{len(test_cases)}: ❌ 不一致！最大误差={max_diff:.6f} (shape={shape})")
            raise AssertionError(f"一致性验证失败（用例 {idx+1}）")
    
    print("=== tensor 用例一致性验证通过！===")

def validate_consistency_channel():
    torch.manual_seed(SEED)
    
    # 测试用例：(shape, axis, quant_min, quant_max)
    test_cases = [
        ((4096, 4096), 0, -128, 127),
        ((4096, 4096), 1, -128, 127),
        ((128, 255), 0, -128, 127),
        ((32, 1024, 4096), 1, -128, 127),
        ((32, 1024, 4096), 2, -128, 127),
    ]
    
    for idx, (shape, axis, qmin, qmax) in enumerate(test_cases):
        input_tensor = torch.randn(shape, dtype=torch.float16, device=DEVICE) * 5.0
        
        channel_size = shape[axis]
        scale = torch.rand(channel_size, dtype=torch.float32, device=DEVICE) * 0.1 + 0.01
        zero_point = torch.randint(qmin, qmax, (channel_size,), dtype=torch.int32, device=DEVICE)

        custom_out = op.fake_quantize_per_channel_affine_custom(
            input_tensor, scale, zero_point, axis, qmin, qmax
        )

        torch_out = torch.fake_quantize_per_channel_affine(
            input_tensor, scale, zero_point, axis, qmin, qmax
        )

        rtol, atol = _tol(input_tensor.dtype)
        is_close = torch.allclose(custom_out, torch_out, rtol=rtol, atol=atol)
        if is_close:
            print(f"用例 {idx+1}/{len(test_cases)}: ✅ 一致 (shape={shape}, axis={axis})")
        else:
            diff = torch.abs(custom_out - torch_out)
            max_diff = diff.max().item()
            print(
                f"用例 {idx+1}/{len(test_cases)}: ❌ 不一致！"
                f"最大误差={max_diff:.6f} (shape={shape}, axis={axis}, atol={atol}, rtol={rtol})"
            )
            raise AssertionError(f"一致性验证失败（用例 {idx+1}）")
    
    print("=== channel 用例一致性验证通过！===")

def benchmark_performance_tensor():
    benchmark_shapes = [
        (1, 128), (8, 256), (32, 512), (100, 1000), (1, 4096), (1024, 1024), (32, 2048, 4096),
    ]
    scale = 0.01
    zero_point = 0
    quant_min = -128
    quant_max = 127
    
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    results = []
    
    for shape in benchmark_shapes:
        input_tensor = torch.randn(shape, dtype=torch.float32, device=DEVICE)
        num_elements = input_tensor.numel()
        
        warmup_operator_tensor(input_tensor, scale, zero_point, quant_min, quant_max)
        
        start_event.record()
        for _ in range(TEST_ITER):
            op_out = op.fake_quantize_per_tensor_affine_custom(
                input_tensor, scale, zero_point, quant_min, quant_max
            )
        end_event.record()

        custom_time_ms = get_kernel_time_ms(start_event, end_event)
        custom_avg_time_us = (custom_time_ms / TEST_ITER) * 1000
        custom_throughput = (num_elements * TEST_ITER) / (custom_time_ms / 1000) / 1e6

        start_event.record()
        for _ in range(TEST_ITER):
            torch_out = torch.fake_quantize_per_tensor_affine(
                input_tensor, scale, zero_point, quant_min, quant_max
            )
        end_event.record()

        torch_time_ms = get_kernel_time_ms(start_event, end_event)
        torch_avg_time_us = (torch_time_ms / TEST_ITER) * 1000
        torch_throughput = (num_elements * TEST_ITER) / (torch_time_ms / 1000) / 1e6
        
        results.append({
            "shape": shape,
            "num_elements": num_elements,
            "custom_avg_time_us": custom_avg_time_us,
            "torch_avg_time_us": torch_avg_time_us,
            "custom_throughput": custom_throughput,
            "torch_throughput": torch_throughput,
            "speedup": torch_avg_time_us / custom_avg_time_us
        })
        
        # print(f"Shape {shape} | 自定义算子: {custom_avg_time_us:.2f} μs | PyTorch: {torch_avg_time_us:.2f} μs | 加速比: {torch_avg_time_us / custom_avg_time_us:.2f}x | 吞吐量: {custom_throughput:.2f} Melem/s")
    
    # 打印汇总结果
    print("=== tensor 性能测试汇总 ===")
    print(f"测试迭代次数: {TEST_ITER} | 预热迭代次数: {WARMUP_ITER}")
    print("-" * 100)
    print(f"{'Shape':<20} | {'Num(K)':<10} | {'Custom(μs)':<10} | {'PyTorch(μs)':<12} | {'Speed':<9} | {'吞吐量(Melem/s)':<15}")
    print("-" * 100)
    for res in results:
        elem_m = f"{res['num_elements']/1e3:.2f}"
        print(f"{str(res['shape']):<20} | {elem_m:<10} | {res['custom_avg_time_us']:<10.2f} | {res['torch_avg_time_us']:<12.2f} | {res['speedup']:<8.2f}x | {res['custom_throughput']:.2f}")
    print("-" * 100)

def benchmark_performance_channel():
    benchmark_configs = [
        {"shape": (4096, 4096), "axis": 0},
        {"shape": (4096, 4096), "axis": 1},
        {"shape": (128, 255), "axis": 0},
        {"shape": (32, 1024, 4096), "axis": 2},
    ]
    
    qmin, qmax = -128, 127
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    results = []
    
    for config in benchmark_configs:
        shape, axis = config["shape"], config["axis"]
        input_tensor = torch.randn(shape, dtype=torch.float16, device=DEVICE)
        num_elements = input_tensor.numel()
        
        channel_size = shape[axis]
        scale = torch.rand(channel_size, dtype=torch.float32, device=DEVICE) * 0.1 + 0.01
        zero_point = torch.randint(qmin, qmax, (channel_size,), dtype=torch.int32, device=DEVICE)
        
        warmup_operator_channel(input_tensor, scale, zero_point, axis, qmin, qmax)

        start_event.record()
        for _ in range(TEST_ITER):
            _ = op.fake_quantize_per_channel_affine_custom(
                input_tensor, scale, zero_point, axis, qmin, qmax
            )
        end_event.record()
        custom_time_ms = get_kernel_time_ms(start_event, end_event)
        custom_avg_us = (custom_time_ms / TEST_ITER) * 1000
        custom_throughput = (num_elements * TEST_ITER) / (custom_time_ms / 1000) / 1e6

        start_event.record()
        for _ in range(TEST_ITER):
            _ = torch.fake_quantize_per_channel_affine(
                input_tensor, scale, zero_point, axis, qmin, qmax
            )
        end_event.record()
        torch_time_ms = get_kernel_time_ms(start_event, end_event)
        torch_avg_us = (torch_time_ms / TEST_ITER) * 1000
        torch_throughput = (num_elements * TEST_ITER) / (torch_time_ms / 1000) / 1e6
        
        results.append({
            "shape": shape,
            "axis": axis,
            "num_elements": num_elements,
            "custom_avg_us": custom_avg_us,
            "torch_avg_us": torch_avg_us,
            "custom_throughput": custom_throughput,
            "speedup": torch_avg_us / custom_avg_us
        })
        
        # print(f"Shape {str(shape):<18} Axis {axis} | 自定义: {custom_avg_us:.2f} μs | PyTorch: {torch_avg_us:.2f} μs | 加速比: {torch_avg_us / custom_avg_us:.2f}x")

    # 打印汇总结果表格
    print("=== channel 性能测试汇总 ===")
    print(f"测试迭代: {TEST_ITER} | 预热迭代: {WARMUP_ITER}")
    print("-" * 115)
    print(f"{'Shape':<20} | {'Axis':<5} | {'Num(K)':<10} | {'Custom(μs)':<12} | {'PyTorch(μs)':<12} | {'Speed':<9} | {'吞吐量(Melem/s)':<15}")
    print("-" * 115)
    for res in results:
        elem_m = f"{res['num_elements']/1e3:.2f}"
        print(f"{str(res['shape']):<20} | {res['axis']:<5} | {elem_m:<10} | {res['custom_avg_us']:<12.2f} | {res['torch_avg_us']:<12.2f} | {res['speedup']:<8.2f}x | {res['custom_throughput']:<15.2f}")
    print("-" * 115)

if __name__ == "__main__":
    if torch.cuda.is_available():
        print(f"测试环境: DEVICE={DEVICE} | PyTorch版本={torch.__version__}")
        
        # 1. tensor一致性验证
        print("=============== fake quantize tensor affine 测试 =================")
        try:
            validate_consistency_tensor()
        except AssertionError as e:
            print(f"一致性验证失败：{e}")
            exit(1)
        
        # 2. tensor性能测试
        benchmark_performance_tensor()

        # 3. channel一致性验证
        print("=============== fake quantize channel affine 测试 =================")
        try:
            validate_consistency_channel()
        except AssertionError as e:
            print(f"一致性验证失败：{e}")
            exit(1)
        
        # 4. channel性能测试
        benchmark_performance_channel()
    else:
        print("Error: CUDA is not available.")