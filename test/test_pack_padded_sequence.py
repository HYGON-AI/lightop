# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn as nn
import numpy as np
from lightop import op

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
WARMUP_ITER = 100
TEST_ITER = 1000
SEED = 42

def get_kernel_time_ms(start_event, end_event):
    end_event.synchronize()
    elapsed_time_ms = start_event.elapsed_time(end_event)
    return elapsed_time_ms

def custom_pack_padded_sequence(input_tensor, lengths):
    lengths_cpu = lengths.cpu().contiguous()
    data, batch_sizes = op.pack_padded_sequence_custom(input_tensor, lengths_cpu, False) # batch_first=False
    return data, batch_sizes

def warmup_operator(input_tensor, lengths):
    for _ in range(WARMUP_ITER):
        _ = custom_pack_padded_sequence(input_tensor, lengths)
        _ = nn.utils.rnn.pack_padded_sequence(input_tensor, lengths, batch_first=False, enforce_sorted=True)
    torch.cuda.synchronize()

# ===================== 一致性验证 =====================
def validate_consistency():
    print("\n=== 一致性验证 (Pack Padded Sequence) ===")
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    
    test_cases = [
        (128, 32, 1024),
        (100, 16, 1025),
        (50, 8, 33),
        (20, 4, 4),
        (256, 64, 512),
    ]
    
    for idx, (T, B, D) in enumerate(test_cases):
        lengths = torch.randint(1, T + 1, (B,)).sort(descending=True).values.to('cpu')
        
        input_tensor = torch.randn(T, B, D, dtype=torch.float16, device=DEVICE)
        
        native_packed = nn.utils.rnn.pack_padded_sequence(
            input_tensor, lengths, batch_first=False, enforce_sorted=True
        )
        custom_data, custom_batch_sizes = custom_pack_padded_sequence(input_tensor, lengths)
        data_match = torch.allclose(native_packed.data, custom_data)
        batch_sizes_match = torch.equal(native_packed.batch_sizes, custom_batch_sizes)
        
        assert data_match, f"用例 {idx+1}: Packed Data 不一致！(T={T}, B={B}, D={D})"
        assert batch_sizes_match, f"用例 {idx+1}: Batch Sizes 不一致！(T={T}, B={B}, D={D})"
        
        print(f"用例 {idx+1}/{len(test_cases)}: ✅ 全部一致 (T={T}, B={B}, D={D})")

# ===================== 性能测试 =====================
def benchmark_performance():
    print("=== 性能测试 (Pack Padded Sequence) ===")
    
    benchmark_shapes = [
        (512, 128, 1024),
        (512, 128, 1025),
        (1024, 64, 4096),
        (128, 32, 128),
    ]
    
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    results = []
    
    for T, B, D in benchmark_shapes:
        lengths = torch.randint(int(T*0.8), T + 1, (B,)).sort(descending=True).values.to('cpu')
        input_tensor = torch.randn(T, B, D, dtype=torch.float16, device=DEVICE)
        total_valid_elements = lengths.sum().item() * D
        # 读 Input (有效部分) + 写 Output (有效部分) = 2 倍数据量
        # FP16 = 2 Bytes
        total_bytes_processed = total_valid_elements * 2 * 2 
        
        warmup_operator(input_tensor, lengths)

        # 1. lightop time
        start_event.record()
        for _ in range(TEST_ITER):
            _ = custom_pack_padded_sequence(input_tensor, lengths)
        end_event.record()
        
        custom_time_ms = get_kernel_time_ms(start_event, end_event)
        custom_avg_us = (custom_time_ms / TEST_ITER) * 1000
        # 显存带宽 (GB/s)
        custom_bw = total_bytes_processed / (custom_avg_us * 1e-6) / 1e9
        
        # 2. PyTorch time
        start_event.record()
        for _ in range(TEST_ITER):
            _ = nn.utils.rnn.pack_padded_sequence(input_tensor, lengths, batch_first=False, enforce_sorted=True)
        end_event.record()
        
        pt_time_ms = get_kernel_time_ms(start_event, end_event)
        pt_avg_us = (pt_time_ms / TEST_ITER) * 1000
        pt_bw = total_bytes_processed / (pt_avg_us * 1e-6) / 1e9
        
        # 记录结果
        results.append({
            "config": f"T={T},B={B},D={D}",
            "elements": total_valid_elements,
            "custom_us": custom_avg_us,
            "pt_us": pt_avg_us,
            "custom_bw": custom_bw,
            "pt_bw": pt_bw,
            "speedup": pt_avg_us / custom_avg_us
        })
        
        print(f"Config {T}x{B}x{D} | Custom: {custom_avg_us:.2f} μs | PyTorch: {pt_avg_us:.2f} μs | 加速比: {pt_avg_us/custom_avg_us:.2f}x | 带宽: {custom_bw:.1f} GB/s")
    
    # 打印汇总表格
    print("\n" + "="*120)
    print(f"{'性能测试汇总 | 测试迭代: ' + str(TEST_ITER) + ' | 数据类型: FP16':^120}")
    print("-" * 120)
    print(f"{'Config (TxBxD)':<20} | {'Num(M)':<12} | {'Custom(μs)':<12} | {'PyTorch(μs)':<12} | {'speed':<11} | {'Custom带宽(GB/s)':<18}")
    print("-" * 120)
    for res in results:
        elem_m = f"{res['elements']/1e6:.2f}"
        print(f"{res['config']:<20} | {elem_m:<12} | {res['custom_us']:<12.2f} | {res['pt_us']:<12.2f} | {res['speedup']:<10.2f}x | {res['custom_bw']:<18.1f}")
    print("-" * 120)

if __name__ == "__main__":
    if torch.cuda.is_available():
        print(f"测试环境: DEVICE={DEVICE} | PyTorch版本={torch.__version__}")
        
        # 1. 一致性验证
        try:
            validate_consistency()
        except AssertionError as e:
            print(f"一致性验证失败: {e}")
            exit(1)
            
        # 2. 性能测试
        benchmark_performance()
    else:
        print("Error: CUDA is not available.")
