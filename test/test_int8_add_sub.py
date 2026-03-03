import torch
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

def warmup_operator(a, b):
    for _ in range(WARMUP_ITER):
        _ = op.add_int8(a, b, saturate=True)
        _ = op.sub_int8(a, b, saturate=True)
        
        _ = torch.clamp(a.int() + b.int(), -128, 127).to(torch.int8)
        _ = torch.clamp(a.int() - b.int(), -128, 127).to(torch.int8)
        _ = a + b
        _ = a - b
    torch.cuda.synchronize()

# ===================== 一致性验证 =====================
def validate_consistency():
    print("\n=== 一致性验证 (Add & Sub) ===")
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    
    test_cases = [
        (1, 128), 
        (8, 255),
        (32, 512), 
        (100, 1000), 
        (1, 4096), 
        (32, 2048, 4096),
        (32, 1024, 4096 - 3),
    ]
    
    for idx, shape in enumerate(test_cases):
        a = torch.randint(-128, 128, shape, dtype=torch.int8, device=DEVICE)
        b = torch.randint(-128, 128, shape, dtype=torch.int8, device=DEVICE)
        
        torch_add_sat = torch.clamp(a.int() + b.int(), -128, 127).to(torch.int8)
        custom_add_sat = op.add_int8(a, b)
        assert torch.equal(custom_add_sat, torch_add_sat), f"用例 {idx+1}: 加法不一致！(shape={shape})"

        torch_sub_sat = torch.clamp(a.int() - b.int(), -128, 127).to(torch.int8)
        custom_sub_sat = op.sub_int8(a, b)
        assert torch.equal(custom_sub_sat, torch_sub_sat), f"用例 {idx+1}: 减法不一致！(shape={shape})"
        
        print(f"用例 {idx+1}/{len(test_cases)}: ✅ 全部一致 (shape={shape})")

# ===================== 性能测试 =====================
def benchmark_performance():
    print("=== 性能测试 ADD ===")
    
    benchmark_shapes = [
        (1, 4096), 
        (32, 512, 512), 
        (64, 1024, 4096), 
        (32, 2048, 8192),
    ]
    
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    results = []
    
    for shape in benchmark_shapes:
        a = torch.randint(-128, 128, shape, dtype=torch.int8, device=DEVICE)
        b = torch.randint(-128, 128, shape, dtype=torch.int8, device=DEVICE)
        num_elements = a.numel()
        
        # 读A, 读B, 写Out
        total_bytes_processed = num_elements * 3 
        
        warmup_operator(a, b)
        
        # 1. lightop time
        start_event.record()
        for _ in range(TEST_ITER):
            _ = op.add_int8(a, b, saturate=True)
        end_event.record()
        
        custom_sat_time_ms = get_kernel_time_ms(start_event, end_event)
        custom_sat_avg_us = (custom_sat_time_ms / TEST_ITER) * 1000
        # 显存带宽 (GB/s) = (Total Bytes) / (Time in seconds) / 1e9
        custom_sat_bw = total_bytes_processed / (custom_sat_avg_us * 1e-6) / 1e9
        
        # 2. PyTorch time
        start_event.record()
        for _ in range(TEST_ITER):
            _ = torch.clamp(a.int() + b.int(), -128, 127).to(torch.int8)
        end_event.record()
        
        pt_sat_time_ms = get_kernel_time_ms(start_event, end_event)
        pt_sat_avg_us = (pt_sat_time_ms / TEST_ITER) * 1000
        pt_sat_bw = total_bytes_processed / (pt_sat_avg_us * 1e-6) / 1e9

        # 3. PyTorch [理论性能]
        start_event.record()
        for _ in range(TEST_ITER):
            _ = a + b
        end_event.record()
        
        pt_wrap_time_ms = get_kernel_time_ms(start_event, end_event)
        pt_wrap_avg_us = (pt_wrap_time_ms / TEST_ITER) * 1000
        pt_wrap_bw = total_bytes_processed / (pt_wrap_avg_us * 1e-6) / 1e9
        
        # 记录结果
        results.append({
            "shape": shape,
            "num_elements": num_elements,
            "custom_sat_us": custom_sat_avg_us,
            "pt_sat_us": pt_sat_avg_us,
            "pt_wrap_us": pt_wrap_avg_us,
            "custom_bw": custom_sat_bw,
            "pt_sat_bw": pt_sat_bw,
            "speedup": pt_sat_avg_us / custom_sat_avg_us
        })
        
        # print(f"Shape {shape} | Custom Sat: {custom_sat_avg_us:.2f} μs | PyTorch Sat: {pt_sat_avg_us:.2f} μs | 加速比: {pt_sat_avg_us/custom_sat_avg_us:.2f}x | 实际带宽: {custom_sat_bw:.1f} GB/s")
    
    # 打印汇总表格
    print("="*120)
    print(f"{'性能测试 | 测试迭代: ' + str(TEST_ITER) + ' | 理论带宽参考: Pytorch Wrapped (a+b)':^120}")
    print("-" * 120)
    print(f"{'Shape':<20} | {'num(M)':<10} | {'Custom Sat(μs)':<16} | {'PyTorch Sat(μs)':<16} | {'PyTorch Wrap(μs)':<18} | {'speed':<11} | {'Custom带宽(GB/s)':<16}")
    print("-" * 120)
    for res in results:
        shape_str = str(res['shape']).replace(' ', '')
        elem_m = f"{res['num_elements']/1e6:.2f}"
        print(f"{shape_str:<20} | {elem_m:<10} | {res['custom_sat_us']:<16.2f} | {res['pt_sat_us']:<16.2f} | {res['pt_wrap_us']:<18.2f} | {res['speedup']:<10.2f}x | {res['custom_bw']:<16.1f}")
    print("-" * 120)

if __name__ == "__main__":
    if torch.cuda.is_available():
        print(f"测试环境: DEVICE={DEVICE} | PyTorch版本={torch.__version__}")
        
        # 1.一致性验证
        try:
            validate_consistency()
        except AssertionError as e:
            print(f"一致性验证失败：{e}")
            exit(1)
        
        # 2. 性能测试
        benchmark_performance()
    else:
        print("Error: CUDA is not available.")