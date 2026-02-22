import torch
import time
from lightop import op
from vllm import _custom_ops as vllm_ops
def benchmark_op(func, output_tensor, input_tensor, warmup=10, runs=100):
    # 预热
    for _ in range(warmup):
        func(output_tensor, input_tensor)
    torch.cuda.synchronize()
    
    # 性能测试
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    start_event.record()
    for _ in range(runs):
        func(output_tensor, input_tensor)
    end_event.record()
    
    torch.cuda.synchronize()
    
    # elapsed_time微秒 (us)
    avg_time_us = (start_event.elapsed_time(end_event) * 1000) / runs
    return avg_time_us

def main():
    # 测试参数配置
    dtype = torch.bfloat16
    device = "cuda"
    d = 4096
    num_tokens_list = [32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]
    
    print("开始验证精度与性能测试...\n")
    print("| num_tokens | d | 精度验证 | 基准耗时 (us) | 自定义算子耗时 (us) | 相对提升 |")
    print("| :--- | :--- | :--- | :--- | :--- | :--- |")
    
    for num_tokens in num_tokens_list:
        # 初始化输入与输出 Tensor
        # input最后维是 2*d
        input_tensor = torch.randn((num_tokens, 2 * d), dtype=dtype, device=device)
        output_baseline = torch.empty((num_tokens, d), dtype=dtype, device=device)
        output_custom = torch.empty((num_tokens, d), dtype=dtype, device=device)
        
        # 1. 精度验证
        torch.ops._C.silu_and_mul(output_baseline, input_tensor)
        op.silu_and_mul_opt(output_custom, input_tensor)
     
        is_correct = torch.allclose(output_baseline, output_custom, atol=1e-3, rtol=1e-3)
        acc_str = "✅ Pass" if is_correct else "❌ Fail"
        
        if not is_correct:
            max_diff = torch.max(torch.abs(output_baseline - output_custom)).item()
            acc_str += f" (Max diff: {max_diff:.4f})"
        
        # 2. 性能测试
        baseline_time = benchmark_op(torch.ops._C.silu_and_mul, output_baseline, input_tensor)
        
        # 测试自定义算子性能
        custom_time = benchmark_op(op.silu_and_mul_opt, output_custom, input_tensor)
        
        # 计算加速比
        speedup = baseline_time / custom_time if custom_time > 0 else 0
        
        # 打印表格行
        print(f"| {num_tokens:<10} | {d:<4} | {acc_str:<8} | {baseline_time:<12.2f} | {custom_time:<17.2f} | {speedup:<7.2f}x |")

if __name__ == "__main__":
    assert torch.cuda.is_available(), "需要可用的 CUDA 环境来运行此测试"
    main()