import argparse
import torch
import torch.nn.functional as F
# 假设 lightop 已经正确安装
from lightop import op

def benchmark_op(func, warmup=10, runs=100):
    """
    通用测速函数，传入一个无参可执行函数进行 benchmark
    """
    for _ in range(warmup):
        func()
    torch.cuda.synchronize()
    
    # 使用正确的 CUDA Event 进行测速
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    start_event.record()
    for _ in range(runs):
        func()
    end_event.record()
    
    torch.cuda.synchronize()
    
    # 计算均值 (微秒 us)
    avg_time_us = (start_event.elapsed_time(end_event) * 1000) / runs
    return avg_time_us

def main():
    parser = argparse.ArgumentParser(description="Benchmark op.relu2_opt op.")
    # [168, 336, 672, 1344, 2688, 5376]
    parser.add_argument("--d_list", type=int, nargs='+', default=[168, 336, 672], help="List of hidden sizes (d) to benchmark")
    args = parser.parse_args()

    dtype = torch.bfloat16  
    device = "cuda"
    d_list = args.d_list
    
    num_tokens_list = [506, 8, 16, 32, 64, 128, 256, 506, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 100000]
    
    print(f"开始验证 op.relu2_opt 精度与性能测试... (Data Type: BF16)\n")
    
    # 更新表头，增加 PyTorch 耗时和加速比
    header = f"| {'d (hidden)':<10} | {'num_tokens':<10} | {'精度验证':<12} | {'Custom(us)':<11} | {'Torch(us)':<10} | {'加速比':<7} | {'带宽(GB/s)':<11} |"
    separator = "|" + "-"*12 + "|" + "-"*12 + "|" + "-"*14 + "|" + "-"*13 + "|" + "-"*12 + "|" + "-"*9 + "|" + "-"*13 + "|"
    
    print(header)
    print(separator)
    
    # 外层循环遍历不同的 hidden size
    for d in d_list:
        # 内层循环遍历不同的 token 数量
        for num_tokens in num_tokens_list:
            # 1. 形状修改：输入输出维度完全一致，均为 d
            input_tensor = torch.randn((num_tokens, d), dtype=dtype, device=device)
            output_tensor = torch.empty_like(input_tensor)
            
            # 调用自定义 CUDA 算子进行一次热身和正确性计算
            op.relu2_opt(output_tensor, input_tensor)
            
            # 2. PyTorch 基线对照计算
            t_output = torch.square(F.relu(input_tensor.view(-1, d)))
            
            is_correct = torch.allclose(output_tensor, t_output, atol=1e-2, rtol=1e-2)
            acc_str = "✅ Pass" if is_correct else "❌ Fail"
            
            if not is_correct:
                max_diff = torch.max(torch.abs(output_tensor - t_output)).item()
                acc_str += f" (Max diff: {max_diff:.4f})"
                
            # 3. 性能测试：使用 lambda 包装调用
            custom_func = lambda: op.relu2_opt(output_tensor, input_tensor)
            torch_func = lambda: torch.square(F.relu(input_tensor.view(-1, d)))
            
            custom_time = benchmark_op(custom_func)
            torch_time = benchmark_op(torch_func)
            
            # 计算加速比
            speedup = torch_time / custom_time if custom_time > 0 else 0.0

            # 4. 带宽计算：读 1 个 d，写 1 个 d，共 2 份数据。bfloat16 占 2 字节。
            total_bytes = num_tokens * 2 * d * 2
            bandwidth_gbs = (total_bytes / 1e9) / (custom_time / 1e6) if custom_time > 0 else 0
            
            # 打印格式化结果
            row = f"| {d:<10} | {num_tokens:<10} | {acc_str:<10} | {custom_time:<11.2f} | {torch_time:<10.2f} | {speedup:<6.2f}x | {bandwidth_gbs:<11.2f} |"
            print(row)
            
        # 在不同 d 的测试之间打印一个分隔行，让表格更具可读性
        print(separator)

if __name__ == "__main__":
    main()