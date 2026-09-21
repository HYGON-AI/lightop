import torch
import time

# 导入自定义算子
from lightop import op

def test_top_k_prefill_duplicate_scores():
    """Regression test: duplicate scores must not leave output slots unwritten."""
    top_k = 512
    row_lengths = [528, 1024, 4096]
    num_rows = len(row_lengths)
    num_cols = max(row_lengths)

    # Quantized/indexer logits commonly contain duplicate values. The previous
    # insertion-sort ranking mapped equal values to the same output slot.
    logits = torch.zeros((num_rows, num_cols), dtype=torch.float32, device="cuda")
    logits[:, ::7] = 1.0
    row_starts = torch.zeros(num_rows, dtype=torch.int32, device="cuda")
    row_ends = torch.tensor(row_lengths, dtype=torch.int32, device="cuda")
    indices = torch.empty((num_rows, top_k), dtype=torch.int32, device="cuda")

    op.top_k_per_row_prefill(
        logits, row_starts, row_ends, indices, num_rows,
        logits.stride(0), logits.stride(1), top_k
    )
    torch.cuda.synchronize()

    for row, row_len in enumerate(row_lengths):
        actual = indices[row]
        assert torch.all(actual >= 0), (row, actual.min().item())
        assert torch.all(actual < row_len), (row, actual.max().item(), row_len)
        assert torch.unique(actual).numel() == top_k
        expected_values = torch.topk(logits[row, :row_len], top_k).values
        actual_values = logits[row].gather(0, actual.to(torch.int64)).sort(descending=True).values
        assert torch.equal(actual_values, expected_values)


def test_top_k():
    # 测试参数配置
    num_rows = 1
    cols_list = [8338]
    top_k = 2048     # top_k 可以根据您的实际场景更改
    next_n = 1     # 解码阶段通常 next_n=1，这样 rowEnd == seq_len

    print("开始测试 top_k_per_row_decode 算子...\n")
    
    # 打印表格 Header
    print(f"| {'列数 (Cols)':<10} | {'精度 (Pass)':<10} | {'PyTorch (us)':<15} | {'Custom (us)':<15} | {'加速比':<10} |")
    print(f"|{'-'*12}|{'-'*12}|{'-'*17}|{'-'*17}|{'-'*12}|")
    
    for num_cols in cols_list:
        # 1. 准备随机数据
        # C++ 中要求 logits 是 float32
        logits = torch.randn((num_rows, num_cols), dtype=torch.float32, device="cuda")
        
        # C++ 中 seqLens 和 indices 的指针使用了 .data_ptr<int>()，因此必须使用 int32
        seq_lens = torch.full((num_rows,), num_cols, dtype=torch.int32, device="cuda")
        custom_indices = torch.empty((num_rows, top_k), dtype=torch.int32, device="cuda")
        
        stride0 = logits.stride(0)
        stride1 = logits.stride(1)
        
        # ==========================================
        # 2. 精度验证
        # ==========================================
        # PyTorch 基准测试
        pt_vals, pt_indices = torch.topk(logits, top_k, dim=1, largest=True, sorted=True)
        
        # 运行自定义算子
        op.top_k_per_row_decode(
            logits, next_n, seq_lens, custom_indices, 
            num_rows, stride0, stride1, top_k
        )
        
        # 将自定义算子提取出的 indices 转换一下类型以进行 gather
        custom_vals = torch.gather(logits, 1, custom_indices.to(torch.int64))
        
        # 因为算子返回的 top-K 可能是无序的，我们需要对 custom_vals 进行排序后再与 PyTorch 结果对比
        custom_vals_sorted, _ = torch.sort(custom_vals, dim=1, descending=True)
        
        # 使用提取出的 value 进行比较更为安全，避免相同分数带来索引顺序不同的误报，同时增加一点点 atol 容差
        is_correct = torch.allclose(pt_vals, custom_vals_sorted, atol=1e-5)
        pass_str = "✅ Pass" if is_correct else "❌ Fail"
        
        # ==========================================
        # 3. 性能测试 (us)
        # ==========================================
        
        # --- PyTorch 性能测试 ---
        # 预热 10 轮
        for _ in range(10):
            torch.topk(logits, top_k, dim=1, largest=True, sorted=True)
        torch.cuda.synchronize()
        
        # 正式运行 100 轮
        start_time = time.perf_counter()
        for _ in range(100):
            torch.topk(logits, top_k, dim=1, largest=True, sorted=True)
        torch.cuda.synchronize()
        pt_time_us = (time.perf_counter() - start_time) * 1e6 / 100
        
        # --- Custom 算子性能测试 ---
        # 预热 10 轮
        for _ in range(10):
            op.top_k_per_row_decode(
                logits, next_n, seq_lens, custom_indices, 
                num_rows, stride0, stride1, top_k
            )
        torch.cuda.synchronize()
        
        # 正式运行 100 轮
        start_time = time.perf_counter()
        for _ in range(100):
            op.top_k_per_row_decode(
                logits, next_n, seq_lens, custom_indices, 
                num_rows, stride0, stride1, top_k
            )
        torch.cuda.synchronize()
        custom_time_us = (time.perf_counter() - start_time) * 1e6 / 100
        
        # 计算加速比并打印该行结果
        ratio = pt_time_us / custom_time_us if custom_time_us > 0 else 0
        print(f"| {num_cols:<10} | {pass_str:<10} | {pt_time_us:<15.2f} | {custom_time_us:<15.2f} | {ratio:<10.2f} |")

if __name__ == "__main__":
    test_top_k_prefill_duplicate_scores()
    test_top_k()
