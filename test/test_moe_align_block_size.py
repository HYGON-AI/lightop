import torch
import math
import os
import time
import numpy as np
import pandas as pd

# 尝试从 vllm._custom_ops 库导入预编译的通用 moe_align_block_size 函数
try:
    from lightop import op as ops
    moe_align_block_size = ops.moe_align_block_size
    print("成功从 vllm._custom_ops 导入 moe_align_block_size。")
except ImportError:
    print("警告：无法从 vllm 库导入 _custom_ops。")
    print("请确保 vllm 已被正确编译和安装，并且当前 Python 环境可以访问它。")
    print("测试将无法运行 CUDA C++ 实现部分。")
    moe_align_block_size = None
except AttributeError:
    print("警告：在 vllm._custom_ops 中未找到 moe_align_block_size。")
    print("请确保 vllm 已被正确编译和安装。")
    print("测试将无法运行 CUDA C++ 实现部分。")
    moe_align_block_size = None

# --- 1. Python 基准实现 ---
def moe_align_block_size_python(topk_ids: torch.Tensor, num_experts: int, block_size: int):
    """
    moe_align_block_size 的纯 Python/PyTorch 实现，用于验证。

    Args:
        topk_ids (torch.Tensor): 形状为 [num_tokens * top_k] 的张量，包含专家 ID。
        num_experts (int): 专家总数。
        block_size (int): 内存对齐的块大小。

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            - sorted_token_ids: 排序和填充后的 token 索引。
            - experts_ids: 每个计算块对应的专家 ID。
            - num_tokens_post_pad: 包含填充后总 token 数的张量。
    """
    # 确保输入在 CPU 上，方便操作
    topk_ids_cpu = topk_ids.cpu()

    # 步骤 1: 统计每个专家的 token 数量
    expert_counts = torch.zeros(num_experts, dtype=torch.long)
    unique_ids, counts = torch.unique(topk_ids_cpu, return_counts=True)
    expert_counts[unique_ids.long()] = counts

    # 步骤 2: 计算每个专家填充后的 token 数量和累积和
    expert_counts_padded = torch.empty_like(expert_counts)
    for i in range(num_experts):
        expert_counts_padded[i] = math.ceil(expert_counts[i] / block_size) * block_size
    
    cumsum_padded = torch.cumsum(torch.cat((torch.tensor([0], dtype=torch.long), expert_counts_padded)), dim=0)
    
    total_tokens_post_pad = cumsum_padded[-1].item()
    if total_tokens_post_pad == 0:
        return torch.tensor([], dtype=torch.int32, device=topk_ids.device), torch.tensor([], dtype=torch.int32, device=topk_ids.device), torch.tensor([0], dtype=torch.int32, device=topk_ids.device)

    num_tokens_post_pad_tensor = torch.tensor([total_tokens_post_pad], dtype=torch.int32, device=topk_ids.device)

    # 步骤 3 & 5: 创建 sorted_token_ids 并进行 token 重排
    sorted_token_ids = torch.full((total_tokens_post_pad,), -1, dtype=torch.int32)
    current_expert_counts = torch.zeros(num_experts, dtype=torch.long)
    for i in range(topk_ids_cpu.numel()):
        expert_id = topk_ids_cpu[i].item()
        final_pos = cumsum_padded[expert_id] + current_expert_counts[expert_id]
        sorted_token_ids[final_pos] = i
        current_expert_counts[expert_id] += 1

    # 步骤 4: 创建 experts_ids
    num_blocks = total_tokens_post_pad // block_size
    experts_ids = torch.full((num_blocks,), -1, dtype=torch.int32)
    for i in range(num_experts):
        start_block = cumsum_padded[i] // block_size
        end_block = cumsum_padded[i+1] // block_size
        for block_idx in range(start_block, end_block):
            experts_ids[block_idx] = i

    return sorted_token_ids.to(topk_ids.device), experts_ids.to(topk_ids.device), num_tokens_post_pad_tensor


# --- 2. 测试和验证 ---
def run_test():
    if not torch.cuda.is_available():
        print("CUDA 不可用，跳过测试。")
        return
    if moe_align_block_size is None:
        print("vllm CUDA 扩展未成功导入，跳过测试。")
        return

    # --- 通用测试参数 ---
    num_experts = 256 # 只测试 256 个专家
    top_k = 8
    block_size = 64
    device = "cuda"
    num_tokens_list = [1, 5, 10, 18, 26, 35, 45, 58, 72, 90,
    110, 135, 160, 190, 225, 260, 300, 350, 410, 480,
    550, 630, 720, 810, 900, 1000, 1110, 1230, 1350, 1480,
    1600, 1750, 1900, 2050, 2200, 2350, 2500, 2650, 2800, 2950,
    3100, 3300, 3500, 3750, 4000, 4250, 4500, 4800, 5100, 5500
    ]
    # num_tokens_list = [1]
    all_accuracy_tests_passed = True
    performance_results = []
    
    # --- 精度测试 ---
    print(f"\n--- 精度测试 (通用 moe_align_block_size, num_experts={num_experts},block_size={block_size}) ---")
    for i, num_tokens in enumerate(num_tokens_list):
        print(f"  [{i+1}/{len(num_tokens_list)}] 正在测试 num_tokens = {num_tokens}...")
        topk_ids = torch.randint(0, num_experts, (num_tokens * top_k,), device=device, dtype=torch.int64)
        (sorted_token_ids_py, experts_ids_py, num_tokens_post_pad_py) = moe_align_block_size_python(topk_ids, num_experts, block_size)
        
        total_tokens_post_pad_cuda = num_tokens_post_pad_py.item()
        if total_tokens_post_pad_cuda == 0:
            sorted_token_ids_cuda = torch.tensor([], dtype=torch.int32, device=device)
            experts_ids_cuda = torch.tensor([], dtype=torch.int32, device=device)
            num_tokens_post_pad_tensor_cuda = torch.tensor([0], dtype=torch.int32, device=device)
        else:
            sorted_token_ids_cuda = torch.full((total_tokens_post_pad_cuda,), -1, dtype=torch.int32, device=device)
            num_blocks_cuda = total_tokens_post_pad_cuda // block_size
            experts_ids_cuda = torch.full((num_blocks_cuda,), -1, dtype=torch.int32, device=device)
            num_tokens_post_pad_tensor_cuda = torch.zeros(1, dtype=torch.int32, device=device)
            moe_align_block_size(topk_ids, num_experts, block_size, sorted_token_ids_cuda, experts_ids_cuda, num_tokens_post_pad_tensor_cuda)
            torch.cuda.synchronize()

        pad_match = torch.equal(num_tokens_post_pad_py, num_tokens_post_pad_tensor_cuda)
        experts_ids_match = torch.equal(experts_ids_py, experts_ids_cuda)
        py_valid_ids = sorted_token_ids_py[sorted_token_ids_py != -1]
        cuda_valid_ids = sorted_token_ids_cuda[sorted_token_ids_cuda != -1]
        # 比较排序后的有效ID集合是否一致
        sorted_ids_match = torch.equal(py_valid_ids.sort().values, cuda_valid_ids.sort().values)

        current_run_passed = pad_match and experts_ids_match and sorted_ids_match
        if not current_run_passed:
            print(f"    ❌ 验证失败!")
            all_accuracy_tests_passed = False
            if not pad_match: print(f"      - 填充后总 Token 数不一致: Py={num_tokens_post_pad_py.item()}, CUDA={num_tokens_post_pad_tensor_cuda.item()}")
            if not experts_ids_match: print(f"      - experts_ids 不一致")
            if not sorted_ids_match: print(f"      - sorted_token_ids (有效值) 不一致")
            break
    
    if not all_accuracy_tests_passed:
        print("\n\n😭😭😭 精度测试失败！由于至少一个精度测试未通过，跳过性能测试。 😭😭😭")
        return
    print("\n\n🎉🎉🎉 所有精度测试通过！moe_align_block_size 实现与 Python 基准结果一致。 🎉🎉🎉")

    # --- 性能测试 ---
    print(f"\n\n--- 性能测试 (通用 moe_align_block_size, num_experts={num_experts}) ---")
    warmup_runs = 10
    test_runs = 100
    excel_header = ['Num Tokens', 'CUDA Time (us)']
    print(f"\n{excel_header[0]:>12} | {excel_header[1]:>18}")
    print("-" * 35)

    for n_tokens in num_tokens_list:
        topk_ids = torch.randint(0, num_experts, (n_tokens * top_k,), device=device, dtype=torch.int64)
        _, _, num_tokens_post_pad_py = moe_align_block_size_python(topk_ids, num_experts, block_size)
        total_tokens_post_pad = num_tokens_post_pad_py.item()
        
        if total_tokens_post_pad > 0:
            sorted_token_ids_cuda = torch.empty((total_tokens_post_pad,), dtype=torch.int32, device=device)
            num_blocks = total_tokens_post_pad // block_size
            experts_ids_cuda = torch.empty((num_blocks,), dtype=torch.int32, device=device)
            num_tokens_post_pad_tensor_cuda = torch.empty(1, dtype=torch.int32, device=device)
        
        # 预热
        for _ in range(warmup_runs):
            if total_tokens_post_pad > 0:
                moe_align_block_size(topk_ids, num_experts, block_size, sorted_token_ids_cuda, experts_ids_cuda, num_tokens_post_pad_tensor_cuda)
        torch.cuda.synchronize()

        # 计时
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        cuda_time_us = 0
        if total_tokens_post_pad > 0:
            start_event.record()
            for _ in range(test_runs):
                moe_align_block_size(topk_ids, num_experts, block_size, sorted_token_ids_cuda, experts_ids_cuda, num_tokens_post_pad_tensor_cuda)
            end_event.record()
            torch.cuda.synchronize()
            cuda_time_us = start_event.elapsed_time(end_event) * 1000 / test_runs
        
        print(f"{n_tokens:12d} | {cuda_time_us:18.2f}")
        performance_results.append([n_tokens, f"{cuda_time_us:.2f}"])

    # --- 保存结果到 Excel ---
    output_excel_file = f'moe_align_block_size_exp{num_experts}_performance.xlsx'
    try:
        df = pd.DataFrame(performance_results, columns=excel_header)
        df.to_excel(output_excel_file, index=False)
        print(f"\n✅ 性能测试结果已成功保存到 Excel 文件: {output_excel_file}")
    except Exception as e:
        print(f"\n❌ 保存 Excel 文件失败: {e}")
        print("请确保已安装 `pandas` 和 `openpyxl` 库 (`pip install pandas openpyxl`)")

if __name__ == "__main__":
    run_test()

