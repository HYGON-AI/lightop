import torch
import math
import os
import time
import numpy as np
import pandas as pd

# 尝试从 lightop 库导入预编译的通用 moe_align_block_size 函数
# 注意：你需要将编译好的库命名为 lightop 并确保它在 Python 的搜索路径中
try:
    from lightop import op as ops
    moe_align_block_size = ops.moe_align_block_size
    print("成功从 lightop 库导入 moe_align_block_size。")
except ImportError:
    print("警告：无法从 lightop 库导入 op。")
    print("请确保自定义算子已被正确编译和安装，并且当前 Python 环境可以访问它。")
    print("测试将无法运行 CUDA C++ 实现部分。")
    moe_align_block_size = None
except AttributeError:
    print("警告：在 lightop.op 中未找到 moe_align_block_size。")
    print("请确保自定义算子已被正确编译和安装。")
    print("测试将无法运行 CUDA C++ 实现部分。")
    moe_align_block_size = None

# --- 1. Python 基准实现 ---
def moe_align_block_size_python(
    topk_ids: torch.Tensor,
    num_experts: int,
    block_size: int,
    expert_map: torch.Tensor = None
):
    """
    moe_align_block_size 的纯 Python/PyTorch 实现，用于验证。
    现在支持可选的 expert_map。

    Args:
        topk_ids (torch.Tensor): 形状为 [num_tokens * top_k] 的张量，包含逻辑专家 ID。
        num_experts (int): 专家总数。
        block_size (int): 内存对齐的块大小。
        expert_map (torch.Tensor, optional): 形状为 [num_experts] 的张量，
                                            用于将逻辑专家 ID 映射到物理专家 ID。

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            - sorted_token_ids: 排序和填充后的 token 索引。
            - experts_ids: 每个计算块对应的物理专家 ID。
            - num_tokens_post_pad: 包含填充后总 token 数的张量。
    """
    topk_ids_cpu = topk_ids.cpu()

    # 如果有 expert_map，则将逻辑 ID 映射到物理 ID
    if expert_map is not None:
        expert_map_cpu = expert_map.cpu()
        mapped_topk_ids_cpu = expert_map_cpu[topk_ids_cpu.long()]
    else:
        mapped_topk_ids_cpu = topk_ids_cpu

    # 步骤 1: 统计每个物理专家的 token 数量
    expert_counts = torch.zeros(num_experts, dtype=torch.long)
    unique_ids, counts = torch.unique(mapped_topk_ids_cpu, return_counts=True)
    expert_counts[unique_ids.long()] = counts

    # 步骤 2: 计算每个专家填充后的 token 数量和累积和
    expert_counts_padded = torch.empty_like(expert_counts)
    for i in range(num_experts):
        expert_counts_padded[i] = math.ceil(expert_counts[i] / block_size) * block_size

    cumsum_padded = torch.cumsum(torch.cat((torch.tensor([0], dtype=torch.long), expert_counts_padded)), dim=0)

    total_tokens_post_pad = cumsum_padded[-1].item()
    if total_tokens_post_pad == 0:
        return torch.tensor([], dtype=torch.int32, device=topk_ids.device), \
               torch.tensor([], dtype=torch.int32, device=topk_ids.device), \
               torch.tensor([0], dtype=torch.int32, device=topk_ids.device)

    num_tokens_post_pad_tensor = torch.tensor([total_tokens_post_pad], dtype=torch.int32, device=topk_ids.device)
    total_count = topk_ids_cpu.numel()
    # 步骤 3 & 5: 创建 sorted_token_ids 并进行 token 重排
    sorted_token_ids = torch.full((total_tokens_post_pad,), total_count, dtype=torch.int32)
    current_expert_counts = torch.zeros(num_experts, dtype=torch.long)
    for i in range(topk_ids_cpu.numel()):
        logical_expert_id = topk_ids_cpu[i].item()
        physical_expert_id = expert_map_cpu[logical_expert_id].item() if expert_map is not None else logical_expert_id

        final_pos = cumsum_padded[physical_expert_id] + current_expert_counts[physical_expert_id]
        sorted_token_ids[final_pos] = i
        current_expert_counts[physical_expert_id] += 1

    # 步骤 4: 创建 experts_ids，其值为物理专家 ID
    num_blocks = total_tokens_post_pad // block_size
    experts_ids = torch.full((num_blocks,), -1, dtype=torch.int32)
    for i in range(num_experts): # i 在这里代表物理专家 ID
        start_block = cumsum_padded[i] // block_size
        end_block = cumsum_padded[i+1] // block_size
        for block_idx in range(start_block, end_block):
            experts_ids[block_idx] = i

    return sorted_token_ids.to(topk_ids.device), experts_ids.to(topk_ids.device), num_tokens_post_pad_tensor


# --- 2. 测试和验证 ---
def run_test():
    # --- 测试控制变量 ---
    # 设置为 True 来测试带 expert_map 的情况
    # 设置为 False 来测试不带 expert_map 的情况
    USE_EXPERT_MAP = False

    if not torch.cuda.is_available():
        print("CUDA 不可用，跳过测试。")
        return
    if moe_align_block_size is None:
        print("自定义 CUDA 算子未成功导入，跳过测试。")
        return

    # --- 通用测试参数 ---
    num_experts = 128
    top_k = 8
    block_size = 16
    device = "cuda"
    num_tokens_list = [1,2,5,7,16,32,64,128,256,323,512,1024,2048,4096,8192,12800]
    all_accuracy_tests_passed = True
    performance_results = []

    # --- 根据控制变量创建 expert_map ---
    expert_map = None
    if USE_EXPERT_MAP:
        print("正在创建一个随机 expert_map 用于测试...")
        expert_map = torch.randperm(num_experts, device=device, dtype=torch.int32)
        print("expert_map 创建完成。")
    else:
        print("将不使用 expert_map 进行测试。")

    # --- 精度测试 ---
    test_mode_str = "带 expert_map" if USE_EXPERT_MAP else "无 expert_map"
    print(f"\n--- 精度测试 ({test_mode_str}, num_experts={num_experts}, block_size={block_size}) ---")
    for i, num_tokens in enumerate(num_tokens_list):
        print(f"  [{i+1}/{len(num_tokens_list)}] 正在测试 num_tokens = {num_tokens}...")
        topk_ids = torch.randint(0, num_experts, (num_tokens * top_k,), device=device, dtype=torch.int64)
        
        # Python 参考实现
        (sorted_token_ids_py, experts_ids_py, num_tokens_post_pad_py) = moe_align_block_size_python(
            topk_ids, num_experts, block_size, expert_map=expert_map
        )
        num_local_tokens = [num_tokens]
        num_local_tokens_tensor = torch.tensor(num_local_tokens, device=device, dtype=torch.int32)
        # CUDA 实现
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
            
            moe_align_block_size(
                topk_ids, num_experts, block_size, sorted_token_ids_cuda,
                experts_ids_cuda, num_tokens_post_pad_tensor_cuda, expert_map,None,num_local_tokens_tensor
            )
            torch.cuda.synchronize()

        # 验证结果
        pad_match = torch.equal(num_tokens_post_pad_py, num_tokens_post_pad_tensor_cuda)
        experts_ids_match = torch.equal(experts_ids_py, experts_ids_cuda)
        
        py_valid_ids = sorted_token_ids_py[sorted_token_ids_py != -1]
        cuda_valid_ids = sorted_token_ids_cuda[sorted_token_ids_cuda != -1]
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
        print(f"\n\n😭😭😭 {test_mode_str} 模式精度测试失败！跳过性能测试。 😭😭😭")
        return
    print(f"\n\n🎉🎉🎉 所有 {test_mode_str} 模式精度测试通过！ 🎉🎉🎉")

    # --- 性能测试 ---
    print(f"\n\n--- 性能测试 ({test_mode_str}, num_experts={num_experts}) ---")
    warmup_runs = 10
    test_runs = 100
    excel_header = ['Num Tokens', 'CUDA Time (us)']
    print(f"\n{excel_header[0]:>12} | {excel_header[1]:>18}")
    print("-" * 35)

    for n_tokens in num_tokens_list:
        topk_ids = torch.randint(0, num_experts, (n_tokens * top_k,), device=device, dtype=torch.int64)
        _, _, num_tokens_post_pad_py = moe_align_block_size_python(
            topk_ids, num_experts, block_size, expert_map=expert_map
        )
        total_tokens_post_pad = num_tokens_post_pad_py.item()
        
        if total_tokens_post_pad > 0:
            sorted_token_ids_cuda = torch.empty((total_tokens_post_pad,), dtype=torch.int32, device=device)
            num_blocks = total_tokens_post_pad // block_size
            experts_ids_cuda = torch.empty((num_blocks,), dtype=torch.int32, device=device)
            num_tokens_post_pad_tensor_cuda = torch.empty(1, dtype=torch.int32, device=device)
        num_local_tokens = [n_tokens]
        num_local_tokens_tensor = torch.tensor(num_local_tokens, device=device, dtype=torch.int32)
        # 预热
        for _ in range(warmup_runs):
            if total_tokens_post_pad > 0:
                moe_align_block_size(
                    topk_ids, num_experts, block_size, sorted_token_ids_cuda,
                     experts_ids_cuda, num_tokens_post_pad_tensor_cuda, expert_map,None,num_local_tokens_tensor
                )
        torch.cuda.synchronize()

        # 计时
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        cuda_time_us = 0
        if total_tokens_post_pad > 0:
            start_event.record()
            for _ in range(test_runs):
                moe_align_block_size(
                    topk_ids, num_experts, block_size, sorted_token_ids_cuda,
                    experts_ids_cuda, num_tokens_post_pad_tensor_cuda, expert_map,None,num_local_tokens_tensor
                )
            end_event.record()
            torch.cuda.synchronize()
            cuda_time_us = start_event.elapsed_time(end_event) * 1000 / test_runs

        print(f"{n_tokens:12d} | {cuda_time_us:18.2f}")
        performance_results.append([n_tokens, f"{cuda_time_us:.2f}"])

    # --- 保存结果到 Excel ---
    map_str = "_with_map" if USE_EXPERT_MAP else "_no_map"
    output_excel_file = f'moe_align_block_size_exp{num_experts}{map_str}_performance.xlsx'
    try:
        df = pd.DataFrame(performance_results, columns=excel_header)
        df.to_excel(output_excel_file, index=False)
        print(f"\n✅ 性能测试结果已成功保存到 Excel 文件: {output_excel_file}")
    except Exception as e:
        print(f"\n❌ 保存 Excel 文件失败: {e}")
        print("请确保已安装 `pandas` 和 `openpyxl` 库 (`pip install pandas openpyxl`)")

if __name__ == "__main__":
    run_test()

