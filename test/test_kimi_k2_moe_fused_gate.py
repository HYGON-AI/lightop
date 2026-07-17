import torch
import time
import pandas as pd
from typing import Optional

from lightop.moe import kimi_k2_moe_fused_gate

# ==========================================
# 1. 用户提供的 Python Reference 实现
# ==========================================

def kimi_k2_biased_topk_impl_ref(
    hidden_states: torch.Tensor,
    gating_output: torch.Tensor,
    correction_bias: torch.Tensor,
    topk: int,
    renormalize: bool,
    routed_scaling_factor: float = 1.0,
    apply_routed_scaling_factor_on_output: bool = False,
):
    # Logits Sigmoid
    scores = gating_output.sigmoid()
    num_token = scores.shape[0]

    # Bias add
    tmp_scores = scores.view(num_token, -1) + correction_bias.unsqueeze(0)

    # TopK (Ref 使用 sorted=False，结果是无序的)
    _, topk_ids = torch.topk(tmp_scores, k=topk, dim=-1, sorted=False)
    
    # Gather values
    topk_weights = scores.gather(1, topk_ids)

    # Renormalize
    if renormalize:
        topk_weights_sum = topk_weights.sum(dim=-1, keepdim=True)
        topk_weights = topk_weights / (topk_weights_sum + 1e-20) 
        if apply_routed_scaling_factor_on_output:
            topk_weights *= routed_scaling_factor

    return topk_weights, topk_ids.to(torch.int32)

# ==========================================
# 2. 测试与基准工具函数
# ==========================================

def run_test_case(batch_size, topk=8, num_experts=384):
    device = torch.device("cuda")
    dtype = torch.float32

    kernel_type = 'Small' if batch_size <= 512 else 'Large'
    print(f"\n{'='*60}")
    print(f"测试场景: Batch Size = {batch_size} (对应 {kernel_type} Kernel)")
    print(f"{'='*60}")

    # --------------------------
    # 准备数据
    # --------------------------
    torch.manual_seed(42) # 固定种子以便复现
    input_tensor = torch.randn(batch_size, num_experts, device=device, dtype=dtype)
    bias = torch.randn(num_experts, device=device, dtype=dtype)
    
    renormalize = True
    routed_scaling_factor = 2.0
    apply_scale = True

    # --------------------------
    # 正确性验证 (Correctness)
    # --------------------------
    print(f"正在验证数值正确性...")

    # 1. 运行 Reference
    ref_weights, ref_ids = kimi_k2_biased_topk_impl_ref(
        hidden_states=None,
        gating_output=input_tensor,
        correction_bias=bias,
        topk=topk,
        renormalize=renormalize,
        routed_scaling_factor=routed_scaling_factor,
        apply_routed_scaling_factor_on_output=apply_scale
    )

    # 2. 运行 Custom Op (CUDA)
    cuda_weights, cuda_ids = kimi_k2_moe_fused_gate(
        input_tensor, 
        bias, 
        topk, 
        renormalize, 
        routed_scaling_factor, 
        apply_scale
    )
    cuda_ids = cuda_ids.to(torch.int32)

    # --------------------------
    # [关键修改] 结果对齐
    # --------------------------
    # 为了忽略输出顺序的影响，我们分别对 Ref 和 CUDA 的结果按权重(Weights)进行降序排列。
    # 这样比较的是“选出的数值集合”是否一致。

    # A. 对 Reference 结果排序
    ref_weights_sorted, ref_sort_indices = torch.sort(ref_weights, dim=1, descending=True)
    # ID 必须跟随 Weight 重新排列
    ref_ids_sorted = torch.gather(ref_ids, 1, ref_sort_indices)

    # B. 对 CUDA 结果排序 (这是你之前缺失的步骤)
    # 即使 Kernel 意图是输出有序的，如果出现微小乱序，这里 sort 后也能对齐
    cuda_weights_sorted, cuda_sort_indices = torch.sort(cuda_weights, dim=1, descending=True)
    # ID 必须跟随 Weight 重新排列
    cuda_ids_sorted = torch.gather(cuda_ids, 1, cuda_sort_indices)

    # --------------------------
    # 对比误差
    # --------------------------
    
    # 1. 检查权重误差
    diff_weights = (cuda_weights_sorted - ref_weights_sorted).abs().max().item()
    
    # 2. 检查 ID 匹配
    # 注意：如果存在两个 Expert 分数完全相同，排序后的 ID 顺序可能仍不一致。
    # 但浮点数完全相等的概率极低，通常可以直接对比。
    mismatch_mask = (cuda_ids_sorted != ref_ids_sorted)
    mismatch_ids = mismatch_mask.sum().item()

    is_passed = True
    # 打印一些详细信息辅助调试
    if diff_weights > 1e-4 or mismatch_ids > 0:
        is_passed = False
        print(f"[失败] 正确性验证失败!")
        print(f"最大权重误差: {diff_weights}")
        print(f"ID 不匹配数量: {mismatch_ids}")
        
        # 找到第一个出错的行
        error_rows = torch.where(mismatch_mask.any(dim=1))[0]
        if len(error_rows) > 0:
            row = error_rows[0].item()
            print(f"\n--- 出错样本 (Row {row}) 详情 ---")
            print("Ref Weights (Sorted): ", ref_weights_sorted[row])
            print("CUDA Weights (Sorted):", cuda_weights_sorted[row])
            print("Ref IDs (Sorted):     ", ref_ids_sorted[row])
            print("CUDA IDs (Sorted):    ", cuda_ids_sorted[row])
            print("CUDA Weights (Raw):   ", cuda_weights[row]) # 看看原始输出是不是乱序的
    else:
        print(f"[通过] 正确性验证通过! 最大权重误差: {diff_weights:.2e}")

    # --------------------------
    # 性能测试 (Performance)
    # --------------------------
    print(f"正在进行性能测试 (CUDA Only)...")
    
    warmup_steps = 10
    test_steps = 100

    # 预热
    for _ in range(warmup_steps):
        _ = kimi_k2_moe_fused_gate(
            input_tensor, bias, topk, renormalize, routed_scaling_factor, apply_scale
        )
    torch.cuda.synchronize()

    # 计时
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    for _ in range(test_steps):
        _ = kimi_k2_moe_fused_gate(
            input_tensor, bias, topk, renormalize, routed_scaling_factor, apply_scale
        )
    end_event.record()
    torch.cuda.synchronize()

    total_time_ms = start_event.elapsed_time(end_event)
    avg_time_us = (total_time_ms / test_steps) * 1000

    print(f"平均执行时间: {avg_time_us:.2f} us")
    
    return {
        "Batch Size": batch_size,
        "Correctness": "Pass" if is_passed else "Fail",
        "Avg Time (us)": avg_time_us
    }

if __name__ == "__main__":
    batch_sizes = [1, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
    results = []

    print(f"开始测试，Batch Sizes: {batch_sizes}")

    for bs in batch_sizes:
        try:
            res = run_test_case(batch_size=bs)
            results.append(res)
        except Exception as e:
            print(f"Batch size {bs} 发生异常: {e}")
            results.append({
                "Batch Size": bs,
                "Correctness": "Error",
                "Avg Time (us)": -1.0
            })

    # 写入 Excel
    try:
        df = pd.DataFrame(results)
        output_file = "kimi_moe_benchmark.xlsx"
        df.to_excel(output_file, index=False)
        print(f"\n{'-'*60}")
        print(f"所有测试完成。结果已保存至: {output_file}")
        print(f"{'-'*60}")
        print(df) # 同时也打印在控制台方便查看
    except Exception as e:
        print(f"\n保存 Excel 失败 (请检查是否安装了 pandas 和 openpyxl): {e}")
        # 如果保存失败，打印 csv 格式到控制台作为备用
        print(pd.DataFrame(results).to_csv(index=False))
