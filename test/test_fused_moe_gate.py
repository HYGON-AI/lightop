import torch
import triton
from vllm import _custom_ops as vllm_ops
from lightop import op as ops
import pdb


def python_reference_moe_gate(scores, bias, num_expert_group, topk_group, topk, n_share_experts_fusion=0, routed_scaling_factor=1.0):
    """
    Python实现的 MoE Gating 逻辑，严格对应提供的 CUDA 代码。
    逻辑特点：
    1. Selection Score = Sigmoid(Input) + Bias
    2. Weight Value    = Sigmoid(Input)
    3. Output          = Normalize(Weight Value of TopK Experts)
    4. 支持共享专家(Shared Experts)逻辑
    """
    # 转 float32 防止精度溢出
    scores_in = scores.float()
    bias_in = bias.float()
    batch_size, num_experts = scores_in.shape
    topk_routed = topk - n_share_experts_fusion

    # 1. 计算 Sigmoid 和 排序分
    probs = torch.sigmoid(scores_in) 
    selection_scores = probs + bias_in

    if num_expert_group > 1:
        experts_per_group = num_experts // num_expert_group

        # Reshape 为 [Batch, Group, ExpertPerGroup]
        selection_reshaped = selection_scores.view(batch_size, num_expert_group, experts_per_group)

        # 2. 组内 Top2 求和 (Group Score)
        group_top2_values, _ = torch.topk(selection_reshaped, k=2, dim=-1)
        group_scores = group_top2_values.sum(dim=-1) # [Batch, Group]

        # 3. 选出 TopK Groups
        _, top_group_indices = torch.topk(group_scores, k=topk_group, dim=-1) # [Batch, TopK_Group]

        # 4. Mask 掉未选中的组
        group_mask = torch.zeros((batch_size, num_expert_group), device=scores.device, dtype=torch.bool)
        group_mask.scatter_(1, top_group_indices, True)
        
        # 扩展回 Expert 维度
        expert_mask = group_mask.unsqueeze(-1).expand(-1, -1, experts_per_group).reshape(batch_size, num_experts)

        # 未选中的专家分数为 -inf
        masked_selection_scores = selection_scores.clone()
        masked_selection_scores[~expert_mask] = float('-inf')
    else:
        # 当 num_expert_group == 1 时，直接跳过组竞争逻辑
        masked_selection_scores = selection_scores

    # 5. 全局 TopK (基于 Selection Score)
    if topk_routed > 0:
        _, final_indices = torch.topk(masked_selection_scores, k=topk_routed, dim=-1) # [Batch, TopK_Routed]
        
        # 6. 取值并归一化 (基于 Probs)
        final_probs = torch.gather(probs, 1, final_indices)
        sum_probs = final_probs.sum(dim=1, keepdim=True) + 1e-20
        final_weights = final_probs / sum_probs
    else:
        final_indices = torch.empty((batch_size, 0), dtype=torch.long, device=scores.device)
        final_weights = torch.empty((batch_size, 0), dtype=torch.float32, device=scores.device)
        sum_probs = torch.zeros((batch_size, 1), dtype=torch.float32, device=scores.device)

    # 7. 共享专家逻辑
    if n_share_experts_fusion > 0:
        # 共享专家ID从 NUM_EXPERTS 开始递增
        share_indices = torch.arange(num_experts, num_experts + n_share_experts_fusion, device=scores.device, dtype=torch.long)
        share_indices = share_indices.unsqueeze(0).expand(batch_size, -1)
        
        # 共享专家权重 = (output_sum / routed_scaling_factor) / output_sum = 1.0 / routed_scaling_factor
        share_weights = torch.full((batch_size, n_share_experts_fusion), 1.0 / routed_scaling_factor, device=scores.device, dtype=torch.float32)
        
        # 拼接到结果中
        final_indices = torch.cat([final_indices, share_indices], dim=-1)
        final_weights = torch.cat([final_weights, share_weights], dim=-1)

    return final_weights.to(scores.dtype), final_indices.to(torch.int32)

# -------------------------------
# 包一层，便于替换/调试
# -------------------------------
def biased_grouped_topk_org_init(scores, bias, num_expert_group, topk_group, topk, n_share_experts_fusion, routed_scaling_factor):
    num_experts = scores.shape[1]
    # 当 group 都为 1 或者 expert 等于 384 时，直接调用 Python Reference
    if (num_expert_group == 1 and topk_group == 1) or num_experts == 384:
        return python_reference_moe_gate(scores, bias, num_expert_group, topk_group, topk, n_share_experts_fusion, routed_scaling_factor)
    return vllm_ops.moe_fused_gate(scores, bias, num_expert_group, topk_group, topk, n_share_experts_fusion, routed_scaling_factor)

def biased_grouped_topk_org_opt(scores, bias, num_expert_group, topk_group, topk, n_share_experts_fusion, routed_scaling_factor):
    return ops.moe_fused_gate(scores, bias, num_expert_group, topk_group, topk, n_share_experts_fusion, routed_scaling_factor)

# -------------------------------
# 正确性验证 (仅 bf16)
# -------------------------------
def check_correctness_once(
    token_num: int,
    num_experts: int,
    num_expert_group: int,
    topk_group: int,
    topk: int,
    n_share_experts_fusion: int, 
    routed_scaling_factor: float,
    rtol=1e-6,
    atol=1e-6
):
    device = torch.device("cuda")
    dtype = torch.float32

    scores = torch.randn((token_num, num_experts), device=device, dtype=dtype)
    bias = torch.randn(num_experts, device=device, dtype=dtype)
    # vllm 使用的还是混合精度，lightop全部强转为了float
    w_init, i_init = biased_grouped_topk_org_init(
        scores.float(), bias.float(), num_expert_group, topk_group, topk, n_share_experts_fusion, routed_scaling_factor
    )
    w_opt, i_opt = biased_grouped_topk_org_opt(
        scores.clone(), bias.clone(), num_expert_group, topk_group, topk, n_share_experts_fusion, routed_scaling_factor
    )
    # import pdb
    # pdb.set_trace()
    assert w_init.shape == w_opt.shape, f"weights shape mismatch: {w_init.shape} vs {w_opt.shape}"
    assert i_init.shape == i_opt.shape, f"indices shape mismatch: {i_init.shape} vs {i_opt.shape}"

    if not torch.equal(i_init, i_opt):
        diff = (i_init != i_opt)
        n_diff = diff.sum().item()
        bad_pos = torch.nonzero(diff)[:10].tolist()
        raise AssertionError(
            f"indices mismatch: {n_diff} elements differ. First few positions: {bad_pos}"
        )

    if not torch.allclose(w_init, w_opt, rtol=rtol, atol=atol):
        max_abs = (w_init - w_opt).abs().max().item()
        max_rel = ((w_init - w_opt).abs() / (w_opt.abs() + 1e-12)).max().item()
        raise AssertionError(
            f"weights not close: max_abs={max_abs:.4e}, max_rel={max_rel:.4e}, "
            f"rtol={rtol}, atol={atol}"
        )

def run_correctness_suite():
    torch.manual_seed(0)
    assert torch.cuda.is_available(), "CUDA is required"
    n_share_experts_fusion = 0
    routed_scaling_factor = 2.5
    num_expert = 512 
    extra = list([1,4,8,32,1024,2048,4096])
    
    cases = [
        (n, num_expert, 1, 1, 22, n_share_experts_fusion, routed_scaling_factor)
        for n in list(extra)
    ]
    for (token_num, ne, ng, tg, tk, nshar, rout) in cases:
        check_correctness_once(
            token_num=token_num,
            num_experts=ne,
            num_expert_group=ng,
            topk_group=tg,
            topk=tk,
            n_share_experts_fusion=nshar, 
            routed_scaling_factor=rout
        )
    print(f"[Correctness] All bf16 tests passed ✅ (num_expert={num_expert})")

# -------------------------------
# 性能基准 (仅 bf16，支持 warmup)
# -------------------------------
token_num_range = list([1,4,8,32,1024,2048,4096])

configs = [(sq,) for sq in token_num_range]

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["token_num"],
        x_vals=[list(_) for _ in configs],
        line_arg="provider",
        line_vals=["init", "opt"],
        line_names=["Init", "Opt"],
        styles=[("blue", "-"), ("red", "-")],
        ylabel="us",
        plot_name="moe-fused-gate-performance",
        args={},
    )
)
def benchmark(token_num, provider, warmup=5):
    dtype = torch.bfloat16
    device = torch.device("cuda")
    num_experts, num_expert_group, topk_group, topk = 512, 1, 1, 22
    n_share_experts_fusion = 0
    routed_scaling_factor = 2.5

    scores = torch.randn((token_num, num_experts), device=device, dtype=dtype)
    bias = torch.rand(num_experts, device=device, dtype=dtype)

    quantiles = [0.5, 0.2, 0.8]

    if provider == "init":
        for _ in range(warmup):
            biased_grouped_topk_org_init(scores.clone(), bias.clone(), num_expert_group, topk_group, topk, n_share_experts_fusion, routed_scaling_factor)
        torch.cuda.synchronize()
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: biased_grouped_topk_org_init(scores.clone(), bias.clone(), num_expert_group, topk_group, topk, n_share_experts_fusion, routed_scaling_factor),
            quantiles=quantiles,
        )
    elif provider == "opt":
        for _ in range(warmup):
            biased_grouped_topk_org_opt(scores.clone(), bias.clone(), num_expert_group, topk_group, topk, n_share_experts_fusion, routed_scaling_factor)
        torch.cuda.synchronize()
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: biased_grouped_topk_org_opt(scores.clone(), bias.clone(), num_expert_group, topk_group, topk, n_share_experts_fusion, routed_scaling_factor),
            quantiles=quantiles,
        )
    else:
        raise ValueError(f"Unknown provider: {provider}")

    return 1000 * ms, 1000 * max_ms, 1000 * min_ms

# -------------------------------
# 入口
# -------------------------------
if __name__ == "__main__":
    run_correctness_suite()
    benchmark.run(print_data=True, warmup=10)