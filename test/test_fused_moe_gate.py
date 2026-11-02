# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

import torch
import triton
from lightop.moe import moe_fused_gate


def biased_grouped_topk_impl(
    hidden_states: torch.Tensor,
    gating_output: torch.Tensor,
    correction_bias: torch.Tensor,
    topk: int,
    renormalize: bool,
    num_expert_group: int = 0,
    topk_group: int = 0,
):
    """Baseline grouped-topk implementation used by this unit test."""
    assert hidden_states.shape[0] == gating_output.shape[0], (
        "Number of tokens mismatch"
    )

    scores = gating_output.sigmoid()
    num_token = scores.shape[0]
    num_experts = scores.shape[1]
    scores_for_choice = (
        scores.view(num_token, -1) + correction_bias.unsqueeze(0)
    )
    group_scores = (
        scores_for_choice.view(num_token, num_expert_group, -1)
        .topk(2, dim=-1)[0]
        .sum(dim=-1)
    )
    group_idx = torch.topk(
        group_scores, k=topk_group, dim=-1, sorted=False
    )[1]
    group_mask = torch.zeros_like(group_scores)
    group_mask.scatter_(1, group_idx, 1)
    score_mask = (
        group_mask.unsqueeze(-1)
        .expand(
            num_token,
            num_expert_group,
            num_experts // num_expert_group,
        )
        .reshape(num_token, -1)
    )
    tmp_scores = scores_for_choice.masked_fill(
        ~score_mask.bool(), float("-inf")
    )
    _, topk_ids = torch.topk(
        tmp_scores, k=topk, dim=-1, sorted=False
    )
    topk_weights = scores.gather(1, topk_ids)

    if renormalize:
        topk_weights = topk_weights / topk_weights.sum(
            dim=-1, keepdim=True
        )

    return topk_weights.to(torch.float32), topk_ids.to(torch.int32)


def requested_baseline_moe_gate(
    scores,
    bias,
    num_expert_group,
    topk_group,
    topk,
    n_share_experts_fusion=0,
    routed_scaling_factor=1.0,
):
    """Add LightOP shared experts and scaling around the requested baseline."""
    batch_size, num_experts = scores.shape
    topk_routed = topk - n_share_experts_fusion
    routed_weights, routed_indices = biased_grouped_topk_impl(
        hidden_states=scores,
        gating_output=scores,
        correction_bias=bias,
        topk=topk_routed,
        renormalize=True,
        num_expert_group=num_expert_group,
        topk_group=topk_group,
    )

    # The default API applies routing scale for every expert count.  Do not
    # duplicate the old group-1 kernel's 192-expert-only scaling bug here.
    route_scale = (
        float(routed_scaling_factor)
        if routed_scaling_factor > 0
        else 1.0
    )
    routed_weights = routed_weights * route_scale

    if n_share_experts_fusion == 0:
        return routed_weights, routed_indices

    shared_indices = torch.arange(
        num_experts,
        num_experts + n_share_experts_fusion,
        device=scores.device,
        dtype=torch.int32,
    ).unsqueeze(0).expand(batch_size, -1)
    shared_base = (
        1.0 / routed_scaling_factor
        if routed_scaling_factor > 0
        else 1.0
    )
    shared_weights = torch.full(
        (batch_size, n_share_experts_fusion),
        shared_base * route_scale,
        device=scores.device,
        dtype=torch.float32,
    )
    return (
        torch.cat([routed_weights, shared_weights], dim=-1),
        torch.cat([routed_indices, shared_indices], dim=-1),
    )


# -------------------------------
# 包一层，便于替换/调试
# -------------------------------
def biased_grouped_topk_org_init(scores, bias, num_expert_group, topk_group, topk, n_share_experts_fusion, routed_scaling_factor):
    return requested_baseline_moe_gate(
        scores,
        bias,
        num_expert_group,
        topk_group,
        topk,
        n_share_experts_fusion,
        routed_scaling_factor,
    )

def biased_grouped_topk_org_opt(scores, bias, num_expert_group, topk_group, topk, n_share_experts_fusion, routed_scaling_factor):
    return moe_fused_gate(scores, bias, num_expert_group, topk_group, topk, n_share_experts_fusion, routed_scaling_factor)

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
    atol=1e-6,
    perf_warmup=20,
    perf_repeat=100,
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

    # The requested baseline uses sorted=False. Compare (expert_id, weight)
    # pairs after canonicalizing by expert ID so output ordering is irrelevant.
    init_order = torch.argsort(i_init, dim=-1)
    opt_order = torch.argsort(i_opt, dim=-1)
    i_init = torch.gather(i_init, 1, init_order)
    i_opt = torch.gather(i_opt, 1, opt_order)
    w_init = torch.gather(w_init, 1, init_order)
    w_opt = torch.gather(w_opt, 1, opt_order)

    if not torch.equal(i_init, i_opt):
        diff = (i_init != i_opt)
        n_diff = diff.sum().item()
        bad_pos = torch.nonzero(diff)[:10].tolist()
        raise AssertionError(
            f"indices mismatch: {n_diff} elements differ. First few positions: {bad_pos}"
        )

    max_abs = (w_init - w_opt).abs().max().item()
    max_rel = ((w_init - w_opt).abs() / (w_opt.abs() + 1e-12)).max().item()
    if not torch.allclose(w_init, w_opt, rtol=rtol, atol=atol):
        raise AssertionError(
            f"weights not close: max_abs={max_abs:.4e}, max_rel={max_rel:.4e}, "
            f"rtol={rtol}, atol={atol}"
        )

    # Measure only the LightOP path after correctness has passed. CUDA events
    # also work on ROCm and report average device elapsed time per invocation.
    for _ in range(perf_warmup):
        biased_grouped_topk_org_opt(
            scores,
            bias,
            num_expert_group,
            topk_group,
            topk,
            n_share_experts_fusion,
            routed_scaling_factor,
        )
    torch.cuda.synchronize()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    for _ in range(perf_repeat):
        biased_grouped_topk_org_opt(
            scores,
            bias,
            num_expert_group,
            topk_group,
            topk,
            n_share_experts_fusion,
            routed_scaling_factor,
        )
    end_event.record()
    end_event.synchronize()
    latency_us = start_event.elapsed_time(end_event) * 1000.0 / perf_repeat
    return max_abs, max_rel, latency_us

def run_correctness_suite():
    torch.manual_seed(0)
    assert torch.cuda.is_available(), "CUDA is required"
    routed_scaling_factor = 2.5
    token_sizes = [1, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]

    routing_configs = [
        # num_experts, num_expert_group, topk_group, topk, shared_experts
        (128, 1, 1, 7, 1),
        (128, 1, 1, 14, 1),
        (254, 2, 1, 7, 2),
        (254, 2, 1, 14, 2),
    ]

    cases = [
        (n, ne, ng, tg, tk, nshared, routed_scaling_factor)
        for ne, ng, tg, tk, nshared in routing_configs
        for n in token_sizes
    ]
    results = []
    for (token_num, ne, ng, tg, tk, nshar, rout) in cases:
        max_abs, max_rel, latency_us = check_correctness_once(
            token_num=token_num,
            num_experts=ne,
            num_expert_group=ng,
            topk_group=tg,
            topk=tk,
            n_share_experts_fusion=nshar, 
            routed_scaling_factor=rout
        )
        results.append(
            (token_num, ne, ng, tg, tk, nshar, max_abs, max_rel, latency_us)
        )
        print(
            f"[PASS] tokens={token_num:5d} experts={ne:3d} "
            f"groups={ng} topk_group={tg} topk={tk:2d} shared={nshar} "
            f"max_abs={max_abs:.4e} max_rel={max_rel:.4e} "
            f"time_us={latency_us:.3f}"
        )
    print(
        f"[Correctness] {len(results)}/{len(cases)} requested "
        "moe_fused_gate cases passed; timing warmup=20 repeat=100"
    )
    return results

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
    num_experts, num_expert_group, topk_group, topk = 192, 1, 1, 8
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
