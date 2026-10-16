# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

import torch

from lightop.moe import moe_fused_gate_sqrtsoftplus_v41_hcu


def _reference(
    logits,
    bias,
    topk,
    num_fused_shared_experts=0,
    renormalize=True,
    routed_scaling_factor=1.0,
    apply_routed_scaling_factor_on_output=False,
):
    softplus = torch.where(
        logits > 20.0,
        logits,
        torch.log1p(torch.exp(logits)),
    )
    activated = torch.sqrt(softplus)
    bias_broadcast = bias if bias.dim() == 2 else bias.unsqueeze(0)
    choice = activated + bias_broadcast
    choice = torch.where(torch.isnan(choice), float("inf"), choice)

    rows, experts = logits.shape
    routed_topk = topk - num_fused_shared_experts
    expert_ids = torch.arange(experts, device=logits.device).unsqueeze(0)
    routed_ids = torch.empty(
        rows, routed_topk, dtype=torch.int32, device=logits.device
    )
    routed_weights = torch.empty(
        rows, routed_topk, dtype=torch.float32, device=logits.device
    )
    remaining = torch.ones_like(choice, dtype=torch.bool)
    for slot in range(routed_topk):
        current = torch.where(remaining, choice, float("-inf"))
        max_value = current.max(dim=1, keepdim=True).values
        candidates = torch.where(
            remaining & (current == max_value), expert_ids, experts + 1
        )
        selected = candidates.min(dim=1).values
        routed_ids[:, slot] = selected.int()
        routed_weights[:, slot] = activated.gather(
            1, selected.unsqueeze(1)
        ).squeeze(1)
        remaining.scatter_(1, selected.unsqueeze(1), False)

    routed_sum = routed_weights.sum(dim=1, keepdim=True)
    weights = torch.empty(rows, topk, dtype=torch.float32, device=logits.device)
    ids = torch.empty(rows, topk, dtype=torch.int32, device=logits.device)
    weights[:, :routed_topk] = routed_weights
    ids[:, :routed_topk] = routed_ids
    if num_fused_shared_experts:
        weights[:, routed_topk:] = routed_sum / routed_scaling_factor
        for offset in range(num_fused_shared_experts):
            ids[:, routed_topk + offset] = experts + offset

    if renormalize:
        norm = torch.where(routed_sum > 0.0, routed_sum, 1.0)
        weights = weights / norm
    if apply_routed_scaling_factor_on_output:
        weights = weights * routed_scaling_factor
    return weights, ids


def _check(
    logits,
    bias,
    topk=6,
    num_fused_shared_experts=0,
    renormalize=True,
    routed_scaling_factor=1.5,
    apply_routed_scaling_factor_on_output=False,
):
    kwargs = dict(
        topk=topk,
        num_fused_shared_experts=num_fused_shared_experts,
        renormalize=renormalize,
        routed_scaling_factor=routed_scaling_factor,
        apply_routed_scaling_factor_on_output=apply_routed_scaling_factor_on_output,
    )
    expected_weights, expected_ids = _reference(logits, bias, **kwargs)
    actual_weights, actual_ids = moe_fused_gate_sqrtsoftplus_v41_hcu(
        logits, bias, **kwargs
    )
    torch.cuda.synchronize()

    torch.testing.assert_close(actual_ids, expected_ids, rtol=0, atol=0)
    torch.testing.assert_close(
        actual_weights,
        expected_weights,
        rtol=1e-5,
        atol=2e-6,
        equal_nan=True,
    )


def _average_us(fn, warmup=10, iterations=100):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        fn()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000.0 / iterations


def _benchmark():
    rows, experts, topk = 1024, 384, 6
    torch.manual_seed(20260923)
    logits = torch.randn(rows, experts, device="cuda", dtype=torch.float32)
    bias = torch.randn(experts, device="cuda", dtype=torch.float32)

    lightop_us = _average_us(
        lambda: moe_fused_gate_sqrtsoftplus_v41_hcu(
            logits, bias, topk, 0, True, 1.5, False
        )
    )
    reference_us = _average_us(
        lambda: _reference(
            logits,
            bias,
            topk,
            renormalize=True,
            routed_scaling_factor=1.5,
        )
    )
    transferred_bytes = (
        rows * experts * logits.element_size()
        + experts * bias.element_size()
        + rows * topk * (torch.finfo(torch.float32).bits // 8)
        + rows * topk * (torch.iinfo(torch.int32).bits // 8)
    )
    effective_gbs = transferred_bytes / (lightop_us * 1.0e3)
    print(
        "benchmark rows=1024 experts=384 topk=6 "
        f"lightop_avg_us={lightop_us:.3f} "
        f"torch_reference_avg_us={reference_us:.3f} "
        f"speedup={reference_us / lightop_us:.3f} "
        f"effective_GB_s={effective_gbs:.3f}"
    )


def test_moe_fused_gate_sqrtsoftplus_v41_hcu():
    assert torch.cuda.is_available(), "CUDA/ROCm is required"
    device = "cuda"

    for experts in (256, 384):
        for rows in (1, 16, 256, 513):
            torch.manual_seed(experts + rows)
            logits = torch.randn(rows, experts, device=device, dtype=torch.float32)
            bias = torch.randn(experts, device=device, dtype=torch.float32)
            _check(logits, bias)

        torch.manual_seed(experts)
        logits = torch.randn(17, experts, device=device, dtype=torch.float32)
        bias = torch.randn(experts, device=device, dtype=torch.float32)
        for topk in range(1, 17):
            _check(logits[:3], bias, topk=topk)
        _check(
            logits,
            bias,
            num_fused_shared_experts=1,
            apply_routed_scaling_factor_on_output=True,
        )
        _check(logits, bias, renormalize=False)

    bias = torch.zeros(384, device=device, dtype=torch.float32)
    zeros = torch.zeros(1, 384, device=device, dtype=torch.float32)
    _check(zeros, bias)
    _, tied_ids = moe_fused_gate_sqrtsoftplus_v41_hcu(zeros, bias, 6)
    torch.testing.assert_close(
        tied_ids,
        torch.arange(6, device=device, dtype=torch.int32).unsqueeze(0),
        rtol=0,
        atol=0,
    )

    extreme = torch.linspace(
        -100.0, 100.0, 384, device=device, dtype=torch.float32
    ).unsqueeze(0)
    _check(extreme, bias)
    extreme_weights, extreme_ids = moe_fused_gate_sqrtsoftplus_v41_hcu(
        extreme, bias, 6
    )
    assert torch.isfinite(extreme_weights).all()
    torch.testing.assert_close(
        extreme_ids,
        torch.arange(383, 377, -1, device=device, dtype=torch.int32).unsqueeze(0),
        rtol=0,
        atol=0,
    )

    nan_logits = zeros.clone()
    nan_logits[0, [1, 9, 17]] = float("nan")
    _check(nan_logits, bias)
    _check(nan_logits, bias, renormalize=False)

    nan_bias = bias.clone()
    nan_bias[[2, 10, 18]] = float("nan")
    _check(zeros, nan_bias)

    negative_inf = torch.full_like(zeros, float("-inf"))
    _check(negative_inf, bias)

    _benchmark()


def test_moe_fused_gate_sqrtsoftplus_v41_hcu_2d_bias():
    """Per-token (2D) bias path used by VL routing: image tokens take one bias,
    text tokens another, selected row-by-row inside the kernel."""
    assert torch.cuda.is_available(), "CUDA/ROCm is required"
    device = "cuda"

    # Correctness vs reference across shapes and option combinations.
    for experts in (256, 384):
        for rows in (1, 16, 256, 513):
            torch.manual_seed(1000 + experts + rows)
            logits = torch.randn(rows, experts, device=device, dtype=torch.float32)
            bias_txt = torch.randn(experts, device=device, dtype=torch.float32)
            bias_vl = torch.randn(experts, device=device, dtype=torch.float32)
            is_img = torch.rand(rows, device=device) < 0.5
            bias2d = torch.where(is_img[:, None], bias_vl, bias_txt).contiguous()
            _check(logits, bias2d)
            _check(
                logits,
                bias2d,
                num_fused_shared_experts=1,
                apply_routed_scaling_factor_on_output=True,
            )
            _check(logits, bias2d, renormalize=False)

    # A 2D bias whose rows are all identical must match the 1D broadcast result
    # bit-exactly (same values, same reduction order).
    torch.manual_seed(77)
    experts, rows = 384, 64
    logits = torch.randn(rows, experts, device=device, dtype=torch.float32)
    bias1d = torch.randn(experts, device=device, dtype=torch.float32)
    bias2d = bias1d.unsqueeze(0).expand(rows, experts).contiguous()
    w1, i1 = moe_fused_gate_sqrtsoftplus_v41_hcu(logits, bias1d, 6)
    w2, i2 = moe_fused_gate_sqrtsoftplus_v41_hcu(logits, bias2d, 6)
    torch.cuda.synchronize()
    torch.testing.assert_close(i1, i2, rtol=0, atol=0)
    torch.testing.assert_close(w1, w2, rtol=0, atol=0)

    # VL partitioning equivalence: one 2D-bias call must equal running each bias
    # on its own rows via separate 1D calls.
    torch.manual_seed(2024)
    rows = 128
    logits = torch.randn(rows, experts, device=device, dtype=torch.float32)
    bias_txt = torch.randn(experts, device=device, dtype=torch.float32)
    bias_vl = torch.randn(experts, device=device, dtype=torch.float32)
    is_img = torch.arange(rows, device=device) % 2 == 0
    bias2d = torch.where(is_img[:, None], bias_vl, bias_txt).contiguous()
    w_all, i_all = moe_fused_gate_sqrtsoftplus_v41_hcu(logits, bias2d, 6)
    w_img, i_img = moe_fused_gate_sqrtsoftplus_v41_hcu(
        logits[is_img].contiguous(), bias_vl, 6
    )
    w_txt, i_txt = moe_fused_gate_sqrtsoftplus_v41_hcu(
        logits[~is_img].contiguous(), bias_txt, 6
    )
    torch.cuda.synchronize()
    torch.testing.assert_close(i_all[is_img], i_img, rtol=0, atol=0)
    torch.testing.assert_close(i_all[~is_img], i_txt, rtol=0, atol=0)
    torch.testing.assert_close(w_all[is_img], w_img, rtol=0, atol=0)
    torch.testing.assert_close(w_all[~is_img], w_txt, rtol=0, atol=0)


if __name__ == "__main__":
    test_moe_fused_gate_sqrtsoftplus_v41_hcu()
    print("moe_fused_gate_sqrtsoftplus_v41_hcu passed")
    test_moe_fused_gate_sqrtsoftplus_v41_hcu_2d_bias()
    print("moe_fused_gate_sqrtsoftplus_v41_hcu 2d bias passed")
