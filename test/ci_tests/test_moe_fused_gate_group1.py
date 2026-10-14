# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Group-1 routing regressions: scale semantics and wave-wide tie selection."""

import pytest
import torch

from lightop import op


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires a CUDA/ROCm device"
)


def _reference(logits, bias, topk, shared=0, scale=2.827, apply_scale=True,
               groups=1, topk_group=1):
    probabilities = logits.float().sigmoid()
    scores = probabilities + bias.float().unsqueeze(0)
    if groups > 1:
        grouped = scores.reshape(logits.shape[0], groups, -1)
        group_scores = grouped.topk(2, dim=-1).values.sum(dim=-1)
        group_ids = torch.argsort(
            group_scores, dim=-1, descending=True, stable=True,
        )[:, :topk_group]
        group_mask = torch.zeros_like(group_scores, dtype=torch.bool)
        group_mask.scatter_(1, group_ids, True)
        expert_mask = group_mask.unsqueeze(-1).expand_as(grouped).reshape_as(scores)
        scores = scores.masked_fill(~expert_mask, -float("inf"))
    routed_k = topk - shared
    # Stable ordering makes the equal-score reference explicit: smaller expert
    # IDs win ties, matching the wave reduction in the fixed group-1 kernel.
    ids = torch.argsort(scores, dim=-1, descending=True, stable=True)[:, :routed_k]
    weights = probabilities.gather(1, ids)
    weights = weights / weights.sum(dim=-1, keepdim=True)
    effective_scale = scale if scale > 0 else 1.0
    if apply_scale:
        weights = weights * effective_scale
    if shared:
        shared_ids = torch.arange(
            logits.shape[1], logits.shape[1] + shared,
            device=logits.device, dtype=ids.dtype,
        ).expand(logits.shape[0], -1)
        shared_weights = torch.full(
            (logits.shape[0], shared),
            1.0 if apply_scale else 1.0 / effective_scale,
            device=logits.device, dtype=torch.float32,
        )
        ids = torch.cat((ids, shared_ids), dim=-1)
        weights = torch.cat((weights, shared_weights), dim=-1)
    return weights, ids.int()


def _assert_pairs(actual, expected):
    actual_weights, actual_ids = actual
    expected_weights, expected_ids = expected
    actual_order = actual_ids.argsort(dim=-1)
    expected_order = expected_ids.argsort(dim=-1)
    torch.testing.assert_close(
        actual_ids.gather(1, actual_order).long(),
        expected_ids.gather(1, expected_order).long(), rtol=0, atol=0,
    )
    torch.testing.assert_close(
        actual_weights.gather(1, actual_order),
        expected_weights.gather(1, expected_order), rtol=2e-5, atol=2e-6,
    )


def _random_inputs(tokens, experts, input_dtype=torch.float32,
                   bias_dtype=torch.float32):
    torch.manual_seed(903 + tokens + experts)
    logits = torch.randn(tokens, experts, device="cuda", dtype=input_dtype)
    bias = torch.randn(experts, device="cuda", dtype=bias_dtype) * 0.1
    return logits, bias


@pytest.mark.parametrize("tokens", [1, 4, 64, 511, 512, 1024])
def test_hy4_group1_default_applies_routing_scale(tokens):
    logits, bias = _random_inputs(tokens, 256)
    # Seven arguments exercise the default apply_scale=True used by HCU.
    actual = op.moe_fused_gate(logits, bias, 1, 1, 8, 0, 2.827)
    _assert_pairs(actual, _reference(logits, bias, 8))
    torch.testing.assert_close(
        actual[0].sum(dim=-1),
        torch.full((tokens,), 2.827, device="cuda"), rtol=2e-6, atol=2e-6,
    )


@pytest.mark.parametrize("output_indices_int64", [False, True])
@pytest.mark.parametrize("tokens", [1, 17, 511, 512])
@pytest.mark.parametrize("apply_scale", [False, True])
def test_group1_index_width_is_explicit_and_value_stable(
    output_indices_int64, tokens, apply_scale,
):
    logits, bias = _random_inputs(tokens, 288, torch.bfloat16, torch.float32)
    actual = op.moe_fused_gate(
        logits, bias, 1, 1, 8, 0, 2.827, apply_scale, output_indices_int64,
    )
    expected_dtype = torch.int64 if output_indices_int64 else torch.int32
    assert actual[1].dtype == expected_dtype
    _assert_pairs(actual, _reference(logits, bias, 8, apply_scale=apply_scale))
    baseline = op.moe_fused_gate(logits, bias, 1, 1, 8, 0, 2.827, apply_scale)
    assert baseline[1].dtype == torch.int32
    torch.testing.assert_close(actual[1].long(), baseline[1].long(), rtol=0, atol=0)
    # Changing the index storage width must not perturb any weight bit.
    torch.testing.assert_close(actual[0], baseline[0], rtol=0, atol=0)


def test_group1_int64_tie_break_matches_int32_and_has_unique_ids():
    experts = 288
    logits = torch.zeros(3, experts, device="cuda", dtype=torch.bfloat16)
    bias = torch.zeros(experts, device="cuda", dtype=torch.float32)
    tied_ids = torch.arange(9, device="cuda") * 8
    bias[tied_ids] = 10.0
    int32_result = op.moe_fused_gate(logits, bias, 1, 1, 8, 0, 2.827, True)
    int64_result = op.moe_fused_gate(
        logits, bias, 1, 1, 8, 0, 2.827, True, True,
    )
    assert int32_result[1].dtype == torch.int32
    assert int64_result[1].dtype == torch.int64
    torch.testing.assert_close(
        int64_result[1], int32_result[1].to(torch.int64), rtol=0, atol=0,
    )
    torch.testing.assert_close(int64_result[0], int32_result[0], rtol=0, atol=0)
    _assert_pairs(int64_result, _reference(logits, bias, 8))
    sorted_ids = int64_result[1].sort(dim=-1).values
    assert bool(torch.all(sorted_ids[:, 1:] != sorted_ids[:, :-1]))


@pytest.mark.parametrize("experts", [128, 160, 192, 256, 288, 384, 512])
@pytest.mark.parametrize("topk", [7, 8, 14, 22])
@pytest.mark.parametrize("shared", [0, 1])
@pytest.mark.parametrize("apply_scale", [False, True])
@pytest.mark.parametrize("output_indices_int64", [False, True])
def test_group1_scale_flag_is_independent_of_expert_count(
    experts, topk, shared, apply_scale, output_indices_int64,
):
    logits, bias = _random_inputs(7, experts)
    actual = op.moe_fused_gate(
        logits, bias, 1, 1, topk, shared, 2.827, apply_scale, output_indices_int64,
    )
    assert actual[1].dtype == (torch.int64 if output_indices_int64 else torch.int32)
    _assert_pairs(
        actual, _reference(logits, bias, topk, shared, 2.827, apply_scale),
    )


@pytest.mark.parametrize("tokens", [1, 511, 512])
@pytest.mark.parametrize("dtypes", [
    (torch.bfloat16, torch.float32),
    (torch.bfloat16, torch.bfloat16),
    (torch.float16, torch.float16),
    (torch.float32, torch.bfloat16),
])
@pytest.mark.parametrize("apply_scale", [False, True])
def test_group1_mixed_dtypes_and_dispatch_boundary(tokens, dtypes, apply_scale):
    logits, bias = _random_inputs(tokens, 256, *dtypes)
    actual = op.moe_fused_gate(logits, bias, 1, 1, 8, 0, 2.827, apply_scale)
    _assert_pairs(actual, _reference(logits, bias, 8, apply_scale=apply_scale))


@pytest.mark.parametrize("experts", [128, 160, 192, 256, 288, 384, 512])
@pytest.mark.parametrize("tokens", [1, 512])
@pytest.mark.parametrize("unequal_probabilities", [False, True])
@pytest.mark.parametrize("output_indices_int64", [False, True])
def test_group1_tied_scores_select_one_expert_per_iteration(
    experts, tokens, unequal_probabilities, output_indices_int64,
):
    # Place one high-scoring expert in each of nine separate lane-owned
    # chunks.  Without an expert-ID tie-break, several winners are invalidated
    # in the same iteration and the kernel selects low-scoring experts next.
    if unequal_probabilities:
        row = torch.linspace(-3, 3, experts, device="cuda")
    else:
        row = torch.zeros(experts, device="cuda")
    logits = row.expand(tokens, -1).contiguous()
    bias = torch.zeros(experts, device="cuda")
    tied_ids = torch.arange(9, device="cuda") * 8
    bias[tied_ids] = 10.0 - row.sigmoid()[tied_ids]

    actual = op.moe_fused_gate(
        logits, bias, 1, 1, 8, 0, 2.827, True, output_indices_int64,
    )
    _assert_pairs(actual, _reference(logits, bias, 8))
    actual_ids = actual[1].long()
    selected_scores = (logits.sigmoid() + bias).gather(1, actual_ids)
    torch.testing.assert_close(
        selected_scores, torch.full_like(selected_scores, 10.0),
        rtol=0, atol=2e-6,
    )
    sorted_ids = actual_ids.sort(dim=-1).values
    assert bool(torch.all(sorted_ids[:, 1:] != sorted_ids[:, :-1]))


def _saturated_tie_inputs(tokens, sign):
    logits = torch.zeros(tokens, 288, device="cuda", dtype=torch.bfloat16)
    bias = torch.zeros(288, device="cuda", dtype=torch.float32)
    tied_ids = torch.arange(9, device="cuda") * 8
    logits[:, tied_ids] = (sign * torch.arange(20, 29, device="cuda")).bfloat16()
    bias[tied_ids] = 1.0
    # Positive logits round sigmoid to 1. Negative logits retain distinct
    # tiny probabilities, but adding the same FP32 bias rounds all scores to 1.
    scores = logits.float().sigmoid() + bias
    assert bool(torch.all(scores[:, tied_ids] == scores[:, tied_ids[:1]]))
    return logits, bias


@pytest.mark.parametrize("sign", [-1, 1])
@pytest.mark.parametrize("tokens", [1, 17, 512])
@pytest.mark.parametrize("apply_scale", [False, True])
@pytest.mark.parametrize("output_indices_int64", [False, True])
def test_group1_saturated_logits_with_equal_bias(
    sign, tokens, apply_scale, output_indices_int64,
):
    logits, bias = _saturated_tie_inputs(tokens, sign)
    actual = op.moe_fused_gate(
        logits, bias, 1, 1, 8, 0, 2.827, apply_scale, output_indices_int64,
    )
    expected_ids = (torch.arange(8, device="cuda") * 8).expand(tokens, -1)
    torch.testing.assert_close(actual[1].long(), expected_ids, rtol=0, atol=0)
    _assert_pairs(actual, _reference(logits, bias, 8, apply_scale=apply_scale))
    torch.testing.assert_close(
        actual[0].sum(dim=-1),
        torch.full((tokens,), 2.827 if apply_scale else 1.0, device="cuda"),
        rtol=2e-6, atol=2e-6,
    )


@pytest.mark.parametrize("apply_scale", [False, True])
@pytest.mark.parametrize("output_indices_int64", [False, True])
def test_group1_index_width_graph_replay_uses_updated_inputs(
    apply_scale, output_indices_int64,
):
    logits, bias = _random_inputs(17, 288, torch.bfloat16, torch.float32)

    def call_gate():
        return op.moe_fused_gate(
            logits, bias, 1, 1, 8, 0, 2.827, apply_scale, output_indices_int64,
        )

    warmup_stream = torch.cuda.Stream()
    warmup_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warmup_stream):
        for _ in range(3):
            call_gate()
    torch.cuda.current_stream().wait_stream(warmup_stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = call_gate()
    output_ptrs = tuple(tensor.data_ptr() for tensor in captured)

    cases = [
        _saturated_tie_inputs(17, 1),
        _saturated_tie_inputs(17, -1),
        (torch.zeros_like(logits), torch.zeros_like(bias)),
        _random_inputs(17, 288, torch.bfloat16, torch.float32),
    ]
    for next_logits, next_bias in cases:
        logits.copy_(next_logits)
        bias.copy_(next_bias)
        graph.replay()
        eager = call_gate()
        torch.cuda.synchronize()
        assert tuple(tensor.data_ptr() for tensor in captured) == output_ptrs
        assert captured[1].dtype == (
            torch.int64 if output_indices_int64 else torch.int32
        )
        for captured_tensor, eager_tensor in zip(captured, eager):
            torch.testing.assert_close(captured_tensor, eager_tensor, rtol=0, atol=0)
        _assert_pairs(captured, _reference(logits, bias, 8, apply_scale=apply_scale))


def test_int64_unsupported_specialization_fails_before_launch():
    logits, bias = _random_inputs(2, 384)
    # This grouped layout is valid for the legacy int32 path, but it has no
    # direct-int64 specialization. LightOp owns the capability check.
    baseline = op.moe_fused_gate(logits, bias, 4, 2, 8, 0, 2.827)
    assert baseline[1].dtype == torch.int32
    with pytest.raises(RuntimeError, match="output_indices_int64 is currently supported"):
        op.moe_fused_gate(logits, bias, 4, 2, 8, 0, 2.827, True, True)
    torch.cuda.synchronize()


@pytest.mark.parametrize("experts,groups,topk_group,topk,shared", [
    (64, 1, 1, 8, 0),
    (255, 1, 1, 8, 0),
    (256, 1, 1, 4, 0),
    (256, 1, 1, 16, 0),
    (256, 2, 1, 8, 0),
    (128, 8, 1, 8, 0),
    (256, 16, 1, 8, 0),
    (384, 4, 4, 17, 0),
    (384, 4, 1, 1, 0),
    (384, 4, 1, 8, 1),
    (896, 1, 1, 8, 0),
    (896, 1, 1, 16, 1),
    (254, 2, 1, 8, 2),
    (254, 2, 1, 7, 0),
])
def test_unsupported_configurations_fail_before_launch(
    experts, groups, topk_group, topk, shared,
):
    logits, bias = _random_inputs(2, experts)
    with pytest.raises(RuntimeError, match="moe_fused_gate: unsupported configuration"):
        op.moe_fused_gate(logits, bias, groups, topk_group, topk, shared, 2.827)
    torch.cuda.synchronize()


@pytest.mark.parametrize("groups,topk_group,topk,shared", [
    (0, 1, 8, 0),
    (3, 1, 8, 0),
    (1, 0, 8, 0),
    (1, 2, 8, 0),
    (1, 1, 0, 0),
    (1, 1, 8, -1),
    (1, 1, 8, 8),
    (1, 1, 512, 0),
])
def test_invalid_routing_parameters_fail_before_launch(groups, topk_group, topk, shared):
    logits, bias = _random_inputs(2, 256)
    with pytest.raises(RuntimeError, match="moe_fused_gate:"):
        op.moe_fused_gate(logits, bias, groups, topk_group, topk, shared, 2.827)
    torch.cuda.synchronize()


def test_group1_validates_tensor_contract_and_empty_batch():
    logits, bias = _random_inputs(2, 256)
    with pytest.raises(RuntimeError, match="bias must have shape"):
        op.moe_fused_gate(logits, bias[:-1], 1, 1, 8, 0, 2.827)
    with pytest.raises(RuntimeError, match="must be contiguous"):
        noncontiguous = torch.zeros(2, 512, device="cuda")[:, ::2]
        op.moe_fused_gate(noncontiguous, bias, 1, 1, 8, 0, 2.827)
    with pytest.raises(RuntimeError, match="unsupported input/bias dtype"):
        op.moe_fused_gate(logits.double(), bias.double(), 1, 1, 8, 0, 2.827)
    with pytest.raises(RuntimeError, match="routed_scaling_factor must be finite"):
        op.moe_fused_gate(logits, bias, 1, 1, 8, 0, float("nan"))
    weights, ids = op.moe_fused_gate(logits[:0], bias, 1, 1, 8, 0, 2.827)
    assert weights.shape == ids.shape == (0, 8)
    assert weights.dtype == torch.float32 and ids.dtype == torch.int32
    assert weights.device == ids.device == logits.device
    _, int64_ids = op.moe_fused_gate(
        logits[:0], bias, 1, 1, 8, 0, 2.827, True, True,
    )
    assert int64_ids.shape == (0, 8)
    assert int64_ids.dtype == torch.int64


@pytest.mark.parametrize("experts,groups,topk_group,topk,shared", [
    (128, 4, 2, 1, 0),
    (128, 4, 2, 16, 1),
    (256, 8, 4, 1, 0),
    (256, 8, 4, 32, 1),
    (384, 4, 2, 2, 0),
    (384, 4, 2, 16, 0),
    (384, 8, 4, 2, 0),
    (384, 8, 4, 32, 0),
    (384, 16, 8, 2, 0),
    (384, 16, 8, 64, 0),
    (254, 2, 1, 7, 2),
    (254, 2, 2, 7, 2),
    (254, 2, 1, 14, 2),
    (254, 2, 2, 14, 2),
    (896, 1, 1, 16, 0),
])
@pytest.mark.parametrize("tokens", [1, 512])
@pytest.mark.parametrize("tied", [False, True])
@pytest.mark.parametrize("apply_scale", [False, True])
def test_other_supported_specializations(
    experts, groups, topk_group, topk, shared, tokens, tied, apply_scale,
):
    logits, bias = _random_inputs(tokens, experts)
    if tied:
        logits.zero_()
        bias.zero_()
    actual = op.moe_fused_gate(
        logits, bias, groups, topk_group, topk, shared, 2.827, apply_scale,
    )
    _assert_pairs(actual, _reference(
        logits, bias, topk, shared, 2.827, apply_scale, groups, topk_group,
    ))


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
