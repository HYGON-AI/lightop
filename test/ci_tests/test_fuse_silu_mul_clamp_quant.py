# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

import os

import pytest
import torch
import torch.nn.functional as F

from lightop import (
    fuse_silu_mul_clamp_quant,
    fuse_silu_mul_clamp_quant_ep,
)


def _reference(
    input: torch.Tensor,
    limit: float,
    use_gelu: bool = False,
    smooth_scale: torch.Tensor | None = None,
    round_silu: bool = True,
):
    hidden = input.shape[-1] // 2
    if use_gelu:
        gate = input[..., :hidden]
        up = input[..., hidden:]
        activated = F.gelu(gate, approximate="none") * up
        activated = activated.clamp(min=-limit, max=limit)
    else:
        gate = input[..., :hidden].clamp(max=limit)
        up = input[..., hidden:].clamp(min=-limit, max=limit)
        if round_silu:
            activated = F.silu(gate) * up
        else:
            activated = (F.silu(gate.float()) * up.float()).to(input.dtype)

    if smooth_scale is not None:
        activated = activated * smooth_scale.to(dtype=input.dtype)

    absmax = activated.float().abs().amax(dim=-1, keepdim=True)
    absmax = absmax.clamp_min(1.0e-10)
    scales = absmax / 127.0
    quantized = torch.round(activated.float() / scales)
    quantized = quantized.clamp(-128, 127).to(torch.int8)
    return quantized, scales


def _assert_quantized_close(actual, expected):
    actual_q, actual_scale = actual
    expected_q, expected_scale = expected
    for a, b in zip(actual, expected):
        assert a.shape == b.shape and a.dtype == b.dtype
    if actual_q.numel():
        assert (actual_q.to(torch.int16) - expected_q.to(torch.int16)).abs().max() <= 1
    torch.testing.assert_close(
        actual_scale, expected_scale, rtol=1.0e-6, atol=1.0e-8
    )


@pytest.fixture
def silu_and_mul_clamp():
    if os.environ.get("LIGHTOP_TEST_SGLANG_INTEGRATION") != "1":
        pytest.skip("set LIGHTOP_TEST_SGLANG_INTEGRATION=1 for SGLang integration")
    from sglang.kernels.ops.moe.dsv4 import silu_and_mul_clamp

    return silu_and_mul_clamp


def test_clamp_quant_activation_exports():
    from lightop import activation

    assert activation.fuse_silu_mul_clamp_quant is fuse_silu_mul_clamp_quant
    assert activation.fuse_silu_mul_clamp_quant_ep is fuse_silu_mul_clamp_quant_ep
    assert "fuse_silu_mul_clamp_quant" in activation.__all__
    assert "fuse_silu_mul_clamp_quant_ep" in activation.__all__


def test_product_rounding_native_capability():
    from lightop.fuse_silu_mul_quant import is_silu_clamp_product_rounding_available

    assert is_silu_clamp_product_rounding_available()


@pytest.mark.parametrize(
    "rows,hidden",
    [(0, 2304), (1, 2304), (16, 2304), (4096, 2304), (7, 128), (7, 513)],
)
@pytest.mark.parametrize("use_smooth", [False, True])
def test_product_rounding_matches_torch(rows, hidden, use_smooth):
    torch.manual_seed(1117)
    x = torch.randn(rows, 2 * hidden, device="cuda", dtype=torch.bfloat16) * 4
    if rows:
        boundaries = torch.tensor(
            [-12, -10, -0.5, 0, 0.5, 9.5, 10, 12], device=x.device, dtype=x.dtype
        )
        x[0, :8] = boundaries
        x[0, hidden : hidden + 8] = boundaries
    if rows > 1:
        x[1].zero_()
    before = x.clone()
    smooth = torch.rand(hidden, device=x.device) + 0.5 if use_smooth else None
    actual = fuse_silu_mul_clamp_quant(
        x, 10.0, smooth_scale=smooth, round_silu=False
    )
    expected = _reference(x, 10.0, smooth_scale=smooth, round_silu=False)
    assert torch.equal(x, before)
    _assert_quantized_close(actual, expected)


@pytest.mark.parametrize("hidden", [128, 513, 2304])
@pytest.mark.parametrize("expect_m", [-1, 1, 3, 32])
@pytest.mark.parametrize("use_smooth", [False, True])
def test_product_rounding_ep_matches_torch(hidden, expect_m, use_smooth):
    torch.manual_seed(1118)
    experts, rows = 4, 9
    x = torch.randn(experts, rows, 2 * hidden, device="cuda", dtype=torch.bfloat16) * 4
    before = x.clone()
    mask = torch.tensor([9, 5, 1, 0], device=x.device, dtype=torch.int32)
    smooth = torch.rand(hidden, device=x.device) + 0.5 if use_smooth else None
    q = torch.full((experts, rows, hidden), 99, device=x.device, dtype=torch.int8)
    scales = torch.full((experts, rows, 1), -1.0, device=x.device)
    actual = fuse_silu_mul_clamp_quant_ep(
        x, 10.0, mask, expect_m, q, scales, smooth_scale=smooth, round_silu=False
    )
    expected = _reference(x, 10.0, smooth_scale=smooth, round_silu=False)
    valid = torch.arange(rows, device=x.device)[None, :] < mask[:, None]
    _assert_quantized_close(
        tuple(t[valid] for t in actual), tuple(t[valid] for t in expected)
    )
    assert torch.equal(x, before)
    assert actual[0].data_ptr() == q.data_ptr()
    assert actual[1].data_ptr() == scales.data_ptr()
    assert torch.all(q[~valid] == 99)
    assert torch.all(scales[~valid] == -1.0)
    dispatched = fuse_silu_mul_clamp_quant(
        x, 10.0, mask_m=mask, expect_m=expect_m,
        smooth_scale=smooth, round_silu=False,
    )
    for a, b in zip(dispatched, actual):
        assert torch.equal(a[valid], b[valid])


def test_product_rounding_graph_replay_matches_torch():
    x = torch.randn(16, 4608, device="cuda", dtype=torch.bfloat16)
    fuse_silu_mul_clamp_quant(x, 10.0, round_silu=False)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = fuse_silu_mul_clamp_quant(x, 10.0, round_silu=False)
    pointers = [t.data_ptr() for t in actual]
    for repeat in range(3):
        x.normal_().mul_(4)
        x[repeat].zero_()
        before = x.clone()
        graph.replay()
        _assert_quantized_close(actual, _reference(x, 10.0, round_silu=False))
        assert torch.equal(x, before)
        assert pointers == [t.data_ptr() for t in actual]


@pytest.mark.parametrize("tokens", [1, 8])
@pytest.mark.parametrize("hidden", [128, 768, 1536, 2048, 12288])
@pytest.mark.parametrize("use_gelu", [False, True])
def test_fuse_silu_mul_clamp_quant_glm_shapes(tokens, hidden, use_gelu):
    torch.manual_seed(1234)
    limit = 10.0
    input = (
        torch.randn((tokens, 2 * hidden), device="cuda", dtype=torch.bfloat16)
        * 4
    )
    # Exercise both clamp boundaries, not only the normal random range.
    input[0, :8] = torch.tensor(
        [-12.0, -10.0, -9.5, 0.0, 9.5, 10.0, 10.5, 12.0],
        device=input.device,
        dtype=input.dtype,
    )
    input[0, hidden : hidden + 8] = torch.tensor(
        [-12.0, -10.0, -9.5, 0.0, 9.5, 10.0, 10.5, 12.0],
        device=input.device,
        dtype=input.dtype,
    )

    actual_q, actual_scale = fuse_silu_mul_clamp_quant(
        input, limit, use_gelu=use_gelu
    )
    expected_q, expected_scale = _reference(input, limit, use_gelu=use_gelu)

    assert torch.max(
        torch.abs(actual_q.to(torch.int16) - expected_q.to(torch.int16))
    ).item() <= 1
    torch.testing.assert_close(
        actual_scale, expected_scale, rtol=1.0e-6, atol=1.0e-8
    )


@pytest.mark.parametrize(
    "rows,hidden", [(0, 2304), (1, 2304), (16, 2304), (4096, 2304), (7, 128)]
)
def test_product_rounding_matches_sglang_clamp_then_native_int8(
    rows, hidden, silu_and_mul_clamp
):
    from lightop.quant import per_token_quant_int8
    torch.manual_seed(1117)
    x = torch.randn(rows, 2 * hidden, device="cuda", dtype=torch.bfloat16) * 4
    if rows:
        boundaries = torch.tensor(
            [-12, -10, -0.5, 0, 0.5, 9.5, 10, 12], device=x.device, dtype=x.dtype
        )
        x[0, :8] = boundaries
        x[0, hidden : hidden + 8] = boundaries
    source = x.clone()
    ref = x.new_empty((rows, hidden))
    if rows:
        silu_and_mul_clamp(x, ref, 10.0)
        expected = per_token_quant_int8(ref)
    else:
        expected = (
            torch.empty((0, hidden), device=x.device, dtype=torch.int8),
            torch.empty((0, 1), device=x.device, dtype=torch.float32),
        )
    actual = fuse_silu_mul_clamp_quant(x, 10.0, round_silu=False)
    assert torch.equal(x, source)
    for a, b in zip(actual, expected):
        assert a.shape == b.shape and a.dtype == b.dtype
        assert torch.equal(
            a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8)
        )


def test_product_rounding_graph_replay_reads_changed_input(silu_and_mul_clamp):
    from lightop.quant import per_token_quant_int8
    x = torch.randn(16, 4608, device="cuda", dtype=torch.bfloat16)
    fuse_silu_mul_clamp_quant(x, 10.0, round_silu=False)
    ref = x.new_empty((16, 2304))
    silu_and_mul_clamp(x, ref, 10.0)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = fuse_silu_mul_clamp_quant(x, 10.0, round_silu=False)
    pointers = [t.data_ptr() for t in actual]
    for repeat in range(3):
        x.normal_()
        x[repeat].zero_()
        before = x.clone()
        graph.replay()
        silu_and_mul_clamp(x, ref, 10.0)
        expected = per_token_quant_int8(ref)
        assert torch.equal(x, before)
        assert pointers == [t.data_ptr() for t in actual]
        for a, b in zip(actual, expected):
            assert torch.equal(
                a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8)
            )


@pytest.mark.parametrize("expect_m", [-1, 1, 3, 32])
@pytest.mark.parametrize("use_smooth", [False, True])
def test_product_rounding_ep_matches_sglang_with_mask(
    expect_m, use_smooth, silu_and_mul_clamp
):
    from lightop.quant import per_token_quant_int8
    torch.manual_seed(1118)
    experts, rows, hidden = 4, 9, 2304
    x = torch.randn(experts, rows, 2 * hidden, device="cuda", dtype=torch.bfloat16) * 4
    before = x.clone()
    mask = torch.tensor([9, 5, 1, 0], device=x.device, dtype=torch.int32)
    smooth = torch.rand(hidden, device=x.device) + 0.5 if use_smooth else None
    ref = x.new_empty((experts * rows, hidden))
    silu_and_mul_clamp(x.reshape(experts * rows, 2 * hidden), ref, 10.0)
    if smooth is not None:
        ref = ref * smooth.to(dtype=x.dtype)
    expected = per_token_quant_int8(ref.reshape(experts, rows, hidden))
    q = torch.full((experts, rows, hidden), 99, device=x.device, dtype=torch.int8)
    scales = torch.full((experts, rows, 1), -1.0, device=x.device)
    actual = fuse_silu_mul_clamp_quant_ep(
        x,
        10.0,
        mask,
        expect_m,
        q,
        scales,
        smooth_scale=smooth,
        round_silu=False,
    )
    valid = torch.arange(rows, device=x.device)[None, :] < mask[:, None]
    assert torch.equal(x, before)
    assert actual[0].data_ptr() == q.data_ptr()
    assert actual[1].data_ptr() == scales.data_ptr()
    for a, b in zip(actual, expected):
        assert torch.equal(a[valid].view(torch.uint8), b[valid].view(torch.uint8))
    assert torch.all(q[~valid] == 99)
    assert torch.all(scales[~valid] == -1.0)


def test_fuse_silu_mul_clamp_quant_zero_row_scale():
    input = torch.zeros((1, 4096), device="cuda", dtype=torch.bfloat16)

    actual_q, actual_scale = fuse_silu_mul_clamp_quant(input, 10.0)

    assert torch.count_nonzero(actual_q).item() == 0
    torch.testing.assert_close(
        actual_scale,
        torch.full_like(actual_scale, 1.0e-10 / 127.0),
        rtol=1.0e-6,
        atol=0,
    )


@pytest.mark.parametrize("hidden", [128, 513, 1536])
@pytest.mark.parametrize("expect_m", [-1, 1, 3, 32])
@pytest.mark.parametrize("use_gelu", [False, True])
def test_fuse_silu_mul_clamp_quant_ep_masked(hidden, expect_m, use_gelu):
    torch.manual_seed(2026)
    experts, tokens = 4, 9
    limit = 10.0
    mask_m = torch.tensor(
        [tokens, 5, 1, 0], device="cuda", dtype=torch.int32
    )
    input = (
        torch.randn(
            (experts, tokens, 2 * hidden),
            device="cuda",
            dtype=torch.bfloat16,
        )
        * 4
    )

    actual_q, actual_scale = fuse_silu_mul_clamp_quant_ep(
        input, limit, mask_m, expect_m, use_gelu=use_gelu
    )
    dispatched_q, dispatched_scale = fuse_silu_mul_clamp_quant(
        input,
        limit,
        mask_m=mask_m,
        expect_m=expect_m,
        use_gelu=use_gelu,
    )
    expected_q, expected_scale = _reference(input, limit, use_gelu=use_gelu)
    valid = (
        torch.arange(tokens, device="cuda").unsqueeze(0)
        < mask_m.unsqueeze(1)
    )

    assert torch.max(
        torch.abs(
            actual_q[valid].to(torch.int16)
            - expected_q[valid].to(torch.int16)
        )
    ).item() <= 1
    torch.testing.assert_close(
        actual_scale[valid],
        expected_scale[valid],
        rtol=1.0e-6,
        atol=1.0e-8,
    )
    assert torch.equal(dispatched_q[valid], actual_q[valid])
    torch.testing.assert_close(
        dispatched_scale[valid],
        actual_scale[valid],
        rtol=0,
        atol=0,
    )


def test_fuse_silu_mul_clamp_quant_ep_requires_mask():
    input = torch.empty((2, 4, 256), device="cuda", dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="mask_m is required"):
        fuse_silu_mul_clamp_quant(input, 10.0, expect_m=2)


def test_fuse_gelu_mul_clamp_smooth_quant_matches_reference_ordering():
    torch.manual_seed(20260807)
    input = (
        torch.randn((17, 2048), device="cuda", dtype=torch.bfloat16) * 4
    )
    smooth_scale = torch.empty(
        (1024,), device="cuda", dtype=torch.float32
    ).uniform_(0.25, 1.75)

    actual_q, actual_scale = fuse_silu_mul_clamp_quant(
        input,
        8.0,
        use_gelu=True,
        smooth_scale=smooth_scale,
    )
    expected_q, expected_scale = _reference(
        input,
        8.0,
        use_gelu=True,
        smooth_scale=smooth_scale,
    )

    assert torch.max(
        torch.abs(actual_q.to(torch.int16) - expected_q.to(torch.int16))
    ).item() <= 1
    torch.testing.assert_close(
        actual_scale, expected_scale, rtol=1.0e-6, atol=1.0e-8
    )


def test_fuse_gelu_mul_clamp_smooth_quant_ep_masked():
    torch.manual_seed(20260808)
    experts, tokens, hidden = 16, 8, 1024
    input = torch.randn(
        (experts, tokens, 2 * hidden),
        device="cuda",
        dtype=torch.bfloat16,
    )
    smooth_scale = torch.empty(
        (hidden,), device="cuda", dtype=torch.float32
    ).uniform_(0.5, 2.0)
    mask_m = torch.tensor(
        [8, 7, 6, 5, 4, 3, 2, 1, 8, 6, 4, 2, 7, 5, 3, 0],
        device="cuda",
        dtype=torch.int32,
    )

    actual_q, actual_scale = fuse_silu_mul_clamp_quant_ep(
        input,
        8.0,
        mask_m,
        expect_m=4,
        use_gelu=True,
        smooth_scale=smooth_scale,
    )
    expected_q, expected_scale = _reference(
        input,
        8.0,
        use_gelu=True,
        smooth_scale=smooth_scale,
    )
    valid = (
        torch.arange(tokens, device="cuda").unsqueeze(0)
        < mask_m.unsqueeze(1)
    )

    assert torch.max(
        torch.abs(
            actual_q[valid].to(torch.int16)
            - expected_q[valid].to(torch.int16)
        )
    ).item() <= 1
    torch.testing.assert_close(
        actual_scale[valid],
        expected_scale[valid],
        rtol=1.0e-6,
        atol=1.0e-8,
    )


def test_fuse_gelu_mul_clamp_quant_default_preserves_legacy_api():
    torch.manual_seed(20260809)
    input = torch.randn((3, 256), device="cuda", dtype=torch.bfloat16)

    default_q, default_scale = fuse_silu_mul_clamp_quant(
        input, 8.0, use_gelu=True
    )
    none_q, none_scale = fuse_silu_mul_clamp_quant(
        input, 8.0, use_gelu=True, smooth_scale=None
    )

    assert torch.equal(default_q, none_q)
    torch.testing.assert_close(default_scale, none_scale, rtol=0, atol=0)
