# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Independent oracle, padding preservation, and live-count graph checks."""

import pytest
import torch

from lightop import op
from lightop.activation import swiglu_oai_fp8_quant_masked

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a GPU")


def _buffers(x):
    q = torch.empty(
        (*x.shape[:2], x.shape[-1] // 2), device=x.device, dtype=torch.float8_e4m3fn
    )
    q.view(torch.uint8).fill_(165)
    scales = torch.full((*x.shape[:2], 1), -123.0, device=x.device)
    return q, scales


def _check(x, counts, q, scales, alpha=1.702, limit=7.0):
    gate, up = x.float().chunk(2, dim=-1)
    gate = gate.clamp(max=limit)
    activated = gate * torch.sigmoid(alpha * gate) * (up.clamp(-limit, limit) + 1)
    expected_scale = activated.abs().amax(dim=-1, keepdim=True).clamp_min(1e-10) / 448
    normalized = (activated / expected_scale).clamp(-448, 448)
    expected = normalized.to(torch.float8_e4m3fn).float()
    for expert, raw_count in enumerate(counts.cpu().tolist()):
        count = max(0, min(raw_count, x.shape[1]))
        torch.testing.assert_close(
            scales[expert, :count],
            expected_scale[expert, :count],
            rtol=2e-6,
            atol=1e-12,
        )
        # Native exp2 and Torch sigmoid can straddle an FP8 rounding midpoint.
        # Check against an independent FP32 oracle within one E4M3 step.
        torch.testing.assert_close(
            q[expert, :count].float(), expected[expert, :count], rtol=0.13, atol=2**-9
        )
        actual_bits = q[expert, :count].view(torch.uint8).to(torch.int16)
        expected_bits = (
            expected[expert, :count]
            .to(torch.float8_e4m3fn)
            .view(torch.uint8)
            .to(torch.int16)
        )
        if count:
            assert ((actual_bits & 127) - (expected_bits & 127)).abs().max().item() <= 1
        both_nonzero = ((actual_bits & 127) != 0) & ((expected_bits & 127) != 0)
        assert torch.equal(
            actual_bits[both_nonzero] & 128, expected_bits[both_nonzero] & 128
        )
        assert (q[expert, count:].view(torch.uint8) == 165).all()
        assert (scales[expert, count:] == -123.0).all()


@pytest.mark.parametrize("experts", [0, 1, 4, 16])
@pytest.mark.parametrize("capacity", [0, 1, 33])
@pytest.mark.parametrize("width", [8, 128, 2304, 3072, 4096])
@pytest.mark.parametrize("distribution", ["normal", "zero", "clamp"])
def test_oracle_and_padding(experts, capacity, width, distribution):
    torch.manual_seed(20261027)
    x = torch.randn(experts, capacity, 2 * width, device="cuda", dtype=torch.bfloat16)
    if distribution == "zero":
        x.zero_()
    elif distribution == "clamp":
        pattern = torch.tensor(
            [-20, -7.0625, -7, -1, 0, 7, 7.0625, 20], device=x.device
        )
        x.copy_(pattern.repeat(x.numel() // 8).reshape_as(x))
    counts = torch.tensor(
        [(-2, capacity + 3, 0, max(0, capacity - 1))[i % 4] for i in range(experts)],
        device=x.device,
        dtype=torch.int32,
    )
    original = x.clone()
    q, scales = _buffers(x)
    returned = swiglu_oai_fp8_quant_masked(x, counts, q, scales)
    assert returned[0] is q and returned[1] is scales
    _check(x, counts, q, scales)
    assert torch.equal(x, original)


@pytest.mark.parametrize("experts", [4, 16])
def test_graph_uses_updated_inputs_and_counts(experts):
    x = torch.randn(experts, 33, 6144, device="cuda", dtype=torch.bfloat16)
    counts = torch.zeros(experts, device=x.device, dtype=torch.int32)
    q, scales = _buffers(x)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        swiglu_oai_fp8_quant_masked(x, counts, q, scales)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            swiglu_oai_fp8_quant_masked(x, counts, q, scales)
        for live in [0, 1, 17, 33, 2]:
            x.mul_(-0.75)
            counts.fill_(live)
            q.view(torch.uint8).fill_(165)
            scales.fill_(-123)
            graph.replay()
            _check(x, counts, q, scales)
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()


def test_flat_and_general_schedules_agree():
    x = torch.randn(16, 33, 6144, device="cuda", dtype=torch.bfloat16)
    counts = torch.arange(16, device=x.device, dtype=torch.int32) * 3
    q1, s1 = _buffers(x)
    q2, s2 = _buffers(x)
    op.swiglu_oai_fp8_quant_masked(x, q1, s1, counts, 1.702, 7.0, 32, 256)
    op.swiglu_oai_fp8_quant_masked_flat(x, q2, s2, counts, 1.702, 7.0, 256, 256)
    assert torch.equal(q1.view(torch.uint8), q2.view(torch.uint8))
    assert torch.equal(s1, s2)


@pytest.mark.parametrize("width", [0, 7, 4104])
def test_invalid_width_rejected(width):
    x = torch.empty(4, 1, width * 2, device="cuda", dtype=torch.bfloat16)
    counts = torch.ones(4, device=x.device, dtype=torch.int32)
    with pytest.raises(RuntimeError, match="width"):
        swiglu_oai_fp8_quant_masked(x, counts)


@pytest.mark.parametrize("alpha,limit", [(0, 7), (1.702, -1), (float("inf"), 7)])
def test_invalid_activation_rejected(alpha, limit):
    x = torch.empty(4, 1, 16, device="cuda", dtype=torch.bfloat16)
    counts = torch.ones(4, device=x.device, dtype=torch.int32)
    with pytest.raises(RuntimeError, match="positive finite"):
        swiglu_oai_fp8_quant_masked(x, counts, alpha=alpha, limit=limit)


def test_noncontiguous_counts_rejected():
    x = torch.empty(4, 1, 16, device="cuda", dtype=torch.bfloat16)
    counts = torch.ones(8, device=x.device, dtype=torch.int32)[::2]
    with pytest.raises(RuntimeError, match="counts"):
        swiglu_oai_fp8_quant_masked(x, counts)


@pytest.mark.parametrize("alpha,limit", [(0.5, 3.0), (2.0, 10.0)])
def test_nondefault_activation(alpha, limit):
    x = torch.randn(16, 3, 6144, device="cuda", dtype=torch.bfloat16) * 10
    counts = torch.full((16,), 3, device=x.device, dtype=torch.int32)
    q, scales = _buffers(x)
    swiglu_oai_fp8_quant_masked(x, counts, q, scales, alpha=alpha, limit=limit)
    _check(x, counts, q, scales, alpha=alpha, limit=limit)


@pytest.mark.parametrize(
    "fault",
    [
        "dtype",
        "output_dtype",
        "output_shape",
        "counts_dtype",
        "counts_length",
        "scales_dtype",
        "alias",
        "input_alignment",
        "output_alignment",
    ],
)
def test_invalid_storage_contract(fault):
    x = torch.empty(4, 2, 16, device="cuda", dtype=torch.bfloat16)
    counts = torch.ones(4, device=x.device, dtype=torch.int32)
    q, scales = _buffers(x)
    if fault == "dtype":
        x = x.float()
    elif fault == "output_dtype":
        q = q.float()
    elif fault == "output_shape":
        q = torch.empty(4, 2, 16, device=x.device, dtype=q.dtype)
    elif fault == "counts_dtype":
        counts = counts.long()
    elif fault == "counts_length":
        counts = counts[:3]
    elif fault == "scales_dtype":
        scales = scales.double()
    elif fault == "alias":
        scales = (
            q.view(torch.uint8).reshape(-1)[:32].view(torch.float32).reshape(4, 2, 1)
        )
    elif fault == "input_alignment":
        x = torch.empty(x.numel() + 1, device=x.device, dtype=x.dtype)[1:].reshape_as(x)
    elif fault == "output_alignment":
        q = torch.empty(q.numel() + 1, device=x.device, dtype=q.dtype)[1:].reshape_as(q)
    with pytest.raises(RuntimeError):
        swiglu_oai_fp8_quant_masked(x, counts, q, scales)
