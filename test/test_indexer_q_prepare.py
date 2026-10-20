# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Q-only BF16 rotary/Hadamard preparation and dynamic HIP graph replay."""
import pytest
import torch
from lightop import op


def require_hcu():
    if not torch.cuda.is_available():
        pytest.skip("requires HCU")
    arch = getattr(torch.cuda.get_device_properties(0), "gcnArchName", "")
    if arch.split(":")[0] != "gfx936":
        pytest.skip("validated on gfx936")


def reference(query, positions, cache, gate, neox):
    cos, sin = cache[positions.long()].split(32, -1)
    q = query[..., :64]
    if neox:
        cos, sin = cos.repeat(1, 2), sin.repeat(1, 2)
        partner = torch.cat((-q[..., 32:], q[..., :32]), -1)
    else:
        cos, sin = cos.repeat_interleave(2, -1), sin.repeat_interleave(2, -1)
        partner = torch.stack((-q[..., 1::2], q[..., ::2]), -1).flatten(-2)
    rotated = query.clone()
    rotated[..., :64] = q * cos[:, None, :] + partner * sin[:, None, :]
    hadamard = torch.ones(1, 1, device=query.device, dtype=torch.float64)
    while hadamard.shape[0] < 128:
        hadamard = torch.cat((torch.cat((hadamard, hadamard), 1),
                              torch.cat((hadamard, -hadamard), 1)), 0)
    # Retain the original BF16 GEMM output and BF16 normalization roundings.
    # FP64 makes the reference independent of a particular BLAS reduction tree.
    transformed = torch.nn.functional.linear(rotated.double(), hadamard).bfloat16()
    transformed = transformed * (128 ** -.5)
    maximum = transformed.float().abs().amax(-1)
    scale = maximum / 127
    inverse = torch.where(maximum == 0, 0., 127 / maximum)
    quantized = torch.round(transformed.float() * inverse[..., None]).clamp(-128, 127).char()
    weights = ((gate * (32 ** -.5)) * (128 ** -.5)) * scale
    return quantized, weights


def invoke(q, positions, cache, gate, neox):
    return op.indexer_q_rope_hadamard_quant(
        q, positions, cache, gate, neox, 128 ** -.5, 32 ** -.5, 128 ** -.5
    )


def compare(actual, expected):
    # Different FP32 Hadamard reduction orders can cross a BF16 rounding tie.
    torch.testing.assert_close(actual[0].short(), expected[0].short(), rtol=0, atol=1)
    error = (actual[1] - expected[1]).abs().max()
    assert error <= .008 * expected[1].abs().max().clamp_min(1e-20)


@pytest.mark.parametrize("rows", [5, 10, 15, 20])
@pytest.mark.parametrize("neox", [False, True])
@pytest.mark.parametrize("stride", [128, 160])
@pytest.mark.parametrize("position_dtype", [torch.int32, torch.int64])
def test_values_and_input_preservation(rows, neox, stride, position_dtype):
    require_hcu()
    torch.manual_seed(20261020 + rows)
    storage = torch.randn(rows, 32, stride, device="cuda", dtype=torch.bfloat16)
    query = storage[..., :128]
    positions = torch.arange(rows, device="cuda", dtype=position_dtype) * 31
    cache = torch.randn(1024, 64, device="cuda", dtype=torch.bfloat16)
    gate = torch.randn(rows, 32, device="cuda", dtype=torch.float32)
    for zero in [False, True]:
        if zero:
            query.zero_()
        originals = [x.clone() for x in [storage, positions, cache, gate]]
        compare(invoke(query, positions, cache, gate, neox),
                reference(query, positions, cache, gate, neox))
        for actual, original in zip([storage, positions, cache, gate], originals):
            assert torch.equal(actual, original)


@pytest.mark.parametrize("rows", [10, 15, 20])
@pytest.mark.parametrize("neox", [False, True])
def test_dynamic_graph_replay(rows, neox):
    require_hcu()
    query = torch.randn(rows, 32, 128, device="cuda", dtype=torch.bfloat16)
    positions = torch.arange(rows, device="cuda", dtype=torch.int64)
    cache = torch.randn(1024, 64, device="cuda", dtype=torch.bfloat16)
    gate = torch.randn(rows, 32, device="cuda", dtype=torch.float32)
    for _ in range(3):
        invoke(query, positions, cache, gate, neox)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = invoke(query, positions, cache, gate, neox)
    for _ in range(5):
        query.copy_(torch.randn_like(query))
        positions.add_(17)
        cache.copy_(torch.randn_like(cache))
        gate.copy_(torch.randn_like(gate))
        graph.replay()
        torch.cuda.synchronize()
        compare(output, reference(query, positions, cache, gate, neox))


def test_empty_and_invalid_inputs():
    require_hcu()
    query = torch.empty(20, 32, 128, device="cuda", dtype=torch.bfloat16)
    positions = torch.arange(20, device="cuda", dtype=torch.int64)
    cache = torch.empty(1024, 64, device="cuda", dtype=torch.bfloat16)
    gate = torch.empty(20, 32, device="cuda", dtype=torch.float32)
    q, w = invoke(query[:0], positions[:0], cache, gate[:0], False)
    assert q.shape == (0, 32, 128) and w.shape == (0, 32)
    for args in [(query.float(), positions, cache, gate),
                 (query, positions.cpu(), cache, gate),
                 (query, positions, cache.float(), gate),
                 (query, positions, cache, gate.bfloat16()),
                 (query[:, :16], positions, cache, gate),
                 (query, positions, cache[:, :32], gate)]:
        with pytest.raises(RuntimeError):
            invoke(*args, False)
