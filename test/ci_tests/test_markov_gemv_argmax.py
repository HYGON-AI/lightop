# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import torch.nn.functional as F

from lightop.sampling import markov_gemv_add_argmax


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")


@pytest.mark.parametrize("logit_dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("batch", [1, 2, 3, 4])
@pytest.mark.parametrize("vocab", [1, 17, 1025, 133120])
def test_markov_projection(batch, vocab, logit_dtype):
    generator = torch.Generator(device="cuda").manual_seed(20261022)
    x = torch.randint(-4, 5, (batch, 256), generator=generator, device="cuda")
    w = torch.randint(-2, 3, (vocab, 256), generator=generator, device="cuda")
    x, w = x.bfloat16(), w.bfloat16()
    # A real Markov position has gaps between batch rows.
    base = torch.randn(batch, 8, vocab, generator=generator, device="cuda").to(logit_dtype)[:, 3]
    expected = F.linear(x, w) + base
    logits, tokens = markov_gemv_add_argmax(x, w, base)
    torch.testing.assert_close(logits, expected, rtol=0, atol=0)
    torch.testing.assert_close(tokens, expected.argmax(-1), rtol=0, atol=0)


@pytest.mark.parametrize("logit_dtype", [torch.bfloat16, torch.float32])
def test_argmax_ties_and_nan(logit_dtype):
    x = torch.zeros(4, 256, device="cuda", dtype=torch.bfloat16)
    w = torch.zeros(1025, 256, device="cuda", dtype=torch.bfloat16)
    base = torch.zeros(4, 1025, device="cuda", dtype=logit_dtype)
    base[:, 100] = base[:, 700] = 2
    base[0, 19] = base[0, 800] = float("nan")
    base[1, :] = -float("inf")
    base[2, 999] = float("inf")
    base[3, 3] = base[3, 800] = float("inf")
    logits, tokens = markov_gemv_add_argmax(x, w, base)
    torch.testing.assert_close(logits, base, rtol=0, atol=0, equal_nan=True)
    torch.testing.assert_close(tokens, base.argmax(-1), rtol=0, atol=0)


@pytest.mark.parametrize("logit_dtype", [torch.bfloat16, torch.float32])
def test_argmax_nan_payloads_and_signed_zero(logit_dtype):
    x = torch.zeros(4, 256, device="cuda", dtype=torch.bfloat16)
    w = torch.zeros(1025, 256, device="cuda", dtype=torch.bfloat16)
    base = torch.full((4, 1025), -1.0, device="cuda", dtype=logit_dtype)
    bits = base.view(torch.int32 if logit_dtype == torch.float32 else torch.int16)
    # A negative NaN with a different payload must win by its earlier index.
    bits[0, 19] = -0x003EDCBB if logit_dtype == torch.float32 else -63
    bits[0, 800] = 0x7FC12345 if logit_dtype == torch.float32 else 0x7FC2
    base[1, 2], base[1, 700] = -0.0, 0.0
    base[2, :] = -float("inf")
    base[3, 63] = base[3, 512] = float("inf")
    expected = F.linear(x, w) + base
    logits, tokens = markov_gemv_add_argmax(x, w, base)
    torch.testing.assert_close(logits, expected, rtol=0, atol=0, equal_nan=True)
    torch.testing.assert_close(tokens, expected.argmax(-1), rtol=0, atol=0)


@pytest.mark.parametrize("logit_dtype", [torch.bfloat16, torch.float32])
def test_argmax_subnormals(logit_dtype):
    x = torch.zeros(4, 256, device="cuda", dtype=torch.bfloat16)
    w = torch.zeros(1025, 256, device="cuda", dtype=torch.bfloat16)
    base = torch.full((4, 1025), -1.0, device="cuda", dtype=logit_dtype)
    if logit_dtype == torch.float32:
        bits = base.view(torch.int32)
        smallest, largest, normal = 1, 0x007FFFFF, 0x00800000
        negative_smallest, negative_largest, negative_zero = -2147483647, -2139095041, -2147483648
    else:
        bits = base.view(torch.int16)
        smallest, largest, normal = 1, 0x007F, 0x0080
        negative_smallest, negative_largest, negative_zero = -32767, -32641, -32768
    bits[0, 17], bits[0, 800] = smallest, largest
    bits[1, 3], bits[1, 900] = negative_smallest, negative_zero
    bits[2, 17], bits[2, 800] = normal, largest
    bits[3, 19], bits[3, 900] = negative_largest, negative_smallest
    expected = F.linear(x, w) + base
    logits, tokens = markov_gemv_add_argmax(x, w, base)
    torch.testing.assert_close(logits, expected, rtol=0, atol=0)
    torch.testing.assert_close(tokens, expected.argmax(-1), rtol=0, atol=0)


@pytest.mark.parametrize("logit_dtype", [torch.bfloat16, torch.float32])
def test_graph_replay_and_stream(logit_dtype):
    x = torch.randint(-4, 5, (4, 256), device="cuda").bfloat16()
    w = torch.randint(-2, 3, (1025, 256), device="cuda").bfloat16()
    base = torch.randn(4, 1025, device="cuda", dtype=logit_dtype)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            markov_gemv_add_argmax(x, w, base)
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        logits, tokens = markov_gemv_add_argmax(x, w, base)
    for _ in range(3):
        base.normal_()
        x.random_(-4, 5)
        graph.replay()
        expected = F.linear(x, w) + base
        torch.testing.assert_close(logits, expected, rtol=0, atol=0)
        torch.testing.assert_close(tokens, expected.argmax(-1), rtol=0, atol=0)


@pytest.mark.parametrize("kind", ["dtype", "rank", "batch", "base_shape"])
def test_reject_invalid_input(kind):
    x = torch.zeros(4, 256, device="cuda", dtype=torch.bfloat16)
    w = torch.zeros(17, 256, device="cuda", dtype=torch.bfloat16)
    base = torch.zeros(4, 17, device="cuda", dtype=torch.bfloat16)
    if kind == "dtype":
        x = x.float()
    elif kind == "rank":
        x = x[:, :255]
    elif kind == "batch":
        x = x.repeat(2, 1)
    else:
        base = base[:, :10]
    with pytest.raises(RuntimeError):
        markov_gemv_add_argmax(x, w, base)
