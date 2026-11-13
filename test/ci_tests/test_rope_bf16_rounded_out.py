# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Exact arithmetic and output-contract regression for the existing RoPE API."""
import pytest
import torch

from lightop.attention import rotary_embedding_deepseek_fuse
from lightop.attention.rope import ROTARY_EMBEDDING_BF16_ROUNDED_OUT_ABI

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")


def test_new_rope_tests_use_categorized_api_imports():
    import ast
    from pathlib import Path
    tree = ast.parse(Path(__file__).read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("lightop"):
            assert node.module in ("lightop.attention", "lightop.attention.rope")
        if isinstance(node, ast.Import):
            assert all(alias.name != "lightop" for alias in node.names)


def reference(positions, query, cache, neox):
    cos, sin = cache[positions].chunk(2, dim=-1)
    cos, sin = cos[:, None, :], sin[:, None, :]
    if neox:
        x, y = query.chunk(2, dim=-1)
        return torch.cat((x * cos + (-y) * sin,
                          y * cos + x * sin), dim=-1)
    x, y = query[..., ::2], query[..., 1::2]
    return torch.stack((x * cos + (-y) * sin,
                        y * cos + x * sin), dim=-1).flatten(-2)


def assert_bits(actual, expected):
    assert actual.shape == expected.shape
    assert torch.equal(actual.contiguous().view(torch.uint8),
                       expected.contiguous().view(torch.uint8))


def rounded(positions, query, key, cache, neox):
    qo = torch.empty(query.shape, dtype=query.dtype, device=query.device)
    ko = None if key is None else torch.empty(key.shape, dtype=key.dtype, device=key.device)
    assert rotary_embedding_deepseek_fuse(
        positions, query, key, 64, cache, neox,
        round_bf16_products=True, query_out=qo, key_out=ko) is None
    return qo, ko


@pytest.mark.parametrize("neox", [False, True])
@pytest.mark.parametrize("tokens,heads,key_heads", [
    (0, 128, 1), (1, 128, 1), (17, 128, 1), (257, 128, 1),
    (513, 7, 3), (129, 8, 0), (40960, 128, 1),
])
def test_rounded_outputs(neox, tokens, heads, key_heads):
    assert ROTARY_EMBEDDING_BF16_ROUNDED_OUT_ABI == 2
    torch.manual_seed(310031)
    q = torch.randn((tokens, heads, 64), device="cuda", dtype=torch.bfloat16)
    k = None if key_heads == 0 else torch.randn(
        (tokens, key_heads, 64), device="cuda", dtype=torch.bfloat16)
    cache = torch.randn((1048576, 64), device="cuda", dtype=torch.bfloat16)
    p = (torch.arange(tokens, device="cuda", dtype=torch.int64) * 4093) % 1048576
    before_q, before_k = q.clone(), None if k is None else k.clone()
    qo, ko = rounded(p, q, k, cache, neox)
    assert_bits(qo, reference(p, q, cache, neox))
    assert_bits(q, before_q)
    if k is not None:
        assert_bits(ko, reference(p, k, cache, neox))
        assert_bits(k, before_k)


@pytest.mark.parametrize("neox", [False, True])
def test_all_bf16_patterns_and_strides(neox):
    raw = torch.arange(65536, device="cuda", dtype=torch.int32).to(torch.int16)
    q = raw.view(torch.bfloat16).reshape(1024, 1, 64).expand(-1, 128, -1)
    # Independent token/head/feature strides and strided positions/cache rows.
    base = torch.empty((1024, 2, 128), device="cuda", dtype=torch.bfloat16)
    k = base[:, 1:2, ::2]
    k.copy_(q[:, :1])
    p = torch.arange(2048, device="cuda", dtype=torch.int64)[::2]
    cache = torch.randn((2048, 128), device="cuda", dtype=torch.bfloat16)[:, :64]
    # Include special values in cache as well as the activations.
    cache[:1024].copy_(raw.view(torch.bfloat16).view(1024, 64))
    qo, ko = rounded(p, q, k, cache, neox)
    assert_bits(qo, reference(p, q, cache, neox))
    assert_bits(ko, reference(p, k, cache, neox))


@pytest.mark.parametrize("neox", [False, True])
def test_graph_changed_inputs_and_stream(neox):
    q = torch.randn((257, 128, 64), device="cuda", dtype=torch.bfloat16)
    k = torch.randn((257, 1, 64), device="cuda", dtype=torch.bfloat16)
    p = torch.arange(257, device="cuda", dtype=torch.int64)
    cache = torch.randn((4096, 64), device="cuda", dtype=torch.bfloat16)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        rounded(p, q, k, cache, neox)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            qo, ko = rounded(p, q, k, cache, neox)
        for _ in range(3):
            q.normal_()
            k.normal_()
            p.add_(17)
            graph.replay()
            assert_bits(qo, reference(p, q, cache, neox))
            assert_bits(ko, reference(p, k, cache, neox))
    torch.cuda.current_stream().wait_stream(stream)


def test_rounded_output_contract_rejects_invalid_inputs():
    p = torch.arange(17, device="cuda", dtype=torch.int64)
    q = torch.randn((17, 128, 64), device="cuda", dtype=torch.bfloat16)
    k = torch.randn((17, 1, 64), device="cuda", dtype=torch.bfloat16)
    cache = torch.randn((128, 64), device="cuda", dtype=torch.bfloat16)
    qo, ko = torch.empty_like(q), torch.empty_like(k)
    args = (p, q, k, 64, cache, True)
    for overrides in [dict(query_out=q), dict(key_out=k), dict(query_out=qo[..., ::2]),
                      dict(query_out=qo.float()), dict(offsets=p)]:
        kw = dict(round_bf16_products=True, query_out=qo, key_out=ko)
        kw.update(overrides)
        with pytest.raises(RuntimeError):
            rotary_embedding_deepseek_fuse(*args, **kw)
    with pytest.raises(RuntimeError):
        rotary_embedding_deepseek_fuse(*args, query_out=qo, key_out=ko)
    with pytest.raises(RuntimeError):
        rotary_embedding_deepseek_fuse(p.int(), *args[1:], round_bf16_products=True,
                                       query_out=qo, key_out=ko)
    with pytest.raises(RuntimeError):
        rotary_embedding_deepseek_fuse(p.cpu(), *args[1:], round_bf16_products=True,
                                       query_out=qo, key_out=ko)


@pytest.mark.parametrize("neox", [False, True])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("tokens,head_size", [(1, 64), (17, 64), (17, 128)])
def test_legacy_inplace_signature_and_offsets(neox, dtype, tokens, head_size):
    torch.manual_seed(310031)
    p = torch.arange(tokens, device="cuda", dtype=torch.int64)
    offset = torch.full_like(p, 3)
    cache = torch.randn((128, head_size), device="cuda", dtype=dtype)
    q = torch.randn((tokens, 8, head_size), device="cuda", dtype=dtype)
    k = torch.randn((tokens, 1, head_size), device="cuda", dtype=dtype)
    # Legacy arithmetic rounds once after the FP32 sum; use tolerance for the
    # original implementation, independently of the new exact-output mode.
    expected_q = reference(p + offset, q.float(), cache.float(), neox).to(dtype)
    expected_k = reference(p + offset, k.float(), cache.float(), neox).to(dtype)
    assert rotary_embedding_deepseek_fuse(p, q, k, head_size, cache, neox, offset) is None
    tolerance = 1e-5 if dtype == torch.float32 else 0.003 if dtype == torch.float16 else 0.02
    torch.testing.assert_close(q, expected_q, rtol=tolerance, atol=tolerance)
    torch.testing.assert_close(k, expected_k, rtol=tolerance, atol=tolerance)


@pytest.mark.parametrize("tokens", [17,3584,5120])
@pytest.mark.parametrize("neox", [False,True])
def test_packed_output_preserves_prefix_and_graph_updates(tokens, neox):
    q = torch.randn((tokens,128,64), device="cuda", dtype=torch.bfloat16)
    k = torch.randn((tokens,1,64), device="cuda", dtype=torch.bfloat16)
    positions = torch.arange(tokens, device="cuda", dtype=torch.int64)
    cache = torch.randn((tokens+64,64), device="cuda", dtype=torch.bfloat16)
    packed = torch.empty((tokens,128,576), device="cuda", dtype=torch.bfloat16)
    ko = torch.empty_like(k)
    def invoke():
        rotary_embedding_deepseek_fuse(
            positions,q,k,64,cache,neox,round_bf16_products=True,
            query_out=packed[...,512:],key_out=ko)
    invoke()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        invoke()
    for _ in range(3):
        q.normal_();k.normal_();positions.add_(1);packed[..., :512].fill_(3)
        graph.replay()
        assert_bits(packed[...,512:],reference(positions,q,cache,neox))
        assert_bits(ko,reference(positions,k,cache,neox))
        assert torch.all(packed[..., :512]==3)
