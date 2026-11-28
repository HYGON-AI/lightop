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


def _native_api():
    import lightop.sampling as sampling

    if not torch.version.hip or torch.cuda.get_device_properties(0).gcnArchName.split(":")[0] != "gfx938":
        pytest.skip("native FP8 MMAC requires gfx938")
    return sampling


def _native_chain(base, anchor, embedding, xs, weight, ws, world=1, tile_rows=128,
                  packed_argmax=False, packed_merge=False, k_stage=256):
    api = _native_api()
    rows, steps, total = base.shape
    width = total // world
    parts = (width + tile_rows - 1) // tile_rows
    partials = [torch.empty(rows, parts, 2, device="cuda") for _ in range(world)]
    compact = [torch.empty(rows, (parts + 31) // 32, 2, device="cuda") for _ in range(world)]
    out = torch.empty(rows, steps, device="cuda", dtype=torch.int64)
    prev = anchor
    for step in range(steps):
        for rank in range(world):
            start = min(rank * width, embedding.size(0))
            end = min((rank + 1) * width, embedding.size(0))
            api.dsv41_markov_fp8_partials(
                base[:, step, rank * width : (rank + 1) * width], prev,
                embedding, xs, weight[start:end], ws[start:end],
                vocab_start=start, vocab_size=embedding.size(0), out=partials[rank],
                tile_rows=tile_rows, packed_argmax=packed_argmax, k_stage=k_stage,
            )
            if world > 1:
                api.dsv41_markov_compact_partials(partials[rank], out=compact[rank])
        gathered = torch.stack(compact) if world > 1 else partials[0]
        api.dsv41_markov_merge_partials(
            gathered, rows=rows, world_size=world, out=out[:, step],
            packed_argmax=packed_merge,
        )
        prev = out[:, step]
    return out


def _native_reference(base, anchor, embedding, xs, weight, ws):
    prev, tokens = anchor.long(), []
    for step in range(base.size(1)):
        dot = embedding.float()[prev] @ weight.float().T
        bias = ((dot * xs[prev]) * ws.T).bfloat16().float()
        prev = (bias + base[:, step, : weight.size(0)].float()).argmax(-1)
        tokens.append(prev)
    return torch.stack(tokens, 1)


@pytest.mark.parametrize("batch", [1, 7, 16, 17, 32])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("world,vocab,width", [(1, 257, 385), (4, 8197, 4096)])
@pytest.mark.parametrize("tile_rows", [128, 64])
@pytest.mark.parametrize("packed_argmax", [False, True])
@pytest.mark.parametrize("k_stage", [256, 64])
def test_dsv41_fp8_chain(batch, dtype, world, vocab, width, tile_rows, packed_argmax, k_stage):
    _native_api()
    torch.manual_seed(41)
    # Dyadic operands give exact dot products, exposing mapping/rounding errors
    # independently of reduction-order differences between GEMM libraries.
    embedding = (torch.randint(-8, 9, (vocab, 256), device="cuda") / 2).to(torch.float8_e4m3fn)
    weight = (torch.randint(-8, 9, (vocab, 256), device="cuda") / 2).to(torch.float8_e4m3fn)
    xs = (torch.rand(vocab, 2, device="cuda") * 0.1 + 0.001)[:, :1]
    ws = (torch.rand(vocab, 2, device="cuda") * 0.1 + 0.001)[:, :1]
    anchor = torch.arange(batch * 2, device="cuda", dtype=torch.int32)[::2]
    base = torch.randn(batch, 5, width * world, device="cuda").to(dtype)
    base[:, :, vocab:] = torch.nan
    actual = _native_chain(base, anchor, embedding, xs, weight, ws, world, tile_rows,
                           packed_argmax, packed_merge=packed_argmax, k_stage=k_stage)
    expected = _native_reference(base, anchor, embedding, xs, weight, ws)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("tile_rows", [128, 64])
@pytest.mark.parametrize("packed_argmax", [False, True])
@pytest.mark.parametrize("k_stage", [256, 64])
def test_dsv41_fp8_scores_and_mask(tile_rows, packed_argmax, k_stage):
    api = _native_api()
    vocab, batch, width = 257, 17, 512
    torch.manual_seed(42)
    embedding = (torch.randint(-4, 5, (vocab, 256), device="cuda") / 2).to(torch.float8_e4m3fn)
    weight = (torch.randint(-4, 5, (vocab, 256), device="cuda") / 2).to(torch.float8_e4m3fn)
    xs, ws = torch.rand(vocab, 1, device="cuda"), torch.rand(vocab, 1, device="cuda")
    prev = torch.arange(batch, device="cuda")
    base = torch.randn(batch, 3, width, device="cuda")[:, 1]
    base[:, vocab:] = torch.nan
    parts = width // tile_rows
    partials = torch.empty(batch, parts, 2, device="cuda")
    api.dsv41_markov_fp8_partials(base, prev, embedding, xs, weight, ws,
                                vocab_start=0, vocab_size=vocab, out=partials, tile_rows=tile_rows,
                                packed_argmax=packed_argmax, k_stage=k_stage)
    dot = embedding.float()[prev] @ weight.float().T
    scores = torch.full((batch, width), -torch.inf, device="cuda")
    scores[:, :vocab] = ((dot * xs[prev]) * ws.T).bfloat16().float() + base[:, :vocab]
    values, ids = scores.view(batch, parts, tile_rows).max(-1)
    ids += torch.arange(parts, device="cuda") * tile_rows
    ids[:, (vocab + tile_rows - 1) // tile_rows:] = 2147483647
    torch.testing.assert_close(partials[:, :, 0], values, rtol=0, atol=0)
    torch.testing.assert_close(partials[:, :, 1].contiguous().view(torch.int32).long(), ids, rtol=0, atol=0)


@pytest.mark.parametrize("tile_rows", [128, 64])
@pytest.mark.parametrize("packed_argmax", [False, True])
@pytest.mark.parametrize("packed_merge", [False, True])
@pytest.mark.parametrize("k_stage", [256, 64])
def test_dsv41_fp8_nan_ties_and_fp32_add(tile_rows, packed_argmax, packed_merge, k_stage):
    _native_api()
    vocab, rows, width, world = 8197, 7, 4096, 4
    embedding = torch.zeros(vocab, 256, device="cuda", dtype=torch.float8_e4m3fn)
    weight = torch.zeros_like(embedding)
    scales = torch.ones(vocab, 1, device="cuda")
    base = torch.full((rows, 3, width * world), -torch.inf, device="cuda")
    base[:, :, vocab:] = torch.nan
    base[0, :, [127, 128, 4096]] = 5
    base[1, :, [128, 4096]] = torch.nan
    base[3, :, 0], base[3, :, 1] = 1.0, 1.001
    base[4, :, 0], base[4, :, 4096] = torch.nan, torch.inf
    base[5, :, 0], base[5, :, 1] = -0.0, 0.0
    base[6, :, 129] = torch.inf
    anchor = torch.zeros(rows, device="cuda", dtype=torch.int64)
    actual = _native_chain(base, anchor, embedding, scales, weight, scales, world,
                           tile_rows, packed_argmax, packed_merge, k_stage)
    expected = torch.tensor([127, 128, 0, 1, 0, 0, 129], device="cuda")[:, None].expand(-1, 3)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("tile_rows", [128, 64])
@pytest.mark.parametrize("packed_argmax", [False, True])
@pytest.mark.parametrize("packed_merge", [False, True])
@pytest.mark.parametrize("k_stage", [256, 64])
def test_dsv41_fp8_graph_replay_and_transport_zero(tile_rows, packed_argmax, packed_merge, k_stage):
    api = _native_api()
    vocab, rows = 257, 17
    embedding = torch.zeros(vocab, 256, device="cuda")
    weight = torch.zeros_like(embedding)
    for token in range(3):
        embedding[token, token] = 1
        weight[(token + 1) % 3, token] = 4
    embedding, weight = embedding.to(torch.float8_e4m3fn), weight.to(torch.float8_e4m3fn)
    scales = torch.ones(vocab, 1, device="cuda")
    base = torch.zeros(rows, 5, 384, device="cuda")
    anchor = torch.arange(rows, device="cuda") % 3
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            _native_chain(base, anchor, embedding, scales, weight, scales, tile_rows=tile_rows,
                          packed_argmax=packed_argmax, packed_merge=packed_merge, k_stage=k_stage)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        actual = _native_chain(base, anchor, embedding, scales, weight, scales, tile_rows=tile_rows,
                          packed_argmax=packed_argmax, packed_merge=packed_merge, k_stage=k_stage)
    for shift in (1, 2):
        anchor.copy_((torch.arange(rows, device="cuda") + shift) % 3)
        base[:, 2, shift] = 8
        graph.replay()
        expected = _native_reference(base, anchor, embedding, scales, weight, scales)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    pairs = torch.zeros(rows, 1, 2, device="cuda")
    pairs[:, :, 1] = -0.0
    out = torch.empty(rows, 5, device="cuda", dtype=torch.int64)
    api.dsv41_markov_merge_partials(pairs, rows=rows, world_size=1, out=out[:, 3],
                                    packed_argmax=packed_merge)
    torch.testing.assert_close(out[:, 3], torch.zeros_like(out[:, 3]), rtol=0, atol=0)


@pytest.mark.parametrize("kind", ["base_dtype", "scale_stride", "prev_dtype", "out_shape", "grad", "tile_rows", "tile_shape", "k_stage"])
def test_dsv41_fp8_reject_invalid_input(kind):
    api = _native_api()
    embedding = torch.zeros(257, 256, device="cuda", dtype=torch.float8_e4m3fn)
    scales = torch.ones(257, 1, device="cuda")
    base = torch.zeros(7, 384, device="cuda")
    prev = torch.zeros(7, device="cuda", dtype=torch.int64)
    out = torch.empty(7, 3, 2, device="cuda")
    if kind == "base_dtype":
        base = base.half()
    elif kind == "scale_stride":
        scales = scales[:1].expand(257, 1)
    elif kind == "prev_dtype":
        prev = prev.float()
    elif kind == "out_shape":
        out = out[:, :2].contiguous()
    elif kind == "grad":
        base.requires_grad_()
    tile_rows = 32 if kind == "tile_rows" else 64 if kind == "tile_shape" else 128
    k_stage = 32 if kind == "k_stage" else 256
    with pytest.raises(RuntimeError):
        api.dsv41_markov_fp8_partials(base, prev, embedding, scales, embedding, scales,
                                    vocab_start=0, vocab_size=257, out=out, tile_rows=tile_rows, k_stage=k_stage)


@pytest.mark.parametrize("tile_rows", [128, 64])
@pytest.mark.parametrize("k_stage", [256, 64])
def test_dsv41_fp8_packed_score_ordering(tile_rows, k_stage):
    """Key ordering must preserve NaNs, zeros, infinities and subnormal winners."""
    api = _native_api()
    vocab, width, rows = 257, 384, 8
    embedding = torch.zeros(vocab, 256, device="cuda", dtype=torch.float8_e4m3fn)
    scales = torch.ones(vocab, 1, device="cuda")
    base = torch.full((rows, width), -torch.inf, device="cuda")
    base[1].zero_()
    base.view(torch.int32)[1, 0] = -2147483648
    base.view(torch.int32)[2, 63:65] = torch.tensor([1, 2], device="cuda", dtype=torch.int32)
    base[3, 63:65] = torch.tensor([-torch.finfo(torch.float32).max,
                                  torch.finfo(torch.float32).max], device="cuda")
    base.view(torch.int32)[4, 63:65] = torch.tensor([2143289345, 2143289601], device="cuda", dtype=torch.int32)
    base.view(torch.int32)[5, 63:65] = torch.tensor([-4194303, 2143289345], device="cuda", dtype=torch.int32)
    base[6, 63:65] = torch.inf
    base[7, 63:65] = 5
    base[:, vocab:] = torch.nan
    parts = width // tile_rows
    out = torch.empty(rows, parts, 2, device="cuda")
    api.dsv41_markov_fp8_partials(
        base, torch.zeros(rows, device="cuda", dtype=torch.int64),
        embedding, scales, embedding, scales, vocab_start=0,
        vocab_size=vocab, out=out, tile_rows=tile_rows, packed_argmax=True, k_stage=k_stage,
    )
    scores = base + 0.0
    scores[:, vocab:] = -torch.inf
    values, ids = scores.view(rows, parts, tile_rows).max(-1)
    ids += torch.arange(parts, device="cuda") * tile_rows
    ids[:, (vocab + tile_rows - 1) // tile_rows:] = 2147483647
    torch.testing.assert_close(out[:, :, 0], values, rtol=0, atol=0, equal_nan=True)
    torch.testing.assert_close(out[:, :, 1].contiguous().view(torch.int32).long(), ids, rtol=0, atol=0)


@pytest.mark.parametrize("world", [1, 4])
@pytest.mark.parametrize("parts", [1, 31, 32, 2020, 2021])
def test_dsv41_fp8_merge_rank_layout_and_sentinels(world, parts):
    """Real service candidate count, strided output and TP rank-major ordering."""
    api = _native_api()
    rows = 7
    torch.manual_seed(43)
    pairs = torch.empty(world, rows, parts, 2, device="cuda")
    pairs[..., 0].normal_()
    # Include invalid/NaN lanes, duplicated IDs, ties and zero transport bits.
    ids = torch.arange(world * parts, device="cuda", dtype=torch.int32).view(world, 1, parts)
    ids = ids.expand(world, rows, parts).clone()
    ids[:, 0].fill_(2147483647)
    pairs[:, 0, :, 0] = torch.nan
    pairs[:, 1, :, 0] = -torch.inf
    pairs[:, 2, :, 0] = 0
    pairs[0, 2, 0, 0] = -0.0
    pairs[:, 3, :, 0] = torch.inf
    pairs[0, 4, 0, 0] = torch.nan
    pairs[-1, 4, -1, 0] = torch.nan
    pairs[..., 1] = ids.view(torch.float32)
    pairs[0, 5, 0, 0] = torch.inf
    pairs[0, 5, 0, 1] = -0.0  # low ID 0 with a Lamport sign sentinel
    ids[0, 5, 0] = 0
    values = pairs[..., 0].permute(1, 0, 2).reshape(rows, -1)
    flat_ids = ids.permute(1, 0, 2).reshape(rows, -1)
    valid = flat_ids != 2147483647
    nan = valid & torch.isnan(values)
    has_nan = nan.any(-1, keepdim=True)
    maximum = torch.where(valid & ~nan, values, -torch.inf).max(-1, keepdim=True).values
    winners = torch.where(has_nan, nan, valid & (values == maximum))
    expected = torch.where(winners, flat_ids, 2147483647).min(-1).values.long()
    for packed in (False, True):
        tokens = torch.full((rows, 5), -7, device="cuda", dtype=torch.int64)
        actual = api.dsv41_markov_merge_partials(
            pairs, rows=rows, world_size=world, out=tokens[:, 3], packed_argmax=packed,
        )
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(tokens[:, [0, 1, 2, 4]],
                                   torch.full_like(tokens[:, [0, 1, 2, 4]], -7), rtol=0, atol=0)



def _benchmark_dsv41(args):
    """Compare the five-step vendor chain and native tile sizes on identical input.

    Run --benchmark --batches 7 for the deployed B7/BF16 case. Without --batches,
    cover 1,7,16,17,32. Graph replay excludes capture-time allocation; timing
    rounds rotate the order of the three implementations. --world-size 8 adds
    a sequential shard/compact/stack simulation, not real collective timing.
    Native 128 remains the default API behavior; 64 is opt-in. Use
    --tile-rows 64 --reductions packed --merge-reductions pair packed
    --service-baseline isolates merge tuning after packed partial. To tune
    staging, fix both reductions to packed and add --k-stages 256 64.
    K64 reduces the weight LDS buffer but adds barriers/register lifetime;
    it is an experimental candidate, not an assumed speedup.
    The latter imports installed SGLang only when requested and measures its
    actual FP8 linear + sharded_greedy_step, avoiding the materialized add
    and torch.argmax overhead of the standalone unfused comparison.
    """
    import functools
    import statistics

    import deepgemm
    import lightop
    from lightop.quant import per_token_quant_fp8

    service_linear = service_greedy = None
    if args.service_baseline:
        # Explicit integration benchmark only. Core CI/reference tests remain
        # independent of SGLang, and this calls its actual serving operators.
        from types import SimpleNamespace
        from sglang.srt.layers.quantization.dsv41_channelwise_fp8 import linear
        from sglang.kernels.ops.speculative.dspark.sharded_greedy import sharded_greedy_step

        service_linear, service_greedy = linear, sharded_greedy_step
        service_group = SimpleNamespace(world_size=1)

    _native_api()
    print("build", lightop.__file__, lightop.op.__file__, torch.__version__, torch.version.hip)
    import inspect

    partial_api = lightop.sampling.dsv41_markov_fp8_partials
    print("partial_wrapper", inspect.getfile(partial_api), inspect.signature(partial_api))
    if "packed" in args.reductions and (
        "packed_argmax" not in inspect.signature(partial_api).parameters
        or "packed_argmax" not in (lightop.op.dsv41_markov_fp8_partials.__doc__ or "")
    ):
        raise RuntimeError("Update both the LightOp wrapper and extension for packed_argmax")
    if (
        "k_stage" not in inspect.signature(partial_api).parameters
        or "k_stage" not in (lightop.op.dsv41_markov_fp8_partials.__doc__ or "")
    ):
        raise RuntimeError("Update both the LightOp wrapper and extension for k_stage")
    merge_api = lightop.sampling.dsv41_markov_merge_partials
    print("merge_wrapper", inspect.getfile(merge_api), inspect.signature(merge_api))
    if "packed" in args.merge_reductions and (
        "packed_argmax" not in inspect.signature(merge_api).parameters
        or "packed_argmax" not in (lightop.op.dsv41_markov_merge_partials.__doc__ or "")
    ):
        raise RuntimeError("Update both the LightOp wrapper and extension for packed merge")
    print("device", torch.cuda.get_device_properties(0))
    vocab, steps, world = 129280, 5, args.world_size
    dtype = torch.bfloat16 if args.base_dtype == "bf16" else torch.float32
    print("config", "base_dtype", str(dtype), "vocab", vocab, "steps", steps,
          "world_size", world, "tile_rows", args.tile_rows, "reductions", args.reductions,
          "rounds", args.rounds, "replays", args.replays,
          "service_baseline", args.service_baseline, "merge_reductions", args.merge_reductions,
          "k_stages", args.k_stages)
    for rows in args.batches:
        torch.manual_seed(41)
        embedding = torch.randn(vocab, 256, device="cuda", dtype=torch.bfloat16)
        eq, xs = per_token_quant_fp8(embedding)
        weight = (torch.randn(vocab, 256, device="cuda") * 16).clamp(-448, 448).to(torch.float8_e4m3fn)
        ws = torch.rand(vocab, 1, device="cuda") * 0.02 + 0.01
        base = torch.randn(rows, steps, vocab, device="cuda").to(dtype)
        anchor = torch.arange(rows, device="cuda")

        def baseline():
            prev, tokens = anchor, []
            tiles, n = 8, vocab // 8
            for step in range(steps):
                xq, scale = per_token_quant_fp8(embedding[prev])
                bias = torch.empty(tiles, rows, n, device="cuda", dtype=torch.bfloat16)
                deepgemm.fp8_gemm(
                    (xq.unsqueeze(0).expand(tiles, -1, -1).contiguous(),
                     scale.unsqueeze(0).expand(tiles, -1, -1).contiguous()),
                    (weight.view(tiles, n, 256), ws.view(tiles, n, 1)), bias,
                )
                prev = (bias.transpose(0, 1).reshape(rows, vocab).float() + base[:, step].float()).argmax(-1)
                tokens.append(prev)
            return torch.stack(tokens, 1)

        def service_baseline():
            prev, tokens = anchor, []
            for step in range(steps):
                latent = F.embedding(prev, embedding)
                bias = service_linear(latent, weight, ws)
                prev = service_greedy(
                    bias, base[:, step], group=service_group, vocab_start=0
                )
                tokens.append(prev)
            return torch.stack(tokens, 1)

        variants = [("unfused_deepgemm", baseline)]
        if args.service_baseline:
            variants.append(("service_vendor", service_baseline))
        variants.extend(
            (f"native_fp8_{tile}_{mode}_merge_{merge_mode}_k{k_stage}", functools.partial(
                _native_chain, base, anchor, eq, xs, weight, ws,
                world=world, tile_rows=tile, packed_argmax=mode == "packed",
                packed_merge=merge_mode == "packed", k_stage=k_stage))
            for tile in args.tile_rows for mode in args.reductions
            for merge_mode in args.merge_reductions for k_stage in args.k_stages
        )
        expected = service_baseline() if args.service_baseline else baseline()
        graphs, results = {}, {}
        for name, fn in variants:
            torch.testing.assert_close(fn(), expected, rtol=0, atol=0)
            for _ in range(10):
                fn()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                result = fn()
            graph.replay()
            torch.testing.assert_close(result, expected, rtol=0, atol=0)
            graphs[name], results[name] = graph, result
        print(rows, "token_parity", "PASS", flush=True)

        samples = {name: [] for name, _ in variants}
        names = list(samples)
        for round_index in range(args.rounds):
            shift = round_index % len(names)
            for name in names[shift:] + names[:shift]:
                begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                begin.record()
                for _ in range(args.replays):
                    graphs[name].replay()
                end.record()
                end.synchronize()
                samples[name].append(begin.elapsed_time(end) * 1000 / args.replays)
        for name in names:
            torch.testing.assert_close(results[name], expected, rtol=0, atol=0)
            values = samples[name]
            mean = statistics.mean(values)
            # Logical problem bytes, shared across implementations; not measured
            # HBM traffic. Shard simulation also includes compact/stack overhead.
            effective_bytes = steps * (vocab * 256 + rows * vocab * base.element_size())
            grids = (rows + 15) // 16 if name.startswith("native_fp8") else 1
            print(rows, name, "chain_mean_us", mean, "step_mean_us", mean / steps,
                  "p50_us", statistics.median(values),
                  "p90_us", sorted(values)[max(0, (9 * len(values) + 9) // 10 - 1)],
                  "stdev_us", statistics.stdev(values),
                  "problem_GB/s", effective_bytes / mean / 1000,
                  "weight_read_grids", grids, flush=True)


if __name__ == "__main__":
    import sys

    if "--benchmark" in sys.argv[1:]:
        import argparse

        parser = argparse.ArgumentParser(description="DSV4.1 FP8 Markov paired graph benchmark")
        parser.add_argument("--benchmark", action="store_true", required=True)
        parser.add_argument("--batches", type=int, nargs="+", default=[1, 7, 16, 17, 32], choices=range(1, 33))
        parser.add_argument("--tile-rows", type=int, nargs="+", default=[128, 64], choices=[128, 64])
        parser.add_argument("--reductions", nargs="+", default=["pair", "packed"], choices=["pair", "packed"])
        parser.add_argument("--merge-reductions", nargs="+", default=["pair"], choices=["pair", "packed"])
        parser.add_argument("--k-stages", nargs="+", type=int, default=[256], choices=[256, 64])
        parser.add_argument("--base-dtype", default="bf16", choices=["bf16", "fp32"])
        parser.add_argument("--service-baseline", action="store_true",
                            help="Also measure actual SGLang FP8 linear + sharded greedy (TP1)")
        parser.add_argument("--world-size", type=int, default=1, choices=[1, 8])
        parser.add_argument("--rounds", type=int, default=10)
        parser.add_argument("--replays", type=int, default=100)
        args = parser.parse_args()
        if args.rounds < 2 or args.replays < 1:
            parser.error("--rounds must be >=2 and --replays must be >=1")
        if args.service_baseline and args.world_size != 1:
            parser.error("--service-baseline requires --world-size 1")
        args.tile_rows = list(dict.fromkeys(args.tile_rows))
        args.reductions = list(dict.fromkeys(args.reductions))
        args.merge_reductions = list(dict.fromkeys(args.merge_reductions))
        args.k_stages = list(dict.fromkeys(args.k_stages))
        _benchmark_dsv41(args)
    else:
        raise SystemExit(pytest.main([__file__, *sys.argv[1:]]))
