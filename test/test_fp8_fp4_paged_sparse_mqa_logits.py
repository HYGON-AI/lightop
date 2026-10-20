# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""MXFP4 paged sparse MQA precision and performance tests.

Run regressions with pytest -q test/test_fp8_fp4_paged_sparse_mqa_logits.py.
Benchmark with python test/test_fp8_fp4_paged_sparse_mqa_logits.py
--contexts 32768 65536 131072 --output test-results/paged-sparse-mqa.json.
Benchmark timing includes the complete scorer and its internal preparation,
but excludes metadata generation, the FP32 oracle and external Q packing.
"""

import argparse
import gc
import json
import math
import random
import statistics
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

INT_MAX = 2147483647
MIB = 1024**2
DEVICE = "cuda"
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.version.hip is None,
    reason="requires an HCU with the installed LightOp extension",
)
torch.backends.cuda.matmul.allow_tf32 = False


@dataclass(frozen=True)
class Case:
    id: str
    rows: int = 3
    heads: int = 32
    page: int = 64
    block: int = 8
    capacity: int = 8
    context: int = 79
    overlap: str = "partial"
    numerics: str = "normal"
    layout: str = "padded"
    requests: int = 1
    causal: bool = False
    benchmark: bool = False
    cp_world: int = 1
    cp_rank: int = 0
    prefix: int = 0
    one_shot: bool = False


SMOKE = [
    Case("smoke-r3"),
    Case("smoke-empty", rows=2, context=0),
    Case("smoke-tail", rows=8, context=65, requests=2, causal=True),
    Case("smoke-weights", rows=2, heads=4, numerics="fp32_weights"),
]


def case_matrix():
    # Orthogonal coverage, deliberately not a Cartesian product.
    result = list(SMOKE)
    for r, k, length in [
        (1, 4, 31),
        (2, 8, 65),
        (3, 32, 1027),
        (8, 128, 4097),
        (32, 128, 8191),
        (128, 32, 65535),
        (512, 8, 257),
        (1024, 8, 257),
        (4096, 8, 257),
    ]:
        result.append(
            Case(
                f"rows-{r}",
                rows=r,
                capacity=k,
                context=length,
                requests=2 if r >= 8 else 1,
                causal=r >= 8,
            )
        )
    for h in range(4, 33, 4):
        result.append(
            Case(
                f"heads-{h}", rows=3, heads=h, capacity=32, context=4099, overlap="full"
            )
        )
    for p in (8, 16, 32, 64, 128, 256):
        result.append(
            Case(
                f"page-{p}",
                rows=8,
                page=p,
                capacity=32,
                context=2049,
                requests=2,
                causal=True,
            )
        )
    for k in (4, 8, 12, 32, 64, 128, 256, 512, 1024, 2044, 2048, 2052, 4096):
        result.append(
            Case(f"capacity-{k}", rows=3, capacity=k, context=131072, overlap="none")
        )
    for length in (1, 7, 8, 9, 65535, 65536, 65537, 131072, 1048576):
        result.append(
            Case(
                f"context-{length}",
                rows=3,
                capacity=128,
                context=length,
                causal=length > 128,
            )
        )
    for overlap in ("none", "partial", "full"):
        result.append(
            Case(
                f"overlap-{overlap}",
                rows=8,
                capacity=128,
                context=65537,
                overlap=overlap,
                requests=2,
            )
        )
    for layout in ("packed", "padded", "broadcast", "q3d"):
        result.append(
            Case(f"stride-{layout}", rows=3, capacity=32, context=3073, layout=layout)
        )
    for mode in (
        "qspan11",
        "qspan12",
        "kspan11",
        "kspan12",
        "mixed_pair_kspan",
        "wide_scales",
        "fp32_weights",
        "signed_weights",
        "zero_q",
        "zero_k",
        "zero_weights",
        "ties",
        "reciprocal",
        "reverse_reciprocal",
        "tiny_dot",
    ):
        result.append(
            Case(
                f"numerics-{mode}",
                rows=3,
                capacity=32,
                context=2051,
                overlap="partial",
                numerics=mode,
            )
        )
    for p, k in ((16, 8), (64, 128), (256, 2052)):
        result.append(
            Case(
                f"block16-p{p}-k{k}",
                rows=3,
                page=p,
                block=16,
                capacity=k,
                context=65539,
            )
        )
    return result


def fp4_values(payload):
    """Decode nibbles only, independent of kernel bit tricks and LUT layouts."""
    raw = payload.view(torch.uint8)
    codes = torch.stack((raw & 15, raw >> 4), dim=-1).flatten(-2).long()
    lut = torch.tensor(
        [
            0.0,
            0.5,
            1.0,
            1.5,
            2.0,
            3.0,
            4.0,
            6.0,
            -0.0,
            -0.5,
            -1.0,
            -1.5,
            -2.0,
            -3.0,
            -4.0,
            -6.0,
        ],
        device=payload.device,
        dtype=torch.float32,
    )
    return lut[codes]


def scale_bytes(packed):
    return torch.stack(
        [(packed.to(torch.int64) >> (8 * i)) & 255 for i in range(4)], dim=-1
    ).to(torch.uint8)


def pack_scales(scales):
    packed = sum(scales[..., i].to(torch.int64) << (8 * i) for i in range(4))
    return packed.to(torch.int32)


def decode_fp4(payload, scales):
    values = fp4_values(payload).unflatten(-1, (4, 32))
    exponents = scales.to(torch.int32) - 127
    return torch.ldexp(values, exponents[..., None]).flatten(-2)


def torch_pack_query(raw):
    """Reference of old fake_quant_fp4, packing its exact BF16 grid once.

    The scale floor is the fake quantizer's 6*2^-126 floor. The separate legacy
    index-K packer's 1e-4 floor must not be applied to a Prefill query.
    """
    blocks = raw.float().unflatten(-1, (4, 32))
    amax = blocks.abs().amax(dim=-1).clamp_min(6.0 * 2.0**-126)
    normalized = (amax * (1.0 / 6.0)).contiguous()
    bits = normalized.view(torch.int32)
    exponents = ((bits >> 23) & 255) + ((bits & 0x7FFFFF) != 0).int()
    scales = (exponents << 23).contiguous().view(torch.float32)
    scaled = (blocks / scales[..., None]).clamp(-6, 6)
    magnitude = scaled.abs()
    step = torch.where(magnitude < 2, 0.5, torch.where(magnitude < 4, 1.0, 2.0))
    rounded = torch.round(magnitude / step) * step
    positive_lut = torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], device=raw.device
    )
    codes = torch.searchsorted(positive_lut, rounded.contiguous()).to(torch.uint8)
    codes = codes | (((scaled < 0) & (codes != 0)).to(torch.uint8) << 3)
    codes = codes.flatten(-2)
    payload = (codes[..., ::2] | (codes[..., 1::2] << 4)).contiguous().view(torch.int8)
    sf = pack_scales(exponents.to(torch.uint8))
    fake_q = (rounded * scaled.sign() * scales[..., None]).flatten(-2).bfloat16()
    return payload.unsqueeze(1), sf.unsqueeze(1), fake_q.contiguous()


def candidate_list(c, lengths, request_ids, seed):
    rows, prior = [], {}
    for row, (length, request) in enumerate(zip(lengths, request_ids)):
        available = (length + c.block - 1) // c.block
        n = min(c.capacity, available)
        rng = random.Random(seed + request * 7919 + (row // 2) * 103)
        if n == available:
            selected = list(range(n))
        elif c.overlap == "none":
            pool = list(range(row % 2, available, 2))
            selected = (
                rng.sample(pool, n)
                if len(pool) >= n
                else rng.sample(range(available), n)
            )
        elif c.overlap == "full":
            selected = rng.sample(range(available), n)
        elif row % 2 and request in prior:
            keep = [v for v in prior[request] if v < available]
            selected = set(keep[: min(len(keep), n * 3 // 4)])
            while len(selected) < n:
                selected.add(rng.randrange(available))
            selected = list(selected)
        else:
            selected = rng.sample(range(available), n)
        # Include the newest block to exercise every context tail.
        if n and available - 1 not in selected:
            selected[-1] = available - 1
        selected = sorted(set(selected))
        while len(selected) < n:
            extra = rng.randrange(available)
            if extra not in selected:
                selected.append(extra)
        selected.sort()
        prior[request] = selected
        rows.append(selected + [INT_MAX] * (c.capacity - n))
    return torch.tensor(rows, dtype=torch.int32, device=DEVICE)


def make_inputs(c, seed, num_sms):
    torch.manual_seed(seed)
    assert c.heads in range(4, 33, 4) and c.page % c.block == 0
    counts = [c.rows // c.requests] * c.requests
    counts[-1] += c.rows - sum(counts)
    lengths, request_ids = [], []
    for request, count in enumerate(counts):
        for j in range(count):
            if c.cp_world > 1:
                assert c.requests == 1 and 0 <= c.cp_rank < c.cp_world
                length = c.prefix + c.cp_rank + 1 + c.cp_world * j
                assert (
                    0 < length <= c.context
                ), "captured CP row exceeds global KV extent"
            else:
                length = max(0, c.context - (count - 1 - j)) if c.causal else c.context
            if c.requests > 1:
                length = max(0, length - request * (c.page + 3))
            lengths.append(length)
            request_ids.append(request)
    pages_per_request = max(1, math.ceil(c.context / c.page))
    useful_pages = c.requests * pages_per_request
    num_pages = useful_pages + max(3, useful_pages // 13)
    stride = c.page * 68
    if c.layout != "packed":
        stride = math.ceil(stride / 512) * 512 + 512
    free, _ = torch.cuda.mem_get_info()
    input_estimate = num_pages * stride + c.rows * (
        c.heads * 392 + c.capacity * 4 + pages_per_request * 4
    )
    if input_estimate + 64 * MIB > free:
        raise MemoryError(
            f"input requires {input_estimate / MIB:.1f} MiB; available {free / MIB:.1f} MiB"
        )
    mapping = torch.randperm(num_pages, device=DEVICE, dtype=torch.int32)[
        :useful_pages
    ].view(c.requests, -1)
    ids = torch.tensor(request_ids, dtype=torch.int32, device=DEVICE)
    table_storage = torch.empty(
        (c.rows, pages_per_request + 3), dtype=torch.int32, device=DEVICE
    )
    table = table_storage[:, :pages_per_request]
    table.copy_(mapping[ids.long()])
    storage = torch.empty((num_pages, stride), dtype=torch.uint8, device=DEVICE)
    payload = storage[:, : c.page * 64].view(num_pages, c.page, 64)
    sf = storage[:, c.page * 64 : c.page * 68].view(num_pages, c.page, 4)
    # Chunk generation avoids a second million-token cache allocation.
    for start in range(0, num_pages, 1024):
        end = min(num_pages, start + 1024)
        payload[start:end].random_(0, 256)
        sf[start:end].random_(124, 131)
    kv = storage.as_strided((num_pages, c.page, 1, 68), (stride, 68, 68, 1))
    raw_q = (torch.randn((c.rows, c.heads, 128), device=DEVICE) * 0.8).bfloat16()
    weight_storage = torch.empty(
        (c.rows, c.heads + 8), device=DEVICE, dtype=torch.float32
    )
    weights = weight_storage[:, : c.heads]
    weights.uniform_(0.05, 1.0)
    cand = candidate_list(c, lengths, request_ids, seed)
    mode = c.numerics
    if mode in ("qspan11", "qspan12"):
        qbytes = torch.full(
            (c.rows, c.heads, 64), 0x77, dtype=torch.uint8, device=DEVICE
        )
        qscales = torch.tensor(
            [115, 115 + int(mode[-2:]), 115, 115 + int(mode[-2:])],
            dtype=torch.uint8,
            device=DEVICE,
        ).expand(c.rows, c.heads, -1)
        raw_q = decode_fp4(qbytes, qscales).bfloat16()
    if mode in ("kspan11", "kspan12", "mixed_pair_kspan"):
        span = 12 if mode == "mixed_pair_kspan" else int(mode[-2:])
        sf.copy_(
            torch.tensor(
                [119, 119 + span, 119, 119 + span], device=DEVICE, dtype=torch.uint8
            )
        )
        if mode == "mixed_pair_kspan":
            sf.fill_(127)
            # Only Q1's exclusive candidate has wide K scales. Q0's fast-path
            # eligibility must not be poisoned by its paired query's union.
            exclusive = sorted(
                set(cand[1].cpu().tolist()) - set(cand[0].cpu().tolist())
            )
            exclusive = [v for v in exclusive if v != INT_MAX]
            assert exclusive, "mixed-pair probe needs an exclusive Q1 candidate"
            logical = exclusive[0] * c.block
            physical = int(mapping[0, logical // c.page])
            sf[physical, logical % c.page : logical % c.page + c.block].copy_(
                torch.tensor([119, 131, 119, 131], device=DEVICE, dtype=torch.uint8)
            )
    if mode == "wide_scales":
        sf.random_(110, 143)
        per_group = torch.tensor([111, 142, 119, 135], device=DEVICE, dtype=torch.int32)
        raw_q = (
            torch.ldexp(
                raw_q.float().unflatten(-1, (4, 32)),
                (per_group - 127)[None, None, :, None],
            )
            .flatten(-2)
            .bfloat16()
        )
    if mode == "fp32_weights":
        raw_q.fill_(1)
        payload.fill_(0x22)
        sf.fill_(127)
        weights.copy_(0.50005 + torch.arange(c.heads, device=DEVICE) * 0.00002)
        assert not torch.equal(weights, weights.bfloat16().float())
    if mode == "signed_weights":
        weights[:, ::2].neg_()
    if mode == "zero_q":
        raw_q.zero_()
    if mode == "zero_k":
        payload.zero_()
    if mode == "zero_weights":
        weights.zero_()
    if mode == "ties":
        raw_q.fill_(1)
        payload.fill_(0x22)
        sf.fill_(127)
        weights.fill_(0.5)
    qp, qsf, fake_q = torch_pack_query(raw_q)
    representation_probe = mode in ("reciprocal", "reverse_reciprocal", "tiny_dot")
    if representation_probe:
        # These are format-domain probes beyond a BF16 Prefill source's range.
        # The new extreme FP32 path must preserve a finite joint-scale dot.
        qp.fill_(0x11 if mode == "tiny_dot" else 0x77)
        payload.fill_(0x11 if mode == "tiny_dot" else 0x77)
        qsf.fill_(
            pack_scales(
                torch.full(
                    (1, 4),
                    254 if mode == "reciprocal" else 0,
                    device=DEVICE,
                    dtype=torch.uint8,
                )
            ).item()
        )
        sf.fill_(
            0
            if mode == "reciprocal"
            else (254 if mode == "reverse_reciprocal" else 127)
        )
        weights.fill_(1.0)
    if c.layout in ("padded", "broadcast"):
        qstorage = torch.empty(
            (c.rows, 1, c.heads + 1, 80), device=DEVICE, dtype=torch.int8
        )
        qview = qstorage[:, :, : c.heads, :64]
        qview.copy_(qp)
        qp = qview
        sstorage = torch.empty(
            (c.rows, 1, c.heads + 3), device=DEVICE, dtype=torch.int32
        )
        sview = sstorage[:, :, : c.heads]
        sview.copy_(qsf)
        qsf = sview
    if c.layout == "broadcast":
        qp = qp[:1].expand(c.rows, -1, -1, -1)
        qsf = qsf[:1].expand(c.rows, -1, -1)
        weights = weights[:1].expand(c.rows, -1)
        fake_q = fake_q[:1].expand(c.rows, -1, -1).contiguous()
        raw_q = raw_q[:1].expand(c.rows, -1, -1).contiguous()
    if c.layout == "q3d":
        qp, qsf = qp[:, 0], qsf[:, 0]
    return dict(
        case=c,
        lengths=lengths,
        request_ids=request_ids,
        counts=counts,
        lens=torch.tensor(lengths, dtype=torch.int32, device=DEVICE),
        ids=ids,
        mapping=mapping,
        table=table,
        payload=payload,
        sf=sf,
        kv=kv,
        q=qp,
        qsf=qsf,
        raw_q=raw_q,
        fake_q=fake_q,
        weights=weights,
        cand=cand,
        num_sms=num_sms,
        input_bytes=input_estimate,
        representation_probe=representation_probe,
    )


def slice_rows(x, start, end):
    result = dict(x)
    for name in (
        "lens",
        "ids",
        "table",
        "q",
        "qsf",
        "raw_q",
        "fake_q",
        "weights",
        "cand",
    ):
        result[name] = x[name][start:end]
    result["lengths"] = x["lengths"][start:end]
    result["request_ids"] = x["request_ids"][start:end]
    result["case"] = replace(x["case"], rows=end - start)
    return result


def row_chunk(x, budget_bytes):
    c = x["case"]
    width = max(4, math.ceil(c.context / 4) * 4)
    compact = math.ceil(c.capacity * c.block / 512) * 512
    # Full output, TopK scratch, masks, candidate tables, bounded oracle tiles.
    per_row = width * 12 + compact * 12 + c.heads * 1024 * 8 + c.capacity * 24
    free, _ = torch.cuda.mem_get_info()
    usable = min(budget_bytes, max(0, free - 64 * MIB))
    if usable < per_row:
        raise MemoryError(
            f"one reference/baseline row needs {per_row / MIB:.1f} MiB; budget {usable / MIB:.1f} MiB"
        )
    return max(1, min(128, c.rows, usable // per_row))


def metadata(op, x, blocks=None):
    c = x["case"]
    schedule_ids = (
        torch.arange(c.rows, device=DEVICE, dtype=torch.int32) // 2
        if (c.benchmark or c.one_shot) and c.requests == 1
        else x["ids"]
    )
    return op.get_paged_sparse_mqa_logits_metadata(
        x["lens"],
        x["table"],
        schedule_ids,
        c.page,
        x["cand"] if blocks is None else blocks,
        x["q"].dtype,
        c.block,
        x["num_sms"],
    )


def compact_mapping(x):
    c = x["case"]
    columns = torch.arange(c.capacity * c.block, device=DEVICE)
    logical = x["cand"].long()[:, columns // c.block] * c.block + columns % c.block
    live = (logical >= 0) & (logical < x["lens"][:, None])
    return logical, live


def oracle(x, columns_per_tile=256):
    """Independent mathematical FP32 reference with bounded K tiles.

    Dot products are accumulated separately over four groups of 32 coordinates.
    Applying Q and K's *joint* power-of-two exponent after each unscaled dot
    avoids overflow of one operand in reciprocal-scale format-domain probes.
    It is also independent of MMAC tile layouts and consumer head reductions.
    """
    c = x["case"]
    qp = x["q"][:, 0] if x["q"].dim() == 4 else x["q"]
    qs = x["qsf"][:, 0] if x["qsf"].dim() == 3 else x["qsf"]
    q = fp4_values(qp).unflatten(-1, (4, 32))
    qexp = scale_bytes(qs).int() - 127
    logical, live = compact_mapping(x)
    result = torch.empty(logical.shape, dtype=torch.float32, device=DEVICE)
    for start in range(0, logical.size(1), columns_per_tile):
        end = min(logical.size(1), start + columns_per_tile)
        positions = logical[:, start:end].clamp(0, max(0, c.context - 1))
        physical = x["table"].gather(1, positions // c.page).long()
        kp = x["payload"][physical, positions % c.page]
        ks = x["sf"][physical, positions % c.page]
        k = fp4_values(kp).unflatten(-1, (4, 32))
        dot = torch.zeros(
            (c.rows, c.heads, end - start), dtype=torch.float32, device=DEVICE
        )
        for group in range(4):
            unscaled = torch.bmm(q[:, :, group], k[:, :, group].transpose(1, 2))
            exponent = qexp[:, :, group, None] + ks[:, None, :, group].int() - 127
            dot.add_(torch.ldexp(unscaled, exponent))
        scores = (dot.clamp_min_(0) * x["weights"][:, :, None]).sum(dim=1)
        result[:, start:end] = scores.masked_fill(~live[:, start:end], -math.inf)
    return result


def topk_remap(x, scores, topk, compact=True):
    take = min(topk, scores.size(1))
    values, ids = torch.topk(scores, take, dim=1)
    if compact:
        c = x["case"]
        ids = x["cand"].long().gather(1, ids // c.block) * c.block + ids % c.block
    ids = ids.masked_fill(~torch.isfinite(values), INT_MAX)
    if take < topk:
        ids = torch.nn.functional.pad(ids, (0, topk - take), value=INT_MAX)
    return (
        ids.sort(dim=1).values.masked_fill(ids.sort(dim=1).values == INT_MAX, -1).int()
    )


def check_topk(x, scores, reference, selected, topk, atol, rtol):
    logical, live = compact_mapping(x)
    assert selected.shape == (scores.size(0), topk) and selected.dtype == torch.int32
    ordered = selected.masked_fill(selected < 0, INT_MAX)
    assert bool(
        (ordered[:, 1:] >= ordered[:, :-1]).all()
    ), "TopK logical IDs must be sorted before -1 padding"
    valid = selected >= 0
    assert bool(
        (~valid[:, 1:] | (ordered[:, 1:] != ordered[:, :-1])).all()
    ), "duplicate TopK ID"
    expected_count = live.sum(1).clamp_max(topk)
    assert torch.equal(
        valid.sum(1), expected_count
    ), "TopK invalid/-1 padding cardinality differs"
    exact_reference = topk_remap(x, reference, topk)
    exact_rows = (selected == exact_reference).all(1)
    tie_rows = 0
    for row in range(scores.size(0)):
        count = int(expected_count[row])
        if not count:
            continue
        valid_ids = logical[row, live[row]]
        valid_scores = reference[row, live[row]]
        ids = selected[row, :count].long()
        pos = torch.searchsorted(valid_ids, ids)
        assert bool((pos < valid_ids.numel()).all()) and torch.equal(
            valid_ids[pos], ids
        ), "TopK selected a noncandidate/tail ID"
        cutoff = torch.topk(valid_scores, count).values[-1]
        tolerance = atol + rtol * valid_scores.abs().max()
        chosen_scores = valid_scores[pos]
        assert bool(
            (chosen_scores >= cutoff - tolerance).all()
        ), "TopK ID lies below oracle's cutoff"
        strict = valid_ids[valid_scores > cutoff + tolerance]
        hit = torch.searchsorted(ids, strict)
        assert bool((hit < count).all()) and torch.equal(
            ids[hit], strict
        ), "TopK lost an oracle ID above all tie tolerance"
        tie_rows += int(bool((valid_scores == cutoff).sum() > 1))
    return dict(
        exact_id_rows=int(exact_rows.sum()),
        rows=scores.size(0),
        oracle_cutoff_tie_rows=tie_rows,
        tie_aware_ids="PASS",
    )


def compare(x, got, ref, label, args):
    assert (
        got.dtype == torch.float32 and got.shape == ref.shape
    ), f"{label}: FP32 shape mismatch"
    assert torch.equal(
        torch.isneginf(got), torch.isneginf(ref)
    ), f"{label}: invalid/tail -inf differs"
    finite = torch.isfinite(ref)
    assert bool(
        torch.isfinite(ref[~torch.isneginf(ref)]).all()
    ), "oracle produced a nonfinite live score"
    assert bool(torch.isfinite(got[finite]).all()), f"{label}: live NaN/Inf"
    a, b = got[finite], ref[finite]
    maximum = float((a - b).abs().max()) if a.numel() else 0.0
    refmax = float(b.abs().max()) if b.numel() else 0.0
    # Normalize before squaring so wide but finite FP32 logits cannot overflow
    # the error statistic; reduction and all arithmetic remain FP32.
    normalization = max(refmax, float(a.abs().max()) if a.numel() else 0.0, 1e-30)
    aa, bb = a / normalization, b / normalization
    energy = float((aa.square() + bb.square()).sum())
    diff = float((aa - bb).square().sum()) / energy if energy else 0.0
    assert (
        diff <= args.calc_diff
    ), f"{label}: calc_diff={diff:.3g} > {args.calc_diff:.3g}"
    assert (
        maximum <= args.atol + args.rtol * refmax
    ), f"{label}: max_abs={maximum:.4g}, refmax={refmax:.4g}"
    selected = topk_remap(x, got, args.topk)
    ids = check_topk(x, got, ref, selected, args.topk, args.atol, args.rtol)
    return dict(
        label=label,
        calc_diff=diff,
        max_abs_error=maximum,
        reference_abs_max=refmax,
        bitwise_equal=torch.equal(got, ref),
        topk=ids,
    )


def validate_metadata_header(x, meta):
    c = x["case"]
    blocks_per_split = 640 // c.block
    max_splits = c.rows * math.ceil(c.capacity / blocks_per_split)
    capacity = (
        16
        + max_splits * (16 + blocks_per_split * 8)
        + math.ceil(max_splits / x["num_sms"]) * x["num_sms"] * 16
    )
    assert meta.dtype == torch.uint8 and meta.dim() == 1 and meta.is_contiguous()
    assert meta.numel() == capacity, "official metadata allocation ABI changed"
    splits, waves, unaligned, reserved = meta[:16].view(torch.int32).cpu().tolist()
    assert 0 <= splits <= max_splits and waves >= 1 and unaligned == 0 and reserved == 0
    return dict(
        bytes=meta.numel(),
        splits=splits,
        waves=waves,
        num_sms=x["num_sms"],
        reserved=reserved,
        official_q_group=2,
    )


def timing_stats(values):
    values = sorted(values)
    mean = statistics.mean(values)
    return dict(
        p50_ms=statistics.median(values),
        p90_ms=values[math.ceil(len(values) * 0.9) - 1],
        mean_ms=mean,
        min_ms=values[0],
        max_ms=values[-1],
        stdev_ms=statistics.pstdev(values),
        cv=statistics.pstdev(values) / mean if mean else 0.0,
    )


def measure(fn, warmup, repeats, graph):
    for _ in range(warmup):
        result = fn()
    torch.cuda.synchronize()
    captured = graph_obj = None
    timed = fn
    if graph:
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                result = fn()
        stream.synchronize()
        graph_obj = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph_obj, stream=stream):
            captured = fn()
        for _ in range(warmup):
            graph_obj.replay()
        torch.cuda.synchronize()
        timed = graph_obj.replay
    pairs = [
        (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
        for _ in range(repeats)
    ]
    for begin, end in pairs:
        begin.record()
        result = timed()
        end.record()
    torch.cuda.synchronize()
    values = [begin.elapsed_time(end) for begin, end in pairs]
    del result, captured, graph_obj
    return timing_stats(values)


def _score(op, inputs, planned, *, whole_blocks=False, output=None, is_prefill=True):
    case = inputs["case"]
    tail = (
        {}
        if whole_blocks
        else dict(context_lens=inputs["lens"], sparse_kv_block_indices=inputs["cand"])
    )
    return op.fp8_fp4_paged_sparse_mqa_logits(
        (inputs["q"], inputs["qsf"]),
        inputs["kv"],
        inputs["weights"],
        planned,
        case.capacity,
        case.block,
        output=output,
        num_sms=inputs["num_sms"],
        is_prefill=is_prefill,
        **tail,
    )


def _check_output(inputs, actual, *, whole_blocks=False):
    args = SimpleNamespace(atol=1e-5, rtol=5e-6, calc_diff=5e-12, topk=32)
    assert actual.dtype == torch.float32
    assert actual.shape == (
        inputs["case"].rows,
        inputs["case"].capacity * inputs["case"].block,
    )
    for start in range(0, actual.shape[0], 32):
        end = min(actual.shape[0], start + 32)
        part = slice_rows(inputs, start, end)
        reference = oracle(part)
        got = actual[start:end]
        if whole_blocks:
            logical, valid = compact_mapping(part)
            block_live = (
                part["cand"].long() * part["case"].block < part["lens"][:, None]
            )
            block_live &= part["cand"] >= 0
            padded = ~block_live.repeat_interleave(part["case"].block, dim=1)
            assert torch.isneginf(got[padded]).all()
            # Whole-block scoring delegates the causal tail to its caller.
            got = got.masked_fill(~valid, -math.inf)
        compare(part, got, reference, "paged-sparse/oracle", args)


@pytest.mark.parametrize("case", case_matrix(), ids=lambda case: case.id)
def test_prefill_logits_against_fp32(case):
    import lightop.attention as op

    sms = torch.cuda.get_device_properties(0).multi_processor_count
    inputs = make_inputs(case, 713300, sms)
    planned = metadata(op, inputs)
    validate_metadata_header(inputs, planned)
    result = _score(op, inputs, planned)
    _check_output(inputs, result)
    if case.id.startswith("smoke"):
        _check_output(inputs, _score(op, inputs, planned, is_prefill=False))
    width = case.capacity * case.block
    storage = torch.empty(
        (case.rows + 1, math.ceil(width / 512) * 512 + 16), device=DEVICE
    )
    supplied = storage[1:, :width]
    returned = _score(op, inputs, planned, output=supplied)
    assert returned.data_ptr() == supplied.data_ptr()
    _check_output(inputs, returned)


@pytest.mark.parametrize("context", [32768, 65536, 98304, 131072])
def test_prepared_whole_block_paths(context):
    import lightop.attention as op

    sms = torch.cuda.get_device_properties(0).multi_processor_count
    case = Case(
        "whole-block",
        rows=64,
        capacity=2048,
        context=context,
        layout="packed",
        causal=True,
        benchmark=True,
    )
    inputs = make_inputs(case, 713300, sms)
    planned = metadata(op, inputs)
    _check_output(
        inputs, _score(op, inputs, planned, whole_blocks=True), whole_blocks=True
    )


def test_graph_replay_refreshes_q_k_weights_and_preserves_output_storage():
    import lightop.attention as op

    sms = torch.cuda.get_device_properties(0).multi_processor_count
    case = Case(
        "graph", rows=64, capacity=2048, context=4096, layout="packed", benchmark=True
    )
    inputs = make_inputs(case, 713300, sms)
    planned = metadata(op, inputs)
    output = torch.empty((case.rows, case.capacity * case.block), device=DEVICE)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            _score(op, inputs, planned, output=output, whole_blocks=True)
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        captured = _score(op, inputs, planned, output=output, whole_blocks=True)
    for mutate in [False, True]:
        if mutate:
            inputs["q"].zero_()
            inputs["payload"].bitwise_xor_(0x88)
            inputs["weights"].neg_()
        graph.replay()
        torch.cuda.synchronize()
        assert captured.data_ptr() == output.data_ptr()
        _check_output(inputs, captured, whole_blocks=True)


def test_candidate_isolation():
    import lightop.attention as op

    sms = torch.cuda.get_device_properties(0).multi_processor_count
    case = Case("isolation", rows=4, capacity=4, context=1024, layout="packed")
    inputs = make_inputs(case, 713300, sms)
    planned = metadata(op, inputs)
    original = _score(op, inputs, planned).clone()
    selected = set(inputs["cand"].reshape(-1).cpu().tolist())
    unselected = next(b for b in range(case.context // case.block) if b not in selected)

    def mutate_block(block):
        token = block * case.block
        page = int(inputs["mapping"][0, token // case.page])
        offset = token % case.page
        inputs["payload"][page, offset : offset + case.block].bitwise_xor_(0x77)

    mutate_block(unselected)
    assert torch.equal(original, _score(op, inputs, planned))
    chosen = next(b for b in selected if b != INT_MAX)
    mutate_block(chosen)
    changed = _score(op, inputs, planned)
    affected = (inputs["cand"] == chosen).repeat_interleave(case.block, dim=1)
    assert torch.equal(original[~affected], changed[~affected])
    _check_output(inputs, changed)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--contexts", type=int, nargs="+", default=[4096, 16384, 32768, 65536, 131072]
    )
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--mode", choices=["eager", "graph", "both"], default="both")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.warmup < 10 or args.repeats < 100:
        parser.error("warmup must be >=10 and repeats must be >=100")
    import lightop.attention as op

    sms = torch.cuda.get_device_properties(0).multi_processor_count
    modes = ["eager", "graph"] if args.mode == "both" else [args.mode]
    records = []
    for context in args.contexts:
        rows = min(context, 32768) // 8
        if rows <= 0:
            parser.error("context must be >=8")
        case = Case(
            "benchmark",
            rows=rows,
            capacity=2048,
            context=context,
            layout="packed",
            causal=True,
            benchmark=True,
            cp_world=8,
            prefix=max(0, context - 32768),
            one_shot=True,
        )
        inputs = make_inputs(case, 713300, sms)
        planned = metadata(op, inputs)
        output = _score(op, inputs, planned, whole_blocks=True)
        _check_output(inputs, output, whole_blocks=True)
        logical, valid = compact_mapping(inputs)
        useful_flops = 2 * case.heads * 128 * int(valid.sum())
        timings = {}
        for mode in modes:
            stats = measure(
                lambda: _score(op, inputs, planned, whole_blocks=True, output=output),
                args.warmup,
                args.repeats,
                mode == "graph",
            )
            stats["useful_tflops"] = useful_flops / (stats["p50_ms"] * 1e9)
            timings[mode] = stats
        records.append(
            dict(
                context=context,
                rows=rows,
                useful_dot_flops=useful_flops,
                timings=timings,
            )
        )
        del inputs, planned, output, logical, valid
        gc.collect()
        torch.cuda.empty_cache()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(records, indent=2) + "\n")


if __name__ == "__main__":
    main()
