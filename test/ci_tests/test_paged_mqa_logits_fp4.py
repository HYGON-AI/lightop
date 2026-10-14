# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Public FP4 paged logits accuracy, input guards and live CUDA/HIP graph replay.

The contract is DeepGEMM's `fp8_fp4_paged_mqa_logits` with clean_logits false:
column t of row b scores pool slot block_table[b, t // TB] * TB + t % TB, and
columns at or past the row's visible length are left unwritten. Accuracy
checks compare each row's visible prefix only; one test asserts the tail is
left untouched.

Acceptance mirrors DeepGEMM's MXFP4 MQA logits tests. DeepGEMM compares its
kernel against a reference built from the *dequantized* operands using
calc_diff, with ref_diff_tol() = 3e-5 whenever a BF16 reduce is involved
(tests/test_attention.py); LightOp's exact BF16 MMAC fallback qualifies.
calc_diff alone is a global cosine measure, so every case additionally carries
a per-element bound derived from the FP32 accumulator's unit roundoff over 128
channels and `heads` head products.
"""

import pytest
import torch

# DeepGEMM tests/test_attention.py::ref_diff_tol(has_bf16=True).
MAX_DIFF = 3e-5
# Worst-case FP32 accumulation growth factor, 4 * (channels + heads) * 2**-24.
UNIT_ROUNDOFF = 2.0 ** -24
# calc_diff is a global cosine, so a uniform relative bias b only moves it by
# about b**2/2; the per-element bound is a worst case over absolute
# contributions. Neither rejects a small uniform bias, so the bulk of the
# relative-error distribution is checked too: 64 FP32 ulps at the 90th
# percentile still rejects a 1e-4 bias by 26x.
BULK_QUANTILE = 0.9
BULK_TOL = 64 * UNIT_ROUNDOFF
# Percentiles are only meaningful with enough scored logits behind them.
BULK_MIN_SAMPLES = 256

# Index-K pool page: 64 slots, all 64-byte E2M1 payloads then all 4-byte scales.
PAGE_SLOTS = 64
PAGE_BYTES = PAGE_SLOTS * 68
# E2M1 values by 4-bit code, sign in bit 3.
CODES = [0, .5, 1, 1.5, 2, 3, 4, 6, -0., -.5, -1, -1.5, -2, -3, -4, -6]


def get_operator():
    from lightop.attention import paged_mqa_logits_fp4
    return paged_mqa_logits_fp4


def calc_diff(x, y):
    """DeepGEMM deep_gemm/testing/numeric.py::calc_diff."""
    x, y = x.double(), y.double()
    denominator = (x * x + y * y).sum()
    if denominator == 0:
        return 0.0
    return float(1 - 2 * (x * y).sum() / denominator)


def assemble_pool(payload_rows, scale_rows):
    """[pages*64, 64] payload and [pages*64, 4] scale rows -> [pages, 64, 1, 68]."""
    pages = payload_rows.shape[0] // PAGE_SLOTS
    flat = torch.cat((payload_rows.reshape(pages, -1), scale_rows.reshape(pages, -1)), dim=1)
    return flat.view(pages, PAGE_SLOTS, 1, 68)


def payload(pool):
    """Writable [pages, 64, 64] view of the pool's E2M1 payload rows."""
    return pool.reshape(pool.shape[0], PAGE_BYTES)[:, :PAGE_SLOTS * 64].unflatten(1, (PAGE_SLOTS, 64))


def scales(pool):
    """Writable [pages, 64, 4] view of the pool's UE8M0 scale rows."""
    return pool.reshape(pool.shape[0], PAGE_BYTES)[:, PAGE_SLOTS * 64:].unflatten(1, (PAGE_SLOTS, 4))


def quantize_mxfp4(x):
    """[..., 128] floats -> packed E2M1 [..., 64] uint8 and UE8M0 [..., 4] uint8.

    OCP MX rounding: the scale is the smallest power of two that brings the
    block max under 6, values round to nearest on the E2M1 grid.
    """
    x = x.float().reshape(*x.shape[:-1], 4, 32)
    exps = torch.ceil(torch.log2(x.abs().amax(-1, keepdim=True).clamp_min(6 * 2.**-126) / 6))
    exps = exps.clamp(-125, 125)
    v = (x / torch.exp2(exps)).clamp(-6, 6)
    step = torch.where(v.abs() < 2, .5, torch.where(v.abs() < 4, 1., 2.))
    mag = (v.abs() / step).round() * step
    grid = torch.tensor(CODES[:8], device=x.device)
    idx = (mag.unsqueeze(-1) == grid).float().argmax(-1)
    codes = (idx + 8 * (v < 0) * (idx > 0)).to(torch.uint8).flatten(-2)
    packed = codes[..., 0::2] | (codes[..., 1::2] << 4)
    return packed, (exps.squeeze(-1) + 127).to(torch.uint8)


def pack_q(packed, scale_bytes):
    """[B, H, 64] payload and [B, H, 4] scale bytes -> (q [B,1,H,64] int8, q_sf [B,1,H] int32)."""
    batch, heads = packed.shape[:2]
    q = packed.reshape(batch, 1, heads, 64).view(torch.int8).contiguous()
    q_sf = scale_bytes.contiguous().view(torch.int32).reshape(batch, 1, heads).contiguous()
    return q, q_sf


def make_case(batch, heads, length, table_block_size=64, seed=0, scale_lo=118,
              scale_hi=138, realistic=False, block_jitter=1, strided=False):
    """Random pool, random block table over it, and lens that exercise edges.

    The default scale window spans 20 E8M0 steps, so a token's four block
    scales routinely differ by more than the FP8 fold can carry, and both
    dispatches run. `realistic=True` instead quantizes random KV, giving the
    block-scale distribution production traffic produces; `block_jitter` is
    the per-32-block exponent spread on top of the per-token magnitude.
    """
    torch.manual_seed(seed)
    blocks = max(1, (length + table_block_size - 1) // table_block_size)
    units_per_page = PAGE_SLOTS // table_block_size
    pages = (batch * blocks + units_per_page - 1) // units_per_page + 2
    slots = pages * PAGE_SLOTS

    if realistic:
        kv = torch.randn(slots, 4, 32, device="cuda")
        kv *= torch.exp2(torch.randint(-8, 9, (slots, 1, 1), device="cuda").float())
        if block_jitter:
            kv *= torch.exp2(torch.randint(-block_jitter, block_jitter + 1,
                                           (slots, 4, 1), device="cuda").float())
        pool = assemble_pool(*quantize_mxfp4(kv.reshape(slots, 128)))
    else:
        pool = assemble_pool(
            torch.randint(0, 256, (slots, 64), device="cuda", dtype=torch.uint8),
            torch.randint(scale_lo, scale_hi, (slots, 4), device="cuda", dtype=torch.uint8))

    # Every entry is a distinct block, deliberately out of order.
    order = torch.randperm(pages * units_per_page, device="cuda")[:batch * blocks]
    table = order.reshape(batch, blocks).to(torch.int32)

    q, q_sf = pack_q(*quantize_mxfp4(torch.randn(batch, heads, 128, device="cuda")))
    weights = torch.randn(batch, heads, device="cuda")
    lens = torch.full((batch,), length, device="cuda", dtype=torch.int32)
    if batch > 1:
        lens[0] = 0
        lens[1] = max(0, length - 7)

    if strided:
        # Pages with padding between them, and head weights with a row stride.
        buf = torch.zeros(pages, PAGE_BYTES + 16, device="cuda", dtype=torch.uint8)
        buf[:, :PAGE_BYTES].copy_(pool.reshape(pages, PAGE_BYTES))
        pool = buf[:, :PAGE_BYTES].unflatten(1, (PAGE_SLOTS, 1, 68))
        wbuf = torch.zeros(batch, heads + 8, device="cuda")
        wbuf[:, :heads].copy_(weights)
        weights = wbuf[:, :heads]
    return [q, q_sf, pool, weights, lens, table]


def table_width(case, table_block_size):
    return case[5].shape[1] * table_block_size


def visible_lengths(case, max_context_len, table_block_size):
    """The kernel's visible_len: the context, capped by the output and the table."""
    cap = min(max_context_len, table_width(case, table_block_size))
    return [max(0, min(int(n), cap)) for n in case[4].tolist()]


def dequantize(packed, scale_bytes):
    """[n, 64] packed E2M1 and [n, 4] UE8M0 -> FP64 [n, 128]."""
    lut = torch.tensor(CODES, device=packed.device, dtype=torch.float64)
    codes = torch.stack((packed & 15, packed >> 4), dim=-1).flatten(-2).long()
    return lut[codes] * torch.exp2(scale_bytes.double() - 127).repeat_interleave(32, dim=-1)


def reference_with_bound(case, max_context_len, table_block_size):
    """FP64 logits and the sum of absolute contributions per logit.

    Each per-32 block scale multiplies its own operand, the 128 scaled products
    form one dot, and relu, head weighting and the head sum follow with no
    further scale. The second tensor bounds FP32 accumulation error, since a
    dot of n terms carries at most n * 2**-24 * sum|terms|.
    """
    q, q_sf, pool, weights, lens, table = case
    batch, heads = weights.shape
    q_values = dequantize(q.view(torch.uint8).reshape(batch, heads, 64),
                          q_sf.view(torch.uint8).reshape(batch, heads, 4))
    pool_payload, pool_scales = payload(pool), scales(pool)
    out = torch.zeros(batch, max_context_len, device=q.device, dtype=torch.float64)
    mag = torch.zeros_like(out)
    for b, n in enumerate(visible_lengths(case, max_context_len, table_block_size)):
        if not n:
            continue
        t = torch.arange(n, device=q.device)
        slot = table[b, t // table_block_size].long() * table_block_size + t % table_block_size
        page, offset = slot // PAGE_SLOTS, slot % PAGE_SLOTS
        k = dequantize(pool_payload[page, offset], pool_scales[page, offset])
        w = weights[b].double()
        out[b, :n] = ((q_values[b] @ k.T).relu() * w[:, None]).sum(0)
        mag[b, :n] = ((q_values[b].abs() @ k.abs().T) * w.abs()[:, None]).sum(0)
    return out, mag


def assert_logits_close(got, case, max_context_len, table_block_size, max_diff=MAX_DIFF):
    """Compare the visible prefix of every row against the FP64 reference."""
    heads = case[3].shape[1]
    expected, mag = reference_with_bound(case, max_context_len, table_block_size)
    visible = torch.zeros(expected.shape, device=got.device, dtype=torch.bool)
    for b, n in enumerate(visible_lengths(case, max_context_len, table_block_size)):
        visible[b, :n] = True
    got = got[:, :max_context_len].double().masked_fill(~visible, 0)
    expected = expected.masked_fill(~visible, 0)
    assert torch.isfinite(got).all(), "visible logits must be finite"
    diff = calc_diff(got, expected)
    assert diff < max_diff, f"calc_diff={diff:.3e} exceeds {max_diff:.1e}"
    tol = 4 * (128 + heads) * UNIT_ROUNDOFF * mag
    err = (got - expected).abs()
    bad = visible & (err > tol)
    if bad.any():
        idx = tuple(bad.nonzero()[0].tolist())
        raise AssertionError(
            f"{int(bad.sum())} logits exceed the FP32 accumulation bound; first at "
            f"{idx}: got {float(got[idx])!r} expected {float(expected[idx])!r} "
            f"tol {float(tol[idx]):.3e}")
    # A bias too small for either bound above still has to show up here.
    scored = visible & (expected != 0)
    if int(scored.sum()) >= BULK_MIN_SAMPLES:
        bulk = float(torch.quantile((err[scored] / expected.abs()[scored]).float(), BULK_QUANTILE))
        assert bulk <= BULK_TOL, (
            f"{BULK_QUANTILE:.0%} of logits carry more than {bulk:.3e} relative error "
            f"(limit {BULK_TOL:.1e}); the accumulation is biased rather than rounding")


def run(op, case, max_context_len, table_block_size=64, schedule_meta=None, out=None, **override):
    q, q_sf, pool, weights, lens, table = case
    args = dict(q=q, q_sf=q_sf, kv_cache=pool, weights=weights, context_lens=lens,
                block_table=table, schedule_meta=schedule_meta,
                max_context_len=max_context_len, table_block_size=table_block_size, out=out)
    args.update(override)
    return op(**args)


def check(op, case, table_block_size=64, **kwargs):
    width = table_width(case, table_block_size)
    got = run(op, case, width, table_block_size, **kwargs)
    assert got.shape == (case[0].shape[0], width) and got.dtype == torch.float32
    assert_logits_close(got, case, width, table_block_size)
    return got


@pytest.fixture(scope="module")
def op():
    if not torch.cuda.is_available():
        pytest.skip("GPU required")
    arch = torch.cuda.get_device_properties(0).gcnArchName
    if not arch.startswith(("gfx928", "gfx936", "gfx938")):
        pytest.skip("FP4 paged logits require gfx928/936/938")
    return get_operator()


def test_abi_marker():
    """Consumers feature-detect on this before trusting the signature."""
    from lightop import attention
    assert attention.PAGED_MQA_LOGITS_FP4_ABI >= 3


@pytest.mark.parametrize("heads", [16, 32, 64])
@pytest.mark.parametrize("table_block_size,length",
                         [(64, 1), (64, 63), (64, 257), (8, 1), (8, 63), (8, 4097)])
def test_parity(op, heads, table_block_size, length):
    check(op, make_case(3, heads, length, table_block_size, seed=heads + length, strided=True),
          table_block_size)


@pytest.mark.parametrize("heads", [16, 32])
@pytest.mark.parametrize("length", [2049, 49152])
def test_realistic_mxfp4_pool(op, heads, length):
    """True MXFP4 KV, so block-scale spans follow the data, not a test knob."""
    check(op, make_case(1, heads, length, seed=heads + length, realistic=True))


@pytest.mark.parametrize("scale_lo,scale_hi", [(110, 146), (120, 133), (100, 156)])
def test_wide_block_scale(op, scale_lo, scale_hi):
    """Block scales far wider than the FP8 fold can carry.

    The window stops at 56 E8M0 steps because the shared FP32 accumulator, here
    and in DeepGEMM, overflows once a single token's K values span the FP32
    range; that is a property of the format, not of the dispatch.
    """
    check(op, make_case(2, 32, 4097, seed=91, scale_lo=scale_lo, scale_hi=scale_hi))


@pytest.mark.parametrize("span", [0, 1, 5, 6, 10, 11, 12])
def test_block_scale_span_boundary(op, span):
    """The gfx938 FP8 fold is exact up to a span of 11 and must hand over beyond.

    Both sides have to produce the same accuracy, otherwise the dispatch itself
    is observable in the output.
    """
    case = make_case(2, 32, 4097, seed=5)
    pool_scales = scales(case[2])
    pool_scales.random_(127 - span, 128)
    pool_scales[..., 0] = 127
    check(op, case)


def test_schedule_persistent(op):
    """The split-KV schedule selects the persistent kernel and must match."""
    from lightop.attention import get_paged_mqa_logits_metadata
    case = make_case(4, 32, 8193, seed=23)
    case[4].copy_(torch.tensor([0, 1, 4097, 8193], device="cuda", dtype=torch.int32))
    cus = torch.cuda.get_device_properties(0).multi_processor_count
    schedule = get_paged_mqa_logits_metadata(case[4], 64, cus)
    check(op, case, schedule_meta=schedule)


def test_output_row_stride(op):
    """Without out, rows are padded to the 256-position split like DeepGEMM."""
    case = make_case(2, 16, 300, seed=2)
    got = run(op, case, 300)
    assert got.shape == (2, 300) and got.stride(0) == 512
    assert_logits_close(got, case, 300, 64)


def test_out_buffer_and_unwritten_tail(op):
    """A caller-owned out is written only up to each row's visible length."""
    sentinel = 12345.0
    case = make_case(3, 32, 2049, seed=29)
    width = table_width(case, 64)
    case[4].copy_(torch.tensor([0, 130, width], device="cuda", dtype=torch.int32))
    out = torch.full((3, width), sentinel, device="cuda")
    got = run(op, case, width, out=out)
    assert got.data_ptr() == out.data_ptr()
    assert_logits_close(got, case, width, 64)
    for b, n in enumerate(visible_lengths(case, width, 64)):
        assert (out[b, n:] == sentinel).all(), f"row {b} tail was written"


@pytest.mark.parametrize("table_block_size", [64, 8])
def test_graph_replay(op, table_block_size):
    """One captured launch must follow live data, lengths and block tables."""
    case = make_case(3, 32, 16385 if table_block_size == 64 else 4097, table_block_size, seed=3)
    width = table_width(case, table_block_size)
    output = torch.empty(3, width, device="cuda")
    run(op, case, width, table_block_size, out=output)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run(op, case, width, table_block_size, out=output)
    for seed, lens in ((11, [0, 513, width]), (13, [width, 1, 9])):
        new = make_case(3, 32, width, table_block_size, seed=seed)
        for old_tensor, new_tensor in zip(case, new):
            old_tensor.copy_(new_tensor)
        case[4].copy_(torch.tensor(lens, device="cuda", dtype=torch.int32))
        graph.replay()
        assert_logits_close(output, case, width, table_block_size)


def verify_case(draft, heads, length, seed, requests):
    """Speculative verify rows: each request's `draft` rows share its block table
    row and see lengths L, L + 1, ..., with L = 0 and L at the table width among
    the starts, and a request count either side of the row-pairing gate."""
    starts = [0, 1, 255, length - draft + 1, 700, 1300, 1700, 1900][:requests]
    case = make_case(len(starts) * draft, heads, length, seed=seed)
    tables = case[5].reshape(len(starts), draft, -1)[:, :1].expand(-1, draft, -1)
    case[5] = tables.reshape(len(starts) * draft, -1).contiguous()
    lens = [start + i for start in starts for i in range(draft)]
    case[4] = torch.tensor(lens, device="cuda", dtype=torch.int32)
    return case


@pytest.mark.parametrize("requests", [4, 8])
@pytest.mark.parametrize("with_schedule", [False, True])
@pytest.mark.parametrize("heads", [16, 32])
@pytest.mark.parametrize("draft", [2, 3, 4, 6])
def test_rows_per_request_matches_unpaired(op, draft, heads, with_schedule, requests):
    """Scoring a request's rows in pairs over one KV read must not change a bit
    of any row, whatever dispatch (FP8 fold or BF16) each row takes; below the
    pairing gate the same call has to fall back to one row per route."""
    from lightop.attention import get_paged_mqa_logits_metadata, paged_mqa_logits_route_lens
    case = verify_case(draft, heads, 2049, seed=draft * 100 + heads, requests=requests)
    width = table_width(case, 64)
    cus = torch.cuda.get_device_properties(0).multi_processor_count

    def schedule(lens):
        return get_paged_mqa_logits_metadata(lens, 64, cus) if with_schedule else None

    want = run(op, case, width, schedule_meta=schedule(case[4]))
    got = run(op, case, width, rows_per_request=draft,
              schedule_meta=schedule(paged_mqa_logits_route_lens(case[4], draft)))
    for b, n in enumerate(visible_lengths(case, width, 64)):
        assert torch.equal(got[b, :n].view(torch.int32), want[b, :n].view(torch.int32)), f"row {b}"
    assert_logits_close(got, case, width, 64)


def test_route_lens_follow_the_pair_gate(op):
    """The gate decides the dispatch, so the schedule's input has to follow it: a
    row count below it keeps one length per row, at or above it one per row pair."""
    from lightop import op as ext
    from lightop.attention import paged_mqa_logits_route_lens
    gate, draft = ext.PAGED_MQA_LOGITS_FP4_PAIR_MIN_REQUESTS, 6
    below = torch.arange(1, (gate - 1) * draft + 1, device="cuda", dtype=torch.int32)
    assert paged_mqa_logits_route_lens(below, draft).equal(below)
    lens = torch.arange(1, gate * draft + 1, device="cuda", dtype=torch.int32)
    paired = lens.reshape(-1, draft // 2, 2).amax(-1).flatten()
    assert paged_mqa_logits_route_lens(lens, draft).equal(paired)


def test_pairing_needs_heads_at_most_32(op):
    """H = 64 has no paired instantiation: two rows of Q in VGPRs would halve
    occupancy, which measured slower than the unpaired walk."""
    case = make_case(2, 64, 257)
    with pytest.raises(RuntimeError, match="H <= 32"):
        run(op, case, table_width(case, 64), rows_per_request=2)


def test_empty(op):
    case = make_case(2, 32, 64, seed=1)
    assert run(op, case, 0).shape == (2, 0)
    empty = make_case(1, 32, 64, seed=1)
    empty = [t[:0] if t.dim() and t is not empty[2] else t for t in empty]
    assert run(op, empty, 64).shape == (0, 64)


def test_input_guards(op):
    case = make_case(2, 32, 257)
    width = table_width(case, 64)
    bad = {
        "q dtype": dict(q=case[0].float()),
        "q_sf dtype": dict(q_sf=case[1].long()),
        "pool dtype": dict(kv_cache=case[2].to(torch.int16)),
        "pool slot bytes": dict(kv_cache=torch.zeros(2, 64, 1, 72, device="cuda", dtype=torch.uint8)),
        "weights dtype": dict(weights=case[3].bfloat16()),
        "weights heads": dict(weights=case[3][:, :-1]),
        "context_lens dtype": dict(context_lens=case[4].long()),
        "block_table dtype": dict(block_table=case[5].long()),
        "table_block_size": dict(table_block_size=32),
        "out dtype": dict(out=torch.empty(2, width, device="cuda", dtype=torch.bfloat16)),
        "out width": dict(out=torch.empty(2, width - 1, device="cuda")),
        "rows_per_request not dividing the batch": dict(rows_per_request=3),
        "rows_per_request on a candidate table": dict(rows_per_request=2, table_block_size=8),
    }
    for name, override in bad.items():
        with pytest.raises(RuntimeError):
            run(op, case, width, **override)
            pytest.fail(f"{name} was accepted")


@pytest.mark.parametrize("heads", [8, 48])
def test_unsupported_heads(op, heads):
    case = make_case(1, 16, 64)
    case[0] = torch.zeros(1, 1, heads, 64, device="cuda", dtype=torch.int8)
    case[1] = torch.zeros(1, 1, heads, device="cuda", dtype=torch.int32)
    case[3] = torch.zeros(1, heads, device="cuda")
    with pytest.raises(RuntimeError, match="heads must be"):
        run(op, case, 64)


def test_unavailable_operator_error(op, monkeypatch):
    from lightop import op as native
    monkeypatch.delattr(native, "paged_mqa_logits_fp4")
    with pytest.raises(RuntimeError, match="paged_mqa_logits_fp4 unavailable"):
        run(op, make_case(1, 16, 1), 64)


def single_product_case(k_codes, k_scales, heads=16, group=0, q_code=2, span=0):
    """Pool of len(k_codes) slots where each logit is one exact product.

    Head 0 holds q_code in the first channel of `group` at scale 1, slot t holds
    k_codes[t] in the same channel at k_scales[t]. The other three scale groups
    of Q and K are set 127 + span away, carrying no data but widening the
    per-token scale span to steer the dispatch.
    """
    n = len(k_codes)
    pages = (n + PAGE_SLOTS - 1) // PAGE_SLOTS
    pool_payload = torch.zeros(pages * PAGE_SLOTS, 64, device="cuda", dtype=torch.uint8)
    pool_scales = torch.full((pages * PAGE_SLOTS, 4), 127 + span, device="cuda", dtype=torch.uint8)
    pool_payload[:n, 16 * group] = torch.tensor(k_codes, device="cuda", dtype=torch.uint8)
    pool_scales[:n, group] = torch.tensor(k_scales, device="cuda", dtype=torch.uint8)
    q_payload = torch.zeros(1, heads, 64, device="cuda", dtype=torch.uint8)
    q_payload[0, 0, 16 * group] = q_code
    q_scales = torch.full((1, heads, 4), 127 + span, device="cuda", dtype=torch.uint8)
    q_scales[0, :, group] = 127
    weights = torch.zeros(1, heads, device="cuda")
    weights[0, 0] = 1
    q, q_sf = pack_q(q_payload, q_scales)
    lens = torch.tensor([n], device="cuda", dtype=torch.int32)
    table = torch.arange(pages, device="cuda", dtype=torch.int32)[None]
    return [q, q_sf, assemble_pool(pool_payload, pool_scales), weights, lens, table]


@pytest.mark.parametrize("span", [0, 11, 12])
@pytest.mark.parametrize("group", [0, 1, 2, 3])
@pytest.mark.parametrize("q_code", [2, 10])
def test_all_nibbles_exact(op, span, group, q_code):
    """Every K code in every scale group, both Q signs, through both dispatches.

    One product per logit leaves no accumulation error to absorb, so the FP8
    fold (span <= 11 on gfx938) and the BF16 path must both be bit exact.
    """
    case = single_product_case(list(range(16)) * 4, [127] * 64, group=group,
                               q_code=q_code, span=span)
    got = run(op, case, 64)
    expected, _ = reference_with_bound(case, 64, 64)
    torch.testing.assert_close(got, expected.float(), rtol=0, atol=0)


@pytest.mark.parametrize("heads", [16, 32])
def test_all_e8m0_scales(op, heads):
    """Every valid E8M0 scale against every code, one exact product each.

    E8M0 0xFF is NaN in the OCP MX spec and 0xFD/0xFE push the larger E2M1
    magnitudes past the BF16 range, so the sweep stops at 0xFC. Subnormal
    decodes at the bottom of the range are included and must stay exact.
    """
    count = 16 * 253
    case = single_product_case([j % 16 for j in range(count)], [j // 16 for j in range(count)],
                               heads=heads)
    # All four groups of each token share its scale, as a real pool would.
    scales(case[2]).reshape(-1, 4)[:count] = torch.tensor(
        [j // 16 for j in range(count)], device="cuda", dtype=torch.uint8)[:, None]
    got = run(op, case, count)
    expected, _ = reference_with_bound(case, count, 64)
    torch.testing.assert_close(got, expected.float(), rtol=0, atol=0)


@pytest.mark.parametrize("heads", [16, 32, 64])
def test_subnormal_logits_accumulate(op, heads):
    """Full dots whose every partial sum is an FP32 subnormal.

    K scales 0..3 against Q scales 100..109 put each product below 2^-126, so
    a dot of 128 channels and the head sum both live in the subnormal range,
    where the gfx936 BF16 MMAC flushes its results. The logits still have to
    match the FP64 reference to within one subnormal step.
    """
    case = make_case(2, heads, 1025, seed=17)
    scales(case[2]).random_(0, 4)
    case[1].view(torch.uint8).reshape(2, heads, 4).random_(100, 110)
    width = table_width(case, 64)
    got = run(op, case, width)
    expected, _ = reference_with_bound(case, width, 64)
    n = visible_lengths(case, width, 64)[1]
    want = expected[1, :n].float()
    assert int((want.abs() > 2.0 ** -140).sum()) > n // 4, "the case must produce subnormal logits"
    torch.testing.assert_close(got[1, :n], want, rtol=0, atol=2.0 ** -148)


def test_out_of_range_scales(op):
    """Records what happens past the representable range, rather than hiding it.

    E8M0 0xFD..0xFF scaled by the larger E2M1 magnitudes exceeds the BF16
    range. That is outside the operator's domain -- a conforming MXFP4
    quantizer never emits these scales -- so an element past BF16, or at scale
    0xFF, decodes to zero on every arch instead of leaking inf or NaN.
    """
    count = 16 * 3
    k_scales = [253 + j // 16 for j in range(count)]
    case = single_product_case([j % 16 for j in range(count)], k_scales)
    got = run(op, case, count)
    assert torch.isfinite(got).all(), "out-of-range scales must not leak inf or NaN"
    # Anything the exact product keeps inside FP32 still has to be exact.
    expected, _ = reference_with_bound(case, count, 64)
    nan_scale = torch.tensor(k_scales, device=expected.device).eq(255).expand_as(expected)
    in_range = (expected.abs() < 2.0 ** 128) & ~nan_scale
    torch.testing.assert_close(got[in_range], expected[in_range].float(), rtol=0, atol=0)
    torch.testing.assert_close(got[~in_range], torch.zeros_like(got[~in_range]), rtol=0, atol=0)


def test_fp32_accumulator_cancellation(op):
    """A catastrophically cancelling dot is bounded, not exact.

    DeepGEMM accumulates the whole K dimension in one FP32 accumulator, so a
    term 2**28 below the running partial sum is legitimately lost. Q holds
    +2**20, 2**-8 and -2**20 in three scale groups against K = 1 everywhere;
    the kernel must stay inside the FP32 bound and not invent a larger error.
    """
    q_payload = torch.zeros(1, 16, 64, device="cuda", dtype=torch.uint8)
    q_scales = torch.full((1, 16, 4), 127, device="cuda", dtype=torch.uint8)
    q_payload[0, 0, 0], q_scales[0, 0, 0] = 0x2, 147       # 1.0 * 2**20
    q_payload[0, 0, 16], q_scales[0, 0, 1] = 0x1, 120      # 0.5 * 2**-7
    q_payload[0, 0, 32], q_scales[0, 0, 2] = 0xA, 147      # -1.0 * 2**20
    q, q_sf = pack_q(q_payload, q_scales)
    # K = 1.0 in every channel: code 2 in both nibbles, scale 1.
    pool = assemble_pool(torch.full((PAGE_SLOTS, 64), 0x22, device="cuda", dtype=torch.uint8),
                         torch.full((PAGE_SLOTS, 4), 127, device="cuda", dtype=torch.uint8))
    weights = torch.zeros(1, 16, device="cuda")
    weights[0, 0] = 1
    case = [q, q_sf, pool, weights, torch.ones(1, device="cuda", dtype=torch.int32),
            torch.zeros(1, 1, device="cuda", dtype=torch.int32)]
    expected, mag = reference_with_bound(case, 1, 64)
    assert float(expected[0, 0]) == 2.0 ** -8
    got = float(run(op, case, 1)[0, 0])
    assert abs(got - 2.0 ** -8) <= 4 * (128 + 16) * UNIT_ROUNDOFF * float(mag[0, 0])
