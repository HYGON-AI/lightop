"""Public FP4 logits accuracy, input guards and live CUDA/HIP graph replay.

Acceptance mirrors DeepGEMM's MXFP4 MQA logits tests. DeepGEMM compares its
kernel against a reference built from the *dequantized* operands using
calc_diff, with ref_diff_tol() = 3e-5 whenever a BF16 reduce or BF16 logit
dtype is involved (tests/test_attention.py). LightOp's head weights are BF16,
so 3e-5 is the matching bound. calc_diff alone is a global cosine measure, so
every case additionally carries a per-element bound derived from the FP32
accumulator's unit roundoff over 128 channels and `heads` head products.
"""

import pytest
import torch

# DeepGEMM tests/test_attention.py::ref_diff_tol(has_bf16=True).
MAX_DIFF = 3e-5
# Worst-case FP32 accumulation growth factor, 4 * (channels + heads) * 2**-24.
UNIT_ROUNDOFF = 2.0 ** -24
# calc_diff is a global cosine, so a uniform relative bias b only moves it by
# about b**2/2; at b = 1e-4 that is 5e-9, three orders under MAX_DIFF. The
# per-element bound below is a worst case over the sum of absolute
# contributions, which relu and the head sum leave roughly a thousand times
# above the error the kernel actually makes. Neither rejects a small uniform
# bias, so the bulk of the relative-error distribution is checked too: measured
# across realistic, narrow and 56-step-wide block scales the 90th percentile
# never exceeded 3.2e-7 (five FP32 ulps), while the tail legitimately reaches
# 1e-2 on logits where the heads cancel. 64 ulps keeps an order of magnitude of
# headroom over the observed bulk and still rejects a 1e-4 bias by 26x.
BULK_QUANTILE = 0.9
BULK_TOL = 64 * UNIT_ROUNDOFF
# Percentiles are only meaningful with enough scored logits behind them.
BULK_MIN_SAMPLES = 256

# The public FP4 kernel has a BF16 implementation on gfx928/gfx936/gfx938,
# while its optional FP4 -> FP8 dispatch is selected only for gfx938.  Keep
# this capability table in the test so a hardware limitation is reported as a
# skip instead of an accuracy failure when the exhaustive E8M0 sweep selects
# the FP8 path.
FP4_ARCHES = ("gfx928", "gfx936", "gfx938")
FP4_TO_FP8_ARCHES = ("gfx938",)


def _device_arch() -> str:
    properties = torch.cuda.get_device_properties(0)
    return (getattr(properties, "gcnArchName", "") or "").split(":", 1)[0]


def _require_fp4_to_fp8_support() -> None:
    arch = _device_arch()
    if not any(arch.startswith(prefix) for prefix in FP4_TO_FP8_ARCHES):
        pytest.skip(
            "FP4-to-FP8 MQA dispatch requires gfx938; "
            f"got {arch or 'unknown'}"
        )


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


def fake_quant_fp4(q):
    """Independent per-32 E2M1 / E8M0 query generation, returned as BF16."""
    x = q.float().reshape(*q.shape[:-1], 4, 32)
    scale = torch.exp2(torch.ceil(torch.log2(x.abs().amax(-1, keepdim=True).clamp_min(6 * 2.**-126) / 6)))
    v = (x / scale).clamp(-6, 6)
    step = torch.where(v.abs() < 2, .5, torch.where(v.abs() < 4, 1., 2.))
    return ((v.abs() / step).round() * step * v.sign() * scale).reshape_as(q).bfloat16()


def quantize_kv_to_table(kv, page_size, pages):
    """Pack BF16 KV rows into the pool layout as true MXFP4 (E2M1 + E8M0).

    This produces the block-scale distribution real indexer traffic has, where
    the four per-32 scales of a token differ by whatever the data demands
    rather than by a hand-picked span.
    """
    x = kv.float().reshape(-1, 4, 32)
    exps = torch.ceil(torch.log2(x.abs().amax(-1, keepdim=True).clamp_min(6 * 2.**-126) / 6))
    exps = exps.clamp(-125, 125)
    v = (x / torch.exp2(exps)).clamp(-6, 6)
    step = torch.where(v.abs() < 2, .5, torch.where(v.abs() < 4, 1., 2.))
    mag = (v.abs() / step).round() * step
    codes = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6], device=kv.device)
    idx = (mag.reshape(-1, 1) == codes).float().argmax(-1).reshape(mag.shape)
    nibble = (idx + 8 * (v < 0) * (idx > 0)).to(torch.uint8).reshape(-1, 128)
    payload = nibble[:, 0::2] | (nibble[:, 1::2] << 4)

    table = torch.zeros((pages, page_size * 68), device=kv.device, dtype=torch.uint8)
    # Each page stores all E2M1 payload rows first, then all E8M0 scale rows.
    table[:, :page_size * 64] = payload.reshape(pages, page_size * 64)
    table[:, page_size * 64:] = (exps.reshape(-1, 4) + 127).to(torch.uint8).reshape(pages, page_size * 4)
    return table


def make_case(batch, heads, length, page_size, seed=0, strided=False,
              scale_lo=118, scale_hi=138, realistic=False, block_jitter=1):
    """Random pool contents.

    The default scale window spans 20 E8M0 steps, so a token's four block
    scales routinely differ by far more than any fast path can fold, and both
    dispatches are exercised. `realistic=True` instead quantizes random BF16
    KV, giving the block-scale distribution production traffic produces.

    `block_jitter` is the per-32-block exponent spread applied on top of the
    per-token magnitude. It decides the intra-token scale span, which is what
    the FP8 fold's span<=11 condition tests, so it is a parameter rather than a
    hidden assumption: reporting a fast-path hit rate off block_jitter=0 would
    describe a pool whose four blocks are near-identical by construction.
    """
    torch.manual_seed(seed)
    pages = max(1, (batch * max(length, 1) + page_size - 1) // page_size + 2)
    if realistic:
        kv = torch.randn(pages * page_size, 4, 32, device="cuda", dtype=torch.bfloat16)
        kv *= torch.exp2(torch.randint(-8, 9, (pages * page_size, 1, 1), device="cuda").float())
        if block_jitter:
            kv *= torch.exp2(torch.randint(-block_jitter, block_jitter + 1,
                                           (pages * page_size, 4, 1), device="cuda").float())
        table = quantize_kv_to_table(kv.reshape(-1, 128), page_size, pages)
    else:
        table = torch.empty((pages, page_size * 68), device="cuda", dtype=torch.uint8)
        table[:, :page_size * 64].random_(0, 256)
        table[:, page_size * 64:].random_(scale_lo, scale_hi)
    q = torch.randn(batch, heads, 128, device="cuda", dtype=torch.bfloat16)
    weights = torch.randn(batch, heads, device="cuda", dtype=torch.bfloat16)
    if strided:
        qbuf = torch.empty(batch, heads, 144, device="cuda", dtype=torch.bfloat16)
        qbuf[:, :, :128].copy_(q)
        q = qbuf[:, :, :128]
        wbuf = torch.empty(batch, heads + 8, device="cuda", dtype=torch.bfloat16)
        wbuf[:, :heads].copy_(weights)
        weights = wbuf[:, :heads]
        tbuf = torch.empty(pages, page_size * 68 + 16, device="cuda", dtype=torch.uint8)
        tbuf[:, :page_size * 68].copy_(table)
        table = tbuf[:, :page_size * 68]
    slots = torch.randint(0, pages * page_size, (batch, length), device="cuda", dtype=torch.int64)
    lens = torch.full((batch,), length, device="cuda", dtype=torch.int64)
    if batch > 1:
        lens[0] = 0
        lens[1] = max(0, length - 7)
    # Invisible slot garbage must never be dereferenced.
    slots.masked_fill_(torch.arange(length, device="cuda")[None] >= lens[:, None], -0x123456789)
    return q, weights, slots, lens, table, page_size


def reference(q, weights, slots, lens, table, page_size):
    """FP64 evaluation of DeepGEMM's MXFP4 MQA logits semantics.

    Each per-32 block scale multiplies its K operand, the 128 scaled products
    form one dot, and relu, head weighting and the head sum follow with no
    further scale. Evaluating all of that in FP64 leaves the candidate's FP32
    accumulation error as the only difference under test.
    """
    value, _ = reference_with_bound(q, weights, slots, lens, table, page_size)
    return value


def reference_with_bound(q, weights, slots, lens, table, page_size):
    """Reference logits plus the sum of absolute contributions per logit.

    The second tensor bounds how much FP32 accumulation error the candidate is
    allowed: an FP32 dot of n terms carries at most n * 2**-24 * sum|terms|.
    """
    batch, heads, _ = q.shape
    length = slots.shape[1]
    out = torch.full((batch, length), -torch.inf, device=q.device)
    mag = torch.zeros((batch, length), device=q.device)
    lut = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6,
                        -0., -.5, -1, -1.5, -2, -3, -4, -6], device=q.device)
    for b in range(batch):
        n = min(max(0, int(lens[b])), length)
        if not n:
            continue
        slot = slots[b, :n]
        page, offset = slot // page_size, slot % page_size
        payload = table[page[:, None], offset[:, None] * 64 + torch.arange(64, device=q.device)]
        codes = torch.stack((payload & 15, payload >> 4), dim=-1).reshape(n, 128).long()
        exps = table[page[:, None], page_size * 64 + offset[:, None] * 4 + torch.arange(4, device=q.device)]
        scale = torch.exp2(exps.double() - 127).repeat_interleave(32, dim=1)
        # The block scale belongs to the operand, exactly as DeepGEMM's
        # block-scaled UMMA applies it before the accumulator.
        k = lut[codes].double() * scale
        qd, wd = q[b].double(), weights[b].double()
        dot = qd @ k.T
        out[b, :n] = (dot.relu() * wd[:, None]).sum(dim=0).float()
        mag[b, :n] = ((qd.abs() @ k.abs().T) * wd.abs()[:, None]).sum(dim=0).float()
    return out, mag


def assert_logits_close(got, case, heads, max_diff=MAX_DIFF):
    expected, mag = reference_with_bound(*case)
    neg_inf = expected == -torch.inf
    assert torch.equal(got == -torch.inf, neg_inf), "visibility mask mismatch"
    finite = ~neg_inf & torch.isfinite(expected) & torch.isfinite(got)
    diff = calc_diff(got.masked_fill(~finite, 0), expected.masked_fill(~finite, 0))
    assert diff < max_diff, f"calc_diff={diff:.3e} exceeds {max_diff:.1e}"
    tol = 4 * (128 + heads) * UNIT_ROUNDOFF * mag.double()
    err = (got.double() - expected.double()).abs()
    bad = finite & (err > tol)
    if bad.any():
        idx = bad.nonzero()[0].tolist()
        raise AssertionError(
            f"{int(bad.sum())} logits exceed the FP32 accumulation bound; "
            f"first at {idx}: got {float(got[tuple(idx)])!r} "
            f"expected {float(expected[tuple(idx)])!r} tol {float(tol[tuple(idx)]):.3e}")
    # -inf and NaN positions must agree exactly.
    assert torch.equal(torch.isnan(got), torch.isnan(expected)), "NaN pattern mismatch"
    # A bias too small for either bound above still has to show up here.
    scored = finite & (expected != 0)
    if int(scored.sum()) >= BULK_MIN_SAMPLES:
        rel = (err[scored] / expected.double().abs()[scored]).float()
        bulk = float(torch.quantile(rel, BULK_QUANTILE))
        assert bulk <= BULK_TOL, (
            f"{BULK_QUANTILE:.0%} of logits carry more than {bulk:.3e} relative "
            f"error (limit {BULK_TOL:.1e}); the accumulation is biased rather "
            "than merely rounding")


@pytest.fixture(scope="module")
def op():
    if not torch.cuda.is_available():
        pytest.skip("GPU required")
    arch = _device_arch()
    if not any(arch.startswith(prefix) for prefix in FP4_ARCHES):
        pytest.skip(
            "FP4 MMAC requires gfx928/gfx936/gfx938; "
            f"got {arch or 'unknown'}"
        )
    return get_operator()


@pytest.mark.parametrize("heads", [8, 16, 32, 64])
@pytest.mark.parametrize("page_size,length", [(1, 1), (3, 31), (4, 17), (8, 63), (16, 65), (64, 129), (128, 257)])
def test_parity(op, heads, page_size, length):
    case = make_case(3, heads, length, page_size, seed=heads+length, strided=True)
    assert_logits_close(op(*case), case, heads)


@pytest.mark.parametrize("heads", [8, 16, 32, 64])
@pytest.mark.parametrize("page_size,length", [(3, 31), (64, 257), (128, 4097)])
def test_fp4_query_parity(op, heads, page_size, length):
    case = list(make_case(3, heads, length, page_size, seed=heads+length, strided=True))
    case[0].copy_(fake_quant_fp4(case[0]))
    assert_logits_close(op(*case), case, heads)


@pytest.mark.parametrize("heads", [8, 32])
@pytest.mark.parametrize("length", [257, 49152])
def test_realistic_mxfp4_pool(op, heads, length):
    """True MXFP4 KV, so block-scale spans follow the data, not a test knob."""
    case = list(make_case(1, heads, length, 64, seed=heads + length, realistic=True))
    case[0].copy_(fake_quant_fp4(case[0]))
    assert_logits_close(op(*case), case, heads)


@pytest.mark.parametrize("scale_lo,scale_hi", [(110, 146), (120, 133), (100, 156)])
def test_wide_block_scale(op, scale_lo, scale_hi):
    """Block scales far wider than any fast path can fold into its operand.

    The window stops at 56 E8M0 steps because the shared FP32 accumulator,
    here and in DeepGEMM, overflows once a single token's K values span the
    FP32 range; that is a property of the format, not of the dispatch.
    """
    case = list(make_case(1, 32, 49152, 64, seed=91, scale_lo=scale_lo, scale_hi=scale_hi))
    case[0].copy_(fake_quant_fp4(case[0]))
    assert_logits_close(op(*case), case, 32)


@pytest.mark.parametrize("span", [0, 1, 5, 6, 10, 11, 12])
def test_block_scale_span_boundary(op, span):
    """The FP8 fold is exact up to a span of 11 and must hand over beyond it.

    The pivot the per-block deltas are measured from is free within
    [max_scale - 6, min_scale + 5], so the window is 11 wide rather than the 5
    a pivot pinned to max_scale allows. 5 and 6 are kept because they were the
    old boundary and must no longer change anything; 11 and 12 are the real
    one. Both sides have to produce the same accuracy, otherwise the dispatch
    itself is observable in the output.
    """
    case = list(make_case(1, 32, 49152, 64, seed=5, scale_lo=127, scale_hi=128))
    case[0].copy_(fake_quant_fp4(case[0]))
    scales = case[4][:, 64 * 64:]
    scales.random_(127 - span, 128)
    scales.reshape(-1, 4)[:, 0] = 127
    assert_logits_close(op(*case), case, 32)


@pytest.mark.parametrize("batch,length", [(0, 8), (2, 0)])
def test_empty(op, batch, length):
    case = make_case(batch, 32, length, 16)
    result = op(*case)
    assert result.shape == (batch, length) and result.dtype == torch.float32


def test_graph_replay(op):
    case = make_case(3, 32, 16385, 64)
    output = torch.empty(3, 16385, device="cuda")
    op(*case, output)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        op(*case, output)
    for seed in (3, 7, 13):
        new = make_case(3, 32, 16385, 64, seed=seed)
        if seed == 7:
            # Replay must transition between the FP8 and BF16 dispatches using
            # the same captured launch and the same addresses.
            new[0].copy_(fake_quant_fp4(new[0]))
        for old_tensor, new_tensor in zip(case[:-1], new[:-1]):
            old_tensor.copy_(new_tensor)
        # Live device visibility changes must reveal previously masked slots.
        case[2][0].copy_(new[2][1])
        case[3][0] = new[3][1]
        graph.replay()
        assert_logits_close(output, case, 32)


@pytest.mark.parametrize("length", [65539, 1048580])
def test_reserved_capacity_graph_replay(op, length):
    # A graph with a large fixed output bound must rewrite the complete tail
    # while live visibility, Q eligibility and the page contents change.
    case = list(make_case(3, 32, 1025, 64, seed=17))
    q, weights, _, lens, table, page = case
    valid_slots = torch.randint(0, table.shape[0]*page, (3, 1025),
                                device="cuda", dtype=torch.int64)
    slots = torch.full((3, length), -0x123456789, device="cuda", dtype=torch.int64)
    slots[:, :1025].copy_(valid_slots)
    case[2] = slots
    q.copy_(fake_quant_fp4(q))
    lens.fill_(1025)
    output = torch.empty(3, length, device="cuda")
    op(*case, output)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        op(*case, output)
    for step, live_lens in enumerate(([1, 257, 513], [1025]*3, [-1, 0, 1025], [513, 65, 1])):
        lens.copy_(torch.tensor(live_lens, device="cuda", dtype=torch.int64))
        slots[:, :1025].copy_(valid_slots)
        slots.masked_fill_(torch.arange(length, device="cuda")[None] >= lens[:, None], -0x123456789)
        weights.neg_()
        q.normal_()
        if step % 2 == 0:
            q.copy_(fake_quant_fp4(q))
        table[:, :page*64].random_(0, 256)
        graph.replay()
        assert_logits_close(output, case, 32)


def test_input_guards(op):
    case = list(make_case(2, 32, 17, 4))
    for index, replacement in ((0, case[0].float()), (1, case[1].float()),
                               (2, case[2].int()), (3, case[3].int()),
                               (4, case[4][:, :-1]), (5, 0)):
        bad = case.copy()
        bad[index] = replacement
        with pytest.raises(RuntimeError):
            op(*bad)
    with pytest.raises(RuntimeError):
        op(*case, torch.empty(2, 17, device="cuda", dtype=torch.bfloat16))


def test_unsupported_shape_errors(op):
    case = list(make_case(1, 8, 1, 64))
    bad_dim = torch.empty(1, 8, 64, device="cuda", dtype=torch.bfloat16)
    with pytest.raises(RuntimeError, match="head_dim must be 128"):
        op(bad_dim, *case[1:])
    bad_heads = list(case)
    bad_heads[0] = torch.empty(1, 12, 128, device="cuda", dtype=torch.bfloat16)
    bad_heads[1] = torch.empty(1, 12, device="cuda", dtype=torch.bfloat16)
    with pytest.raises(RuntimeError, match="heads must be one of"):
        op(*bad_heads)


def test_unavailable_operator_error(op, monkeypatch):
    from lightop import op as native
    case = make_case(1, 8, 1, 64)
    monkeypatch.delattr(native, "paged_mqa_logits_fp4")
    with pytest.raises(RuntimeError, match="paged_mqa_logits_fp4 unavailable"):
        op(*case)


def test_all_nibbles_and_block_scales(op):
    case = list(make_case(1, 8, 16 * 23, 4))
    q, w, slots, lens, table, page = case
    table.zero_()
    q.zero_()
    w.zero_()
    # A single exact nonzero product per head probes each scale block, both
    # signs and every nibble independently of cancellation in a GEMM.
    for h in range(8):
        q[0, h, (h // 2) * 32] = 1 if h % 2 == 0 else -1
        w[0, h] = 1
    slots[0] = torch.arange(slots.shape[1], device="cuda")
    for j in range(slots.shape[1]):
        p, offset = j // page, j % page
        table[p, offset * 64:offset * 64+64] = (j % 16) * 17
        table[p, page * 64 + offset * 4:page * 64 + offset * 4+4] = 116 + j // 16
    # One product per logit leaves no accumulation error to absorb.
    torch.testing.assert_close(op(*case), reference(*case), rtol=0, atol=0)


@pytest.mark.parametrize("heads", [8, 32])
def test_all_e8m0_scales(op, heads):
    """Every valid E8M0 scale against every nibble, one exact product each.

    E8M0 0xFF is NaN in the OCP MX spec and 0xFD/0xFE push the larger E2M1
    magnitudes past the BF16 range, so the sweep stops at 0xFC, which is also
    where decode_fp4x4's exponent fold stops. Subnormal decodes at the bottom
    of the range are included and must stay exact.
    """
    # This sweep deliberately covers the full scale range and is used to
    # validate the FP4 -> FP8 hand-over.  gfx936/gfx928 do not provide that
    # hardware path; skip the whole sweep instead of turning the limitation
    # into a false numerical regression. Other BF16 FP4 coverage remains active.
    _require_fp4_to_fp8_support()
    scales = 253
    case = list(make_case(1, heads, 16 * scales, 64))
    q, w, slots, lens, table, page = case
    q.zero_()
    q[0, 0, 0] = 1
    w.zero_()
    w[0, 0] = 1
    slots[0].copy_(torch.arange(slots.shape[1], device="cuda"))
    for j in range(slots.shape[1]):
        p, offset = j // page, j % page
        table[p, offset*64:offset*64+64] = (j % 16) * 17
        table[p, page*64+offset*4:page*64+offset*4+4] = j // 16
    if heads == 32:
        # Select the public FP8 dispatch and exercise its per-wave hand-over
        # for the extreme scales, with garbage in every invisible slot.
        capacity = torch.full((1, 49152), -0x123456789, device="cuda", dtype=torch.int64)
        capacity[:, :slots.shape[1]].copy_(slots)
        case[2] = capacity
    # One product per logit leaves no accumulation error to absorb.
    torch.testing.assert_close(op(*case), reference(*case), rtol=0, atol=0)


def test_out_of_range_scales_saturate_to_zero(op):
    """Records what happens past the representable range, rather than hiding it.

    E8M0 0xFD..0xFF scaled by the larger E2M1 magnitudes exceeds the BF16
    range, so decode_fp4 yields an infinity. The gfx938 BF16 MMAC returns zero
    for an infinite operand instead of propagating it, so these logits come out
    as zero. That is outside the operator's domain -- a conforming MXFP4
    quantizer never emits these scales -- but it must stay finite and
    deterministic rather than leaking an inf or NaN into the indexer.
    """
    case = list(make_case(1, 8, 16 * 3, 64))
    q, w, slots, lens, table, page = case
    q.zero_()
    q[0, 0, 0] = 1
    w.zero_()
    w[0, 0] = 1
    slots[0].copy_(torch.arange(slots.shape[1], device="cuda"))
    for j in range(slots.shape[1]):
        p, offset = j // page, j % page
        table[p, offset*64:offset*64+64] = (j % 16) * 17
        table[p, page*64+offset*4:page*64+offset*4+4] = 253 + j // 16
    got = op(*case)
    assert torch.isfinite(got).all(), "out-of-range scales must not leak inf or NaN"
    assert (got >= 0).all()
    # Anything the exact product keeps inside BF16 still has to be exact.
    expected = reference(*case)
    in_range = expected.abs() < 2.0 ** 127
    torch.testing.assert_close(got[in_range], expected[in_range], rtol=0, atol=0)


def test_fp32_accumulator_cancellation(op):
    """A catastrophically cancelling dot is bounded, not exact.

    DeepGEMM accumulates the whole K dimension in one FP32 accumulator, so a
    term 2**28 below the running partial sum is legitimately lost. The kernel
    must stay inside the FP32 bound, and must not invent a larger error.
    """
    case = list(make_case(1, 32, 1, 64))
    q, weights, slots, lens, table, page = case
    q.zero_()
    q[0, 0, 0] = 2.**20
    q[0, 0, 1] = 2.**-8
    q[0, 0, 64] = -2.**20
    weights.zero_()
    weights[0, 0] = 1
    table.zero_()
    table[:, page*64:].fill_(127)
    table[0, 0] = 0x22
    table[0, 32] = 0x02
    slots.zero_()
    lens.fill_(1)
    expected, mag = reference_with_bound(*case)
    assert float(expected[0, 0]) == 2.**-8
    got = float(op(*case)[0, 0])
    assert abs(got - 2.**-8) <= 4 * (128 + 32) * UNIT_ROUNDOFF * float(mag[0, 0])
