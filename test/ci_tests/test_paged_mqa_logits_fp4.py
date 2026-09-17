"""Public FP4 logits parity, input guards and live CUDA/HIP graph replay."""

import pytest
import torch


def get_operator():
    from lightop.attention import paged_mqa_logits_fp4
    return paged_mqa_logits_fp4


def fake_quant_fp4(q):
    """Independent per-32 E2M1 / E8M0 query generation, returned as BF16."""
    x = q.float().reshape(*q.shape[:-1], 4, 32)
    scale = torch.exp2(torch.ceil(torch.log2(x.abs().amax(-1, keepdim=True).clamp_min(6 * 2.**-126) / 6)))
    v = (x / scale).clamp(-6, 6)
    step = torch.where(v.abs() < 2, .5, torch.where(v.abs() < 4, 1., 2.))
    return ((v.abs() / step).round() * step * v.sign() * scale).reshape_as(q).bfloat16()


def make_case(batch, heads, length, page_size, seed=0, strided=False):
    torch.manual_seed(seed)
    pages = max(1, (batch * max(length, 1) + page_size - 1) // page_size + 2)
    table = torch.empty((pages, page_size * 68), device="cuda", dtype=torch.uint8)
    table[:, :page_size * 64].random_(0, 256)
    table[:, page_size * 64:].random_(125, 130)
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


def reference_bf16_round(x):
    """Single ties-to-even BF16 rounding of FP64, including subnormals.

    A direct Torch FP64-to-BF16 cast first rounds to FP32 on this backend.
    Quantize on the BF16 grid in FP64 to avoid that double rounding.
    """
    exponent = torch.frexp(x.abs())[1]
    step = torch.exp2((exponent - 8).double()).clamp_min(2.**-133)
    return ((x / step).round() * step).to(torch.bfloat16)


def reference(q, weights, slots, lens, table, page_size):
    batch, heads, _ = q.shape
    length = slots.shape[1]
    out = torch.full((batch, length), -torch.inf, device=q.device)
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
        scale = torch.exp2(exps.float() - 127).repeat_interleave(32, dim=1)
        k = (lut[codes] * scale).to(torch.bfloat16)
        # Independent FP64 dot and head sum avoid inheriting the candidate's
        # accumulation layout or using a weak cosine-only correctness gate.
        dot = reference_bf16_round(q[b].double() @ k.double().T)
        product = (dot.relu() * weights[b, :, None]).to(torch.bfloat16)
        out[b, :n] = reference_bf16_round(product.double().sum(dim=0)).float()
    return out


@pytest.fixture(scope="module")
def op():
    if not torch.cuda.is_available():
        pytest.skip("GPU required")
    arch = getattr(torch.cuda.get_device_properties(0), "gcnArchName", "")
    if not arch.startswith(("gfx928", "gfx936", "gfx938")):
        pytest.skip("FP4 MMAC requires gfx928/936/938")
    return get_operator()


@pytest.mark.parametrize("heads", [8, 16, 32, 64])
@pytest.mark.parametrize("page_size,length", [(1, 1), (3, 31), (4, 17), (8, 63), (16, 65), (64, 129), (128, 257)])
def test_parity(op, heads, page_size, length):
    case = make_case(3, heads, length, page_size, seed=heads+length, strided=True)
    torch.testing.assert_close(op(*case), reference(*case), rtol=0, atol=0)


@pytest.mark.parametrize("heads", [8, 16, 32, 64])
@pytest.mark.parametrize("page_size,length", [(3, 31), (64, 257), (128, 4097)])
def test_fp4_query_parity(op, heads, page_size, length):
    case = list(make_case(3, heads, length, page_size, seed=heads+length, strided=True))
    case[0].copy_(fake_quant_fp4(case[0]))
    torch.testing.assert_close(op(*case), reference(*case), rtol=0, atol=0)


def test_head_reduction_rounding_regression(op):
    case = list(make_case(8, 32, 65536, 64, seed=8+32+65536))
    case[2].random_(0, case[4].shape[0]*64)
    # This single token hit a BF16 tie after FP32 head reduction, while the
    # exact sum lies below it: expected 126.5, not 127.0.
    case[0] = case[0][6:7]
    case[1] = case[1][6:7]
    case[2] = case[2][6:7, 36993:36994].contiguous()
    case[3] = torch.ones(1, device="cuda", dtype=torch.int64)
    expected = reference(*case)
    assert float(expected[0, 0]) == 126.5
    torch.testing.assert_close(op(*case), expected, rtol=0, atol=0)


def test_bf16_dot_rounding_regression(op):
    case = list(make_case(3, 32, 16385, 64, seed=3))
    case[0] = case[0][2:3]
    case[1] = case[1][2:3]
    case[2] = case[2][2:3, 16352:16353].contiguous()
    case[3] = torch.ones(1, device="cuda", dtype=torch.int64)
    expected = reference(*case)
    assert float(expected[0, 0]) == 2.890625
    torch.testing.assert_close(op(*case), expected, rtol=0, atol=0)


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
            # Replay transitions between the exact FP8 path and arbitrary
            # BF16 fallback using the same captured launch and addresses.
            new[0].copy_(fake_quant_fp4(new[0]))
        for old_tensor, new_tensor in zip(case[:-1], new[:-1]):
            old_tensor.copy_(new_tensor)
        # Live device visibility changes must reveal previously masked slots.
        case[2][0].copy_(new[2][1])
        case[3][0] = new[3][1]
        graph.replay()
        torch.testing.assert_close(output, reference(*case), rtol=0, atol=0)


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
        torch.testing.assert_close(output, reference(*case), rtol=0, atol=0)


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
    torch.testing.assert_close(op(*case), reference(*case), rtol=0, atol=0)


@pytest.mark.parametrize("heads", [8, 32])
def test_all_e8m0_scales(op, heads):
    case = list(make_case(1, heads, 16 * 256, 64))
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
        # Select the public FP8 dispatch and exercise its per-wave fallback
        # for the extreme scales, with garbage in every invisible slot.
        capacity = torch.full((1, 49152), -0x123456789, device="cuda", dtype=torch.int64)
        capacity[:, :slots.shape[1]].copy_(slots)
        case[2] = capacity
    expected = reference(*case)
    torch.testing.assert_close(op(*case), expected, rtol=0, atol=0, equal_nan=True)


@pytest.mark.parametrize("scale_low,scale_high", [(110, 146), (120, 133)])
def test_wide_block_scale_cancellation(op, scale_low, scale_high):
    # Exact FP8 operands alone do not ensure an exact sum of the four scaled
    # partial dots. This input previously produced 50 incorrect logits.
    case = list(make_case(1, 32, 49152, 64, seed=91))
    case[0].copy_(fake_quant_fp4(case[0]))
    case[4][:, 64*64:].random_(scale_low, scale_high)
    torch.testing.assert_close(op(*case), reference(*case), rtol=0, atol=0)


@pytest.mark.parametrize("heads", [8, 32])
def test_bf16_query_cancellation(op, heads):
    case = list(make_case(1, heads, 1, 64))
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
    if heads == 32:
        case[2] = torch.full((1, 49152), -0x123456789, device="cuda", dtype=torch.int64)
        case[2][0, 0] = 0
    expected = reference(*case)
    assert float(expected[0, 0]) == 2.**-8
    torch.testing.assert_close(op(*case), expected, rtol=0, atol=0)


def test_weighted_head_cancellation(op):
    case = list(make_case(1, 32, 49152, 64))
    q, weights, slots, lens, table, page = case
    q.zero_()
    q[0, :, 0] = 1
    weights.zero_()
    weights[0, :3] = torch.tensor([2.**20, 2.**-8, -2.**20], device="cuda", dtype=torch.bfloat16)
    table.zero_()
    table[:, page*64:].fill_(127)
    table[0, 0] = 0x02
    slots.zero_()
    lens.fill_(1)
    expected = reference(*case)
    assert float(expected[0, 0]) == 2.**-8
    torch.testing.assert_close(op(*case), expected, rtol=0, atol=0)
