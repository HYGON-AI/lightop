"""DSV4.1 candidate indexer level one: block top-k over dense logits.

Compared exactly against a torch reference of the documented contract:
block keys are the max over 8 positions, the newest block is always kept, ties
keep the lowest block id, and kept blocks come out ascending as pool slots / 8.
"""

import pytest
import torch

BLOCK = 8


def get_operator():
    from lightop.attention import dsv41_candidate_block_table
    return dsv41_candidate_block_table


def reference(logits, seq_lens, page_table, page_size, topk):
    batch = logits.shape[0]
    per_page = page_size // BLOCK
    phys = torch.zeros(batch, topk, dtype=torch.int32)
    valid = torch.zeros(batch, dtype=torch.int32)
    logits, seq_lens, page_table = logits.cpu(), seq_lens.cpu(), page_table.cpu()
    for b in range(batch):
        length = min(int(seq_lens[b]), logits.shape[1])
        n = (length + BLOCK - 1) // BLOCK
        if n == 0:
            continue
        if n <= topk:
            kept = torch.arange(n)
        else:
            keys = logits[b, :(n - 1) * BLOCK].reshape(n - 1, BLOCK).amax(1)
            keys = torch.cat((keys, torch.tensor([float("inf")])))
            # Stable descending order keeps the lowest block id among ties.
            order = torch.sort(keys, descending=True, stable=True).indices
            kept = order[:topk].sort().values
        blocks = page_table[b, kept // per_page].long() * per_page + kept % per_page
        phys[b, :len(kept)] = blocks.int()
        valid[b] = BLOCK * (len(kept) - 1) + (length - 1) % BLOCK + 1
    return phys, valid


def make_case(batch, width, page_size, seed, ties=False):
    torch.manual_seed(seed)
    stride = (width + 255) // 256 * 256
    logits = torch.randn(batch, stride, device="cuda")[:, :width]
    if ties:
        # Few distinct values, so the radix select must resolve ties by id.
        logits = logits.mul(2).round()
    pages = (width + page_size - 1) // page_size
    pool_pages = batch * pages + 3
    page_table = torch.randperm(pool_pages, device="cuda")[:batch * pages].reshape(batch, pages).int()
    return logits, page_table


def run(op, logits, seq_lens, page_table, page_size, topk):
    batch = logits.shape[0]
    phys = torch.full((batch, topk), -1, device="cuda", dtype=torch.int32)
    valid = torch.full((batch,), -1, device="cuda", dtype=torch.int32)
    op(logits, seq_lens, page_table, page_size, BLOCK, phys, valid)
    return phys, valid


@pytest.fixture(scope="module")
def op():
    if not torch.cuda.is_available():
        pytest.skip("GPU required")
    return get_operator()


@pytest.mark.parametrize("page_size", [64, 8])
@pytest.mark.parametrize("width,topk", [(1024, 2048), (20000, 2048), (131072, 2048), (4096, 64)])
@pytest.mark.parametrize("ties", [False, True])
def test_matches_reference(op, page_size, width, topk, ties):
    logits, page_table = make_case(4, width, page_size, seed=width + topk, ties=ties)
    seq_lens = torch.tensor([0, 1, width // 2 + 3, width], device="cuda", dtype=torch.int32)
    phys, valid = run(op, logits, seq_lens, page_table, page_size, topk)
    ref_phys, ref_valid = reference(logits, seq_lens, page_table, page_size, topk)
    torch.testing.assert_close(valid.cpu(), ref_valid, rtol=0, atol=0)
    torch.testing.assert_close(phys.cpu(), ref_phys, rtol=0, atol=0)


def test_keeps_newest_block(op):
    """The newest block is kept even when its logits are the smallest."""
    width, topk = 4096, 16
    logits, page_table = make_case(1, width, 64, seed=3)
    logits[0, -BLOCK:] = -1e30
    seq_lens = torch.tensor([width - 3], device="cuda", dtype=torch.int32)
    phys, valid = run(op, logits, seq_lens, page_table, 64, topk)
    newest = (width - 3 - 1) // BLOCK
    per_page = 64 // BLOCK
    expected = int(page_table[0, newest // per_page]) * per_page + newest % per_page
    assert int(phys[0, -1]) == expected
    assert int(valid[0]) == BLOCK * (topk - 1) + (width - 3 - 1) % BLOCK + 1


def test_nan_block_never_outranks_newest(op):
    """A block whose logits are all NaN ranks as -inf, below every finite block."""
    width, topk = 4096, 16
    logits, page_table = make_case(1, width, 64, seed=4)
    logits[0, :topk * BLOCK] = float("nan")
    seq_lens = torch.tensor([width], device="cuda", dtype=torch.int32)
    phys, valid = run(op, logits, seq_lens, page_table, 64, topk)
    ref_phys, ref_valid = reference(logits.nan_to_num(nan=float("-inf")), seq_lens, page_table, 64, topk)
    torch.testing.assert_close(valid.cpu(), ref_valid, rtol=0, atol=0)
    torch.testing.assert_close(phys.cpu(), ref_phys, rtol=0, atol=0)


# 131072 positions = 16384 blocks, past the LDS key capacity: the workspace path.
@pytest.mark.parametrize("width", [20000, 131072])
def test_graph_replay(op, width):
    topk = 2048
    logits, page_table = make_case(3, width, 64, seed=5)
    seq_lens = torch.tensor([width, 100, width - 3000], device="cuda", dtype=torch.int32)
    phys = torch.empty(3, topk, device="cuda", dtype=torch.int32)
    valid = torch.empty(3, device="cuda", dtype=torch.int32)
    op(logits, seq_lens, page_table, 64, BLOCK, phys, valid)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        op(logits, seq_lens, page_table, 64, BLOCK, phys, valid)
    for seed in (7, 9):
        new_logits, new_table = make_case(3, width, 64, seed=seed)
        logits.copy_(new_logits)
        page_table.copy_(new_table)
        seq_lens.copy_(torch.tensor([seed * 1000, width, 9], device="cuda", dtype=torch.int32))
        graph.replay()
        ref_phys, ref_valid = reference(logits, seq_lens, page_table, 64, topk)
        torch.testing.assert_close(valid.cpu(), ref_valid, rtol=0, atol=0)
        torch.testing.assert_close(phys.cpu(), ref_phys, rtol=0, atol=0)


def test_input_guards(op):
    logits, page_table = make_case(2, 1024, 64, seed=1)
    seq_lens = torch.full((2,), 1024, device="cuda", dtype=torch.int32)
    phys = torch.empty(2, 64, device="cuda", dtype=torch.int32)
    valid = torch.empty(2, device="cuda", dtype=torch.int32)
    with pytest.raises(RuntimeError, match="block_size must be 8"):
        op(logits, seq_lens, page_table, 64, 16, phys, valid)
    with pytest.raises(RuntimeError, match="page_size"):
        op(logits, seq_lens, page_table, 12, BLOCK, phys, valid)
    with pytest.raises(RuntimeError, match="cover"):
        op(logits, seq_lens, page_table[:, :1], 64, BLOCK, phys, valid)
    with pytest.raises(RuntimeError, match="seq_lens"):
        op(logits, seq_lens.long(), page_table, 64, BLOCK, phys, valid)
