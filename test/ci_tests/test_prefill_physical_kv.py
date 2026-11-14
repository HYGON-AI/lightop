import pytest
import torch

from lightop.kvcache import prefill_gather_physical_kv

ENTRY_BYTES = 656
HEAD_DIM = 576
NOPE_DIM = 512
NUM_GROUPS = 4
GROUP_SIZE = 128
FP8_DTYPE = torch.float8_e4m3fn

def _make_packed_cache(
    num_physical_tokens: int,
    entry_bytes: int = ENTRY_BYTES,
    head_dim: int = HEAD_DIM,
) -> torch.Tensor:
    torch.manual_seed(20260717)
    rope_dim = head_dim - NOPE_DIM
    assert rope_dim >= 0
    assert entry_bytes == NOPE_DIM + NUM_GROUPS * 4 + rope_dim * 2

    fp8_values = (
        torch.randn(
            num_physical_tokens,
            NOPE_DIM,
            device="cuda",
            dtype=torch.float32,
        )
        * 3.0
    ).to(FP8_DTYPE)
    scales = (
        0.25
        + torch.rand(
            num_physical_tokens,
            NUM_GROUPS,
            device="cuda",
            dtype=torch.float32,
        )
        * 1.75
    ).contiguous()

    packed = torch.empty(
        (num_physical_tokens, entry_bytes),
        device="cuda",
        dtype=torch.uint8,
    )
    packed[:, :NOPE_DIM].copy_(fp8_values.view(torch.uint8))
    packed[:, NOPE_DIM : NOPE_DIM + NUM_GROUPS * 4].copy_(
        scales.view(torch.uint8)
    )
    if rope_dim > 0:
        rope = torch.randn(
            num_physical_tokens,
            rope_dim,
            device="cuda",
            dtype=torch.bfloat16,
        ).contiguous()
        packed[:, NOPE_DIM + NUM_GROUPS * 4 :].copy_(rope.view(torch.uint8))
    return packed.view(num_physical_tokens, 1, 1, entry_bytes)

def _kv_reference(
    packed_cache: torch.Tensor,
    page_table_flat: torch.Tensor,
) -> torch.Tensor:
    entry_bytes = packed_cache.shape[-1]
    rope_bytes = entry_bytes - NOPE_DIM - NUM_GROUPS * 4
    assert rope_bytes >= 0 and rope_bytes % 2 == 0
    head_dim = NOPE_DIM + rope_bytes // 2

    rows = packed_cache.view(-1, entry_bytes).index_select(
        0, page_table_flat.to(torch.int64)
    )
    fp8_values = rows[:, :NOPE_DIM].contiguous().view(FP8_DTYPE).float()
    scales = (
        rows[:, NOPE_DIM : NOPE_DIM + NUM_GROUPS * 4]
        .contiguous()
        .view(torch.float32)
    )
    nope = (
        fp8_values.view(-1, NUM_GROUPS, GROUP_SIZE) * scales[:, :, None]
    ).reshape(-1, NOPE_DIM).to(torch.bfloat16)

    output = torch.empty(
        (page_table_flat.numel(), 1, head_dim),
        device="cuda",
        dtype=torch.bfloat16,
    )
    output[:, 0, :NOPE_DIM] = nope
    if head_dim > NOPE_DIM:
        rope = rows[:, NOPE_DIM + NUM_GROUPS * 4 :].contiguous().view(
            torch.bfloat16
        )
        output[:, 0, NOPE_DIM:] = rope
    return output

@pytest.mark.parametrize("physical_rows", [1, 65, 257])
def test_physical_kv_sharing_and_graph_updates(physical_rows):
    cache = _make_packed_cache(physical_rows)
    logical = torch.arange(physical_rows * 3, device="cuda", dtype=torch.int32) % physical_rows
    logical[-2:] = torch.tensor([-1, physical_rows + 1], device="cuda", dtype=torch.int32)
    indices = torch.arange(4096, device="cuda", dtype=torch.int32).view(2, 2048)
    indices.remainder_(logical.numel())
    indices[:, -3:] = torch.tensor([-1, logical.numel(), -9], device="cuda", dtype=torch.int32)
    prefill_gather_physical_kv(cache, logical, indices)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output, mapped = prefill_gather_physical_kv(cache, logical, indices)
    for _ in range(3):
        logical.copy_(torch.roll(logical, 1))
        cache.copy_(_make_packed_cache(physical_rows))
        graph.replay()
        reference = _kv_reference(cache, torch.arange(physical_rows, device="cuda", dtype=torch.int32))
        assert torch.equal(output[:physical_rows].view(torch.uint8), reference.view(torch.uint8))
        assert torch.count_nonzero(output[physical_rows]) == 0
        expected = torch.full_like(indices, -1)
        valid = (indices >= 0) & (indices < logical.numel())
        expected[valid] = logical[indices[valid].long()]
        bad_physical = valid & ((expected < 0) | (expected >= physical_rows))
        expected[bad_physical] = physical_rows
        assert torch.equal(mapped, expected)
        assert output.shape == (physical_rows + 1, 1, 576)
        assert output.numel() < logical.numel() * 576 + 576
