import pytest
import torch

def _mla_preprocess_api():
    from lightop import kvcache

    if not torch.cuda.is_available():
        pytest.skip("GPU required")
    if not hasattr(kvcache, "main_kv_loc_to_compact"):
        pytest.fail("LightOp native preprocessing extension is not built")
    return kvcache


def _compact_reference(loc, table, page_size):
    valid = loc >= 0
    safe = loc.clamp_min(0)
    page = torch.div(safe, page_size, rounding_mode="floor").long()
    compact = table.index_select(0, page.reshape(-1)).reshape(page.shape)
    result = compact.to(loc.dtype) * page_size + safe % page_size
    return torch.where(valid & (compact >= 0), result, torch.full_like(result, -1))


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("n", [0, 1, 65, 1032, 40960])
def test_main_kv_compact_locations(dtype, n):
    op = _mla_preprocess_api()
    table = torch.randperm(1024, device="cuda").int()
    table[17] = -1
    loc = torch.randint(0, 65536, (n,), device="cuda", dtype=dtype)
    loc[::7] = -3
    for _ in range(3):
        expected = _compact_reference(loc, table, 64)
        actual = op.main_kv_loc_to_compact(loc, table, 64)
        assert torch.equal(expected, actual)
        table.copy_(torch.roll(table, 1))


@pytest.mark.parametrize("n", [0, 1, 1032, 24576, 40960])
@pytest.mark.parametrize("mirror", [False, True])
def test_mla_dual_store_concat_bytes(n, mirror):
    op = _mla_preprocess_api()
    pair = torch.empty((n, 1, 576), device="cuda", dtype=torch.bfloat16)
    kn, kr = pair[..., :512], pair[..., 512:]
    loc = torch.randperm(n, device="cuda").long() + 64
    table = torch.randperm((n + 255) // 64 + 32, device="cuda").int()
    mapped = _compact_reference(loc, table, 64)
    a = torch.full((n + 256, 1, 656), 73, device="cuda", dtype=torch.uint8)
    b = torch.full((table.numel() * 64, 1, 656), 73, device="cuda", dtype=torch.uint8)
    c, d = torch.empty_like(a), torch.empty_like(b)
    output = torch.empty_like(pair)
    for _ in range(3):
        pair.normal_()
        a.fill_(73); b.fill_(73); c.fill_(73); d.fill_(73)
        op.fused_quantize_and_store_mla_kv_cache(kn, kr, a, loc, "fp8_e4m3", 1e-6)
        if mirror:
            op.fused_quantize_and_store_mla_kv_cache(kn, kr, b, mapped, "fp8_e4m3", 1e-6)
        op.fused_quantize_and_store_mla_kv_cache_dual(
            kn, kr, c, loc, d if mirror else None,
            mapped if mirror else None, output, 1e-6)
        assert torch.equal(a, c)
        if mirror:
            assert torch.equal(b, d)
        assert torch.equal(output.view(torch.uint8), pair.view(torch.uint8))


def test_mla_dual_graph_input_updates():
    op = _mla_preprocess_api()
    n = 1032
    pair = torch.randn((n, 1, 576), device="cuda", dtype=torch.bfloat16)
    loc = torch.randperm(n, device="cuda").long() + 64
    table = torch.randperm(64, device="cuda").int()
    cache = torch.empty((n + 256, 1, 656), device="cuda", dtype=torch.uint8)
    mirror = torch.empty((4096, 1, 656), device="cuda", dtype=torch.uint8)
    output = torch.empty_like(pair)
    for _ in range(3):
        mapped = op.main_kv_loc_to_compact(loc, table, 64)
        op.fused_quantize_and_store_mla_kv_cache_dual(pair[..., :512], pair[..., 512:], cache, loc, mirror, mapped, output, 1e-6)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        mapped_graph = op.main_kv_loc_to_compact(loc, table, 64)
        op.fused_quantize_and_store_mla_kv_cache_dual(pair[..., :512], pair[..., 512:], cache, loc, mirror, mapped_graph, output, 1e-6)
    for _ in range(3):
        pair.normal_(); loc.copy_(torch.roll(loc, 3)); table.copy_(torch.roll(table, 1))
        expected_a = torch.full_like(cache, 73); expected_b = torch.full_like(mirror, 73)
        cache.fill_(73); mirror.fill_(73)
        mapped = _compact_reference(loc, table, 64)
        assert mapped.unique().numel() == n
        op.fused_quantize_and_store_mla_kv_cache(pair[..., :512], pair[..., 512:], expected_a, loc, "fp8_e4m3", 1e-6)
        op.fused_quantize_and_store_mla_kv_cache(pair[..., :512], pair[..., 512:], expected_b, mapped, "fp8_e4m3", 1e-6)
        graph.replay()
        assert torch.equal(cache, expected_a) and torch.equal(mirror, expected_b)
        assert torch.equal(output.view(torch.uint8), pair.view(torch.uint8))
