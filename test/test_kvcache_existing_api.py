"""Compatibility and correctness checks for LightOp KV-cache exports."""

import pytest
import torch

import lightop
import lightop.kvcache


EXISTING_KVCACHE_APIS = [
    "dcu_alloc_extend_kernel",
    "fused_metadata_kernel_general",
    "reshape_and_cache_cuda",
]


@pytest.mark.parametrize("name", EXISTING_KVCACHE_APIS)
def test_existing_kvcache_api_is_direct_reexport(name):
    assert getattr(lightop.kvcache, name) is getattr(lightop, name)


def test_kvcache_public_api_inventory():
    assert set(lightop.kvcache.__all__) == set(EXISTING_KVCACHE_APIS)


def test_dcu_alloc_extend_kernel_page_mapping():
    device = "cuda"
    pre_lens = torch.tensor([3, 4], dtype=torch.int64, device=device)
    seq_lens = torch.tensor([6, 7], dtype=torch.int64, device=device)
    last_locs = torch.tensor([10, 20], dtype=torch.int64, device=device)
    free_pages = torch.tensor([100, 101], dtype=torch.int64, device=device)
    output = torch.empty(6, dtype=torch.int64, device=device)

    lightop.kvcache.dcu_alloc_extend_kernel(
        pre_lens,
        seq_lens,
        last_locs,
        free_pages,
        output,
        bs=2,
        page_size=4,
    )

    expected = torch.tensor(
        [11, 400, 401, 404, 405, 406],
        dtype=torch.int64,
        device=device,
    )
    assert torch.equal(output, expected)
