"""Compatibility and correctness checks for LightOp KV-cache exports."""

import pytest
import torch

import lightop
from lightop import op as lightop_native
import lightop.kvcache


EXISTING_KVCACHE_APIS = [
    "hcu_alloc_extend_kernel",
    "fused_metadata_kernel_general",
    "reshape_and_cache_cuda",
]

NATIVE_KVCACHE_APIS = [
    "fuse_act_quant_and_store_index_k_cache",
    "fuse_qk_quant_and_store_index_k_cache",
    "fused_quantize_and_store_mla_kv_cache",
    "quantize_nope_fp8_rope_bf16_pack_store",
]


@pytest.mark.parametrize("name", EXISTING_KVCACHE_APIS)
def test_existing_kvcache_api_is_direct_reexport(name):
    assert getattr(lightop.kvcache, name) is getattr(lightop, name)


@pytest.mark.parametrize("name", NATIVE_KVCACHE_APIS)
def test_native_kvcache_api_is_direct_binding(name):
    assert getattr(lightop.kvcache, name) is getattr(lightop_native, name)


def test_kvcache_public_api_inventory():
    assert set(lightop.kvcache.__all__) == set(
        EXISTING_KVCACHE_APIS + NATIVE_KVCACHE_APIS
    )


def test_hcu_alloc_extend_kernel_page_mapping():
    device = "cuda"
    pre_lens = torch.tensor([3, 4], dtype=torch.int64, device=device)
    seq_lens = torch.tensor([6, 7], dtype=torch.int64, device=device)
    last_locs = torch.tensor([10, 20], dtype=torch.int64, device=device)
    free_pages = torch.tensor([100, 101], dtype=torch.int64, device=device)
    output = torch.empty(6, dtype=torch.int64, device=device)

    lightop.kvcache.hcu_alloc_extend_kernel(
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
