"""Runtime checks for the remaining categorized attention operators."""

import pytest
import torch

from lightop import gemmopt
from lightop import op as lightop_native
from lightop.attention import (
    fuse_layernorm_rotary_embedding,
    fused_deepseek_v4_qnorm_rope_kvnorm_rope,
    fused_deepseek_v4_qnorm_rope_kvnorm_rope_quant_insert_int32,
    get_paged_mqa_logits_metadata,
    paged_mqa_logits,
)


def _require_cuda() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA/ROCm device is required")


def _cos_sin_cache(max_position: int = 8) -> torch.Tensor:
    cache = torch.zeros(
        (max_position, 64), device="cuda", dtype=torch.float32
    )
    cache[:, :32] = 1.0
    return cache


def test_remaining_attention_bindings_are_direct_reexports() -> None:
    assert get_paged_mqa_logits_metadata is (
        gemmopt.get_paged_mqa_logits_metadata
    )
    assert paged_mqa_logits is gemmopt.paged_mqa_logits
    assert fuse_layernorm_rotary_embedding is (
        lightop_native.fuse_layernorm_rotary_embedding
    )
    assert fused_deepseek_v4_qnorm_rope_kvnorm_rope is (
        lightop_native.fused_deepseek_v4_qnorm_rope_kvnorm_rope
    )
    assert fused_deepseek_v4_qnorm_rope_kvnorm_rope_quant_insert_int32 is (
        lightop_native
        .fused_deepseek_v4_qnorm_rope_kvnorm_rope_quant_insert_int32
    )


def test_paged_mqa_metadata_and_logits_runtime() -> None:
    _require_cuda()
    context_lens = torch.tensor([32], device="cuda", dtype=torch.int32)
    metadata = get_paged_mqa_logits_metadata(
        context_lens,
        64,
        torch.cuda.get_device_properties(0).multi_processor_count,
    )
    assert metadata.is_cuda

    q = torch.randn(
        (1, 1, 32, 128), device="cuda", dtype=torch.bfloat16
    )
    kv_cache = torch.randn(
        (1, 64, 1, 128), device="cuda", dtype=torch.bfloat16
    )
    weights = torch.randn((1, 32), device="cuda", dtype=torch.float32)
    block_table = torch.zeros((1, 1), device="cuda", dtype=torch.int32)
    logits = paged_mqa_logits(
        q,
        kv_cache,
        weights,
        context_lens,
        block_table,
        None,
        64,
    )
    torch.cuda.synchronize()
    assert logits.shape == (1, 64)
    assert torch.isfinite(logits[:, :32]).all()
    assert torch.isneginf(logits[:, 32:]).all()


def test_fuse_layernorm_rotary_embedding_matches_native() -> None:
    _require_cuda()
    torch.manual_seed(11)
    positions = torch.tensor([0, 1], device="cuda", dtype=torch.int64)
    cache = _cos_sin_cache()
    query = torch.randn((2, 2, 64), device="cuda", dtype=torch.float16)
    key = torch.randn((2, 1, 64), device="cuda", dtype=torch.float16)
    query_native = query.clone()
    key_native = key.clone()

    args = (positions, query, key, 64, cache, False)
    fuse_layernorm_rotary_embedding(
        *args, None, None, None, None, None, None, 1e-5
    )
    lightop_native.fuse_layernorm_rotary_embedding(
        positions,
        query_native,
        key_native,
        64,
        cache,
        False,
        None,
        None,
        None,
        None,
        None,
        None,
        1e-5,
    )
    torch.cuda.synchronize()
    torch.testing.assert_close(query, query_native, rtol=0, atol=0)
    torch.testing.assert_close(key, key_native, rtol=0, atol=0)


def test_fused_deepseek_v4_qnorm_rope_kvnorm_rope_matches_native() -> None:
    _require_cuda()
    torch.manual_seed(12)
    positions = torch.tensor([0, 1], device="cuda", dtype=torch.int64)
    cache = _cos_sin_cache()
    weight = torch.randn((512,), device="cuda", dtype=torch.float32)
    query = torch.randn((2, 2, 512), device="cuda", dtype=torch.bfloat16)
    kv = torch.randn((2, 512), device="cuda", dtype=torch.bfloat16)
    query_native = query.clone()
    kv_native = kv.clone()

    fused_deepseek_v4_qnorm_rope_kvnorm_rope(
        query, kv, weight, positions, cache, 1e-5
    )
    lightop_native.fused_deepseek_v4_qnorm_rope_kvnorm_rope(
        query_native, kv_native, weight, positions, cache, 1e-5
    )
    torch.cuda.synchronize()
    torch.testing.assert_close(query, query_native, rtol=0, atol=0)
    torch.testing.assert_close(kv, kv_native, rtol=0, atol=0)


def test_deepseek_quant_insert_int32_matches_native() -> None:
    _require_cuda()
    torch.manual_seed(13)
    block_size = 4
    positions = torch.tensor([0, 1], device="cuda", dtype=torch.int64)
    slots = torch.tensor([0, 1], device="cuda", dtype=torch.int32)
    cache = _cos_sin_cache()
    weight = torch.randn((512,), device="cuda", dtype=torch.float32)
    query = torch.randn((2, 2, 512), device="cuda", dtype=torch.bfloat16)
    kv = torch.randn((2, 512), device="cuda", dtype=torch.bfloat16)
    query_native = query.clone()
    packed_cache = torch.zeros(
        (1, block_size * 584), device="cuda", dtype=torch.uint8
    )
    packed_cache_native = packed_cache.clone()

    fused_deepseek_v4_qnorm_rope_kvnorm_rope_quant_insert_int32(
        query,
        kv,
        weight,
        packed_cache,
        slots,
        positions,
        cache,
        1e-5,
        block_size,
    )
    (
        lightop_native
        .fused_deepseek_v4_qnorm_rope_kvnorm_rope_quant_insert_int32(
            query_native,
            kv,
            weight,
            packed_cache_native,
            slots,
            positions,
            cache,
            1e-5,
            block_size,
        )
    )
    torch.cuda.synchronize()
    torch.testing.assert_close(query, query_native, rtol=0, atol=0)
    torch.testing.assert_close(
        packed_cache, packed_cache_native, rtol=0, atol=0
    )
