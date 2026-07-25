"""Verify the categorized export for the native DeepSeek RoPE operator."""

import torch

import lightop.attention
from lightop import op as lightop_native
from lightop.attention import rotary_embedding_deepseek_fuse


def test_rotary_embedding_deepseek_fuse_categorized_export() -> None:
    assert (
        rotary_embedding_deepseek_fuse
        is lightop_native.rotary_embedding_deepseek_fuse
    )
    assert "rotary_embedding_deepseek_fuse" in lightop.attention.__all__


def test_rotary_embedding_deepseek_fuse_category_matches_native() -> None:
    torch.manual_seed(3)
    positions = torch.tensor([0, 1, 2, 3], device="cuda")
    q = torch.randn(4, 2, 64, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(4, 1, 64, device="cuda", dtype=torch.bfloat16)
    cos_sin_cache = torch.randn(
        16, 64, device="cuda", dtype=torch.bfloat16
    )
    categorized_q = q.clone()
    categorized_k = k.clone()
    native_q = q.clone()
    native_k = k.clone()

    rotary_embedding_deepseek_fuse(
        positions,
        categorized_q,
        categorized_k,
        64,
        cos_sin_cache,
        False,
        None,
    )
    lightop_native.rotary_embedding_deepseek_fuse(
        positions,
        native_q,
        native_k,
        64,
        cos_sin_cache,
        False,
        None,
    )

    torch.testing.assert_close(categorized_q, native_q, rtol=0, atol=0)
    torch.testing.assert_close(categorized_k, native_k, rtol=0, atol=0)
