import torch
from typing import Optional, Tuple
from . import op  


def rms_rotary_embedding_fuse(
    positions: torch.Tensor,  # [batch_size, seq_len] or [num_tokens]
    query: torch.Tensor,      # [batch_size, seq_len, num_heads * head_size] or [num_tokens, num_heads * head_size]
    key: Optional[torch.Tensor],  # same shape as query or None
    head_size: int,
    cos_sin_cache: torch.Tensor,  # [max_position, rot_dim]
    is_neox: bool,
    weight_q: torch.Tensor,       # [hidden_size] 
    weight_k: torch.Tensor,       # [hidden_size]
    residual_q: Optional[torch.Tensor],
    residual_k: Optional[torch.Tensor],
    epsilon: float = 1e-5
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Python wrapper for the fused RMSNorm + Rotary Embedding kernel.
    This function fuses RMS normalization on Q/K and then applies RoPE.

    Args:
        positions: Tensor of token positions (int64)
        query: Query tensor to be normalized and rotated (in-place)
        key: Optional key tensor to be normalized and rotated
        head_size: per-head hidden dimension
        cos_sin_cache: Precomputed cosine/sine table
        is_neox: Whether to use GPT-NeoX style rotation
        weight_q: RMSNorm gamma for query
        weight_k: RMSNorm gamma for key
        epsilon: Small constant for RMSNorm stability

    # Returns:
        (query, key): Both tensors after fused RMSNorm + RoPE.
    """
    # # 输出张量（in-place修改的 buffer 版本）
    # q_out = query.clone()
    # k_out = key.clone() if key is not None else None

    # 调用 C++ 融合算子
    op.rms_rotary_embedding_fuse(
        positions,
        query,
        key,
        head_size,
        cos_sin_cache,
        is_neox,
        weight_q,
        weight_k,
        residual_q,
        residual_k,
        epsilon
    )
    
    return query, key

def rms_rotary_embedding_fuse_with_kv_store(
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    v: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    page_size: int,
    k_buffer: torch.Tensor,
    v_buffer: torch.Tensor,
    kv_cache_loc: torch.Tensor,
    is_neox: bool,
    weight_q: torch.Tensor,
    weight_k: torch.Tensor,
    residual_q: Optional[torch.Tensor],
    residual_k: Optional[torch.Tensor],
    epsilon: float = 1e-5
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:

    # print("rms_rotary_embedding_fuse_with_kv_store")
    # print("positions", positions.shape)
    # print("query", query.shape)
    # print("key", key.shape)
    # print("v", v.shape)
    # print("cos_sin_cache", cos_sin_cache.shape)
    # print("page_size", page_size)
    # print("k_buffer", k_buffer.shape)
    # print("v_buffer", v_buffer.shape)
    # print("kv_cache_loc", kv_cache_loc.shape)
    # print("is_neox", is_neox)
    # print("interleave", not(is_neox))
    # print("weight_q", weight_q.shape)
    # print("weight_k", weight_k.shape)
    # print("residual_q", residual_q.shape if residual_q is not None else None)
    # print("residual_k", residual_k.shape if residual_k is not None else None)
    # print("epsilon", epsilon)
    op.rms_rotary_embedding_fuse_with_kv_store(
        positions,
        query,
        key,
        v,
        cos_sin_cache,
        page_size,
        k_buffer,
        v_buffer,
        kv_cache_loc,
        not(is_neox),
        weight_q,
        weight_k,
        residual_q,
        residual_k,
        epsilon
    )
    return query, key