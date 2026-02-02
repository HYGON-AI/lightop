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
    head_dim: int,
    page_size: int,
    k_buffer: torch.Tensor,
    v_buffer: torch.Tensor,
    kv_cache_loc: torch.Tensor,
    is_neox: bool,
    weight_q: torch.Tensor,
    weight_k: torch.Tensor,
    output_dtype: torch.dtype,
    residual_q: Optional[torch.Tensor],
    residual_k: Optional[torch.Tensor],
    k_scale: Optional[float],
    v_scale: Optional[float],
    epsilon: float = 1e-5
) -> Tuple[torch.Tensor, torch.Tensor]:
    # 当 v_buffer 为 fp8 (uint8) 时，kernel 会将 q 转为 fp8 写入 q_rope_out，需预先分配
    use_fp8_cache = (v_buffer.dtype == torch.uint8)
    q_rope_out = None
    k_rope_out = None
    v_rope_out = None
    if use_fp8_cache:
        q_rope_out = torch.empty(
            query.shape, dtype=torch.uint8, device=query.device
        ).contiguous()
        k_rope_out = torch.empty(
            key.shape, dtype=torch.uint8, device=key.device
        ).contiguous()
        v_rope_out = torch.empty(
            v.shape, dtype=torch.uint8, device=v.device
        ).contiguous()

    op.rms_rotary_embedding_fuse_with_kv_store(
        positions,
        query,
        key,
        v,
        cos_sin_cache,
        head_dim,
        page_size,
        k_buffer,
        v_buffer,
        kv_cache_loc,
        not(is_neox),
        weight_q,
        weight_k,
        output_dtype,
        residual_q,
        residual_k,
        k_scale,
        v_scale,
        q_rope_out,
        k_rope_out,
        v_rope_out,
        epsilon
    )
    # fp8 时返回 q 的 fp8 结果 (q_rope_out)，否则返回原地修改后的 query
    q_out = q_rope_out if use_fp8_cache else query
    k_out = k_rope_out if use_fp8_cache else key
    v_out = v_rope_out if use_fp8_cache else v
    return q_out, k_out, v_out