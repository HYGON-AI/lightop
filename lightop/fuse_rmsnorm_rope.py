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
