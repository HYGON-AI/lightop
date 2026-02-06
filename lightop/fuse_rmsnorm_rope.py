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
    # 与 query/key 相同 layout，这样 kernel 用 query 的 stride 写 q_rope_out 时布局一致（支持 query 非连续）
    q_rope_out = torch.empty_strided(
        tuple(query.shape), tuple(query.stride()),
        dtype=v_buffer.dtype, device=query.device
    )
    k_rope_out = torch.empty_strided(
        tuple(key.shape), tuple(key.stride()),
        dtype=v_buffer.dtype, device=key.device
    )
    v_rope_out = torch.empty_strided(
        tuple(v.shape), tuple(v.stride()),
        dtype=v_buffer.dtype, device=v.device
    )
    use_fp8_e4m3 = v_buffer.dtype != torch.float8_e4m3fn and v_buffer.dtype != torch.float8_e5m2
    if (head_dim > 128 or use_fp8_e4m3):
        op.rms_rotary_embedding_fuse(
            positions,
            query,
            key,
            head_dim,
            cos_sin_cache,
            is_neox,
            weight_q,
            weight_k,
            residual_q,
            residual_k,
            epsilon
        )
        
        q_rope_out = query.to(v_buffer.dtype)
        k_rope_out = key.to(v_buffer.dtype)
        v_rope_out = v.to(v_buffer.dtype)
        
        loc_id = kv_cache_loc // page_size
        loc_offset = kv_cache_loc % page_size
        if k.dim() == 2:
            kv = q_rope_out.view(k_rope_out.size(0), k_buffer.shape[1], -1)
            vc = k_rope_out.view(v_rope_out.size(0), v_buffer.shape[1], -1)
        else:
            kv = k_rope_out
            vc = v_rope_out
        k_buffer[loc_id,:,loc_offset,:] = kv
        v_buffer[loc_id,:,:,loc_offset] = vc
    else:
        op.rms_rotary_embedding_fuse_with_kv_store(
            positions,
            query,
            key,
            v,
            head_dim,
            cos_sin_cache,
            is_neox,
            weight_q,
            weight_k,
            residual_q,
            residual_k,
            k_buffer,
            v_buffer,
            kv_cache_loc,
            q_rope_out,
            k_rope_out,
            v_rope_out,
            page_size,
            epsilon,
            not (use_fp8_e4m3)
        )
        # v_rope_out = v.to(torch.float).to(v_buffer.dtype)
        # if v.dim() == 2:
        #     vc = v_rope_out.view(v_rope_out.size(0), v_buffer.shape[1], -1)
        # else:
        #     vc = v_rope_out
        # loc_id = kv_cache_loc // page_size
        # loc_offset = kv_cache_loc % page_size
        # v_buffer[loc_id,:,:,loc_offset] = vc
    return q_rope_out, k_rope_out, v_rope_out

def rms_rotary_embedding_fuse_with_kv_store_fake(
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
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # 当 v_buffer 为 fp8 (uint8) 时，kernel 会将 q 转为 fp8 写入 q_rope_out，需预先分配
    use_fp8_cache = (v_buffer.dtype == torch.float8_e4m3fn or v_buffer.dtype == torch.float8_e5m2)

    op.rms_rotary_embedding_fuse(
        positions,
        query,
        key,
        head_dim,
        cos_sin_cache,
        is_neox,
        weight_q,
        weight_k,
        residual_q,
        residual_k,
        epsilon
    )
    
    q = query.to(v_buffer.dtype)
    k = key.to(v_buffer.dtype)
    v = v.to(v_buffer.dtype)
    
    loc_id = kv_cache_loc // page_size
    loc_offset = kv_cache_loc % page_size
    if k.dim() == 2:
        kv = k.view(k.size(0), k_buffer.shape[1], -1)
        vc = v.view(v.size(0), v_buffer.shape[1], -1)
    else:
        kv = k
        vc = v
    k_buffer[loc_id,:,loc_offset,:] = kv
    v_buffer[loc_id,:,:,loc_offset] = vc

    return q, k, v