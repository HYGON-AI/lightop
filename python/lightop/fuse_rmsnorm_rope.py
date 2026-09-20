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
    weight_q: Optional[torch.Tensor],
    weight_k: Optional[torch.Tensor],
    output_dtype: torch.dtype,
    residual_q: Optional[torch.Tensor],
    residual_k: Optional[torch.Tensor],
    k_scale: Optional[float],
    v_scale: Optional[float],
    epsilon: float = 1e-5
) -> Tuple[torch.Tensor, torch.Tensor]:
    dims = query.dim()
    heads = query.shape[dims - 1] // head_dim if dims == 2 else query.shape[dims - 2]
    # q goes straight to attention and never lands in the KV cache, so it must
    # honour output_dtype instead of inheriting the cache dtype. Tying it to
    # v_buffer.dtype silently emitted fp8 bytes that callers expecting the model
    # dtype (e.g. gfx936, where is_nmz_fp8() is False) reinterpreted as bf16.
    q_out_dtype = output_dtype if output_dtype is not None else v_buffer.dtype
    q_rope_out = torch.empty((query.shape[0], heads, head_dim), dtype=q_out_dtype, device=query.device).contiguous()
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

        # Same rule as the fused path: q follows output_dtype, k/v the cache.
        q_rope_out = query.to(q_out_dtype)
        k_rope_out = key.to(v_buffer.dtype)
        v_rope_out = v.to(v_buffer.dtype)

        loc_id = kv_cache_loc // page_size
        loc_offset = kv_cache_loc % page_size
        if key.dim() == 2:
            kv = k_rope_out.view(k_rope_out.size(0), k_buffer.shape[1], -1)
            vc = v_rope_out.view(v_rope_out.size(0), v_buffer.shape[1], -1)
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
    return q_rope_out, k_rope_out, v_rope_out

def gammarms_rotary_embedding_fuse_with_kv_store(
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
    weight_q: Optional[torch.Tensor],
    weight_k: Optional[torch.Tensor],
    output_dtype: torch.dtype,
    residual_q: Optional[torch.Tensor],
    residual_k: Optional[torch.Tensor],
    k_scale: Optional[float],
    v_scale: Optional[float],
    epsilon: float = 1e-5,
    mrope_section_t: int = 0,
    mrope_section_h: int = 0,
    mrope_section_w: int = 0,
    is_interleaved: bool = False
) -> Tuple[torch.Tensor, torch.Tensor]:
    if positions.dim() == 1:
        positions = torch.stack([positions, positions, positions], dim=0).contiguous()
    elif positions.dim() == 2:
        if positions.shape[0] == 3:
            positions = positions.contiguous()
        elif positions.shape[0] == 1:
            positions = positions.expand(3, -1).contiguous()
        else:
            raise RuntimeError(f"Unsupported positions shape for gammarms_rotary_embedding_fuse_with_kv_store: {positions.shape}")
    else:
        raise RuntimeError(f"Unsupported positions dim for gammarms_rotary_embedding_fuse_with_kv_store: {positions.dim()}")

    query = query.contiguous()
    key = key.contiguous()
    v = v.contiguous()
    kv_cache_loc = kv_cache_loc.contiguous()

    dims = query.dim()
    heads = query.shape[dims - 1] // head_dim if dims == 2 else query.shape[dims - 2]
    q_rope_out = torch.empty((query.shape[0], heads, head_dim), dtype=v_buffer.dtype, device=query.device).contiguous()
    k_rope_out = torch.empty(tuple(key.shape), dtype=v_buffer.dtype, device=key.device).contiguous()
    v_rope_out = torch.empty(tuple(v.shape), dtype=v_buffer.dtype, device=v.device).contiguous()

    rot_dim = cos_sin_cache.shape[1]
    if mrope_section_t == 0 and mrope_section_h == 0 and mrope_section_w == 0:
        mrope_section_t = rot_dim // 2

    use_fp8_e4m3 = v_buffer.dtype != torch.float8_e4m3fn and v_buffer.dtype != torch.float8_e5m2
    if (head_dim > 256 or use_fp8_e4m3):
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
        if key.dim() == 2:
            kv = k_rope_out.view(k_rope_out.size(0), k_buffer.shape[1], -1)
            vc = v_rope_out.view(v_rope_out.size(0), v_buffer.shape[1], -1)
        else:
            kv = k_rope_out
            vc = v_rope_out
        k_buffer[loc_id,:,loc_offset,:] = kv
        v_buffer[loc_id,:,:,loc_offset] = vc
    else:
        op.gammarms_rotary_embedding_fuse_with_kv_store(
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
            not (use_fp8_e4m3),
            mrope_section_t,
            mrope_section_h,
            mrope_section_w,
            is_interleaved
        )
    return q_rope_out, k_rope_out, v_rope_out

def split_qkv_rms_rotary_embedding_fuse_with_kv_store_quant(
    positions: torch.Tensor,
    qkv: torch.Tensor,
    q_size: int,
    kv_size: int,
    cos_sin_cache: torch.Tensor,
    head_dim: int,
    page_size: int,
    k_buffer: torch.Tensor,
    v_buffer: torch.Tensor,
    kv_cache_loc: Optional[torch.Tensor],
    is_neox: bool,
    weight_q: Optional[torch.Tensor],
    weight_k: Optional[torch.Tensor],
    output_dtype: torch.dtype,
    kv_cache_dtype: torch.dtype,
    residual_q: Optional[torch.Tensor],
    residual_k: Optional[torch.Tensor],
    k_scale: Optional[float],
    v_scale: Optional[float],
    epsilon: float = 1e-5
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    query_rope_out = torch.empty((qkv.shape[0], q_size), dtype=qkv.dtype, device=qkv.device)
    key_rope_out = torch.empty((qkv.shape[0], kv_size), dtype=qkv.dtype, device=qkv.device)
    v_rope_out = torch.empty((qkv.shape[0], kv_size), dtype=qkv.dtype, device=qkv.device)

    # use_fp8_cache = (kv_cache_loc != None) and (k_buffer.dtype == torch.float8_e4m3fn or k_buffer.dtype == torch.float8_e5m2)
    # use_fp8_e4m3 = use_fp8_cache and (k_buffer.dtype == torch.float8_e4m3fn)
    use_fp8_cache = (kv_cache_loc != None) and (kv_cache_dtype == torch.float8_e4m3fn or kv_cache_dtype == torch.float8_e5m2)
    use_fp8_e4m3 = use_fp8_cache and (kv_cache_dtype == torch.float8_e4m3fn)
    op.split_qkv_rms_rotary_embedding_fuse_with_kv_store_no_quant(
        positions,
        qkv,
        q_size,
        kv_size,
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
        query_rope_out,
        key_rope_out,
        v_rope_out,
        page_size,
        epsilon,
        use_fp8_e4m3,
        use_fp8_cache
    )
    return query_rope_out, key_rope_out, v_rope_out
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
    # q is returned to the caller rather than stored in the KV cache, so it
    # follows output_dtype; only k/v adopt the cache dtype.
    use_fp8_cache = (v_buffer.dtype == torch.float8_e4m3fn or v_buffer.dtype == torch.float8_e5m2)
    q_out_dtype = output_dtype if output_dtype is not None else v_buffer.dtype

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
    dims = query.dim()
    heads = query.shape[dims - 1] // head_dim if dims == 2 else query.shape[dims - 2]
    q = query.to(q_out_dtype).view(-1, heads, head_dim).contiguous()
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

def gammarms_rotary_embedding_fuse_with_kv_store_fake(
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

    # ====== ✅ Gamma RMSNorm hack ======
    # 用 (1 + gamma) 替代原来的 gamma
    weight_q_gamma = weight_q + 1.0
    weight_k_gamma = weight_k + 1.0

    # ====== 原 fused op ======
    op.rms_rotary_embedding_fuse(
        positions,
        query,
        key,
        head_dim,
        cos_sin_cache,
        is_neox,
        weight_q_gamma,
        weight_k_gamma,
        residual_q,
        residual_k,
        epsilon
    )

    dims = query.dim()
    heads = query.shape[dims - 1] // head_dim if dims == 2 else query.shape[dims - 2]

    q = query.to(v_buffer.dtype).view(-1, heads, head_dim).contiguous()
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

    k_buffer[loc_id, :, loc_offset, :] = kv
    v_buffer[loc_id, :, :, loc_offset] = vc

    return q, k, v

def mimo_v2_split_rope_vscale_kv_store(
    positions: torch.Tensor,
    qkv: torch.Tensor,
    q_size: int,
    k_size: int,
    v_size: int,
    cos_sin_cache: torch.Tensor,
    head_dim: int,
    v_head_dim: int,
    k_buffer: torch.Tensor,
    v_buffer: torch.Tensor,
    kv_cache_loc: Optional[torch.Tensor],
    is_neox: bool,
    value_scale: float,
    kv_cache_dtype: Optional[torch.dtype] = None,
    k_scale: Optional[torch.Tensor] = None,
    v_cache_scale: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split MIMO v2 QKV, apply RoPE to Q/K, scale V, and write MHA KV cache."""
    qkv = qkv.contiguous()
    positions = positions.contiguous()
    cos_sin_cache = cos_sin_cache.contiguous()
    if kv_cache_loc is not None:
        kv_cache_loc = kv_cache_loc.contiguous()

    if kv_cache_dtype is None:
        kv_cache_dtype = k_buffer.dtype

    fp8_dtypes = (torch.float8_e4m3fn, torch.float8_e5m2)
    use_fp8_cache = (
        kv_cache_loc is not None
        and (kv_cache_dtype in fp8_dtypes or k_buffer.dtype in fp8_dtypes)
    )
    use_fp8_e4m3 = kv_cache_dtype == torch.float8_e4m3fn

    if kv_cache_loc is not None and not use_fp8_cache:
        if k_buffer.dtype != qkv.dtype or v_buffer.dtype != qkv.dtype:
            raise RuntimeError(
                "mimo_v2_split_rope_vscale_kv_store expects bf16/fp16 cache "
                "buffers to have the same dtype as qkv unless kv_cache_dtype is fp8"
            )

    q_out, k_out, v_out = qkv.split([q_size, k_size, v_size], dim=-1)
    op.mimo_v2_split_rope_vscale_kv_store(
        positions,
        qkv,
        q_size,
        k_size,
        v_size,
        head_dim,
        v_head_dim,
        cos_sin_cache,
        is_neox,
        k_buffer,
        v_buffer,
        kv_cache_loc,
        q_out,
        k_out,
        v_out,
        float(value_scale),
        k_scale,
        v_cache_scale,
        use_fp8_e4m3,
        use_fp8_cache,
    )
    return q_out, k_out, v_out