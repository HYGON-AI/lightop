"""
PCP fused operator for DeepSeek-V4 MLA attention.

Fuses the PCP (Prefill Context Parallelism) prefill path into a single kernel:
  - Q side : per-head RMSNorm (no weight) + GPT-J RoPE → in-place on q
  - KV side: GPT-J RoPE + UE8M0 FP8 quant + paged cache insert → k_cache

Unlike the non-PCP fused kernel, this operator supports **different** token counts
for Q and KV — the typical PCP scenario where Q is a local token shard and KV has
been all-gathered to the full prefill length.
"""

import torch
from . import op
from typing import Optional


# ── Kernel constants (mirrors the CUDA kernel) ─────────────────────────────
HEAD_DIM = 512
ROPE_DIM = 64
NOPE_DIM = HEAD_DIM - ROPE_DIM  # 448


def fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert_pcp(
    q: torch.Tensor,
    kv: torch.Tensor,
    k_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    q_position_ids: torch.Tensor,
    kv_position_ids: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    eps: float = 1e-6,
    cache_block_size: int = 16,
) -> None:
    """
    PCP fused prefill preprocessing for DeepSeek-V4 MLA.

    Q path (local token shard):
        q = RMSNorm(q, per_head, no_weight) + GPT-J RoPE(dims[448:512])
        Result written **in-place** back to ``q``.

    KV path (full-gathered tokens, NO RMSNorm — must be pre-normalized):
        kv → GPT-J RoPE(dims[448:512]) → FP8 quant (NoPE) + bf16 (RoPE)
        → paged cache insert into ``k_cache``.

    Args:
        q:
            Query tensor, shape ``[Nq, H, 512]``, dtype ``bfloat16`` or ``float16``.
            Must be contiguous.  **Modified in-place.**
        kv:
            Key-Value latent tensor, shape ``[Nkv, 512]``, same dtype as ``q``.
            Must be contiguous.  Read-only.  Must already be RMSNorm-ed (by the
            caller, typically ``fused_q_kv_rmsnorm``).
        k_cache:
            Paged KV cache buffer, shape ``[num_blocks, block_stride]``,
            dtype ``uint8``.  **Modified in-place.**
        slot_mapping:
            Slot indices for each KV token, shape ``[Nkv]``, dtype ``int64``
            (or ``int32`` via the ``_int32`` variant).
            Negative slots are skipped (padding).
        q_position_ids:
            RoPE position IDs for Q tokens, shape ``[Nq]``, dtype ``int64``.
        kv_position_ids:
            RoPE position IDs for KV tokens, shape ``[Nkv]``, dtype ``int64``.
            In PCP mode this differs from ``q_position_ids`` because KV is
            all-gathered while Q is a local shard.
        cos_sin_cache:
            Precomputed RoPE cos/sin table, shape ``[max_pos, 64]``,
            dtype ``float32``.  Layout: first 32 columns = cos, last 32 = sin.
        eps:
            Epsilon for RMSNorm (default 1e-6).
        cache_block_size:
            Number of tokens per paged-cache block (must be a power of 2,
            default 16).

    Raises:
        RuntimeError: If any shape, dtype, contiguity, or alignment constraint
            is violated.

    Constraints / Assertions:
        - ``q`` must be 3-D, contiguous CUDA tensor with ``shape[2] == 512``.
        - ``kv`` must be 2-D, contiguous CUDA tensor with ``shape[1] == 512``,
          last-dim contiguous (``stride(1) == 1``), and row stride ≥ 512
          with 16-byte alignment (``stride(0) % 8 == 0``).
        - ``q.dtype == kv.dtype`` (``bfloat16`` or ``float16``).
        - ``k_cache`` must be a CUDA ``uint8`` tensor.
        - ``slot_mapping`` must be CUDA, ``int64``, with ``len == Nkv``.
        - ``q_position_ids`` must be CUDA, ``int64``, with ``len == Nq``.
        - ``kv_position_ids`` must be CUDA, ``int64``, with ``len == Nkv``.
        - ``cos_sin_cache`` must be CUDA, ``float32``, shape ``[max_pos, 64]``.
        - ``eps > 0``.
        - ``cache_block_size`` must be a positive power of 2.
        - ``Nq`` and ``Nkv`` may differ (this is the PCP case).
        - ``1 <= num_heads <= 128``.
        - The caller must ensure ``max(position_ids) < cos_sin_cache.shape[0]``.
        - KV must already be RMSNorm-ed upstream (this kernel applies no KV norm).

    Example:
        >>> # PCP mode: 2048 local Q tokens, 16384 full KV tokens
        >>> fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert_pcp(
        ...     q=q_local,                        # [2048, 128, 512] bf16
        ...     kv=kv_full,                       # [16384, 512] bf16
        ...     k_cache=swa_kv_cache_2d,
        ...     slot_mapping=slot_mapping,        # [16384] int64
        ...     q_position_ids=q_pos,             # [2048] int64
        ...     kv_position_ids=kv_pos,           # [16384] int64
        ...     cos_sin_cache=rotary_emb.cos_sin_cache,
        ...     eps=1e-6,
        ...     cache_block_size=16,
        ... )
    """
    # ── Python-level assertions (friendly error messages) ──────────────────
    _validate_pcp_inputs(
        q, kv, k_cache, slot_mapping,
        q_position_ids, kv_position_ids, cos_sin_cache,
        eps, cache_block_size,
    )

    # ── Dispatch to C++ kernel ─────────────────────────────────────────────
    op.fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert_pcp(
        q,
        kv,
        k_cache,
        slot_mapping,
        q_position_ids,
        kv_position_ids,
        cos_sin_cache,
        eps,
        cache_block_size,
    )


def _validate_pcp_inputs(
    q: torch.Tensor,
    kv: torch.Tensor,
    k_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    q_position_ids: torch.Tensor,
    kv_position_ids: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    eps: float,
    cache_block_size: int,
) -> None:
    """Validate all inputs before dispatching to the CUDA kernel.

    These checks run on the Python side so users get clear, actionable
    error messages.  The C++ kernel also contains TORCH_CHECK guards, but
    those produce less readable diagnostics.
    """

    # ── Device ─────────────────────────────────────────────────────────────
    if not q.is_cuda:
        raise RuntimeError(f"q must be a CUDA tensor, got device={q.device}")
    if not kv.is_cuda:
        raise RuntimeError(f"kv must be a CUDA tensor, got device={kv.device}")
    if not k_cache.is_cuda:
        raise RuntimeError(f"k_cache must be a CUDA tensor, got device={k_cache.device}")
    if not slot_mapping.is_cuda:
        raise RuntimeError(f"slot_mapping must be a CUDA tensor, got device={slot_mapping.device}")
    if not q_position_ids.is_cuda:
        raise RuntimeError(f"q_position_ids must be a CUDA tensor, got device={q_position_ids.device}")
    if not kv_position_ids.is_cuda:
        raise RuntimeError(f"kv_position_ids must be a CUDA tensor, got device={kv_position_ids.device}")
    if not cos_sin_cache.is_cuda:
        raise RuntimeError(f"cos_sin_cache must be a CUDA tensor, got device={cos_sin_cache.device}")

    # ── Contiguity ─────────────────────────────────────────────────────────
    if not q.is_contiguous():
        raise RuntimeError(f"q must be contiguous, got strides {q.stride()}")
    if not kv.is_contiguous():
        raise RuntimeError(f"kv must be contiguous, got strides {kv.stride()}")

    # ── Dtype ──────────────────────────────────────────────────────────────
    if q.dtype != kv.dtype:
        raise RuntimeError(
            f"q and kv must have the same dtype, got q={q.dtype}, kv={kv.dtype}"
        )
    if q.dtype not in (torch.bfloat16, torch.float16):
        raise RuntimeError(
            f"q/kv must be bfloat16 or float16, got {q.dtype}"
        )
    if k_cache.dtype != torch.uint8:
        raise RuntimeError(f"k_cache must be uint8, got {k_cache.dtype}")
    if cos_sin_cache.dtype != torch.float32:
        raise RuntimeError(
            f"cos_sin_cache must be float32, got {cos_sin_cache.dtype}"
        )
    if q_position_ids.dtype != torch.int64:
        raise RuntimeError(
            f"q_position_ids must be int64, got {q_position_ids.dtype}"
        )
    if kv_position_ids.dtype != torch.int64:
        raise RuntimeError(
            f"kv_position_ids must be int64, got {kv_position_ids.dtype}"
        )

    # ── Shape ──────────────────────────────────────────────────────────────
    if q.dim() != 3:
        raise RuntimeError(f"q must be 3-D [Nq, H, 512], got shape {tuple(q.shape)}")
    if q.shape[2] != HEAD_DIM:
        raise RuntimeError(
            f"q last dim must be {HEAD_DIM}, got {q.shape[2]}"
        )

    nq = q.shape[0]
    nh = q.shape[1]

    if kv.dim() != 2:
        raise RuntimeError(f"kv must be 2-D [Nkv, 512], got shape {tuple(kv.shape)}")
    if kv.shape[1] != HEAD_DIM:
        raise RuntimeError(
            f"kv last dim must be {HEAD_DIM}, got {kv.shape[1]}"
        )

    nkv = kv.shape[0]

    if nh < 1 or nh > 128:
        raise RuntimeError(
            f"num_heads must be in [1, 128], got {nh}"
        )

    if slot_mapping.numel() != nkv:
        raise RuntimeError(
            f"slot_mapping length must equal Nkv, "
            f"got {slot_mapping.numel()} vs {nkv}"
        )
    if q_position_ids.numel() != nq:
        raise RuntimeError(
            f"q_position_ids length must equal Nq, "
            f"got {q_position_ids.numel()} vs {nq}"
        )
    if kv_position_ids.numel() != nkv:
        raise RuntimeError(
            f"kv_position_ids length must equal Nkv, "
            f"got {kv_position_ids.numel()} vs {nkv}"
        )

    if cos_sin_cache.dim() != 2 or cos_sin_cache.shape[1] != ROPE_DIM:
        raise RuntimeError(
            f"cos_sin_cache must be 2-D [max_pos, {ROPE_DIM}], "
            f"got shape {tuple(cos_sin_cache.shape)}"
        )

    # ── KV layout ──────────────────────────────────────────────────────────
    # Last dim must be contiguous (stride 1) for vectorized uint4 loads.
    if kv.stride(1) != 1:
        raise RuntimeError(
            f"kv last dim must be contiguous (stride=1), got stride={kv.stride(1)}"
        )
    # Row stride must cover at least 512 elements and be 16-byte aligned.
    if nkv > 1:
        if kv.stride(0) < HEAD_DIM:
            raise RuntimeError(
                f"kv row stride must be >= {HEAD_DIM}, got {kv.stride(0)}"
            )
        if kv.stride(0) % 8 != 0:
            raise RuntimeError(
                f"kv row stride must be 16-byte aligned (%%8==0), got {kv.stride(0)}"
            )

    # ── Scalar params ──────────────────────────────────────────────────────
    if eps <= 0.0:
        raise RuntimeError(f"eps must be positive, got {eps}")
    if cache_block_size <= 0 or (cache_block_size & (cache_block_size - 1)) != 0:
        raise RuntimeError(
            f"cache_block_size must be a positive power of 2, got {cache_block_size}"
        )

    # ── Position range (warning only — caller must guarantee) ──────────────
    # We cannot check position_ids.max() < cos_sin_cache.shape[0] here without
    # a device sync.  The kernel will read out-of-bounds if this is violated.
