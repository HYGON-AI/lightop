# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Gather and optionally dequantize paged KV-cache entries."""

from __future__ import annotations

import torch

from . import op


def gather_and_maybe_dequant_cache(
    src_cache: torch.Tensor,
    dst: torch.Tensor,
    block_table: torch.Tensor,
    cu_seq_lens: torch.Tensor,
    token_to_seq: torch.Tensor,
    num_tokens: int,
    kv_cache_dtype: str,
    scale: torch.Tensor,
    seq_starts: torch.Tensor | None = None,
) -> None:
    """Gather paged KV-cache entries and optionally dequantize FP8.

    The argument order and mutation semantics match vLLM's
    ``gather_and_maybe_dequant_cache`` Python API. Results are written
    in-place to ``dst`` and the function returns ``None``.

    Args:
        src_cache: Cache storage with shape
            ``[num_blocks, block_size, entries...]``. FP8 cache modes use
            one-byte storage.
        dst: Output workspace with shape ``[num_tokens, head_dim]`` and dtype
            ``float32``, ``float16``, or ``bfloat16``.
        block_table: Per-sequence physical cache block ids, stored as int32.
        cu_seq_lens: Cumulative sequence lengths, stored as int32.
        token_to_seq: Mapping from output token id to sequence id, int32.
        num_tokens: Number of output tokens to gather.
        kv_cache_dtype: ``auto``, ``float16``, ``bfloat16``, ``fp8``,
            ``fp8_e4m3``, ``fp8_ds_mla``, or ``fp8_e5m2``.
        scale: Single-element float32 device tensor. FP8 output is computed as
            ``decode(cache_byte) * scale``.
        seq_starts: Optional int32 starting cache position for each sequence.

    Note:
        The optimized native kernel currently supports head dimensions 320
        and 576.
    """
    op.gather_and_maybe_dequant_cache(
        src_cache,
        dst,
        block_table,
        cu_seq_lens,
        token_to_seq,
        num_tokens,
        kv_cache_dtype,
        scale,
        seq_starts,
    )


__all__ = ["gather_and_maybe_dequant_cache"]
