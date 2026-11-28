# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

from .. import op


def markov_gemv_add_argmax(latent, weight, base_logits):
    """Fuse a BF16 rank-256 Markov projection, logits add and greedy argmax.

    Args:
        latent: Contiguous BF16 [BS, 256], with BS in {1, 2, 3, 4}.
        weight: Contiguous, 16-byte-aligned BF16 [vocabulary, 256].
        base_logits: BF16 or FP32 [BS, vocabulary]; the leading stride may contain
            gaps, for example a position selected from [BS, block, vocabulary].

    Returns:
        (corrected_logits, token_ids), shaped [BS, vocabulary] and [BS].
        The projection rounds to BF16 before addition. Corrected logits use
        base_logits.dtype, so FP32 logits do not receive an extra BF16 rounding.
        Equal maxima choose the smallest token index, as torch.argmax does.

    This is a forward-only greedy operator. It does not replace stochastic
    sampling. All buffers use the current device and stream; CUDA Graph replay
    and independent concurrent calls are supported.
    """
    return op.markov_gemv_add_argmax(latent, weight, base_logits)


def dsv41_markov_fp8_partials(base, prev, embedding, embedding_scale, weight,
                              weight_scale, *, vocab_start, vocab_size, out, tile_rows=128,
                              packed_argmax=False, k_stage=256):
    """gfx938 E4M3FN rank-256 lookup/projection/add/top-1 for 1..32 requests.

    FP8 matrices are contiguous and 16-byte aligned. FP32 scales have shape
    [rows, 1] and may have strided rows. ``prev`` is an int32/int64 batch vector;
    valid IDs are in [0, vocab_size). Base BF16/FP32 logits have contiguous,
    non-overlapping vocabulary rows. ``weight`` contains only real shard rows;
    base may include a padded vocabulary tail or an entirely empty shard.

    The projection multiplies token then channel scales in FP32, rounds to
    BF16, then adds base in FP32. NaNs win; ties choose the lowest global ID.
    ``tile_rows`` is 128 (default) or 64. ``out`` is caller-owned contiguous
    FP32 [batch, ceil(base.width/tile_rows), 2]. The optional
    ``packed_argmax`` uses integer-key partial reduction; it defaults off,
    preserves winning IDs and may canonicalize NaN payloads/signed-zero scores.

    ``k_stage=64`` optionally reuses a smaller weight LDS buffer and stages
    the next packet in registers; 256 retains the original full-rank load.

    Pairs carry score/ID-bits, with (-inf, INT_MAX) for invalid tiles. No full
    logits are materialized. Buffers must not alias inputs or each other.
    Forward only, current device/stream, graph safe; no allocations or CPU sync.
    """
    args = (base, prev, embedding, embedding_scale, weight, weight_scale,
            vocab_start, vocab_size, out)
    if k_stage != 256:
        op.dsv41_markov_fp8_partials(*args, tile_rows, packed_argmax, k_stage)
    elif packed_argmax:
        op.dsv41_markov_fp8_partials(*args, tile_rows, True)
    elif tile_rows == 128:
        op.dsv41_markov_fp8_partials(*args)
    else:
        op.dsv41_markov_fp8_partials(*args, tile_rows)


def dsv41_markov_compact_partials(partials, *, out):
    """Compact 32 adjacent partial winners into one TP pair.

    Each pair covers 4096 entries for tile_rows=128, or 2048 for tile_rows=64.

    Input/output are disjoint contiguous FP32 [batch, parts, 2] buffers.
    """
    op.dsv41_markov_compact_partials(partials, out)


def dsv41_markov_merge_partials(partials, *, rows, world_size, out, packed_argmax=False):
    """Merge contiguous rank-major score/ID-bit pairs into a token column.

    ``partials`` holds [world_size, rows, parts, 2], or an equivalent contiguous
    view. ``out`` is a disjoint int64 [rows] view with positive stride, including
    tokens[:, step]. Next-step lookup reads it on the same stream, without CPU
    readback. All-invalid rows produce INT_MAX. No allocation or synchronization.
    Optional ``packed_argmax`` uses integer-key reduction and a rank-local
    address calculation at TP1; the original merge remains the default.
    """
    if packed_argmax:
        op.dsv41_markov_merge_partials(partials, rows, world_size, out, True)
    else:
        op.dsv41_markov_merge_partials(partials, rows, world_size, out)
    return out
