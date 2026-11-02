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
