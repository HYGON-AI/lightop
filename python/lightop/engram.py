# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Engram Prefill predecessor lookup and exact INT64 n-gram hashing on HCU."""

from . import op

__all__ = ["engram_hash_ids_hcu"]


def engram_hash_ids_hcu(
    input_ids, positions, history, row, starts, token_map, multipliers,
    primes, offsets, num_real, pad_id, image_token_id=None, mm_pad_shift=1000000,
    block=0,
):
    """Return hashes [T,2,24] and predecessor tokens [T,4], both INT64.

    IDs/positions/history are contiguous INT32/INT64; other tensors are
    contiguous INT64 on the same GPU.  With ``block == 0`` the caller passes
    row [num_real] and starts [B] for flattened EXTEND requests.  With
    ``block > 0`` the batch is a fixed-block target-verify layout:
    ``request = t // block``, ``offset = t % block``, history is [T/block, 3],
    and row/starts must be None.  Token IDs must index token_map after image
    normalization; primes are positive and products fit INT64, as constructed
    by SGLang.  Padding produces PAD hashes, not zero hashes.  Inputs and
    history are never modified; the caller commits history later.
    """
    return tuple(op.engram_hash_ids_hcu(
        input_ids, positions, history, row, starts, token_map, multipliers,
        primes, offsets, num_real, pad_id,
        -1 if image_token_id is None else image_token_id, mm_pad_shift, block,
    ))
