# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Indexed RoPE and fake FP4 quantization for the DSV4.1 prefill indexer.

``dsv41_indexer_rope_fake_quant(x, freqs_cis, positions)`` accepts contiguous
BF16 queries [T, H, 128], a contiguous complex64 table [P, 32], and contiguous
int32/int64 positions [T], all on the same GPU. Positions must be in [0, P);
values are not checked on host. H must be positive; T may be zero.

The last 64 channels receive RoPE, including the native BF16 round-trip before
per-32 UE8M0/E2M1 fake quantization. The new contiguous BF16 result contains
FP4-grid values, not packed FP4. Inputs are unchanged. This inference operator
uses the current stream and supports graph capture/replay with changed inputs.

Re-export the native callable directly: importing this optional module is the
only extra indirection; the framework call itself has no Python wrapper.
"""

from ._dsv41_prep import dsv41_indexer_rope_fake_quant

__all__ = ["dsv41_indexer_rope_fake_quant"]
