#!/usr/bin/env python3
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Run every lightop.quant migration acceptance case."""

from category_test_runner import run_category


RUNNABLE_TEST_FILES = [
    "test/test_native_categorized_quant_remaining_api.py",
    "test/test_quant_blockwise_scaled_cast_transpose_api.py",
    "test/test_quant_destindex_copy_quantize_int8_cuda.py",
    "test/test_quant_destindex_dequantize_int8_cuda.py",
    "test/test_quant_fp8_to_16.py",
    "test/test_quant_per_token_group_quant_fp8.py",
    "test/test_quant_per_token_quant_fp8.py",
    "test/test_quant_per_token_quant_int8.py",
    "test/test_quant_w4tow16.py",
]

SCRIPT_TEST_FILES = [
    "test/test_per_token_dynamic_quant_int8.py",
    "test/test_quantize_nope_fp8_rope_bf16_pack.py",
]


if __name__ == "__main__":
    raise SystemExit(
        run_category("quant", RUNNABLE_TEST_FILES, SCRIPT_TEST_FILES)
    )
