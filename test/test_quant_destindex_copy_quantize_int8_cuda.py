# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Migration tests for ``destindex_copy_quantize_int8_cuda``."""

import torch

import lightop
import lightop.quant


def test_destindex_copy_quantize_int8_cuda_export_and_smoke() -> None:
    category_api = lightop.quant.destindex_copy_quantize_int8_cuda
    assert category_api is lightop.destindex_copy_quantize_int8_cuda
    assert "destindex_copy_quantize_int8_cuda" in lightop.quant.__all__

    torch.manual_seed(0)
    input_tensor = torch.randn((4, 3, 64), device="cuda", dtype=torch.bfloat16)
    scale_zero = torch.empty((4, 3, 2), device="cuda", dtype=torch.bfloat16)
    output = torch.empty_like(input_tensor, dtype=torch.int8)

    category_api(input_tensor, scale_zero, output)
    torch.cuda.synchronize()

    assert output.dtype is torch.int8
    assert torch.isfinite(scale_zero).all()
    assert (scale_zero[..., 0] > 0).all()
