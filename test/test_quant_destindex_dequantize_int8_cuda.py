# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Migration tests for ``destindex_dequantize_int8_cuda``."""

import torch

import lightop
import lightop.quant


def test_destindex_dequantize_int8_cuda_export_and_reference() -> None:
    category_api = lightop.quant.destindex_dequantize_int8_cuda
    assert category_api is lightop.destindex_dequantize_int8_cuda
    assert "destindex_dequantize_int8_cuda" in lightop.quant.__all__

    torch.manual_seed(1)
    input_tensor = torch.randint(
        -128, 128, (4, 3, 64), device="cuda", dtype=torch.int8
    )
    scale_zero = torch.empty((4, 3, 2), device="cuda", dtype=torch.bfloat16)
    scale_zero[..., 0] = torch.rand((4, 3), device="cuda") + 0.01
    scale_zero[..., 1] = torch.randint(
        0, 256, (4, 3), device="cuda", dtype=torch.int32
    ).to(torch.bfloat16)
    output = torch.empty(input_tensor.shape, device="cuda", dtype=torch.bfloat16)

    category_api(input_tensor, scale_zero, output)
    torch.cuda.synchronize()

    expected = (
        input_tensor.to(torch.int32)
        + 128
        - scale_zero[..., 1, None].to(torch.int32)
    ).to(torch.bfloat16) * scale_zero[..., 0, None]
    torch.testing.assert_close(output, expected, rtol=0, atol=0)
