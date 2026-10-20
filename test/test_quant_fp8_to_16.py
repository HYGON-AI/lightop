# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Migration tests for ``fp8_to_16``."""

import torch

import lightop
import lightop.quant


def test_fp8_to_16_export_and_block_scale_reference() -> None:
    category_api = lightop.quant.fp8_to_16
    assert category_api is lightop.fp8_to_16
    assert "fp8_to_16" in lightop.quant.__all__

    torch.manual_seed(2)
    input_tensor = torch.randn((256, 256), device="cuda").to(torch.float8_e4m3fn)
    scales = torch.tensor(
        [[0.5, 1.0], [1.5, 2.0]], device="cuda", dtype=torch.float32
    )
    output = torch.empty(input_tensor.shape, device="cuda", dtype=torch.float16)

    category_api(output, input_tensor, scales)
    torch.cuda.synchronize()

    expanded_scales = scales.repeat_interleave(128, 0).repeat_interleave(128, 1)
    expected = (input_tensor.float() * expanded_scales).to(torch.float16)
    torch.testing.assert_close(output, expected, rtol=0, atol=0)
