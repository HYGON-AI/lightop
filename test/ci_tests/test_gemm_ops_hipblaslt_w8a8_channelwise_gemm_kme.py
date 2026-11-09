# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

# pytest -q test/ci_tests/test_gemm_ops_hipblaslt_w8a8_channelwise_gemm_kme.py
import inspect

import pytest
import torch

from lightop import gemm_ops
from lightop._lmslim_native.quantize import quant_ops


def test_hipblaslt_w8a8_channelwise_gemm_kme_is_direct_public_binding():
    assert (
        gemm_ops.hipblaslt_w8a8_channelwise_gemm_kme
        is quant_ops.hipblaslt_w8a8_channelwise_gemm_kme
    )
    assert list(
        inspect.signature(gemm_ops.hipblaslt_w8a8_channelwise_gemm_kme).parameters
    ) == [
        "a", "b", "scale_a", "scale_b", "m", "n", "k",
        "transpose_flag", "out_dtype", "bias"
    ]


@pytest.mark.parametrize("out_dtype", [torch.float16, torch.bfloat16])
def test_hipblaslt_w8a8_channelwise_gemm_kme_matches_torch(out_dtype):
    if not torch.cuda.is_available():
        pytest.skip("requires a CUDA/ROCm device")
    arch = torch.cuda.get_device_properties(torch.cuda.current_device()).gcnArchName.split(":")[0]
    if arch != "gfx928":
        pytest.skip(f"native channelwise KME requires gfx928 (found {arch})")

    torch.manual_seed(17)
    m, n, k = 128, 256, 512
    a = torch.randint(-8, 8, (1, m, k), device="cuda", dtype=torch.int8)
    b = torch.randint(-8, 8, (1, n, k), device="cuda", dtype=torch.int8)
    scale_a = torch.rand((1, m, 1), device="cuda", dtype=torch.float32) * 0.05
    scale_b = torch.rand((1, n, 1), device="cuda", dtype=torch.float32) * 0.05
    bias = torch.randn((1, n), device="cuda", dtype=out_dtype)

    ok, actual = gemm_ops.hipblaslt_w8a8_channelwise_gemm_kme(
        a, b, scale_a, scale_b, m, n, k, "NT", out_dtype, bias
    )
    expected = torch.matmul(
        a.float() * scale_a, (b.float() * scale_b).transpose(-1, -2)
    )
    expected = (expected + bias.float().unsqueeze(1)).to(out_dtype)
    torch.cuda.synchronize()

    assert ok
    torch.testing.assert_close(actual, expected, atol=0.125, rtol=0.02)
