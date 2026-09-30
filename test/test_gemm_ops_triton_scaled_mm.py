# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

import inspect

import pytest
import torch

from lightop import gemm_ops
from lightop._lmslim_native.quantize import quant_ops


def test_triton_scaled_mm_is_direct_public_binding():
    assert gemm_ops.triton_scaled_mm is quant_ops.triton_scaled_mm
    assert list(inspect.signature(gemm_ops.triton_scaled_mm).parameters) == [
        "a", "b", "scale_a", "scale_b", "out_dtype", "bias", "best_config"
    ]


@pytest.mark.parametrize("out_dtype", [torch.float16, torch.bfloat16])
def test_triton_scaled_mm_matches_torch(out_dtype):
    if not torch.cuda.is_available():
        pytest.skip("requires a CUDA/ROCm device")

    torch.manual_seed(13)
    m, n, k = 128, 256, 512
    a = torch.randint(-8, 8, (m, k), device="cuda", dtype=torch.int8)
    b = torch.randint(-8, 8, (n, k), device="cuda", dtype=torch.int8).T
    scale_a = torch.rand((m, 1), device="cuda", dtype=torch.float32) * 0.05
    scale_b = torch.rand((n, 1), device="cuda", dtype=torch.float32) * 0.05
    bias = torch.randn((n,), device="cuda", dtype=out_dtype)

    actual = gemm_ops.triton_scaled_mm(
        a, b, scale_a, scale_b, out_dtype, bias, best_config=None
    )
    expected = (
        (a.float() @ b.float()) * (scale_a * scale_b.T)
    ).to(out_dtype) + bias
    torch.cuda.synchronize()

    torch.testing.assert_close(actual, expected, atol=0.125, rtol=0.02)
