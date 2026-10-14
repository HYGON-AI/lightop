# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from lightop._lmslim_native.quantize import quant_ops  # noqa: F401


@pytest.mark.parametrize("out_dtype", [torch.float16, torch.bfloat16])
def test_rocblas_scaled_mm_tn_matches_torch(out_dtype):
    if not torch.cuda.is_available():
        pytest.skip("requires a CUDA/ROCm device")

    torch.manual_seed(17)
    m, n, k = 128, 256, 512
    a = torch.randint(-8, 8, (m, k), device="cuda", dtype=torch.int8)
    b = torch.randint(-8, 8, (k, n), device="cuda", dtype=torch.int8)
    scale_a = torch.rand((m, 1), device="cuda") * 0.05
    scale_b = torch.rand((n, 1), device="cuda") * 0.05
    bias = torch.randn((n,), device="cuda", dtype=out_dtype)

    actual = torch.ops.lmslim._rocblas_scaled_mm_tn(
        a, b, scale_a, scale_b, out_dtype, bias
    )
    expected = (
        (a.float() @ b.float()) * (scale_a * scale_b.T)
    ).to(out_dtype) + bias
    torch.cuda.synchronize()

    torch.testing.assert_close(actual, expected, atol=0.125, rtol=0.02)
