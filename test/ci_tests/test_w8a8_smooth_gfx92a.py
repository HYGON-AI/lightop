# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Validate the gfx92a small-M dispatch against an independent Torch GEMM."""

import pytest
import torch

from lightop.gemm_ops import gemm_w8a8_smooth


@pytest.mark.parametrize("m", [1, 5, 16])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("with_bias", [False, True])
def test_small_m_smooth_gemm_gfx92a(m, dtype, with_bias):
    if not torch.cuda.is_available():
        pytest.skip("An HCU device is required")
    if torch.cuda.get_device_properties(0).gcnArchName.split(":")[0] != "gfx92a":
        pytest.skip("gfx92a dispatch regression")
    generator = torch.Generator(device="cuda").manual_seed(1234 + m)
    n, k = 512, 7168
    a = torch.randint(-32, 32, (m, k), generator=generator, device="cuda", dtype=torch.int8)
    b = torch.randint(-32, 32, (n, k), generator=generator, device="cuda", dtype=torch.int8).t()
    scale_a = torch.rand((m, 1), generator=generator, device="cuda") / 32
    scale_b = torch.rand((n, 1), generator=generator, device="cuda") / 32
    bias = torch.randn((n,), generator=generator, device="cuda", dtype=dtype) if with_bias else None
    available, actual = gemm_w8a8_smooth(a, b, scale_a, scale_b, bias, dtype)
    assert available, f"Missing gfx92a dispatch for M={m}, N={n}, K={k}"
    expected = (a.float() @ b.float()) * scale_a * scale_b.t()
    if bias is not None:
        expected += bias.float()
    torch.testing.assert_close(actual, expected.to(dtype), rtol=1e-2, atol=1e-2)
