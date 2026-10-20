# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Migration test for LMSlim blockwise hipBLASLt GEMM."""

import inspect
import math

import pytest
import torch

from lightop._lmslim_native.quantize.quant_ops import (
    hipblaslt_w8a8_blockwise_gemm as native_api,
)
from lightop.gemm_ops import BlockSize, hipblaslt_w8a8_blockwise_gemm


def test_hipblaslt_w8a8_blockwise_gemm_export() -> None:
    assert hipblaslt_w8a8_blockwise_gemm is native_api
    assert inspect.signature(hipblaslt_w8a8_blockwise_gemm) == inspect.signature(native_api)


def test_hipblaslt_w8a8_blockwise_gemm_matches_torch_reference() -> None:
    if torch.cuda.get_device_name(0) == "BW151":
        pytest.skip("BW151 hipBLASLt has no blockwise FP8 algorithm for this test")

    torch.manual_seed(10)
    m, n, k = 256, 128, 512
    block_size = 128
    a = torch.randn((m, k), device="cuda").to(torch.float8_e4m3fn)
    b = torch.randn((k, n), device="cuda").to(torch.float8_e4m3fn)
    scale_a = torch.rand(
        (m, math.ceil(k / block_size)), device="cuda", dtype=torch.float32
    ) * 0.1
    scale_b = torch.rand(
        (math.ceil(k / block_size), math.ceil(n / block_size)),
        device="cuda",
        dtype=torch.float32,
    ) * 0.1

    status, actual = hipblaslt_w8a8_blockwise_gemm(
        a,
        b,
        scale_a,
        scale_b,
        m,
        n,
        k,
        "NN",
        torch.bfloat16,
        BlockSize.block_128x128,
    )

    k_blocks = torch.arange(k, device="cuda") // block_size
    n_blocks = torch.arange(n, device="cuda") // block_size
    expected = (
        a.float() * scale_a[:, k_blocks]
    ) @ (
        b.float() * scale_b[k_blocks][:, n_blocks]
    )
    torch.cuda.synchronize()

    assert status is True
    torch.testing.assert_close(actual.view(m, n), expected.to(torch.bfloat16), rtol=0, atol=1)
