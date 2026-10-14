# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

import sys

import pytest
import torch
import torch.nn.functional as F

from lightop._lmslim_native.quantize import quant_ops


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires a CUDA/ROCm device"
)


def _pack_int4(values: torch.Tensor) -> torch.Tensor:
    k, n = values.shape
    assert n % 8 == 0
    values = (values.to(torch.int32) & 0xF).view(k, n // 8, 8)
    packed = torch.zeros((k, n // 8), dtype=torch.int32, device=values.device)
    for index in range(8):
        packed |= values[:, :, index] << (index * 4)
    return packed


@pytest.fixture(scope="module")
def awq_case():
    k = n = group_size = 128
    generator = torch.Generator(device="cuda").manual_seed(20260827)
    weight = torch.randn(
        (k, n), generator=generator, device="cuda", dtype=torch.float16
    ) / 8
    scales = torch.empty((1, n), device="cuda", dtype=torch.float16)
    qzeros = torch.empty((1, n // 8), device="cuda", dtype=torch.int32)

    low, high = weight.amin(dim=0), weight.amax(dim=0)
    scale = (high - low) / 15
    scale[scale == 0] = 1
    zero = low / scale
    quant = torch.round(weight / scale - zero).clamp(0, 15)
    scales[0] = scale
    qzeros[0] = _pack_int4(zero.to(torch.int8).unsqueeze(0)).squeeze(0)
    qweight = _pack_int4(quant)

    converted, scale_zero = quant_ops.convert_s4(
        qweight.contiguous(), qzeros.contiguous(), scales.contiguous(), group_size
    )
    permuted = quant_ops.sz_permute(scale_zero)
    return {
        "k": k,
        "n": n,
        "group_size": group_size,
        "weight": converted.reshape(k, -1),
        "scale_zero": permuted.reshape(n, -1),
    }


def test_migrated_helpers_do_not_load_legacy_extension():
    assert "lmslimquant" not in sys.modules
    for name in (
        "awq_gemm",
        "gptq_gemm",
        "gptq_gemm1",
        "gptq_shuffle",
        "convert_s4",
        "sz_permute",
        "awq_dequant_colmajor",
        "dequant_w4_gemm_colmajor",
        "rocblas_sgemm_fp16_tn",
        "rocblas_sgemm_fp16_nn",
        "rocblas_sgemm_int8_tn",
        "rocblas_sgemm_int8_nn",
    ):
        assert callable(getattr(quant_ops, name))
    assert not hasattr(quant_ops, "get_awq_workspace_size")
    assert not hasattr(quant_ops, "dequant_w4_gemm_rowmajor")


def test_awq_dequant_implementations_match(awq_case):
    k = awq_case["k"]
    n = awq_case["n"]
    workspace = torch.zeros(k * n + 1, device="cuda", dtype=torch.float16)
    awq_dequant = quant_ops.awq_dequant_colmajor(
        awq_case["weight"],
        awq_case["scale_zero"],
        k,
        k,
        n,
        awq_case["group_size"],
        workspace,
        workspace.numel() * workspace.element_size(),
    )
    gemm_dequant = quant_ops.dequant_w4_gemm_colmajor(
        awq_case["weight"],
        awq_case["scale_zero"],
        k,
        n,
        awq_case["group_size"],
    )
    torch.testing.assert_close(awq_dequant, gemm_dequant)


def test_awq_gemm_matches_dequantized_linear(awq_case):
    k = awq_case["k"]
    n = awq_case["n"]
    m = 16
    generator = torch.Generator(device="cuda").manual_seed(20260828)
    inputs = torch.randn(
        (m, k), generator=generator, device="cuda", dtype=torch.float16
    ) / 10
    dequant = quant_ops.dequant_w4_gemm_colmajor(
        awq_case["weight"],
        awq_case["scale_zero"],
        k,
        n,
        awq_case["group_size"],
    )
    workspace = quant_ops.GetAWQShareWorkspace()
    actual = quant_ops.awq_gemm(
        inputs,
        awq_case["weight"],
        awq_case["scale_zero"],
        m,
        n,
        k,
        awq_case["group_size"],
        0,
        workspace,
        quant_ops.GetAWQShareWorkspaceSize(),
    )
    expected = F.linear(inputs, dequant)
    torch.testing.assert_close(actual, expected, rtol=1e-3, atol=1e-2)


def test_gptq_public_and_direct_bindings_match():
    generator = torch.Generator(device="cuda").manual_seed(20260829)
    inputs = torch.randn(
        (1, 128), generator=generator, device="cuda", dtype=torch.float16
    )
    qweight = torch.zeros((16, 128), device="cuda", dtype=torch.int32)
    qzeros = torch.zeros((1, 16), device="cuda", dtype=torch.int32)
    scales = torch.ones((1, 128), device="cuda", dtype=torch.float16)
    group_index = torch.zeros((128,), device="cuda", dtype=torch.int32)

    public = quant_ops.gptq_gemm(
        inputs, qweight, qzeros, scales, group_index, False, 4
    )
    direct = quant_ops.gptq_gemm1(
        inputs, qweight, qzeros, scales, group_index, False, 4
    )
    torch.testing.assert_close(public, direct)


def test_gptq_shuffle_accepts_identity_permutation():
    qweight = torch.zeros((16, 128), device="cuda", dtype=torch.int32)
    expected = qweight.clone()
    permutation = torch.arange(128, device="cuda", dtype=torch.int32)
    quant_ops.gptq_shuffle(qweight, permutation, 4)
    torch.testing.assert_close(qweight, expected, rtol=0, atol=0)


@pytest.mark.parametrize("suffix", ["tn", "nn"])
def test_rocblas_sgemm_fp16_matches_torch(suffix):
    generator = torch.Generator(device="cuda").manual_seed(20260830)
    a = torch.randn(
        (16, 64), generator=generator, device="cuda", dtype=torch.float16
    )
    b = torch.randn(
        (64, 32), generator=generator, device="cuda", dtype=torch.float16
    )
    actual = getattr(quant_ops, f"rocblas_sgemm_fp16_{suffix}")(
        a, b, torch.float16
    )
    # The TN entry point accepts weights whose storage is transposed while the
    # tensor keeps its (K, N) metadata, matching the legacy LMSlim contract.
    logical_b = b if suffix == "nn" else b.view(b.size(1), b.size(0)).T
    expected = a @ logical_b
    torch.testing.assert_close(actual, expected, rtol=1e-3, atol=1e-2)


@pytest.mark.parametrize("suffix", ["tn", "nn"])
def test_rocblas_sgemm_int8_matches_torch(suffix):
    generator = torch.Generator(device="cuda").manual_seed(20260831)
    a = torch.randint(
        -8, 8, (16, 64), generator=generator, device="cuda", dtype=torch.int8
    )
    b = torch.randint(
        -8, 8, (64, 32), generator=generator, device="cuda", dtype=torch.int8
    )
    actual = getattr(quant_ops, f"rocblas_sgemm_int8_{suffix}")(
        a, b, torch.int32
    )
    logical_b = b if suffix == "nn" else b.view(b.size(1), b.size(0)).T
    expected = (
        a.cpu().to(torch.int64) @ logical_b.cpu().to(torch.int64)
    ).to(device="cuda", dtype=torch.int32)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
