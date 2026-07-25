import inspect

import pytest
import torch

from lightop import gemm_ops
from lightop._lmslim_native.quantize import quant_ops


def test_hipblaslt_w8a8_gemm_is_direct_public_binding():
    assert gemm_ops.hipblaslt_w8a8_gemm is quant_ops.hipblaslt_w8a8_gemm
    assert list(inspect.signature(gemm_ops.hipblaslt_w8a8_gemm).parameters) == [
        "a",
        "b",
        "scale_a",
        "scale_b",
        "m",
        "n",
        "k",
        "transpose_flag",
        "out_dtype",
        "bias",
    ]


@pytest.mark.parametrize("out_dtype", [torch.float16, torch.bfloat16])
def test_hipblaslt_w8a8_gemm_matches_torch(out_dtype):
    if not torch.cuda.is_available():
        pytest.skip("requires a CUDA/ROCm device")

    torch.manual_seed(7)
    m, n, k = 128, 256, 512
    a = torch.randint(-8, 8, (1, m, k), device="cuda", dtype=torch.int8)
    b = torch.randint(-8, 8, (1, n, k), device="cuda", dtype=torch.int8)
    scale_a = torch.rand((1, m, 1), device="cuda", dtype=torch.float32) * 0.05
    scale_b = torch.rand((1, n, 1), device="cuda", dtype=torch.float32) * 0.05
    bias = torch.randn((1, n), device="cuda", dtype=out_dtype)

    ok, actual = gemm_ops.hipblaslt_w8a8_gemm(
        a, b, scale_a, scale_b, m, n, k, "NT", out_dtype, bias
    )
    torch.cuda.synchronize()

    expected = torch.matmul(
        a.float() * scale_a, (b.float() * scale_b).transpose(-1, -2)
    ).squeeze(0)
    expected = (expected + bias.float().squeeze(0)).to(out_dtype)

    assert ok
    torch.testing.assert_close(actual, expected, atol=0.125, rtol=0.02)
