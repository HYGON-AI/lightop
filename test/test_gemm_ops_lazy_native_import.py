import subprocess
import sys


def test_unused_legacy_lmslimquant_apis_are_removed():
    from lightop._lmslim_native.quantize import quant_ops

    assert not hasattr(quant_ops, "get_awq_workspace_size")
    assert not hasattr(quant_ops, "dequant_w4_gemm_rowmajor")


def test_native_gemm_api_does_not_eagerly_import_lmslimquant():
    code = """
import sys
from lightop.gemm_ops import rocblas_scaled_mm_nn
assert callable(rocblas_scaled_mm_nn)
assert "lmslimquant" not in sys.modules
"""
    subprocess.run([sys.executable, "-c", code], check=True)

def test_rocblas_scaled_mm_nn_falls_back_without_lmslimquant():
    code = """
import torch
from lightop.gemm_ops import rocblas_scaled_mm_nn

for m in (4, 32):
    a = torch.randint(-8, 8, (m, 16), dtype=torch.int8, device="cuda")
    b = torch.randint(-8, 8, (16, 8), dtype=torch.int8, device="cuda")
    scale_a = torch.rand((m, 1), dtype=torch.float32, device="cuda")
    scale_b = torch.rand((8, 1), dtype=torch.float32, device="cuda")
    actual = rocblas_scaled_mm_nn(
        a, b, scale_a, scale_b, torch.float32
    )
    expected = (a.float() @ b.float()) * scale_a * scale_b.T
    torch.testing.assert_close(actual, expected)
"""
    subprocess.run([sys.executable, "-c", code], check=True)
