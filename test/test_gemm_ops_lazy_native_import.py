import subprocess
import sys


def test_existing_gemm_api_does_not_reregister_lmslim_ops():
    code = """
from lmslim.quantize import quant_ops
from lightop.gemm_ops import grougemm_w8a8_pertensor_asm
assert callable(grougemm_w8a8_pertensor_asm)
"""
    subprocess.run([sys.executable, "-c", code], check=True)
