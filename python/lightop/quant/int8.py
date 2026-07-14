"""INT8 quantization APIs."""

from .._lmslim_native.layers.gemm.int8_utils import per_token_quant_int8
from ..wan_quant import (
    destindex_copy_quantize_int8_cuda,
    destindex_dequantize_int8_cuda,
)

__all__ = [
    "destindex_copy_quantize_int8_cuda",
    "destindex_dequantize_int8_cuda",
    "per_token_quant_int8",
]
