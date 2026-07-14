"""FP8 quantization and conversion APIs."""

from ..gemmopt import fp8_to_16
from .._lmslim_native.layers.gemm.fp8_utils import (
    per_token_group_quant_fp8,
    per_token_quant_fp8,
)

__all__ = ["fp8_to_16", "per_token_group_quant_fp8", "per_token_quant_fp8"]
