"""Public binding for the source-preserving LMSlim SiLU-mul quant API."""

from .._lmslim_native.quantize.quant_ops import lm_fuse_silu_mul_quant

__all__ = ["lm_fuse_silu_mul_quant"]
