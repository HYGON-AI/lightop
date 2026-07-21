"""Public bindings for source-preserving LMSlim MoE implementations."""

from .._lmslim_native.layers.fused_moe.fp8_marlin import (
    fused_experts_impl_fp8_marlin,
)
from .._lmslim_native.layers.fused_moe.int8 import fused_experts_impl_int8
from .._lmslim_native.layers.fused_moe.int8_marlin import (
    fused_experts_impl_int8_marlin,
)
from .._lmslim_native.layers.fused_moe.int4 import (
    fused_experts_impl_int4,
    fused_experts_w4a16,
)
from .._lmslim_native.layers.fused_moe.w4a16 import fused_experts_impl_w4a16
from .._lmslim_native.layers.fused_moe.w4a8 import fused_experts_impl_w4a8
from .._lmslim_native.layers.fused_moe.w4a8_marlin import (
    fused_experts_impl_w4a8_marlin,
)
from .._lmslim_native.layers.groupgemm.w4a8 import (
    groupgemm_impl_w4a8,
    groupgemm_impl_w4a8_marlin,
)


def fused_experts_cuda(*args, **kwargs):
    """Load the optional grouped_gemm_int4 CUDA implementation on demand."""
    from .._lmslim_native.layers.fused_moe.int4_cuda import (
        fused_experts_cuda as implementation,
    )

    return implementation(*args, **kwargs)


def fused_experts_impl_cuda(*args, **kwargs):
    """Load the optional grouped_gemm_int4 CUDA implementation on demand."""
    from .._lmslim_native.layers.fused_moe.int4_cuda import (
        fused_experts_impl_cuda as implementation,
    )

    return implementation(*args, **kwargs)


__all__ = [
    "fused_experts_cuda",
    "fused_experts_impl_fp8_marlin",
    "fused_experts_impl_cuda",
    "fused_experts_impl_int4",
    "fused_experts_impl_int8",
    "fused_experts_impl_int8_marlin",
    "fused_experts_impl_w4a16",
    "fused_experts_impl_w4a8",
    "fused_experts_impl_w4a8_marlin",
    "fused_experts_w4a16",
    "groupgemm_impl_w4a8",
    "groupgemm_impl_w4a8_marlin",
]
