# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Three-segment INT8 reduction for non-power-of-two dense widths."""
import triton
import triton.language as tl
from triton.language.extra import libdevice

@triton.jit
def per_token_quant_int8_segments_kernel(X,Q,S,D:tl.constexpr,B:tl.constexpr):
    row=tl.program_id(0)
    col=tl.arange(0,B)
    x0=tl.load(X+row*D+col).to(tl.float32)
    x1=tl.load(X+row*D+B+col).to(tl.float32)
    x2=tl.load(X+row*D+2*B+col).to(tl.float32)
    am=tl.maximum(tl.maximum(tl.abs(x0),tl.abs(x1)),tl.abs(x2))
    amax=tl.maximum(tl.max(am,0),1e-10)
    inv=127/amax
    tl.store(Q+row*D+col,libdevice.nearbyint(x0*inv).to(tl.int8))
    tl.store(Q+row*D+B+col,libdevice.nearbyint(x1*inv).to(tl.int8))
    tl.store(Q+row*D+2*B+col,libdevice.nearbyint(x2*inv).to(tl.int8))
    tl.store(S+row,amax/127)
