"""Combine routed, shared and zero-expert output in MiniMax VERIFY."""

import torch
import triton
import triton.language as tl


@triton.jit
def _apply(routed, shared, zero_expert, output, N: tl.constexpr,
           SCALE: tl.constexpr, BLOCK: tl.constexpr, ADD_ZERO: tl.constexpr):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = index < N
    r = tl.load(routed + index, valid, 0).to(tl.float32)
    s = tl.load(shared + index, valid, 0).to(tl.float32)
    if SCALE != 1.0:
        r = (r * SCALE).to(tl.bfloat16).to(tl.float32)
    summed = (s + r).to(tl.bfloat16)
    if ADD_ZERO:
        z = tl.load(zero_expert + index, valid, 0).to(tl.float32)
        summed = (summed.to(tl.float32) + z).to(tl.bfloat16)
    tl.store(output + index, summed, valid)


def supports(routed, shared):
    return (routed.shape == (128, 6144) and shared is not None and shared.shape == routed.shape
            and routed.dtype == shared.dtype == torch.bfloat16
            and routed.is_cuda and routed.device == shared.device
            and routed.is_contiguous() and shared.is_contiguous())


def merge(routed, shared, zero_expert, scale, *, block=1024):
    if not supports(routed, shared):
        raise ValueError("unsupported VERIFY MoE output geometry")
    if (zero_expert is not None and
            (zero_expert.shape != routed.shape or zero_expert.dtype != routed.dtype
             or zero_expert.device != routed.device or not zero_expert.is_contiguous())):
        raise ValueError("unsupported VERIFY MoE output geometry")
    result = torch.empty_like(routed)
    _apply[(triton.cdiv(routed.numel(), block),)](
        routed, shared, zero_expert, result, routed.numel(), float(scale), block, zero_expert is not None,
        num_warps=4, enable_fp_fusion=False,
    )
    return result
