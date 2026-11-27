"""Fused MiniMax Prefill routed/shared reduction and Gemma RMSNorm."""
import triton
import triton.language as tl


@triton.jit
def _sum_norm(shared, routed, carry_in, weight, carry_out, normalized, eps,
              ROUTED_SCALE: tl.constexpr):
    row = tl.program_id(0)
    col = tl.arange(0, 8192)
    mask = col < 6144
    offset = row * 6144 + col
    a = tl.load(shared + offset, mask, 0).to(tl.float32)
    b = tl.load(routed + offset, mask, 0).to(tl.float32)
    if ROUTED_SCALE != 1.0:
        b = (b * ROUTED_SCALE).to(tl.bfloat16).to(tl.float32)
    rounded_sum = (a + b).to(tl.bfloat16).to(tl.float32)
    total = rounded_sum + tl.load(carry_in + offset, mask, 0).to(tl.float32)
    tl.store(carry_out + offset, total.to(tl.bfloat16), mask)
    variance = tl.sum(total * total, 0) / 6144
    inverse = 1.0 / tl.sqrt(variance + eps)
    scale = tl.load(weight + col, mask, 0).to(tl.float32) + 1.0
    y = (total * inverse * scale).to(tl.bfloat16)
    tl.store(normalized + offset, y, mask)
