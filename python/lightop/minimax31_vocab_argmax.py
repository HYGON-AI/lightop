"""Graph-safe two-stage argmax for large, row-contiguous BF16 vocab logits."""
import torch
import triton
import triton.language as tl


@triton.jit
def _partial(logits, partial, cols: tl.constexpr, tiles: tl.constexpr,
             block: tl.constexpr, shift: tl.constexpr):
    row, tile = tl.program_id(0).to(tl.int64), tl.program_id(1)
    lane = tl.arange(0, block)
    col = tile * block + lane
    raw = tl.load(logits + row * cols + col, col < cols, other=-float("inf"))
    bits = raw.to(tl.uint16, bitcast=True).to(tl.uint32)
    abs_bits = bits & 0x7FFF
    bits = tl.where(abs_bits == 0, 0, bits)
    score = tl.where((bits & 0x8000) != 0, bits ^ 0xFFFF, bits ^ 0x8000)
    score = tl.where(abs_bits > 0x7F80, 0xFFFF, score)
    packed = (score << shift) | (block - 1 - lane).to(tl.uint32)
    winner = tl.max(tl.where(col < cols, packed, 0), 0)
    tl.store(partial + row * tiles + tile, winner)


@triton.jit
def _finish(partial, output, tiles: tl.constexpr, block: tl.constexpr,
            shift: tl.constexpr, reduce_block: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    tile = tl.arange(0, reduce_block)
    packed = tl.load(partial + row * tiles + tile, tile < tiles, other=0).to(tl.uint32)
    score = packed >> shift
    best_score = tl.max(score, 0)
    col = tile * block + block - 1 - (packed & (block - 1)).to(tl.int32)
    winner = tl.min(tl.where((tile < tiles) & (score == best_score), col, 0x7FFFFFFF), 0)
    tl.store(output + row, winner.to(tl.int64))


def argmax(logits, *, block=2048, warps=4, groups=1):
    if not logits.is_cuda or logits.dtype != torch.bfloat16 or logits.ndim != 2 or not logits.is_contiguous():
        return logits.argmax(dim=-1)
    rows, cols = logits.shape
    if cols < 32768 or rows == 0:
        return logits.argmax(dim=-1)
    tiles = triton.cdiv(cols, block)
    partial = torch.empty((rows, tiles), device=logits.device, dtype=torch.uint32)
    output = torch.empty((rows,), device=logits.device, dtype=torch.int64)
    _partial[(rows, tiles)](logits, partial, cols, tiles, block, block.bit_length() - 1,
                            num_warps=warps, waves_per_eu=groups)
    _finish[(rows,)](partial, output, tiles, block, block.bit_length() - 1,
                     triton.next_power_of_2(tiles), num_warps=1)
    return output
