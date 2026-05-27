"""
Test script for quantize_nope_fp8_rope_bf16_pack.

Baseline: Triton reference implementation.
C++ implementation wrapped in NopeFp8RopeBf16Pack.
"""
import torch
import triton
import triton.language as tl
from dataclasses import dataclass
from typing import Any
from lightop import op

fp8_dtype = torch.float8_e4m3fn


@dataclass
class NopeFp8RopeBf16Pack:
    k_nope_fp8: torch.Tensor
    k_rope_bf16: torch.Tensor
    scale_k_nope_ue8m0: torch.Tensor

    def __post_init__(self):
        assert self.k_nope_fp8.shape[-1] == 448
        assert self.k_rope_bf16.shape[-1] == 64
        assert self.scale_k_nope_ue8m0.shape[-1] == 7

    def slice_pack(self, _slice: Any) -> "NopeFp8RopeBf16Pack":
        return NopeFp8RopeBf16Pack(
            k_nope_fp8=self.k_nope_fp8[_slice],
            k_rope_bf16=self.k_rope_bf16[_slice],
            scale_k_nope_ue8m0=self.scale_k_nope_ue8m0[_slice],
        )


@triton.jit
def _quant_k_cache_fused_kernel(
    k_bf16_ptr,
    k_nope_fp8_ptr,
    k_rope_bf16_ptr,
    scale_k_nope_uint8_ptr,
    k_bf16_stride_0,
    k_nope_fp8_stride_0,
    k_rope_bf16_stride_0,
    scale_stride_0,
    DIM_NOPE: tl.constexpr,
    DIM_ROPE: tl.constexpr,
    TILE_SIZE: tl.constexpr,
    NUM_TILES: tl.constexpr,
    FP8_MIN: tl.constexpr,
    FP8_MAX: tl.constexpr,
    EPS: tl.constexpr,
):
    token_id = tl.program_id(0)
    tile_id = tl.program_id(1)

    if tile_id == NUM_TILES:
        rope_range = tl.arange(0, TILE_SIZE)
        rope_mask = rope_range < DIM_ROPE

        in_rope_offsets = token_id * k_bf16_stride_0 + DIM_NOPE + rope_range
        rope_data = tl.load(k_bf16_ptr + in_rope_offsets, mask=rope_mask, other=0.0)

        out_rope_offsets = token_id * k_rope_bf16_stride_0 + rope_range
        tl.store(k_rope_bf16_ptr + out_rope_offsets, rope_data, mask=rope_mask)
    else:
        tile_range = tl.arange(0, TILE_SIZE)

        in_tile_offsets = token_id * k_bf16_stride_0 + tile_id * TILE_SIZE + tile_range
        x_bf16 = tl.load(k_bf16_ptr + in_tile_offsets)
        x_fp32 = x_bf16.to(tl.float32)

        abs_x = tl.abs(x_fp32)
        max_abs = tl.max(abs_x)
        max_abs_clamped = tl.maximum(max_abs, EPS)
        scale = max_abs_clamped / FP8_MAX

        log2_scale = tl.log2(scale)
        ceil_log2 = tl.math.ceil(log2_scale)
        scale_pow2_fp32 = tl.exp2(ceil_log2)
        scale_inv = 1.0 / scale_pow2_fp32
        x_scaled = x_fp32 * scale_inv
        x_fp8 = tl.clamp(x_scaled, FP8_MIN, FP8_MAX).to(k_nope_fp8_ptr.dtype.element_ty)

        out_fp8_offsets = (
            token_id * k_nope_fp8_stride_0 + tile_id * TILE_SIZE + tile_range
        )
        tl.store(k_nope_fp8_ptr + out_fp8_offsets, x_fp8)

        exponent = ceil_log2.to(tl.int32)
        scale_uint8 = (exponent + 127).to(tl.uint8)

        out_scale_offset = token_id * scale_stride_0 + tile_id
        tl.store(scale_k_nope_uint8_ptr + out_scale_offset, scale_uint8)


def quant_to_nope_fp8_rope_bf16_pack_triton(
    k_bf16: torch.Tensor,
) -> NopeFp8RopeBf16Pack:
    assert k_bf16.dtype == torch.bfloat16
    num_tokens, hidden_dim = k_bf16.shape
    assert hidden_dim == 512
    dim_nope = 448
    dim_rope = 64
    tile_size = 64
    num_tiles = dim_nope // tile_size

    k_bf16 = k_bf16.contiguous()

    k_nope_fp8 = torch.empty(
        (num_tokens, dim_nope), dtype=fp8_dtype, device=k_bf16.device
    )
    k_rope_bf16 = torch.empty(
        (num_tokens, dim_rope), dtype=torch.bfloat16, device=k_bf16.device
    )
    scale_k_nope_ue8m0 = torch.empty(
        (num_tokens, num_tiles), dtype=torch.uint8, device=k_bf16.device
    )

    fp8_dtype_info = torch.finfo(fp8_dtype)

    grid = (num_tokens, num_tiles + 1)
    _quant_k_cache_fused_kernel[grid](
        k_bf16,
        k_nope_fp8,
        k_rope_bf16,
        scale_k_nope_ue8m0,
        k_bf16.stride(0),
        k_nope_fp8.stride(0),
        k_rope_bf16.stride(0),
        scale_k_nope_ue8m0.stride(0),
        DIM_NOPE=dim_nope,
        DIM_ROPE=dim_rope,
        TILE_SIZE=tile_size,
        NUM_TILES=num_tiles,
        FP8_MIN=fp8_dtype_info.min,
        FP8_MAX=fp8_dtype_info.max,
        EPS=1e-8,
    )

    return NopeFp8RopeBf16Pack(
        k_nope_fp8=k_nope_fp8,
        k_rope_bf16=k_rope_bf16,
        scale_k_nope_ue8m0=scale_k_nope_ue8m0,
    )


def quantize_nope_fp8_rope_bf16_pack_cpp(
    k_bf16: torch.Tensor, eps: float = 1e-8
) -> NopeFp8RopeBf16Pack:
    """Call the C++ implementation and wrap result in NopeFp8RopeBf16Pack."""
    result = op.quantize_nope_fp8_rope_bf16_pack(k_bf16, eps)
    k_nope_fp8_u8, k_rope_bf16, scale_k_nope = result
    # C++ returns k_nope_fp8 as uint8; view as fp8 for the dataclass
    k_nope_fp8 = k_nope_fp8_u8.view(fp8_dtype)
    return NopeFp8RopeBf16Pack(
        k_nope_fp8=k_nope_fp8,
        k_rope_bf16=k_rope_bf16,
        scale_k_nope_ue8m0=scale_k_nope,
    )


def dequant_nope_fp8(pack: NopeFp8RopeBf16Pack) -> torch.Tensor:
    """Dequantize k_nope_fp8 * scale -> bf16 for correctness comparison."""
    scale_exp = pack.scale_k_nope_ue8m0.to(torch.float32)  # biased E8M0, [T, 7]
    scale_factor = 2.0 ** (scale_exp - 127.0)  # [T, 7]
    # Tile-wise dequant: each scale applies to 64 elements
    scale_factor = scale_factor.unsqueeze(-1).expand(-1, -1, 64).reshape(-1, 448)  # [T, 448]
    fp32_val = pack.k_nope_fp8.to(torch.float32)
    return (fp32_val * scale_factor).to(torch.bfloat16)


def verify_correctness(k_bf16: torch.Tensor):
    """Compare Triton and C++ outputs."""
    triton_pack = quant_to_nope_fp8_rope_bf16_pack_triton(k_bf16)
    cpp_pack = quantize_nope_fp8_rope_bf16_pack_cpp(k_bf16)

    results = {}

    # k_rope_bf16: exact copy, should match identically
    rope_diff = (triton_pack.k_rope_bf16 != cpp_pack.k_rope_bf16).sum().item()
    results["rope_equal"] = rope_diff == 0
    results["rope_max_diff"] = 0.0 if rope_diff == 0 else (
        (triton_pack.k_rope_bf16.float() - cpp_pack.k_rope_bf16.float()).abs().max().item()
    )

    # scale_k_nope_ue8m0: should match identically
    scale_diff = (triton_pack.scale_k_nope_ue8m0 != cpp_pack.scale_k_nope_ue8m0).sum().item()
    results["scale_equal"] = scale_diff == 0
    results["scale_max_diff"] = scale_diff

    # k_nope_fp8: dequantize both and compare
    triton_dequant = dequant_nope_fp8(triton_pack)
    cpp_dequant = dequant_nope_fp8(cpp_pack)
    diff = (triton_dequant.float() - cpp_dequant.float()).abs()
    results["dequant_max_diff"] = diff.max().item()
    results["dequant_mean_diff"] = diff.mean().item()

    return results


def benchmark():
    torch.manual_seed(42)

    test_cases = [
        ("Decode", 1),
        ("Decode", 4),
        ("Decode", 16),
        ("Prefill", 128),
        ("Prefill", 1024),
        ("Prefill", 4096),
        ("Prefill", 8192),
        ("Prefill", 16384),
    ]

    # Bytes per token: input bf16(512*2) + output fp8(448) + output bf16(64*2) + scale(7)
    bytes_per_token = 512 * 2 + 448 + 64 * 2 + 7

    print(f"\n| {'Stage':<10} | {'Tokens':<8} | {'k_rope':<10} | {'scale':<10} | {'Dequant':<12} | {'Triton(us)':<10} | {'TritonBW':<10} | {'CUDA(us)':<10} | {'CudBW':<10} | {'Speedup':<8} |")
    print(f"|{'-'*12}|{'-'*10}|{'-'*12}|{'-'*12}|{'-'*14}|{'-'*12}|{'-'*12}|{'-'*12}|{'-'*12}|{'-'*10}|")

    for stage, num_tokens in test_cases:
        k_bf16 = torch.randn((num_tokens, 512), dtype=torch.bfloat16, device="cuda") * 2.0

        # Correctness
        ck = verify_correctness(k_bf16)
        rope_str = "Pass" if ck["rope_equal"] else f"Diff:{ck['rope_max_diff']:.2e}"
        scale_str = "Pass" if ck["scale_equal"] else f"Diff:{ck['scale_max_diff']}"
        deq_str = f"max{ck['dequant_max_diff']:.2e}"

        # Triton warmup + timed
        for _ in range(10):
            _ = quant_to_nope_fp8_rope_bf16_pack_triton(k_bf16)
        torch.cuda.synchronize()

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        for _ in range(100):
            _ = quant_to_nope_fp8_rope_bf16_pack_triton(k_bf16)
        end_event.record()
        torch.cuda.synchronize()
        triton_time_us = start_event.elapsed_time(end_event) * 1000.0 / 100.0
        triton_bw = (num_tokens * bytes_per_token) / (triton_time_us * 1e-6) / 1e9

        # C++ warmup + timed
        for _ in range(10):
            _ = quantize_nope_fp8_rope_bf16_pack_cpp(k_bf16)
        torch.cuda.synchronize()

        start_event.record()
        for _ in range(100):
            _ = quantize_nope_fp8_rope_bf16_pack_cpp(k_bf16)
        end_event.record()
        torch.cuda.synchronize()
        cpp_time_us = start_event.elapsed_time(end_event) * 1000.0 / 100.0
        cpp_bw = (num_tokens * bytes_per_token) / (cpp_time_us * 1e-6) / 1e9

        speedup = triton_time_us / cpp_time_us if cpp_time_us > 0 else 0.0

        print(f"| {stage:<10} | {num_tokens:<8} | {rope_str:<10} | {scale_str:<10} | {deq_str:<12} | {triton_time_us:<10.2f} | {triton_bw:<9.2f} | {cpp_time_us:<10.2f} | {cpp_bw:<9.2f} | {speedup:<7.2f}x |")


if __name__ == "__main__":
    benchmark()
