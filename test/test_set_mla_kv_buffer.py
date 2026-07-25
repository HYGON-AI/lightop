import torch
import triton
import triton.language as tl
import time

from lightop.kvcache import fused_quantize_and_store_mla_kv_cache


@triton.jit
def _quantize_k_cache_fast_kernel(
    output_nope_q_ptr, output_nope_s_ptr, output_rope_ptr,
    k_nope_ptr, k_rope_ptr,
    output_nope_q_stride_0: int, output_nope_s_stride_0: int,
    output_rope_stride_0: int, k_nope_stride_0: int, k_rope_stride_0: int,
    NUM_NOPE_BLOCKS: tl.constexpr, GROUP_SIZE: tl.constexpr,
    DIM_NOPE: tl.constexpr, DIM_ROPE: tl.constexpr,
    FP8_MIN: tl.constexpr, FP8_MAX: tl.constexpr,
):
    token_id = tl.program_id(0)
    raw_block_id = tl.program_id(1)

    if raw_block_id < NUM_NOPE_BLOCKS:
        effective_block_id = raw_block_id
        offs = effective_block_id * GROUP_SIZE + tl.arange(0, GROUP_SIZE)
        mask = offs < DIM_NOPE
        ptr = k_nope_ptr + token_id * k_nope_stride_0 + offs

        y = tl.load(ptr, mask=mask, other=0.0).to(tl.float32)

        y_s = tl.max(tl.abs(y)) / FP8_MAX
        y_s_inv = 1.0 / y_s
        y_q = tl.clamp(y * y_s_inv, FP8_MIN, FP8_MAX).to(output_nope_q_ptr.dtype.element_ty)

        dst_q_ptr = output_nope_q_ptr + token_id * output_nope_q_stride_0 + offs
        dst_s_ptr = output_nope_s_ptr + token_id * output_nope_s_stride_0 + effective_block_id

        tl.store(dst_q_ptr, y_q, mask=mask)
        tl.store(dst_s_ptr, y_s)
    else:
        effective_block_id = raw_block_id - NUM_NOPE_BLOCKS
        offs = effective_block_id * GROUP_SIZE + tl.arange(0, GROUP_SIZE)
        mask = offs < DIM_ROPE

        src_ptr = k_rope_ptr + token_id * k_rope_stride_0 + offs
        dst_ptr = output_rope_ptr + token_id * output_rope_stride_0 + offs

        data = tl.load(src_ptr, mask=mask)
        tl.store(dst_ptr, data, mask=mask)

def quantize_k_cache_separate(k_nope: torch.Tensor, k_rope: torch.Tensor, tile_size: int = 128):
    k_nope_2d = k_nope.squeeze(1) if k_nope.ndim == 3 else k_nope
    k_rope_2d = k_rope.squeeze(1) if k_rope.ndim == 3 else k_rope

    num_tokens = k_nope_2d.shape[0]
    dim_nope = k_nope_2d.shape[1]
    dim_rope = k_rope_2d.shape[1]

    num_tiles = dim_nope // tile_size
    nope_part_bytes = dim_nope + num_tiles * 4
    rope_part_bytes = dim_rope * k_rope.element_size()

    nope_part_u8 = torch.empty((num_tokens, nope_part_bytes), dtype=torch.uint8, device=k_nope.device)
    rope_part_u8 = torch.empty((num_tokens, rope_part_bytes), dtype=torch.uint8, device=k_rope.device)

    nope_q_view = nope_part_u8[:, :dim_nope].view(torch.float8_e4m3fn)
    nope_s_view = nope_part_u8[:, dim_nope:].view(torch.float32)
    rope_view = rope_part_u8.view(torch.bfloat16)

    num_blocks_per_token = triton.cdiv(dim_nope + dim_rope, tile_size)
    NUM_NOPE_BLOCKS = dim_nope // tile_size

    _quantize_k_cache_fast_kernel[(num_tokens, num_blocks_per_token)](
        nope_q_view, nope_s_view, rope_view, k_nope_2d, k_rope_2d,
        nope_q_view.stride(0), nope_s_view.stride(0), rope_view.stride(0),
        k_nope_2d.stride(0), k_rope_2d.stride(0),
        NUM_NOPE_BLOCKS=NUM_NOPE_BLOCKS, GROUP_SIZE=tile_size,
        DIM_NOPE=dim_nope, DIM_ROPE=dim_rope,
        FP8_MIN=torch.finfo(torch.float8_e4m3fn).min,
        FP8_MAX=torch.finfo(torch.float8_e4m3fn).max,
    )
    return nope_part_u8, rope_part_u8


@triton.jit
def set_mla_kv_buffer_kernel(
    kv_buffer_ptr, cache_k_nope_ptr, cache_k_rope_ptr, loc_ptr,
    buffer_stride: tl.constexpr, nope_stride: tl.constexpr, rope_stride: tl.constexpr,
    nope_dim: tl.constexpr, rope_dim: tl.constexpr, BLOCK: tl.constexpr,
):
    pid_loc = tl.program_id(0)
    pid_blk = tl.program_id(1)
    base = pid_blk * BLOCK
    offs = base + tl.arange(0, BLOCK)
    total_dim = nope_dim + rope_dim
    mask = offs < total_dim

    loc = tl.load(loc_ptr + pid_loc).to(tl.int64)
    dst_ptr = kv_buffer_ptr + loc * buffer_stride + offs

    if base + BLOCK <= nope_dim:
        src = tl.load(cache_k_nope_ptr + pid_loc * nope_stride + offs, mask=mask)
    elif base >= nope_dim:
        offs_rope = offs - nope_dim
        src = tl.load(cache_k_rope_ptr + pid_loc * rope_stride + offs_rope, mask=mask)
    else:
        is_nope = offs < nope_dim
        is_rope = (offs >= nope_dim) & (offs < (nope_dim + rope_dim))
        src_nope = tl.load(cache_k_nope_ptr + pid_loc * nope_stride + offs, mask=mask & is_nope, other=0)
        src_rope = tl.load(cache_k_rope_ptr + pid_loc * rope_stride + (offs - nope_dim), mask=mask & is_rope, other=0)
        src = tl.where(is_nope, src_nope, src_rope)

    tl.store(dst_ptr, src, mask=mask)

def set_mla_kv_buffer_triton(kv_buffer, loc, cache_k_nope, cache_k_rope):
    nope_dim = cache_k_nope.shape[-1]
    rope_dim = cache_k_rope.shape[-1]
    total_dim = nope_dim + rope_dim
    BLOCK = 128
    grid = (loc.numel(), triton.cdiv(total_dim, BLOCK))

    set_mla_kv_buffer_kernel[grid](
        kv_buffer, cache_k_nope, cache_k_rope, loc,
        kv_buffer.stride(0), cache_k_nope.stride(0), cache_k_rope.stride(0),
        nope_dim, rope_dim, BLOCK=BLOCK,
    )

def benchmark_and_verify():
    torch.manual_seed(42)
    
    # 定义测试场景 (类型, Token数)
    test_cases = [
        ("Decode", 1),
        ("Decode", 4),
        ("Decode", 16),
        ("Prefill", 128),
        ("Prefill", 1024),
        ("Prefill", 4096),
    ]

    print(f"\n| {'Stage':<10} | {'Tokens':<8} | {'Accuracy (Max Diff)':<20} | {'Triton (us)':<12} | {'CUDA (us)':<12} | {'Speedup':<8} |")
    print(f"|{'-'*12}|{'-'*10}|{'-'*22}|{'-'*14}|{'-'*14}|{'-'*10}|")

    for stage, num_tokens in test_cases:
        dim_nope = 512
        dim_rope = 64
        max_blocks = max(8192, num_tokens * 2)
        
        k_nope = torch.randn((num_tokens, dim_nope), dtype=torch.bfloat16, device="cuda") * 2.0
        k_rope = torch.randn((num_tokens, dim_rope), dtype=torch.bfloat16, device="cuda") * 2.0
        loc = torch.randperm(max_blocks, device="cuda", dtype=torch.int64)[:num_tokens]
        
        kv_buffer_triton = torch.zeros((max_blocks, 656), dtype=torch.uint8, device="cuda")
        kv_buffer_cuda = torch.zeros((max_blocks, 656), dtype=torch.uint8, device="cuda")

        nope_u8, rope_u8 = quantize_k_cache_separate(k_nope, k_rope)
        set_mla_kv_buffer_triton(kv_buffer_triton, loc, nope_u8, rope_u8)
        
        fused_quantize_and_store_mla_kv_cache(k_nope, k_rope, kv_buffer_cuda, loc, "fp8_e4m3", 1e-5)
        
        diff = (kv_buffer_triton.float() - kv_buffer_cuda.float()).abs().max().item()
        acc_status = f"Pass (Diff: {diff})" if diff <= 1.0 else f"Fail (Diff: {diff})"

        # 3. 性能测试 (Triton)
        # Warmup
        for _ in range(10):
            n, r = quantize_k_cache_separate(k_nope, k_rope)
            set_mla_kv_buffer_triton(kv_buffer_triton, loc, n, r)
        torch.cuda.synchronize()
        
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        start_event.record()
        for _ in range(100):
            n, r = quantize_k_cache_separate(k_nope, k_rope)
            set_mla_kv_buffer_triton(kv_buffer_triton, loc, n, r)
        end_event.record()
        torch.cuda.synchronize()
        triton_time_us = start_event.elapsed_time(end_event) * 1000.0 / 100.0

        # 4. 性能测试 (CUDA)
        # Warmup
        for _ in range(10):
            fused_quantize_and_store_mla_kv_cache(k_nope, k_rope, kv_buffer_cuda, loc, "fp8_e4m3", 1e-5)
        torch.cuda.synchronize()

        start_event.record()
        for _ in range(100):
            fused_quantize_and_store_mla_kv_cache(k_nope, k_rope, kv_buffer_cuda, loc, "fp8_e4m3", 1e-5)
        end_event.record()
        torch.cuda.synchronize()
        cuda_time_us = start_event.elapsed_time(end_event) * 1000.0 / 100.0

        # 计算加速比
        speedup = triton_time_us / cuda_time_us if cuda_time_us > 0 else 0.0

        # 打印表格行
        print(f"| {stage:<10} | {num_tokens:<8} | {acc_status:<20} | {triton_time_us:<12.2f} | {cuda_time_us:<12.2f} | {speedup:<7.2f}x |")

if __name__ == "__main__":
    benchmark_and_verify()