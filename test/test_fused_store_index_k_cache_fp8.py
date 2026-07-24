import torch
import triton
import triton.language as tl
import math
from typing import Optional, Tuple
import time

# 导入你的底层融合算子
from lightop import op

# ==========================================
# 环境变量 (确保代码可以正常运行)
# ==========================================
_is_fp8_fnuz = False
_is_hip = False
_is_hcu = False

# ==========================================
# 用户提供的基准代码 (Triton Baseline)
# ==========================================
@triton.jit
def _act_quant_triton_kernel(
    X_ptr, Y_ptr, S_ptr,
    N: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    FP8_MAX: tl.constexpr,
    FP8_MIN: tl.constexpr,
    ROUND_SCALE: tl.constexpr,
):
    """
    使用 Triton 实现的按组 (Group-wise) 激活量化 Kernel
    """
    pid = tl.program_id(0)
    num_groups_per_row = N // GROUP_SIZE
    
    # 计算当前块属于哪一行以及哪一个组
    row_idx = pid // num_groups_per_row
    group_idx = pid % num_groups_per_row
    
    # 组内元素的全局偏移量
    offset = row_idx * N + group_idx * GROUP_SIZE + tl.arange(0, GROUP_SIZE)
    x = tl.load(X_ptr + offset)
    
    # 求绝对最大值
    abs_x = tl.math.abs(x)
    amax = tl.max(abs_x)
    amax = tl.maximum(amax, 1e-4)
    
    # 计算 scale
    scale = amax / FP8_MAX
    if ROUND_SCALE:
        # 支持 ue8m0 的 2幂次 向上取整优化 (fast_round_scale 逻辑)
        scale = tl.math.exp2(tl.math.ceil(tl.math.log2(scale)))
        
    # 量化并截断
    inv_scale = 1.0 / scale
    y = x * inv_scale
    y = tl.clamp(y, FP8_MIN, FP8_MAX)
    
    # 写回内存 (Triton 根据 Y_ptr 类型自动 cast 到 fp8)
    tl.store(Y_ptr + offset, y)
    tl.store(S_ptr + pid, scale)

def act_quant(
    x: torch.Tensor, block_size: int = 128, scale_fmt: Optional[str] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Quantizes the input tensor `x` using block-wise quantization.
    Args:
        x (torch.Tensor): The input tensor to be quantized. Must be contiguous and its last dimension size must be divisible by `block_size`.
        block_size (int, optional): The size of the blocks to be used for quantization. Default is 128.
        scale_fmt (Optional[str], optional): The format of the scale. Default is None.
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
            - The quantized tensor with dtype `torch.float8_e4m3fn`.
            - A tensor of scaling factors with dtype `torch.float32`.
    """
    assert x.is_contiguous(), "Input tensor must be contiguous"
    assert (
        x.size(-1) % block_size == 0
    ), f"Last dimension size must be divisible by block_size (block_size={block_size})"
    N = x.size(-1)
    
    if _is_fp8_fnuz:
        y = torch.empty_like(x, dtype=torch.float8_e4m3fnuz)
        fp8_max, fp8_min = 224.0, -224.0
    else:
        y = torch.empty_like(x, dtype=torch.float8_e4m3fn)
        fp8_max, fp8_min = 448.0, -448.0
        
    s = x.new_empty(*x.size()[:-1], N // block_size, dtype=torch.float32)
    
    # 计算网格大小：每个 block_size 作为一个程序执行
    M = x.numel() // N
    total_groups = M * (N // block_size)
    grid = (total_groups,)
    
    _act_quant_triton_kernel[grid](
        x, y, s,
        N=N,
        GROUP_SIZE=block_size,
        FP8_MAX=fp8_max,
        FP8_MIN=fp8_min,
        ROUND_SCALE=(scale_fmt is not None)
    )
    
    return y, s

@triton.jit
def _set_k_and_s_triton_kernel(
    buf_fp8_ptr,
    buf_fp32_ptr,
    loc_ptr,
    index_k_ptr,
    index_k_scale_ptr,
    index_k_ptr_stride_0,
    PAGE_SIZE: tl.constexpr,
    BUF_NUMEL_PER_PAGE: tl.constexpr,
    NUM_K_ELEMS_PER_TOKEN: tl.constexpr,
    S_OFFSET_NBYTES_IN_PAGE: tl.constexpr,
):
    token_id = tl.program_id(0)
    loc = tl.load(loc_ptr + token_id)
    in_k_offsets = token_id * index_k_ptr_stride_0 + tl.arange(0, NUM_K_ELEMS_PER_TOKEN)
    # no need for `mask`, since we read 128B for k and 4B for scale, both pow of 2
    k = tl.load(index_k_ptr + in_k_offsets)
    k_scale = tl.load(index_k_scale_ptr + token_id)
    loc_page_index = loc // PAGE_SIZE
    loc_token_offset_in_page = loc % PAGE_SIZE
    out_k_offsets = (
        loc_page_index * BUF_NUMEL_PER_PAGE
        + loc_token_offset_in_page * NUM_K_ELEMS_PER_TOKEN
        + tl.arange(0, NUM_K_ELEMS_PER_TOKEN)
    )
    # "//4" b/c it is fp32 instead of uint8
    out_s_offset = (
        loc_page_index * BUF_NUMEL_PER_PAGE // 4
        + S_OFFSET_NBYTES_IN_PAGE // 4
        + loc_token_offset_in_page
    )
    tl.store(buf_fp8_ptr + out_k_offsets, k)
    tl.store(buf_fp32_ptr + out_s_offset, k_scale)

def _set_k_and_s_triton(
    buf: torch.Tensor,
    loc: torch.Tensor,
    index_k: torch.Tensor,
    index_k_scale: torch.Tensor,
    page_size: int,
):
    if index_k.dim() > 2:
        index_k = index_k.reshape(-1, index_k.size(-1))
    if index_k_scale.dim() > 2:
        index_k_scale = index_k_scale.reshape(-1, index_k_scale.size(-1))
    
    num_pages, buf_numel_per_page = buf.shape
    (num_tokens_to_write,) = loc.shape
    num_tokens_to_write_, index_head_dim = index_k.shape
    
    if index_k_scale.ndim == 1:
        num_tokens_to_write__ = index_k_scale.shape[0]
        scale_dim = 1
    elif index_k_scale.ndim == 2:
        num_tokens_to_write__, scale_dim = index_k_scale.shape
    else:
        raise ValueError(f"index_k_scale must be 1D or 2D, got shape {index_k_scale.shape}")
        
    if _is_hip and not _is_hcu:
        assert buf_numel_per_page == 1 * (128 + 4)
    else:
        assert buf_numel_per_page == 64 * (128 + 4)
        
    assert num_tokens_to_write == num_tokens_to_write_ == num_tokens_to_write__
    assert index_head_dim == 128
    assert scale_dim == 1
    
    if _is_hip and not _is_hcu:
        assert page_size == 1
    else:
        assert page_size == 64
        
    assert buf.dtype == torch.uint8
    assert loc.dtype == torch.int64, f"{loc.dtype=}"
    
    if _is_fp8_fnuz:
        assert index_k.dtype == torch.float8_e4m3fnuz
    else:
        assert index_k.dtype == torch.float8_e4m3fn
        
    assert index_k_scale.dtype == torch.float32
    assert buf.is_contiguous()
    assert loc.is_contiguous()
    assert index_k.is_contiguous()
    assert index_k_scale.is_contiguous()
    
    if _is_fp8_fnuz:
        buf_fp8 = buf.view(torch.float8_e4m3fnuz)
    else:
        buf_fp8 = buf.view(torch.float8_e4m3fn)
        
    buf_fp32 = buf.view(torch.float32)
    _set_k_and_s_triton_kernel[(num_tokens_to_write,)](
        buf_fp8,
        buf_fp32,
        loc,
        index_k,
        index_k_scale,
        index_k.stride(0),
        PAGE_SIZE=page_size,
        BUF_NUMEL_PER_PAGE=buf_numel_per_page,
        NUM_K_ELEMS_PER_TOKEN=index_head_dim,
        S_OFFSET_NBYTES_IN_PAGE=page_size * index_head_dim,
    )

# ==========================================
# 测试与 Benchmark 逻辑
# ==========================================
def run_benchmark():
    tokens_list = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536]
    group_size = 128
    page_size = 64
    
    print("\n| 序列长度 (Num Tokens) | 精度验证 (Scale真实误差 | FP8 ULP差异) | 基准耗时 (us) | 融合算子耗时 (us) | 加速比 (Speedup) |")
    print("| :--- | :--- | :--- | :--- | :--- |")
    
    for num_tokens in tokens_list:
        # 1. 准备随机数据
        key = torch.randn((num_tokens, group_size), dtype=torch.bfloat16, device="cuda")
        
        # 随机生成 loc 映射，保证在 page 的范围内
        num_pages = math.ceil(num_tokens / page_size)
        loc = torch.randperm(num_tokens, device="cuda").to(torch.int32)
        
        # Buffer 大小: 每页 64 个 Token，每个 Token = 128字节(K) + 4字节(Scale) = 132 Bytes
        buf_numel_per_page = page_size * (group_size + 4)
        
        # 分配 Buffer
        buf_ref = torch.zeros((num_pages, buf_numel_per_page), dtype=torch.uint8, device="cuda")
        buf_fused = torch.zeros_like(buf_ref)
        
        # 统一使用 int64 类型的 loc 以对齐 Triton 算子和修改后的底层 C++
        loc_int64 = loc.to(torch.int64)

        # ========================================
        # 精度验证 (Correctness Check)
        # ========================================
        # Baseline (现在完全基于 Triton)
        k_fp8, k_scale = act_quant(key, block_size=group_size, scale_fmt=None)
        _set_k_and_s_triton(buf_ref, loc_int64, k_fp8, k_scale, page_size)
        
        # Fused Operator
        op.fuse_act_quant_and_store_index_k_cache(
            key, 
            buf_fused, 
            loc_int64, 
            page_size, 
            1e-5,    # eps
            False,   # use_ue8m0
            True     # is_e4m3
        )
        
        # ----------------------------------------
        # 改进精度验证逻辑: 将 Scale 与 K-Cache 拆分计算
        # ----------------------------------------
        s_offset = page_size * group_size # 在物理页中，Scale从这个字节开始
        
        # 1. 提取 Scale 并转回 float32，计算真实数学误差
        scales_ref = buf_ref.view(torch.float32)[:, s_offset//4 :]
        scales_fused = buf_fused.view(torch.float32)[:, s_offset//4 :]
        scale_diff = (scales_ref - scales_fused).abs().max().item()
        
        # 2. 提取 FP8 数据(保留为 uint8)，比较二进制 ULP (Unit in Last Place) 误差
        k_ref_uint8 = buf_ref[:, :s_offset]
        k_fused_uint8 = buf_fused[:, :s_offset]
        k_byte_diff = int((k_ref_uint8.float() - k_fused_uint8.float()).abs().max().item())
        
        diff_str = f"S_err: {scale_diff:.1e} | FP8_ULP: {k_byte_diff}"

        # ========================================
        # 性能测试 (Performance Benchmark)
        # ========================================
        warmup_iters = 10
        bench_iters = 100
        
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        # ---- 基准性能测试 ----
        for _ in range(warmup_iters):
            k_fp8, k_scale = act_quant(key, block_size=group_size, scale_fmt=None)
            _set_k_and_s_triton(buf_ref, loc_int64, k_fp8, k_scale, page_size)
            
        torch.cuda.synchronize()
        start_event.record()
        for _ in range(bench_iters):
            k_fp8, k_scale = act_quant(key, block_size=group_size, scale_fmt=None)
            _set_k_and_s_triton(buf_ref, loc_int64, k_fp8, k_scale, page_size)
        end_event.record()
        torch.cuda.synchronize()
        base_time_us = start_event.elapsed_time(end_event) * 1000 / bench_iters

        # ---- 融合算子性能测试 ----
        for _ in range(warmup_iters):
            op.fuse_act_quant_and_store_index_k_cache(key, buf_fused, loc_int64, page_size, 1e-5, False, True)
            
        torch.cuda.synchronize()
        start_event.record()
        for _ in range(bench_iters):
            op.fuse_act_quant_and_store_index_k_cache(key, buf_fused, loc_int64, page_size, 1e-5, False, True)
        end_event.record()
        torch.cuda.synchronize()
        fused_time_us = start_event.elapsed_time(end_event) * 1000 / bench_iters

        # 加速比
        speedup = base_time_us / fused_time_us if fused_time_us > 0 else 0.0

        # 打印结果
        print(f"| {num_tokens} | {diff_str} | {base_time_us:.2f} | {fused_time_us:.2f} | {speedup:.2f}x |")

if __name__ == "__main__":
    # 为了避免 torch 初始化带来的初次额外耗时，先随便跑一个小 Kernel
    _ = torch.randn(10, device="cuda") * 2
    torch.cuda.synchronize()
    
    run_benchmark()