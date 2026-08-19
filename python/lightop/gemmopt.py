import torch
from typing import Optional, Dict, Any, List, NamedTuple
#from lightop import op
from . import op
import functools
import os
import math
from .config_smooth_gemm import gemm_w8a8_smooth_config
from .config_deepgemm import gemm_w8a8_deepgemm_masked_config
from .config import get_awq_w4a16_gemm_marlin_config


@functools.lru_cache(maxsize=128)
def w8a8_smooth_quant_config(m, n, k):
    """
    根据输入维度(m, n, k)返回最优的配置参数
    
    Args:
        m (int): 输入矩阵的行数
        n (int): 输出矩阵的列数  
        k (int): 输入矩阵的列数/权重矩阵的行数
        
    Returns:
        dict: 包含warps, nperblock, kperwarp, splitk的最优配置
    """
    # 基于tuning结果的最优配置字典
    # value为元组：(warps, nperblock, kperwarp, splitk, time_ms)
    
    # 查找匹配的配置
    key = f"M{m}N{n}K{k}"
    if key in gemm_w8a8_smooth_config.get_w8a8_smooth_quant_config_dict():
        warps, nperblock, kperwarp, splitk, use_n_warps = gemm_w8a8_smooth_config.get_w8a8_smooth_quant_config_dict()[key]
        status = True
        return warps, nperblock, kperwarp, splitk, use_n_warps,status,
    else:
        # 如果没有完全匹配的配置，返回默认配置，并提示警告
        # print(f"警告：未找到维度({m}, {n}, {k})的最优配置，已返回默认配置 (2, 32, 256, 1, False)。如需最佳性能，请先进行tuning。")
        status = False
        return -1, -1, -1, -1, False, status


def round_up_to_next_power_of_2(x):
    if x == 0:
        return 1  # 特殊情况：0 的向上取整是 1
    x -= 1  # 处理 x 已经是 2 的幂的情况
    # 填充所有低位为 1
    x |= x >> 1
    x |= x >> 2
    x |= x >> 4
    x |= x >> 8
    x |= x >> 16
    return x + 1


@functools.lru_cache(maxsize=2048)
def w8a8_smooth_asm_config(m, n, k):
    
    if m < 256:
        adapt_m = math.ceil(m / 16) * 16  # 向上取整到下一个16的倍数
    else:
        adapt_m = round_up_to_next_power_of_2(m)
    key = f"M{adapt_m}N{n}K{k}"
    if key in gemm_w8a8_smooth_config.get_gemm_w8a8_smooth_asm_dict():
        isTunning_slide_block, tunning_slide_block = True, gemm_w8a8_smooth_config.get_gemm_w8a8_smooth_asm_dict()[key]
    else:
        isTunning_slide_block, tunning_slide_block = False, 255
    return isTunning_slide_block, tunning_slide_block

@functools.lru_cache(maxsize=1)
def isNOt_exist_co_dir():
    path_co_file = os.getenv("LIGHTOP_ASM_DIR", "")
    if not os.path.exists(path_co_file):
        return True
    return False

def gemm_w8a8_smooth(a, b, scale_a, scale_b, bias, out_dtype):
    use_gemm_w8s8_smooth = os.getenv("LIGHTOP_GEMM_W8A8_SMOOTH", "1")
    if use_gemm_w8s8_smooth != "1":
        return False, None
    assert a.dim()==2 and b.dim()==2 and a.is_contiguous() and b.is_contiguous()==False
    m, n, k = a.shape[0], b.shape[1], a.shape[1]
    use_gemm_w8s8_smooth_print = os.getenv("LIGHTOP_GEMM_W8A8_SMOOTH_PRINT", "0")
    if use_gemm_w8s8_smooth_print == "1":
        print(f"[LIGHTOP] gemm_w8a8_smooth: m={m}, n={n}, k={k}, scale_a.shape={scale_a.shape}, scale_b.shape={scale_b.shape}, bias={bias}, out_dtype={out_dtype}")
    if m<=16:
        warps, nperblock, kperwarp, splitk, use_n_warps,status = w8a8_smooth_quant_config(m, n, k)
        if status is True:
            output = op.gemm_w8_a8_smooth_colmajor(a, b, scale_a, scale_b, bias, out_dtype, warps, nperblock, kperwarp, splitk, use_n_warps)
            return status, output
        else:
            return False, None
    else:
        if isNOt_exist_co_dir():
            return False, None
        if bias is not None:
            return False, None
        if k < 128:
            return False, None
        else:
            isTunning_slide_block, tunning_slide_block = w8a8_smooth_asm_config(m, n, k)
            output = op.gemm_w8a8_smooth_asm(a, b, scale_a, scale_b, bias, out_dtype,m, n, k, isTunning_slide_block, tunning_slide_block)
            return True, output


def gemm(a,b):
    assert a.dim()==2 and b.dim()==2 and a.is_contiguous()
    return op.gemm_opt(a,b)

def gemm_w4a16(a,b,scale_n,zero_n):
    assert a.dim()==2 and b.dim()==2 and a.is_contiguous() and b.is_contiguous() == False
    return op.gemm_w4_a16(a,b,scale_n,zero_n)

def w4tow16(a,scale_n,zero_n):
    return op.w4_to_w16(a,scale_n,zero_n)

def gemm_w8a16_fp8(a,b,scale,warps=0,n_per_block=0,splitk=0):
    assert a.dim()==2 and b.dim()==2 and a.is_contiguous() and b.is_contiguous()==False
    return op.gemm_w8_a16_fp8_colmajor(a,b,scale,warps,n_per_block,splitk)

def fp8_to_16(out,intput,scale):
    return op.fp8to16(out,intput,scale)

def moe_gemm_w4a8(input: torch.Tensor,
                    b_qweight: torch.Tensor,
                    output: torch.Tensor,
                    a_scale: torch.Tensor,
                    b_scale: torch.Tensor,
                    topk_weights: Optional[torch.Tensor],
                    sorted_token_ids: torch.Tensor,
                    experts_ids: torch.Tensor,
                    num_tokens_post_pad: torch.Tensor, 
                    top_k: int,
                    config: Dict[str, Any]) -> torch.Tensor:
    mode = config['MODE']
    delta = config['DELTA']

    if mode < 1000:
        op.moe_gemm_w4a8(input, b_qweight, output, a_scale, b_scale, topk_weights,
                                        sorted_token_ids, experts_ids, num_tokens_post_pad, top_k, mode, delta)
    else:
        op.moe_w4a8_asm(input, b_qweight, output, a_scale, b_scale, topk_weights,
                                        sorted_token_ids, experts_ids, num_tokens_post_pad, top_k, mode)
    return output

def moe_gemm_marlin_w4a8(
                    input: torch.Tensor,
                    b_qweight: torch.Tensor,
                    output: torch.Tensor,
                    a_scale: torch.Tensor,
                    b_scale: torch.Tensor,
                    topk_weights: Optional[torch.Tensor],
                    sorted_token_ids: torch.Tensor,
                    experts_ids: torch.Tensor,
                    num_tokens_post_pad: torch.Tensor, 
                    top_k: int,
                    locks: Optional[torch.Tensor], # 用于控制 block间归约的 信号量Tensor
                    global_reduce_buffer: Optional[torch.Tensor], # 用于 block间归约的 缓存区Tensor
                    config: Dict[str, Any],
                    is_ep:bool = False) -> torch.Tensor:
    mode = config['MODE']
    delta = config.get('DELTA', 1)

    if mode < 500: # 手写算子实现
        op.moe_gemm_w4a8(input, b_qweight, output, a_scale, b_scale, topk_weights,
                                        sorted_token_ids, experts_ids, num_tokens_post_pad, top_k, mode, delta)
    elif mode >= 500 and mode < 1000: # marlin版 w4a8 算子实现
        block_m = config["BLOCK_SIZE_M"]
        size_m = input.shape[0]
        size_k = input.shape[1]
        size_n = b_qweight.shape[2] // 4

        thread_n = config["thread_n"]
        thread_k = config["thread_k"]
        blocks_per_cu = config["blocks_per_cu"]
        is_reduce = config.get("is_reduce", False)
        op.moe_marlin_w4a8(input,  # A
                                a_scale, # a_scale
                                output, # 输出
                                b_qweight, # B 
                                b_scale, # B_scale
                                locks,
                                global_reduce_buffer,
                                sorted_token_ids, 
                                experts_ids, 
                                num_tokens_post_pad, 
                                topk_weights,
                                block_m,
                                top_k,
                                is_ep,
                                size_m,
                                size_n,
                                size_k,
                                thread_n,
                                thread_k,
                                blocks_per_cu,
                                is_reduce)  

    else: # 汇编版 w4a8 实现
        op.moe_w4a8_marlin_asm(input, b_qweight, output, a_scale, b_scale, topk_weights,
                                        sorted_token_ids, experts_ids, num_tokens_post_pad, top_k, mode)
    return output

def moe_gemm_marlin_w4a16(
                    input: torch.Tensor,
                    b_qweight: torch.Tensor,
                    output: torch.Tensor,
                    b_scale_zero: torch.Tensor,
                    topk_weights: Optional[torch.Tensor],
                    sorted_token_ids: torch.Tensor,
                    experts_ids: torch.Tensor,
                    num_tokens_post_pad: torch.Tensor, 
                    top_k: int,
                    locks: Optional[torch.Tensor], # 用于控制 block间归约的 信号量Tensor
                    global_reduce_buffer: Optional[torch.Tensor], # 用于 block间归约的 缓存区Tensor
                    config: Dict[str, Any],
                    is_ep:bool = False) -> torch.Tensor:
    mode = config['MODE']
    delta = config['DELTA']

    block_m = config["BLOCK_SIZE_M"]
    size_m = input.shape[0]
    size_k = input.shape[1]
    size_n = b_qweight.shape[2] // 4

    thread_n = config["thread_n"]
    thread_k = config["thread_k"]
    blocks_per_cu = config["blocks_per_cu"]
    is_reduce = config.get("is_reduce", False)
    op.moe_marlin_w4a8(
                        input,  # A
                        a_scale, # a_scale
                        output, # 输出
                        b_qweight, # B 
                        b_scale, # B_scale
                        locks,
                        global_reduce_buffer,
                        sorted_token_ids, 
                        experts_ids, 
                        num_tokens_post_pad, 
                        topk_weights,
                        block_m,
                        top_k,
                        is_ep,
                        size_m,
                        size_n,
                        size_k,
                        thread_n,
                        thread_k,
                        blocks_per_cu,
                    )  

    # if mode < 500: # 手写算子实现
    #     op.moe_gemm_w4a8(input, b_qweight, output, a_scale, b_scale, topk_weights,
    #                                     sorted_token_ids, experts_ids, num_tokens_post_pad, top_k, mode, delta)
    # elif mode >= 500 and mode < 1000: # marlin版 w4a8 算子实现
    #     block_m = config["BLOCK_SIZE_M"]
    #     size_m = input.shape[0]
    #     size_k = input.shape[1]
    #     size_n = b_qweight.shape[2] // 4

    #     thread_n = config["thread_n"]
    #     thread_k = config["thread_k"]
    #     blocks_per_cu = config["blocks_per_cu"]
    #     is_reduce = config.get("is_reduce", False)
    #     op.moe_marlin_w4a8(input,  # A
    #                             a_scale, # a_scale
    #                             output, # 输出
    #                             b_qweight, # B 
    #                             b_scale, # B_scale
    #                             locks,
    #                             global_reduce_buffer,
    #                             sorted_token_ids, 
    #                             experts_ids, 
    #                             num_tokens_post_pad, 
    #                             topk_weights,
    #                             block_m,
    #                             top_k,
    #                             is_ep,
    #                             size_m,
    #                             size_n,
    #                             size_k,
    #                             thread_n,
    #                             thread_k,
    #                             blocks_per_cu,
    #                             is_reduce)  

    # else: # 汇编版 w4a8 实现
    #     op.moe_w4a8_marlin_asm(input, b_qweight, output, a_scale, b_scale, topk_weights,
    #                                     sorted_token_ids, experts_ids, num_tokens_post_pad, top_k, mode)
    return output

def moe_gemm_marlin_w16a16(
                    input: torch.Tensor,
                    b_qweight: torch.Tensor,
                    output: torch.Tensor,
                    topk_weights: Optional[torch.Tensor],
                    sorted_token_ids: torch.Tensor,
                    experts_ids: torch.Tensor,
                    num_tokens_post_pad: torch.Tensor, 
                    top_k: int,
                    config: Dict[str, Any],
                    is_ep:bool = False) -> torch.Tensor:
    mode = config['MODE']
    delta = config.get('DELTA', 1)

    if mode < 1000:
        op.moe_gemm_w16a16(input, b_qweight, output,  topk_weights,
                                            sorted_token_ids, experts_ids, num_tokens_post_pad, top_k, mode, delta)
    else:
        op.moe_marlin_w16a16_asm(input, b_qweight, output,  topk_weights,
                                            sorted_token_ids, experts_ids, num_tokens_post_pad, top_k, mode, delta)
    return output

def moe_gemm_w16a16(
                    input: torch.Tensor,
                    b_qweight: torch.Tensor,
                    output: torch.Tensor,
                    topk_weights: Optional[torch.Tensor],
                    sorted_token_ids: torch.Tensor,
                    experts_ids: torch.Tensor,
                    num_tokens_post_pad: torch.Tensor, 
                    top_k: int,
                    config: Dict[str, Any],
                    is_ep:bool = False) -> torch.Tensor:
    mode = config['MODE']
    delta = config['DELTA']

    if mode < 1000:
        op.moe_gemm_w16a16(input, b_qweight, output,  topk_weights,
                                            sorted_token_ids, experts_ids, num_tokens_post_pad, top_k, mode, delta)
    else:
        op.moe_w16a16_asm(input, b_qweight, output,  topk_weights,
                                            sorted_token_ids, experts_ids, num_tokens_post_pad, top_k, mode, delta)
    return output

def moe_gemm_w8a8(input: torch.Tensor,
                    b_qweight: torch.Tensor,
                    output: torch.Tensor,
                    a_scale: torch.Tensor,
                    b_scale: torch.Tensor,
                    topk_weights: Optional[torch.Tensor],
                    sorted_token_ids: torch.Tensor,
                    experts_ids: torch.Tensor,
                    num_tokens_post_pad: torch.Tensor, 
                    top_k: int,
                    config: Dict[str, Any]) -> torch.Tensor:
    mode = config['MODE']
    delta = config['DELTA']

    if mode < 1000:
        op.moe_gemm_w8a8(input, b_qweight, output, a_scale, b_scale, topk_weights,
                                        sorted_token_ids, experts_ids, num_tokens_post_pad, top_k, mode, delta)
    else:
        op.moe_w8a8_asm(input, b_qweight, output, a_scale, b_scale, topk_weights,
                                        sorted_token_ids, experts_ids, num_tokens_post_pad, top_k, mode, delta)
    return output

def moe_gemm_marlin_w8a8(input: torch.Tensor,
                    b_qweight: torch.Tensor,
                    output: torch.Tensor,
                    a_scale: torch.Tensor,
                    b_scale: torch.Tensor,
                    topk_weights: Optional[torch.Tensor],
                    sorted_token_ids: torch.Tensor,
                    experts_ids: torch.Tensor,
                    num_tokens_post_pad: torch.Tensor, 
                    top_k: int,
                    config: Dict[str, Any]) -> torch.Tensor:
    mode = config['MODE']
    delta = config['DELTA']

    if mode < 1000:
        op.moe_gemm_marlin_w8a8(input, b_qweight, output, a_scale, b_scale, topk_weights,
                                        sorted_token_ids, experts_ids, num_tokens_post_pad, top_k, mode, delta)
    else:
        op.moe_marlin_w8a8_asm(input, b_qweight, output, a_scale, b_scale, topk_weights,
                                        sorted_token_ids, experts_ids, num_tokens_post_pad, top_k, mode, delta)
    return output

def moe_gemm_marlin_w8a8_fp8(input: torch.Tensor,
                    b_qweight: torch.Tensor,
                    output: torch.Tensor,
                    a_scale: torch.Tensor,
                    b_scale: torch.Tensor,
                    topk_weights: Optional[torch.Tensor],
                    sorted_token_ids: torch.Tensor,
                    experts_ids: torch.Tensor,
                    num_tokens_post_pad: torch.Tensor, 
                    top_k: int,
                    config: Dict[str, Any]) -> torch.Tensor:
    mode = config['MODE']
    delta = config['DELTA']

    if mode < 1000:
        op.moe_gemm_marlin_w8a8_fp8(input, b_qweight, output, a_scale, b_scale, topk_weights,
                                        sorted_token_ids, experts_ids, num_tokens_post_pad, top_k, mode, delta)
    else:
        op.moe_marlin_w8a8_fp8_asm(input, b_qweight, output, a_scale, b_scale, topk_weights,
                                        sorted_token_ids, experts_ids, num_tokens_post_pad, top_k, mode, delta)
    return output

def moe_gemm_marlin_w8a8_blockwise_fp8(input: torch.Tensor,
                    b_qweight: torch.Tensor,
                    output: torch.Tensor,
                    a_scale: torch.Tensor,
                    b_scale: torch.Tensor,
                    topk_weights: Optional[torch.Tensor],
                    sorted_token_ids: torch.Tensor,
                    experts_ids: torch.Tensor,
                    num_tokens_post_pad: torch.Tensor, 
                    top_k: int,
                    config: Dict[str, Any]) -> torch.Tensor:
    mode = config['MODE']
    delta = config['DELTA']

    op.moe_gemm_marlin_w8a8_blockwise_fp8(input, b_qweight, output, a_scale, b_scale, topk_weights,
                                        sorted_token_ids, experts_ids, num_tokens_post_pad, top_k, mode, delta)

    # if mode < 1000:
    #     op.moe_gemm_marlin_w8a8_blockwise_fp8(input, b_qweight, output, a_scale, b_scale, topk_weights,
    #                                     sorted_token_ids, experts_ids, num_tokens_post_pad, top_k, mode, delta)
    # else:
    #     op.moe_marlin_w8a8_fp8_asm(input, b_qweight, output, a_scale, b_scale, topk_weights,
    #                                     sorted_token_ids, experts_ids, num_tokens_post_pad, top_k, mode, delta)
    return output

def moe_groupgemm_w4a8(input: torch.Tensor,
                        b_qweight: torch.Tensor,
                        output: torch.Tensor,
                        a_scale: torch.Tensor,
                        b_scale: torch.Tensor,
                        problem_size: torch.Tensor,
                        experts_offsets: torch.Tensor,
                        config: Dict[str, Any]) -> torch.Tensor:

    mode = config['MODE']
    op.moe_groupgemm_w4a8_asm(input, b_qweight, output, a_scale, b_scale,
                                        problem_size, experts_offsets, mode)
    return output

def moe_groupgemm_marlin_w4a8(input: torch.Tensor,
                               b_qweight: torch.Tensor,
                               output: torch.Tensor,
                               a_scale: torch.Tensor,
                               b_scale: torch.Tensor,
                               problem_size: torch.Tensor,
                               experts_offsets: torch.Tensor,
                               config: Dict[str, Any]) -> torch.Tensor:

    mode = config['MODE']
    op.moe_groupgemm_marlin_w4a8_asm(input, b_qweight, output, a_scale, b_scale,
                                        problem_size, experts_offsets, mode)
    return output


@functools.lru_cache(maxsize=2048)
def w8a8_deepgemm_masked_asm_config(m):
    
    if m < 16:
        adapt_m = 16  # 向上取整到下一个16的倍数
    else:
        adapt_m = round_up_to_next_power_of_2(m)
    key = f"EXPECTED_M{adapt_m}"
    if key in gemm_w8a8_deepgemm_masked_config.get_deepgemm_w8a8_smooth_asm_dict():
        MODE = gemm_w8a8_deepgemm_masked_config.get_deepgemm_w8a8_smooth_asm_dict()[key]
    else:
        MODE = 1000
    return MODE


def m_grouped_w4a8_gemm_nt_masked(a: tuple[torch.Tensor, torch.Tensor],
                                  b: tuple[torch.Tensor, torch.Tensor],
                                  d: torch.Tensor,
                                  masked_m: torch.Tensor,
                                  expected_m_per_group: int,
                                  config: Optional[Dict[str, Any]] = {"MODE": 1000,}) -> torch.Tensor:

    mode = config['MODE']
    input, a_scale = a
    b_qweight, b_scale = b
    output = d
    op.m_grouped_marlin_w4a8_gemm_nt_masked_asm(input, b_qweight, output, a_scale, b_scale,
                                        masked_m, expected_m_per_group, mode)
    return output


def m_grouped_fp8_gemm_nt_masked(a: tuple[torch.Tensor, torch.Tensor],
                                  b: tuple[torch.Tensor, torch.Tensor],
                                  d: torch.Tensor,
                                  masked_m: torch.Tensor,
                                  expected_m_per_group: int,
                                  config: Optional[Dict[str, Any]] = {"MODE": 1000,}) -> torch.Tensor:

    mode = w8a8_deepgemm_masked_asm_config(expected_m_per_group)
    input, a_scale = a
    b_qweight, b_scale = b
    output = d
    op.m_grouped_marlin_fp8_gemm_nt_masked_asm(input, b_qweight, output, a_scale, b_scale,
                                        masked_m, expected_m_per_group, mode)
    return output


def m_grouped_w8a8_gemm_nt_masked(a: tuple[torch.Tensor, torch.Tensor],
                                  b: tuple[torch.Tensor, torch.Tensor],
                                  d: torch.Tensor,
                                  masked_m: torch.Tensor,
                                  expected_m_per_group: int,
                                  config: Optional[Dict[str, Any]] = {"MODE": 1000,}) -> torch.Tensor:

    mode = w8a8_deepgemm_masked_asm_config(expected_m_per_group)
    input, a_scale = a
    b_qweight, b_scale = b
    output = d
    op.m_grouped_marlin_w8a8_gemm_nt_masked_asm(input, b_qweight, output, a_scale, b_scale,
                                        masked_m, expected_m_per_group, mode)
    return output


def m_grouped_bf16_gemm_nt_masked(a: torch.Tensor,
                                  b: torch.Tensor,
                                  d: torch.Tensor,
                                  masked_m: torch.Tensor,
                                  expected_m_per_group: int,
                                  config: Optional[Dict[str, Any]] = {"MODE": 1000,}) -> torch.Tensor:

    # mode = w8a8_deepgemm_masked_asm_config(expected_m_per_group)
    mode = config['MODE']
    op.m_grouped_bf16_gemm_nt_masked_asm(a, b, d, masked_m, expected_m_per_group, mode)
    return d


def gemm_w8a8_asm(a, b, scale_a, scale_b, block_size, out_dtype, layout = 'TN'):
    assert a.dim()==2 and b.dim()==2 and a.is_contiguous() and b.is_contiguous()
    assert block_size[0] == 128 and block_size[1] == 128
    m, n, k = a.shape[0], b.shape[0], a.shape[1]
    # print(f"gemm_w8a8_block: m={m}, n={n}, k={k}, scale_a.shape={scale_a.shape}, scale_b.shape={scale_b.shape}, out_dtype={out_dtype}")
    
    return op.gemm_w8a8_asm(a, b, scale_a, scale_b, None, None, None, out_dtype, m, n, k, 0)

def gemm_w8a8_wgrad_asm(a, b, scale_a, scale_b, c, accumulate, block_size, out_dtype, layout = 'TN'):
    assert a.dim()==2 and b.dim()==2 and a.is_contiguous() and b.is_contiguous() and scale_a.is_contiguous and scale_b.is_contiguous
    assert block_size[0] == 128 and block_size[1] == 128
    m, n, k = a.shape[0], b.shape[0], a.shape[1]
    # print(f"gemm_w8a8_block: m={m}, n={n}, k={k}, scale_a.shape={scale_a.shape}, scale_b.shape={scale_b.shape}, out_dtype={out_dtype}")
    
    if accumulate:
        return op.gemm_w8a8_asm(a, b, scale_a, scale_b, c, c, None, out_dtype, m, n, k, 1)
    return op.gemm_w8a8_asm(a, b, scale_a, scale_b, None, None, None, out_dtype, m, n, k, 1)

def gemm_w8a8_xgrad_asm(a, b, scale_a, scale_b, block_size, out_dtype, layout = 'TN'):
    assert a.dim()==2 and b.dim()==2 and a.is_contiguous() and b.is_contiguous()
    assert block_size[0] == 128 and block_size[1] == 128
    m, n, k = a.shape[0], b.shape[1], a.shape[1]
    # print(f"gemm_w8a8_block: m={m}, n={n}, k={k}, scale_a.shape={scale_a.shape}, scale_b.shape={scale_b.shape}, out_dtype={out_dtype}")
    # assert scale_a.shape[0] == ((math.ceil(k / block_size[0]) +3) / 4 * 4) and scale_a.shape[1] == m
    # assert scale_b.shape[1] == ((math.ceil(n / block_size[1]) + 3) / 4 * 4) and scale_b.shape[0] == math.ceil(k / block_size[0])
    
    return op.gemm_w8a8_asm(a, b, scale_a, scale_b, None, None, None, out_dtype, m, n, k, 2)




def moe_marlin_w4a16(a: torch.Tensor,
                    c_or_none: Optional[torch.Tensor],
                    b_q_weight: torch.Tensor,
                    b_scale_zeros: torch.Tensor,
                    g_idx_or_none: Optional[torch.Tensor],
                    perm_or_none: Optional[torch.Tensor],
                    workspace: torch.Tensor,
                    sorted_token_ids: torch.Tensor,
                    expert_ids: torch.Tensor,
                    num_tokens_past_padded: torch.Tensor,
                    topk_weights: torch.Tensor,
                    moe_block_size: int,
                    top_k: int,
                    mul_topk_weights: bool,
                    is_ep: bool,
                    size_m: int,
                    size_n: int,
                    size_k: int,
                    is_k_full: bool,
                    use_atomic_add: bool,
                    use_fp32_reduce: bool,
                    is_zp_float: bool,
                    thread_n: int = -1,
                    thread_k: int = -1,
                    blocks_per_cu: int = -1
                    ) -> torch.Tensor:
    output = op.moe_marlin_w4a16(a, 
                        c_or_none, 
                        b_q_weight, 
                        b_scale_zeros, 
                        g_idx_or_none, 
                        perm_or_none, 
                        workspace, 
                        sorted_token_ids, 
                        expert_ids,
                        num_tokens_past_padded,
                        topk_weights,
                        moe_block_size,
                        top_k,
                        mul_topk_weights,
                        is_ep,
                        size_m,
                        size_n,
                        size_k,
                        is_k_full,
                        use_atomic_add,
                        use_fp32_reduce,
                        is_zp_float,
                        thread_n,
                        thread_k,
                        blocks_per_cu
                        )
    return output

def awq_marlin_repack(b_q_weight: torch.Tensor, size_k: int, size_n: int, num_bits: int) -> torch.Tensor:
    output = op.awq_marlin_repack(b_q_weight, size_k, size_n, num_bits)
    return output

def awq_marlin_repack_w4a8(b_q_weight: torch.Tensor, marlin_weight: torch.Tensor, size_batch:int, size_k: int, size_n: int) -> torch.Tensor:
    output = op.awq_marlin_repack_w4a8(b_q_weight, marlin_weight, size_batch, size_k, size_n)
    return output

def grougemm_w8a8_pertensor_asm(A_list:list[torch.Tensor], B_list:list[torch.Tensor], D_list:list[torch.Tensor], As_list:list[torch.Tensor], Bs_list:list[torch.Tensor], 
                                alphas:list[float], C_list:Optional[list[torch.Tensor]], betas:Optional[list[float]], out_dtype):
    return op.grougemm_w8a8_pertensor_asm(A_list, B_list, D_list, As_list, Bs_list, alphas, C_list, betas, out_dtype)


def grougemm_bf16_fp16(A_list:list[torch.Tensor], B_list:list[torch.Tensor], D_list:list[torch.Tensor],
                  alphas:list[float], C_list:Optional[list[torch.Tensor]], betas:Optional[list[float]], out_dtype, trans_a:bool=False, trans_b:bool=True):
    if trans_a is False and trans_b is True:
        return op.grougemm_bf16_fp16_nt_asm(A_list, B_list, D_list, alphas, C_list, betas, out_dtype)   
    elif trans_a is True and trans_b is False:
        return op.grougemm_bf16_fp16_tn_asm(A_list, B_list, D_list, alphas, C_list, betas, out_dtype)
    elif trans_a is False and trans_b is False:
        return op.grougemm_bf16_fp16_nn_asm(A_list, B_list, D_list, alphas, C_list, betas, out_dtype)
    else:
       raise ValueError(f" GROUPGEMM Unsupported Layout trans_a : {trans_a}, trans_b : {trans_b}")



def m_grouped_w8a8_gemm_nt_contig_asm( a: tuple[torch.Tensor, torch.Tensor],
                                       b: tuple[torch.Tensor, torch.Tensor], 
                                       output:torch.Tensor, m_indices:torch.Tensor, 
                                       config: Optional[Dict[str, Any]] = {"MODE": 1000,}) -> torch.Tensor:
    mode = config['MODE']
    input, a_scale = a
    b_qweight, b_scale = b
    return op.m_grouped_w8a8_gemm_nt_contig_asm(input, b_qweight, output, a_scale, b_scale, m_indices, mode)

def m_grouped_w16a16_gemm_nt_contig_asm(a:torch.Tensor, b:torch.Tensor, output:torch.Tensor, m_indices:torch.Tensor,
                                        config: Optional[Dict[str, Any]] = {"MODE": 1000,}) -> torch.Tensor:
    mode = config['MODE']
    M = a.shape[0]
    E, N = b.shape[0],b.shape[1]
    a_scale = torch.ones(M, dtype=torch.float32, device=a.device)
    b_scale = torch.ones((E,N),dtype=torch.float32, device=b.device)
    return op.m_grouped_w16a16_gemm_nt_contig_asm(a, b, output, a_scale, b_scale, m_indices, mode)

def m_grouped_w8a8_gemm_nt_nopad_contig_asm( a: tuple[torch.Tensor, torch.Tensor],
                                             b: tuple[torch.Tensor, torch.Tensor],
                                             output:torch.Tensor, m_indices:torch.Tensor,
                                             token_per_expert:torch.Tensor) -> torch.Tensor:
    input, a_scale = a
    b_qweight, b_scale = b
    return op.m_grouped_w8a8_gemm_nt_nopad_contig_asm(input, b_qweight, output, a_scale, b_scale, m_indices, token_per_expert)

def groupgemm(a: torch.Tensor, b: torch.Tensor, batch_sizes: torch.Tensor, trans_b: bool = False) -> torch.Tensor:
    outs = op.groupgemm(a, b, batch_sizes, trans_b)
    return outs

def mqa_logits(
    Q: torch.Tensor, 
    K: torch.Tensor, 
    Weights: torch.Tensor, 
    ks: torch.Tensor, 
    ke: torch.Tensor, 
    kv_scale: Optional[torch.Tensor] = None,
    clean_logit: bool = True,
    D_out: Optional[torch.Tensor] = None
) -> torch.Tensor:
    q_seq_len, num_heads, head_dim = Q.shape
    kv_seq_len = K.shape[0]
    
    assert num_heads in [1, 2, 4, 8, 16, 32, 64, 128], "num_heads only support [1, 2, 4, 8, 16, 32, 64, 128]"
    assert head_dim == 128, "head_dim only support 128"
    
    outs = op.mqa_logits(
        Q, K, Weights, ks, ke, 
        q_seq_len, kv_seq_len, num_heads, head_dim, 
        kv_scale, clean_logit, D_out
    )
    return outs


def get_paged_mqa_logits_metadata(
        context_lens: torch.Tensor,
        block_kv: int,
        num_sms: int) -> torch.Tensor:
    schedule_metadata = op.get_paged_mqa_logits_metadata(
        context_lens,
        block_kv,
        num_sms)
    return schedule_metadata

@functools.lru_cache(maxsize=16)
def _paged_mqa_device_info(device_index: int):
    props = torch.cuda.get_device_properties(device_index)
    arch = getattr(props, "gcnArchName", props.name).split(":")[0]
    return arch, props.multi_processor_count


def _use_page_mqa_logits_sparse_mask(
        q: torch.Tensor,
        fused_kv_cache: torch.Tensor,
        weights: torch.Tensor,
        context_lens: torch.Tensor,
        block_table: torch.Tensor,
        schedule_meta: Optional[torch.Tensor],
        max_context_len: int,
        clean_logits: bool) -> bool:
    if not q.is_cuda or q.dim() != 4 or q.shape[0] == 0:
        return False
    if q.shape[1] != 1 or q.shape[3] != 128 or not q.is_contiguous():
        return False
    if fused_kv_cache.dim() != 4 or fused_kv_cache.shape[0] == 0:
        return False
    if fused_kv_cache.shape[1:3] != (64, 1) or not fused_kv_cache.is_contiguous():
        return False
    batch_size, _, num_heads, _ = q.shape
    if weights.dtype != torch.float32 or weights.shape != (batch_size, num_heads):
        return False
    if not weights.is_contiguous():
        return False
    if context_lens.dtype != torch.int32 or context_lens.shape != (batch_size,):
        return False
    if not context_lens.is_contiguous():
        return False
    if block_table.dtype != torch.int32 or block_table.dim() != 2:
        return False
    if block_table.shape[0] != batch_size or block_table.stride(1) != 1:
        return False
    if max_context_len <= 0 or max_context_len != block_table.shape[1] * 64:
        return False
    if not clean_logits:
        return False
    if schedule_meta is not None:
        return False

    arch, num_cus = _paged_mqa_device_info(q.device.index)
    fp8_q_dtypes = {torch.float8_e4m3fn}
    fp8_fnuz_dtype = getattr(torch, "float8_e4m3fnuz", None)
    if fp8_fnuz_dtype is not None:
        fp8_q_dtypes.add(fp8_fnuz_dtype)
    supported_device = (arch, num_cus) in (
        ("gfx936", 80),
        ("gfx938", 64),
    )
    supported_heads = num_heads in (8, 16, 32, 64)
    bf16_path = (
        q.dtype == torch.bfloat16
        and fused_kv_cache.dtype == torch.bfloat16
        and fused_kv_cache.shape[3] == 128
    )
    packed_fp8_path = (
        q.dtype in fp8_q_dtypes | {torch.bfloat16}
        and fused_kv_cache.dtype == torch.uint8
        and fused_kv_cache.shape[3] == 132
    )
    if not (supported_device and supported_heads and
            (bf16_path or packed_fp8_path)):
        return False

    inputs = (
        fused_kv_cache,
        weights,
        context_lens,
        block_table,
    )
    if any(not tensor.is_cuda or tensor.device != q.device
           for tensor in inputs):
        return False

    return True


def paged_mqa_logits(
        q: torch.Tensor,
        fused_kv_cache: torch.Tensor,
        weights: torch.Tensor,
        context_lens: torch.Tensor,
        block_table: torch.Tensor,
        schedule_meta: torch.Tensor,
        max_context_len: int,
        clean_logits: bool = True) -> torch.Tensor:
    if (fused_kv_cache.dtype == torch.uint8
        and q.dtype == torch.float8_e4m3fn):
        arch = torch.cuda.get_device_properties(q.device).gcnArchName.split(":")[0]
        if "gfx936" in arch:
            q = q.to(torch.bfloat16)
    logits = op.paged_mqa_logits(
        q,
        fused_kv_cache,
        weights,
        context_lens,
        block_table,
        schedule_meta,
        max_context_len,
        clean_logits)
    return logits


def page_mqa_logits_sparse_mask(
        q: torch.Tensor,
        fused_kv_cache: torch.Tensor,
        weights: torch.Tensor,
        context_lens: torch.Tensor,
        block_table: torch.Tensor,
        schedule_meta: Optional[torch.Tensor],
        max_context_len: int,
        clean_logits: bool = True,
        num_warps: int = 4,
        persistent_ctas: int = 0):
    """Return sparse paged-MQA values and a one-bit-per-token mask.

    The values tensor has the same logical shape as the dense paged-MQA output.
    Only positions whose corresponding mask bit is one are initialized. The
    int16 mask packs 16 consecutive positions into each element, so it must be
    consumed by ``fast_topk_transform_sparse_mask_fused`` rather than by a
    dense logits consumer.
    """

    if not _use_page_mqa_logits_sparse_mask(
            q, fused_kv_cache, weights, context_lens, block_table,
            schedule_meta, max_context_len, clean_logits):
        raise ValueError(
            "page_mqa_logits_sparse_mask requires a positive batch, heads "
            "8/16/32/64, BF16 or packed FP8 KV, and "
            "gfx936/80CU or gfx938/64CU with schedule_meta=None")
    arch, num_cus = _paged_mqa_device_info(q.device.index)
    # Keep the validated legacy kernel for its tuned shapes. This is a kernel
    # selection, not an input gate; every other supported batch falls through
    # to the independent-row (group_size=1) implementation below.
    use_legacy_kernel = (
        q.dtype == torch.bfloat16
        and fused_kv_cache.dtype == torch.bfloat16
        and q.shape[0] in (8, 40)
        and q.shape[2] == 64
        and (arch, num_cus) == ("gfx936", 80)
    )
    if use_legacy_kernel:
        return op.page_mqa_logits_sparse_mask(
            q,
            fused_kv_cache,
            weights,
            context_lens,
            block_table,
            None,
            max_context_len,
            clean_logits,
            num_warps,
            persistent_ctas)
    return page_mqa_logits_sparse_mask_grouped(
        q,
        fused_kv_cache,
        weights,
        context_lens,
        block_table,
        None,
        max_context_len,
        clean_logits,
        num_warps,
        persistent_ctas,
        group_size=1)


def page_mqa_logits_sparse_mask_grouped(
        q: torch.Tensor,
        fused_kv_cache: torch.Tensor,
        weights: torch.Tensor,
        context_lens: torch.Tensor,
        block_table: torch.Tensor,
        schedule_meta: Optional[torch.Tensor],
        max_context_len: int,
        clean_logits: bool = True,
        num_warps: int = 4,
        persistent_ctas: int = 0,
        group_size: int = 5,
        *,
        _int8_kv: bool = False):
    """Sparse MQA for independent rows or grouped MTP2/3/4 rows.

    For ``group_size=1`` each row is independent. For ``group_size=3/4/5``,
    consecutive rows must belong to one request and have identical block-table
    contents, corresponding to MTP2/3/4. The returned sparse values/mask
    contract is identical to :func:`page_mqa_logits_sparse_mask`. Supported
    heads are 8/16/32/64. The FP8 path accepts the packed uint8 KV+scale layout
    used by :func:`paged_mqa_logits`.
    """

    if group_size not in (1, 3, 4, 5):
        raise ValueError("sparse MQA supports group_size 1, 3, 4, or 5")
    fp8_q_dtypes = {torch.float8_e4m3fn}
    fp8_fnuz_dtype = getattr(torch, "float8_e4m3fnuz", None)
    if fp8_fnuz_dtype is not None:
        fp8_q_dtypes.add(fp8_fnuz_dtype)
    if fused_kv_cache.dtype == torch.uint8 and q.dtype in fp8_q_dtypes:
        q = q.to(torch.bfloat16)
    expected_batch = q.shape[0] if q.dim() > 0 else 0
    if expected_batch <= 0:
        raise ValueError("sparse MQA requires a positive batch")
    if group_size != 1 and expected_batch % group_size != 0:
        raise ValueError(
            "grouped sparse MQA requires batch divisible by group_size")
    if (q.dim() != 4 or q.shape[0] != expected_batch or q.shape[1] != 1 or
            q.shape[2] not in (8, 16, 32, 64) or q.shape[3] != 128 or
            q.dtype != torch.bfloat16 or not q.is_contiguous()):
        raise ValueError(
            "sparse MQA requires contiguous BF16 q "
            "[batch, 1, heads, 128], heads in 8/16/32/64")
    if fused_kv_cache.dtype not in (torch.bfloat16, torch.uint8):
        raise ValueError("grouped sparse MQA supports BF16 or packed FP8 KV")
    if _int8_kv and fused_kv_cache.dtype != torch.uint8:
        raise ValueError("grouped sparse INT8 MQA requires packed uint8 KV+scale")
    expected_kv_width = 132 if fused_kv_cache.dtype == torch.uint8 else 128
    if (fused_kv_cache.dim() != 4 or fused_kv_cache.shape[0] == 0 or
            tuple(fused_kv_cache.shape[1:]) != (64, 1, expected_kv_width) or
            fused_kv_cache.stride(1) != expected_kv_width or
            fused_kv_cache.stride(2) != expected_kv_width or
            fused_kv_cache.stride(3) != 1):
        raise ValueError(
            "grouped sparse MQA requires packed BF16 [blocks,64,1,128] "
            "or FP8+scale uint8 [blocks,64,1,132] KV")
    if (weights.dtype != torch.float32 or
            weights.shape != (expected_batch, q.shape[2]) or
            not weights.is_contiguous()):
        raise ValueError("grouped sparse MQA weights must be contiguous FP32 [batch, heads]")
    if (context_lens.dtype != torch.int32 or
            context_lens.shape != (expected_batch,) or
            not context_lens.is_contiguous()):
        raise ValueError("grouped sparse MQA context_lens must be contiguous int32 [batch]")
    if (block_table.dtype != torch.int32 or block_table.dim() != 2 or
            block_table.shape[0] != expected_batch or block_table.stride(1) != 1 or
            schedule_meta is not None):
        raise ValueError(
            "grouped sparse MQA requires one block-table row per q row "
            "and schedule_meta=None")
    if (not clean_logits or num_warps != 4 or max_context_len <= 0 or
            max_context_len != block_table.shape[1] * 64):
        raise ValueError(
            "grouped sparse MQA requires clean_logits=True, num_warps=4 "
            "and max_context_len equal to block-table capacity")
    arch, num_cus = _paged_mqa_device_info(q.device.index)
    if (arch, num_cus) not in (("gfx936", 80), ("gfx938", 64)):
        raise ValueError(
            "grouped sparse MQA requires gfx936/80CU or gfx938/64CU")
    native_op = (
        op.page_mqa_logits_sparse_mask_grouped_int8
        if _int8_kv
        else op.page_mqa_logits_sparse_mask_grouped
    )
    return native_op(
        q,
        fused_kv_cache,
        weights,
        context_lens,
        block_table,
        None,
        max_context_len,
        clean_logits,
        num_warps,
        persistent_ctas,
        group_size)


def page_mqa_logits_sparse_mask_grouped_int8(
        q: torch.Tensor,
        packed_kv_cache: torch.Tensor,
        weights: torch.Tensor,
        context_lens: torch.Tensor,
        block_table: torch.Tensor,
        schedule_meta: Optional[torch.Tensor],
        max_context_len: int,
        clean_logits: bool = True,
        num_warps: int = 4,
        persistent_ctas: int = 0,
        group_size: int = 5):
    """Internal sparse MQA path for packed INT8 index-K pages.

    Dequantization is performed per KV tile inside the MQA kernel. The public
    BF16/FP8 grouped entry point and its dispatch behavior are unchanged.
    """
    return page_mqa_logits_sparse_mask_grouped(
        q,
        packed_kv_cache,
        weights,
        context_lens,
        block_table,
        schedule_meta,
        max_context_len,
        clean_logits,
        num_warps,
        persistent_ctas,
        group_size,
        _int8_kv=True)


def fused_bmm(x: torch.Tensor,                                       # [batch_size, num_head, p_size], int8
              w: torch.Tensor,                                       # [num_head, l_size, p_size], int8
              x_scale: torch.Tensor,                                 # [batch_size, num_head, 1], fp32
              w_scale: torch.Tensor,                                 # [num_head, l_size, 1], fp32
              bias: Optional[torch.Tensor] = None,                   # [num_head, l_size], None
              dtype: Optional[torch.dtype] = torch.float8_e4m3fn,    # output dtype: fp8 or bf16
              output: Optional[torch.Tensor] = None,                 # [batch_size, num_head, l_size], fp8 or bf16
              transpose_bm: Optional[bool] = False,
              transpose_bm_in: Optional[bool] = False,
              config: Optional[Dict[str, Any]] = {"MODE": 0}) -> torch.Tensor:
    """
    Computes batched 8 bit matrix multiplication Y[i] = X[i] @ W[i]^T with active activation quantization.
    X is quantized to INT8 during computation using per-token grouped quantization.
    W is pre-quantized INT8 with per-batch-element scaling.

    Args:
        x (torch.Tensor): Higher precision input batch with shape (B, M, K) or (M, B, K) if transpose_bm_in=True.
            Quantized to INT8 on-the-fly during GEMM.
        w (torch.Tensor): Pre-quantized INT8 weight batch with shape (B, N, K), internally transposed.
        x_scale (torch.Tensor): Per-batch scale for x with shape (B, M, 1) or (M, B, 1) if transpose_bm_in=True.
        w_scale (torch.Tensor): Per-batch scale for w with shape (B,N,1).
        bias (Optional[torch.Tensor]): Bias batch with shape (B,N).
        dtype (Optional[torch.dtype]): Output datatype (float8_e4m3fn or bf16).
        output (Optional[torch.Tensor]): Pre-allocated output tensor with shape (B, M, N) or (M, B, N) if transpose_bm=True.
        transpose_bm (Optional[bool]): Transpose batch and M dimensions in output.
        transpose_bm_in (Optional[bool]): Transpose batch and M dimensions in input.
        config (Optional[dict]): Kernel tuning parameters eg: {"MODE": 0}.

    Returns:
        torch.Tensor: Output batch with shape (B, M, N) or (M, B, N) if transpose_bm=True.
    """
    
    # Check constraints.
    if not transpose_bm_in:
        batch_size = x.shape[0]
        num_head = x.shape[1]
    else:
        num_head = x.shape[0]
        batch_size = x.shape[1]
    L = w.shape[1]

    assert num_head == w.shape[0], "Incompatible num_head dimensions!!!"

    assert dtype in [
        torch.float8_e4m3fn,
        torch.float8_e5m2,
        torch.bfloat16,
    ], f"Output dtype {dtype} is currently not supported in fused_bmm"

    if output is None:
        if transpose_bm:
            output = torch.empty((num_head, batch_size, L), dtype=dtype, device=x.device)
        else:
            output = torch.empty((batch_size, num_head, L), dtype=dtype, device=x.device)
    else:
        if transpose_bm:
            assert (
                output.shape[0] == num_head and output.shape[1] == batch_size and output.shape[2] == L
            ), "Output dimension error"
        else:
            assert (
                output.shape[0] == batch_size and output.shape[1] == num_head and output.shape[2] == L
            ), "Output dimension error"
    
    mode = config['MODE']
    return op.fused_bmm(x, w, output, x_scale, w_scale, bias, dtype, transpose_bm, transpose_bm_in, mode)


def validate_fuse_rmsnorm_rope_quant_qkv_args(positions, q_c, q_pe, q_quant, q_scale, k_pe, kv_c, kv_c_normed,
                                               weight, cos_sin_cache, slot_mapping, kv_cache,
                                               kv_cache_dtype_str, kv_cache_scale, is_neox_style, epsilon, limit):
    """
    Validate arguments for fuse_rmsnorm_rope_quant_qkv function
    """
    # Constants
    q_c_dim = 128
    rot_dim = 64
    kv_c_dim = 512
    k_stride_head = k_pe.stride(0)

    # Get basic dimensions
    bs = q_pe.size(0)
    num_head = q_pe.size(1)
    q_stride_head = q_pe.stride(1)

    # Validate q_c
    assert q_c.dim() == 3, "q_c must be 3D tensor [bs, num_head, q_c_dim]"
    assert q_c.size(0) == bs, f"q_c.size(0) must be {bs}"
    assert q_c.size(1) == num_head, f"q_c.size(1) must be {num_head}"
    assert q_c.size(2) == q_c_dim, f"q_c.size(2) must be {q_c_dim}"

    # Validate q_pe
    assert q_pe.dim() == 3, "q_pe must be 3D tensor [bs, num_head, rot_dim]"
    assert q_pe.size(0) == bs, f"q_pe.size(0) must be {bs}"
    assert q_pe.size(1) == num_head, f"q_pe.size(1) must be {num_head}"
    assert q_pe.size(2) == rot_dim, f"q_pe.size(2) must be {rot_dim}"

    # Validate q_quant
    assert q_quant.dim() == 3, "q_quant must be 3D tensor [bs, num_head, q_c_dim + rot_dim]"
    assert q_quant.size(0) == bs, f"q_quant.size(0) must be {bs}"
    assert q_quant.size(1) == num_head, f"q_quant.size(1) must be {num_head}"
    assert q_quant.size(2) == q_c_dim + rot_dim, f"q_quant.size(2) must be {q_c_dim + rot_dim}"

    # Validate stride
    assert q_stride_head == q_c_dim + rot_dim, "q_stride_head must be q_c_dim + rot_dim"

    # Validate cos_sin_cache
    assert cos_sin_cache.dim() == 2, "cos_sin_cache must be 2D tensor [max_position, rot_dim]"
    assert cos_sin_cache.size(1) == rot_dim, "cos_sin_cache.size(1) must equal rot_dim"

    # Validate q_scale
    assert q_scale.dtype == torch.float32, "q_scale must be float tensor"

    # Validate k_pe
    assert k_pe.dim() == 2, "k_pe must be 2D tensor [bs, k_stride_head]"
    assert k_pe.size(0) == bs, f"k_pe.size(0) must be {bs}"
    assert k_pe.size(1) == rot_dim, f"k_pe.size(1) must be {rot_dim}"

    # Validate kv_c_normed
    assert kv_c_normed.dim() == 2, "kv_c_normed must be 2D tensor [bs, kv_c_dim]"
    assert kv_c_normed.size(0) == bs, f"kv_c_normed.size(0) must be {bs}"
    assert kv_c_normed.size(1) == kv_c_dim, f"kv_c_normed.size(1) must be {kv_c_dim}"
    assert kv_c_normed.stride(0) == kv_c_dim, "kv_c_normed.stride(0) must be kv_c_dim"

    # Validate kv_c
    assert kv_c.dim() == 2, "kv_c must be 2D tensor [bs, kv_c_dim + rot_dim]"
    assert kv_c.size(0) == bs, f"kv_c.size(0) must be {bs}"
    assert kv_c.size(1) == kv_c_dim, f"kv_c.size(1) must be {kv_c_dim}"
    assert kv_c.stride(0) == k_stride_head, "kv_c.stride(0) must equal k_pe.stride(0)"

    # Validate slot_mapping
    assert slot_mapping.dim() == 1, "slot_mapping must be 1D tensor [bs]"
    assert slot_mapping.size(0) == bs, f"slot_mapping.size(0) must be {bs}"

    # Validate kv_cache
    assert kv_cache.dim() == 3, "kv_cache must be 3D tensor [num_kv_cache_blocks, kv_cache_blocks_size, kv_c_dim + rot_dim]"
    assert kv_cache.size(2) == kv_c_dim + rot_dim, f"kv_cache.size(2) must be {kv_c_dim + rot_dim}"

    # Validate kv_cache_dtype_str
    assert kv_cache_dtype_str in ["fp8_e4m3", "fp8_e5m2"], "kv_cache_dtype_str must be fp8_e4m3 or fp8_e5m2"

    # Validate kv_cache_scale
    assert kv_cache_scale == 1.0, "kv_cache_scale must be 1.0"

    assert limit == -1.0 or (math.isfinite(limit) and limit >= 0.0), \
        "limit must be -1 (disabled) or a finite non-negative value"


def fuse_rmsnorm_rope_quant_qkv(positions, q_c, q_pe, q_quant, q_scale, k_pe, kv_c, kv_c_normed,
                               weight, cos_sin_cache, slot_mapping, kv_cache,
                               kv_cache_dtype_str, kv_cache_scale=1.0, is_neox_style=False,
                               epsilon=1e-6, limit=-1.0):
    """
    Python wrapper for fuse_rmsnorm_rope_quant_qkv with input validation
    """
    # Validate inputs
    validate_fuse_rmsnorm_rope_quant_qkv_args(
        positions, q_c, q_pe, q_quant, q_scale, k_pe, kv_c, kv_c_normed,
        weight, cos_sin_cache, slot_mapping, kv_cache,
        kv_cache_dtype_str, kv_cache_scale, is_neox_style, epsilon, limit
    )

    # Call the C++ function
    return op.fuse_rmsnorm_rope_quant_qkv(
        positions, q_c, q_pe, q_quant, q_scale, k_pe, kv_c, kv_c_normed,
        weight, cos_sin_cache, slot_mapping, kv_cache,
        kv_cache_dtype_str, kv_cache_scale, is_neox_style, epsilon, limit
    )

import torch.nn.functional as F
def gemm_awq_w4a16_marlin(a, b, scales_zeros):
    # assert use_marlin is True, "暂时只支持marlin版"
    M = a.shape[0]
    N = scales_zeros.shape[0]
    K = a.shape[1]
    G = K // scales_zeros.shape[1]
    MarlinTileK = 256 if K % 256 == 0 else 128
    assert N == b.shape[1]/(MarlinTileK/8), "Marlin权重shape异常"
    assert a.dtype == torch.float16, "输入tensor必须是float16"
    assert K%256 == 0 or K%128 == 0, "K必须是256或128的倍数"
    ret = torch.empty((M, N), device=a.device, dtype=a.dtype)
    use_marlin = True
    if M <=16:
        config = get_awq_w4a16_gemm_marlin_config(M,N,K)
        assert config is not None, f"Unsupported shape: M={M}, N={N}, K={K}"
        num_n_warps, num_k_warps, n_per_warp, k_per_warp, splitk = config
        # import pdb; pdb.set_trace()
        op.gemm_awq_w4a16(ret, a, b, scales_zeros, num_n_warps, num_k_warps, n_per_warp, k_per_warp, splitk, use_marlin)
    elif M <= 64:
        config = get_awq_w4a16_gemm_marlin_config(16,N,K)
        assert config is not None, f"Unsupported shape: M={M}, N={N}, K={K}"
        num_n_warps, num_k_warps, n_per_warp, k_per_warp, splitk = config
        for start_idx in range(0, M, 16):
            end_idx = min(start_idx + 16, M)
            op.gemm_awq_w4a16(ret[start_idx:end_idx], a[start_idx:end_idx], b, scales_zeros, num_n_warps, num_k_warps, n_per_warp, k_per_warp, splitk, use_marlin)
    else:
        weight_dequant_fp16 = torch.empty((N, K), device=a.device, dtype=torch.float16)
        op.gemm_awq_w4a16_dequant(weight_dequant_fp16, b, scales_zeros, use_marlin)
        ret = F.linear(a,weight_dequant_fp16)
    return ret


def gemm_awq_w4a16_marlin_dequant(ret, weight, scales_zeros, use_marlin):
    assert use_marlin is True, "暂时只支持marlin版"
    op.gemm_awq_w4a16_dequant(ret, weight, scales_zeros, use_marlin)
    return ret


def awq_gemm_marlin_weight_repack(weight_trans, N, K):
    if K % 256 == 0:
        marlin_tile_k = 256
        weight_marlin = weight_trans.T
        K_quant = K//8
        K_tile = marlin_tile_k // 8
        weight_marlin = weight_marlin.reshape(K_quant // K_tile, K_tile, N)
        weight_marlin = weight_marlin.transpose(1, 2)
        weight_marlin = weight_marlin.reshape(K_quant//K_tile, N*K_tile)
    elif K % 128 == 0:
        marlin_tile_k = 128
        weight_marlin = weight_trans.T
        K_quant = K//8
        K_tile = marlin_tile_k // 8
        weight_marlin = weight_marlin.reshape(K_quant // K_tile, K_tile, N)
        weight_marlin = weight_marlin.transpose(1, 2)
        weight_marlin = weight_marlin.reshape(K_quant//K_tile, N*K_tile)
    else:
        raise ValueError(f"K {K} not supported")
    return weight_marlin
