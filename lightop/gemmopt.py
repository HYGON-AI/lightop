import torch
from typing import Optional, Dict, Any, List, NamedTuple
#from lightop import op
from . import op
import functools
import os
import math
from .config_smooth_gemm import gemm_w8a8_smooth_config
from .config_deepgemm import gemm_w8a8_deepgemm_masked_config


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
    delta = config['DELTA']

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

def mqa_logits(Q: torch.Tensor, K: torch.Tensor, Weights: torch.Tensor, ks:torch.Tensor, ke:torch.Tensor) -> torch.Tensor:
    q_seq_len, num_heads, head_dim = Q.shape
    kv_seq_len = K.shape[0]
    clean_logit = True
    # pdb.set_trace()
    outs = op.mqa_logits(Q, K, Weights, ks, ke, q_seq_len, kv_seq_len, num_heads, head_dim,clean_logit)
    # outs = torch.nn.functional.linear(Q.view(q_seq_len*num_heads, head_dim), K)
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

def paged_mqa_logits(
        q: torch.Tensor,
        fused_kv_cache: torch.Tensor,
        weights: torch.Tensor,
        context_lens: torch.Tensor,
        block_table: torch.Tensor,
        schedule_meta: torch.Tensor,
        max_context_len: int,
        clean_logits: bool = True) -> torch.Tensor:
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

def fused_bmm(q_nope: torch.Tensor,                        # [batch_size, num_head, p_size], int8
              w_uv: torch.Tensor,                          # [num_head, l_size, p_size], int8
              output: torch.Tensor,                        # [batch_size, num_head, l_size], fp8
              a_scale: torch.Tensor,                       # [num_head, batch_size, 1], fp32
              b_scale: torch.Tensor,                       # [num_head, l_size, 1], fp32
              bias: Optional[torch.Tensor] = None,         # [num_head, l_size], None
              config: Optional[Dict[str, Any]] = {"MODE": 0}) -> torch.Tensor:
    mode = config['MODE']
    return op.fused_bmm(q_nope, w_uv, output, a_scale, b_scale, bias, mode)