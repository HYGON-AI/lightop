# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Fused MoE kernel."""
import functools
import json
import os
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import triton
import triton.language as tl

#import vllm.envs as envs
# from vllm import _custom_ops as ops
# from vllm.logger import init_logger

# from vllm.utils import direct_register_custom_op
# from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
#     moe_align_block_size)
from ...vllm_compat.moe_align_block_size import moe_align_block_size
from grouped_gemm_int4  import moe_gemm_w4a16
from grouped_gemm_int4.ops import permute as permute_topK, unpermute as unpermute_topK
import torch.nn.functional as F
# from vllm.platforms import current_platform
# from vllm_hcu.platforms import HCUPlatform as current_platform
# device_name = current_platform.get_device_name()
device_name = torch.cuda.get_device_properties(0).name
# logger = init_logger(__name__)


def config_cuda(M):
    
    k100ai_gemm1_m_to_mode_dict = {
        1: 'M16N16K256NN1NW8B240',
        2: 'M16N16K256NN1NW16B360',
        3: 'M16N16K256NN1NW16B360',
        4: 'M16N16K256NN1NW16B360',
        5: 'M16N16K256NN1NW16B360',
        6: 'M16N16K256NN1NW16B360',
        7: 'M16N16K256NN1NW16B360',
        8: 'M16N16K256NN1NW16B240',
        9: 'M16N16K256NN1NW16B360',
        10: 'M16N16K256NN1NW16B360',
        11: 'M16N16K256NN1NW16B360',
        12: 'M16N16K256NN1NW16B360',
        13: 'M16N16K256NN1NW16B360',
        14: 'M16N16K256NN1NW16B360',
        15: 'M16N16K256NN1NW16B360',
        16: 'M16N16K256NN1NW16B240',
        17: 'M16N16K256NN1NW16B360',
        18: 'M16N16K256NN1NW16B360',
        19: 'M16N16K256NN1NW16B360',
        20: 'M16N16K256NN1NW16B360',
        21: 'M16N16K256NN1NW16B360',
        22: 'M16N16K256NN1NW16B360',
        23: 'M16N16K256NN1NW16B360',
        24: 'M16N16K256NN1NW16B360',
        25: 'M16N16K256NN1NW16B360',
        26: 'M16N16K256NN1NW16B360',
        27: 'M16N16K256NN1NW16B240',
        28: 'M16N16K256NN1NW16B360',
        29: 'M16N16K256NN1NW16B360',
        30: 'M16N16K256NN1NW16B360',
        31: 'M16N16K256NN1NW16B240',
        32: 'M16N16K256NN1NW16B360',
        64: 'M16N16K256NN1NW16B360',
        128: 'M16N16K256NN1NW16B240',
        256: 'M16N16K256NN1NW16B360',
        512: 'M16N16K128NN1NW8B120',
        768: 'M16N32K128NN1NW16B100',
        1024: 'M16N32K128NN1NW16B120',
    }


    k100ai_gemm2_m_to_mode_dict = {
        1: 'M16N32K256NN8NW1B240',
        2: 'M16N32K256NN8NW1B360',
        3: 'M16N32K256NN8NW1B360',
        4: 'M16N32K256NN4NW1B360',
        5: 'M16N32K256NN4NW1B360',
        6: 'M16N32K256NN4NW1B360',
        7: 'M16N32K256NN4NW1B360',
        8: 'M16N32K256NN8NW1B360',
        9: 'M16N32K256NN8NW1B240',
        10: 'M16N32K256NN8NW1B240',
        11: 'M16N32K256NN8NW1B240',
        12: 'M16N32K256NN8NW1B240',
        13: 'M16N32K256NN4NW1B360',
        14: 'M16N32K256NN16NW1B360',
        15: 'M16N32K256NN16NW1B360',
        16: 'M16N32K256NN16NW1B360',
        17: 'M16N32K256NN8NW1B240',
        18: 'M16N32K256NN8NW1B240',
        19: 'M16N32K256NN16NW1B360',
        20: 'M16N32K256NN16NW1B360',
        21: 'M16N32K256NN16NW1B360',
        22: 'M16N32K256NN16NW1B360',
        23: 'M16N32K256NN16NW1B360',
        24: 'M16N32K256NN16NW1B240',
        25: 'M16N32K256NN16NW1B360',
        26: 'M16N32K256NN16NW1B360',
        27: 'M16N32K256NN16NW1B360',
        28: 'M16N32K256NN16NW1B360',
        29: 'M16N64K256NN4NW1B240',
        30: 'M16N32K256NN16NW1B360',
        31: 'M16N32K256NN16NW1B360',
        32: 'M16N32K256NN16NW1B240',
        64: 'M16N32K256NN16NW1B360',
        128: 'M16N64K256NN4NW1B240',
        256: 'M16N32K256NN16NW1B360',
        512: 'M16N64K256NN8NW1B120',
        768: 'M16N64K256NN16NW1B360',
        1024: 'M16N64K256NN16NW1B360',
    }


    bw_gemm1_m_to_mode_dict = {
        1: 'M16N16K256NN1NW8B360',
        2: 'M16N16K256NN1NW4B360',
        3: 'M16N32K256NN1NW8B240',
        4: 'M16N32K256NN1NW4B360',
        5: 'M16N64K256NN1NW4B240',
        6: 'M16N32K256NN1NW8B240',
        7: 'M16N32K256NN1NW8B360',
        8: 'M16N64K256NN1NW4B360',
        9: 'M16N64K256NN1NW4B240',
        10: 'M16N32K256NN1NW8B240',
        11: 'M16N64K256NN1NW4B240',
        12: 'M16N64K256NN1NW4B360',
        13: 'M16N32K256NN1NW8B240',
        14: 'M16N32K256NN1NW8B240',
        15: 'M16N32K256NN1NW8B240',
        16: 'M16N64K256NN1NW4B360',
        17: 'M16N32K256NN1NW8B240',
        18: 'M16N64K256NN1NW4B240',
        19: 'M16N32K256NN1NW8B240',
        20: 'M16N32K256NN1NW8B240',
        21: 'M16N32K256NN1NW8B240',
        22: 'M16N32K256NN1NW8B240',
        23: 'M16N32K256NN1NW8B240',
        24: 'M16N64K256NN1NW4B240',
        25: 'M16N32K256NN1NW8B240',
        26: 'M16N32K256NN1NW8B240',
        27: 'M16N32K256NN1NW8B240',
        28: 'M16N32K256NN1NW8B240',
        29: 'M16N64K256NN1NW4B240',
        30: 'M16N64K256NN1NW4B240',
        31: 'M16N32K256NN1NW8B240',
        32: 'M16N64K256NN1NW4B240',
        64: 'M16N32K256NN1NW8B240',
        128: 'M16N64K256NN1NW4B240',
        256: 'M16N64K256NN1NW4B240',
        512: 'M16N64K256NN1NW4B240',
        768: 'M16N64K256NN1NW4B240',
        1024: 'M16N64K256NN1NW4B240',
    }


    bw_gemm2_m_to_mode_dict = {
        1: 'M16N32K128NN8NW1B240',
        2: 'M16N64K256NN8NW1B240',
        3: 'M16N64K256NN4NW1B360',
        4: 'M16N64K256NN16NW1B240',
        5: 'M16N64K256NN8NW1B240',
        6: 'M16N64K256NN8NW1B240',
        7: 'M16N64K256NN16NW1B240',
        8: 'M16N64K256NN8NW1B240',
        9: 'M16N64K256NN16NW1B360',
        10: 'M16N64K256NN8NW1B240',
        11: 'M16N64K256NN16NW1B360',
        12: 'M16N64K256NN8NW1B240',
        13: 'M16N64K256NN16NW1B240',
        14: 'M16N64K256NN16NW1B360',
        15: 'M16N64K256NN16NW1B240',
        16: 'M16N64K256NN16NW1B240',
        17: 'M16N64K256NN8NW1B240',
        18: 'M16N64K256NN8NW1B240',
        19: 'M16N64K256NN16NW1B240',
        20: 'M16N64K256NN8NW1B240',
        21: 'M16N64K256NN16NW1B240',
        22: 'M16N64K256NN16NW1B360',
        23: 'M16N64K256NN16NW1B360',
        24: 'M16N64K256NN16NW1B240',
        25: 'M16N64K256NN8NW1B240',
        26: 'M16N64K256NN16NW1B240',
        27: 'M16N64K256NN16NW1B240',
        28: 'M16N64K256NN16NW1B240',
        29: 'M16N64K256NN16NW1B240',
        30: 'M16N64K256NN16NW1B240',
        31: 'M16N64K256NN8NW1B240',
        32: 'M16N64K256NN16NW1B240',
        64: 'M16N64K256NN16NW1B240',
        128: 'M16N64K256NN16NW1B240',
        256: 'M16N64K256NN16NW1B240',
        512: 'M16N64K256NN16NW1B240',
        768: 'M16N64K256NN16NW1B240',
        1024: 'M16N64K256NN16NW1B240',
    }

    reference_points = [32, 64, 128, 256, 512, 1024]
    NearestM = -1
    
    if M <= 32:
        NearestM = M
    else:
        NearestM = min(reference_points, key=lambda x: abs(x - M))

    if device_name == "K100_AI":
        mode_1 = k100ai_gemm1_m_to_mode_dict.get(NearestM, k100ai_gemm1_m_to_mode_dict[32])
        mode_2 = k100ai_gemm2_m_to_mode_dict.get(NearestM, k100ai_gemm2_m_to_mode_dict[32])
    else:
        mode_1 = bw_gemm1_m_to_mode_dict.get(NearestM, bw_gemm1_m_to_mode_dict[32])
        mode_2 = bw_gemm2_m_to_mode_dict.get(NearestM, bw_gemm2_m_to_mode_dict[32])

    return mode_1, mode_2
   
def fused_experts_cuda(hidden_states: torch.Tensor,
                  w1: torch.Tensor,
                  w2: torch.Tensor,
                  topk_weights: torch.Tensor,
                  topk_ids: torch.Tensor,
                  inplace: bool = False,
                  activation: str = "silu",
                  use_fp8_w8a8: bool = False,
                  use_int8_w8a16: bool = False,
                  use_int4_w4a16: bool = False,
                  w1_scale: Optional[torch.Tensor] = None,
                  w2_scale: Optional[torch.Tensor] = None,
                  w1_zp: Optional[torch.Tensor] = None,
                  w2_zp: Optional[torch.Tensor] = None,
                  a1_scale: Optional[torch.Tensor] = None,
                  a2_scale: Optional[torch.Tensor] = None,
                  block_shape: Optional[List[int]] = None,
                  expert_map: Optional[torch.Tensor] = None,):
    if inplace:
        fused_experts_impl_cuda(hidden_states, w1, w2, topk_weights, topk_ids, True, activation,
                        use_fp8_w8a8, use_int8_w8a16, use_int4_w4a16, w1_scale,
                        w2_scale, w1_zp, w2_zp, a1_scale, a2_scale, block_shape,
                        expert_map)

        return hidden_states
    else:
        return fused_experts_impl_cuda(hidden_states, w1, w2, topk_weights, topk_ids, False, activation,
                        use_fp8_w8a8, use_int8_w8a16, use_int4_w4a16, w1_scale,
                        w2_scale, w1_zp, w2_zp, a1_scale, a2_scale, block_shape,
                        expert_map)


def fused_experts_impl_cuda(hidden_states: torch.Tensor,
                       w1: torch.Tensor,
                       w2: torch.Tensor,
                       topk_weights: torch.Tensor,
                       topk_ids: torch.Tensor,
                       inplace: bool = False,
                       activation: str = "silu",                       
                       use_fp8_w8a8: bool = False,
                       use_int8_w8a16: bool = False,
                       use_int4_w4a16: bool = True,
                       w1_scale: Optional[torch.Tensor] = None,
                       w2_scale: Optional[torch.Tensor] = None,
                       w1_zp: Optional[torch.Tensor] = None,
                       w2_zp: Optional[torch.Tensor] = None,
                       a1_scale: Optional[torch.Tensor] = None,
                       a2_scale: Optional[torch.Tensor] = None,
                       block_shape: Optional[List[int]] = None,
                       expert_map: Optional[torch.Tensor] = None,):
    # Check constraints.
    assert hidden_states.shape[1] // 2 == w1.shape[
        2], "Hidden size mismatch"


    assert topk_weights.shape == topk_ids.shape, "topk shape mismatch"
    assert hidden_states.is_contiguous(), "Hidden_states must be contiguous"
    assert w1.is_contiguous(), "Expert weights1 must be contiguous"
    assert w2.is_contiguous(), "Expert weights2 must be contiguous"
    assert hidden_states.dtype in [
        torch.float32, torch.float16, torch.bfloat16
    ]

    num_tokens, _ = hidden_states.shape

    E, N, _ = w1.shape
    # We execute the fused_moe kernel in chunks to circumvent this issue:
    # https://github.com/vllm-project/vllm/issues/5938

    M = num_tokens
    topk = topk_ids.shape[1]
    mode_1, mode_2 = config_cuda(M)
    # config = get_config_func(M)

    intermediate_cache1 = torch.empty((M, topk, N),
                                      device=hidden_states.device,
                                      dtype=hidden_states.dtype)
    intermediate_cache2 = torch.empty((M * topk, N // 2),
                                      device=hidden_states.device,
                                      dtype=hidden_states.dtype)
    intermediate_cache3 = torch.empty((M, topk, w2.shape[1]),
                                      device=hidden_states.device,
                                      dtype=hidden_states.dtype)
    
    if hidden_states.dtype == torch.bfloat16:
        compute_type = tl.bfloat16
    elif hidden_states.dtype == torch.float16:
        compute_type = tl.float16
    elif hidden_states.dtype == torch.float32:
        compute_type = tl.float32
    else:
        raise ValueError(f"Unsupported compute_type: {hidden_states.dtype}")

    if inplace:
        out_hidden_states = hidden_states
    else:
        out_hidden_states = torch.empty_like(hidden_states)

    sorted_token_ids, expert_ids, num_tokens_post_padded =  (               
                moe_align_block_size(topk_ids, 16, E, expert_map, hidden_states.shape[0]))
    moe_gemm_w4a16.gemm1_w4a16(sorted_token_ids, #  sorted_token_ids.to(torch.uint16)
                            hidden_states, # hidden_states
                            w1, # w1
                            intermediate_cache1, # gemm1_out
                            num_tokens_post_padded, # 实际专家数
                            expert_ids, # expert_id_vec
                            w1_scale, # scale_zero
                            block_shape[1], # group_size
                            topk=topk, # topk
                            mode=mode_1) # mode=gemm1_mode
    
    
    if activation == "silu":
        torch.ops._C.silu_and_mul(intermediate_cache2, intermediate_cache1.view(-1, N))
    elif activation == "gelu":
        torch.ops._C.gelu_and_mul(intermediate_cache2, intermediate_cache1.view(-1, N))
    else:
        raise ValueError(f"Unsupported FusedMoe activation: {activation}")
    # return intermediate_cache2
    moe_gemm_w4a16.gemm2_w4a16(sorted_token_ids, # sorted_token_ids.to(torch.uint16)
                            intermediate_cache2, # hidden_states
                            w2, # w2
                            intermediate_cache3, # gemm2_out
                            num_tokens_post_padded,
                            expert_ids, # expert_id_vec
                            w2_scale, # scale_zero
                            topk_weights, # topk_weights
                            block_shape[1], # group_size
                            topk=topk, # topk
                            mode=mode_2)   # mode=gemm2_mode
    # ops.moe_sum(intermediate_cache3.view(*intermediate_cache3.shape), out_hidden_states)
    torch.ops._moe_C.moe_sum(intermediate_cache3.view(*intermediate_cache3.shape), out_hidden_states)
    
    
    return out_hidden_states


