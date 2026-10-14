# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

import torch
import time
from typing import Optional, Type, Any, Callable, Dict, List, Tuple
import pandas as pd
import os
import json
import triton
import triton.language as tl
import pandas as pd
import logging
import math
import torch.nn.functional as F
# from per_token_group_test import per_token_group_quant_int8
import functools
import logging
from vllm.platforms import current_platform
import vllm.envs as envs
from vllm.logger import init_logger
from vllm import _custom_ops as ops
# import _custom_ops as ops
from lightop.moe import moe_gemm_w8a8, get_moe_cuda_config
# from vllm.model_executor.layers.fused_moe.fused_moe import moe_kernel_prepare_input
from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
    moe_align_block_size)

M_THRESHOLD = 128

os.environ["TRITON_CACHE_DIR"] = f".triton/cache"

logger = init_logger(__name__)
device_name = current_platform.get_device_name().replace(" ", "_")
num_cus= torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count

from lightop.quant import (
    per_token_group_quant_int8,
    per_token_quant_int8,
)
def moe_kernel_prepare_input(
    A: torch.Tensor,
    B: torch.Tensor,
    A_scale: Optional[torch.Tensor],
    B_scale: Optional[torch.Tensor],
    use_fp8_w8a8: bool,
    use_int8_w8a8: bool,
    use_int8_w8a16: bool,
    use_int4_w4a16: bool,
    per_channel_quant: bool,
    block_shape: Optional[List[int]] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    if use_fp8_w8a8:
        assert B_scale is not None
        if block_shape is None:
            # If weights are per-channel (per_channel_quant=True), then
            # activations apply per-token quantization. Otherwise, assume
            # activation tensor-wise fp8 quantization, dynamic or static
            A, A_scale = ops.scaled_fp8_quant(
                A, A_scale, use_per_token_if_dynamic=per_channel_quant)
        else:
            # activation block-wise fp8 quantization
            assert len(block_shape) == 2
            _, block_k = block_shape[0], block_shape[1]
            A, A_scale = per_token_group_quant_fp8(A, block_k)
            assert triton.cdiv(A.shape[-1], block_k) == A_scale.shape[-1]
            # assert triton.cdiv(B.shape[-2], block_n) == B_scale.shape[-2]
            # assert triton.cdiv(B.shape[-1], block_k) == B_scale.shape[-1]
    elif use_int8_w8a8:
        assert B_scale is not None
        if block_shape is None:
            # activation channel-wise int8 quantization
            assert (per_channel_quant
                    ), "int8 quantization only supports block or channel-wise"
            A, A_scale = per_token_quant_int8(A)
        else:
            # activation block-wise int8 quantization
            assert len(block_shape) == 2
            _, block_k = block_shape[0], block_shape[1]
            A, A_scale = per_token_group_quant_int8(A, block_k)
            assert triton.cdiv(A.shape[-1], block_k) == A_scale.shape[-1]
            # assert triton.cdiv(B.shape[-2], block_n) == B_scale.shape[-2]
            # assert triton.cdiv(B.shape[-1], block_k) == B_scale.shape[-1]
    elif use_int8_w8a16 or use_int4_w4a16:
        assert B_scale is not None
        assert block_shape is None or block_shape[0] == 0
    else:
        assert A_scale is None
        assert B_scale is None

    return A, A_scale

blockM_configs = {
    0 : 16,
    1 : 32,
    2 : 64,
    3 : 16,
    4 : 32,
    6 : 16,
    9 : 32,
    10: 64,
    11: 32,
    13: 64,
    14: 32,
    15: 32,
    16: 64,
    17: 64,
    18: 32,
    19: 64,
    20: 32,
    21: 64,
    22: 64,
    23: 64,
    24: 128,
    25: 128,
    26: 16,
    27: 32,
    28: 16,
    29: 16,
    30: 16,
    31: 32,
    32: 32,
    33: 32,
    35: 64,
    36: 64,
    39: 32,
    40: 64,
    41: 32,
    42: 64,
    43: 32,
    44: 64,
    45: 64,
    46: 64,
    47: 64,
    48: 64,
    49: 128,
    50: 16,
    51: 32,
    52: 64,
    53: 16,
    54: 32,
    56: 16,
    59: 32,
    60: 64,
    61: 32,
    63: 64,
    
    64: 16,
    65: 32,
    66: 64,
    67: 16,
    68: 32,
    69: 16,
    70: 32,
    71: 64,
    74: 32,
    75: 32,
    76: 64,
    77: 64,
    78: 32,
    79: 64,
    80: 32,
    81: 64,
    82: 64,
    83: 64,
    84: 128,
    85: 128,
    86: 16,
    87: 32,
    88: 16,
    89: 16,
    90: 16,
    91: 32,
    92: 32,
    93: 32,
    94: 64,
    95: 64,
    1000: 128,
    1001: 64,
    1002: 32,
}

blockM_configs_decode = {
    0: 16,   # 索引0对应M=16
    1: 16,   # 索引1对应M=32
    2: 16,   # 索引2对应M=64
    3: 16,   # 索引3对应M=16
    4: 16,   # 索引4对应M=32
    5: 16,   # 索引5对应M=64
    6: 16,   # 索引6对应M=16
    7: 16,   # 索引7对应M=32
    8: 16,   # 索引8对应M=64
    9: 16,   # 索引9对应M=32
    10: 16,  # 索引10对应M=64
    11: 16,  # 索引11对应M=32
    12: 16,  # 索引12对应M=64
    13: 16,  # 索引13对应M=64
    14: 16,  # 索引14对应M=32
    15: 16,  # 索引15对应M=32
    16: 16,  # 索引16对应M=64
    17: 16,  # 索引17对应M=64
    18: 16,  # 索引18对应M=32
    19: 16,  # 索引19对应M=64
    20: 16,
    21: 16,
    22: 16,
    23: 16,
    24: 16,
    25: 16,
    26: 16,
    27: 16,
    28: 16,
    29: 16,
    30: 16,
    31: 16,
    32: 16,
    33: 16,
    34: 16,
    35: 16,
    36: 16,
    37: 16,
    38: 16,
    39: 16,
    40: 16,
    41: 16,
    42: 16,
    43: 16
}


def get_moe_config(M, N, K):
    # print("#####################################################")
    # print(f"M:{M}  N:{N}  K:{K}")
    file_name = f"MOE_BLOCKINT8_{N}_{K}_{device_name}_{num_cus}.json"
    config_file_path = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "configs", file_name)
    try:
        with open(config_file_path, 'r') as file:
            config_infos = json.load(file)
            if f"{N}_{K}" in config_infos:
                config_info = config_infos[f"{N}_{K}"]
                int_keys = [int(key) for key in config_info.keys()]
                greater_numbers = [num for num in int_keys if num >= M]
                M = min(greater_numbers) if greater_numbers else None
                config = config_info[f"{M}"] if f"{M}" in config_info else {}
            else:
                config={}
    except FileNotFoundError:
        config = {}
        
    # if not config: 
    #     print("Config is not found! Use default config.")
        
    config = config if config else {
        "BLOCK_SIZE_M": 64,
        "BLOCK_SIZE_N": 128,
        "BLOCK_SIZE_K": 128,
        "GROUP_SIZE_M": 1,
        "kpack": 2,
        "num_stages": 0,
        "num_warps": 8
    }
    return config

def to_int8(tensor: torch.Tensor):
    return torch.round(tensor.clamp(min=-128, max=127)).to(dtype=torch.int8)

# 生成 [m, topk] 的整数张量，每行元素不重复
def generate_unique_int_tensor(m, topk, low=0, high=256):
    # 检查 topk 是否超过范围
    if topk > (high - low):
        raise ValueError(f"topk ({topk}) 不能超过范围 ({high - low})")
    
    # 生成张量
    tensor = torch.zeros((m, topk), dtype=torch.int)
    
    for i in range(m):
        # 对每一行生成不重复的随机整数
        row = torch.randperm(high - low)[:topk] + low
        tensor[i] = row
    
    return tensor

# Adapted from: https://github.com/sgl-project/sglang/pull/2628
def get_config_file_name(E: int,
                         N: int,
                         dtype: Optional[str],
                         block_shape: Optional[List[int]] = None, use_nn_moe: Optional[bool] = False) -> str:
    device_name=torch.cuda.get_device_properties('cuda').name.replace(" ","_")

    dtype_selector = "" if not dtype else f",dtype={dtype}"
    block_shape_selector = ("" if not block_shape or not all(block_shape) else
                            f",block_shape={block_shape}").replace(" ", "")
    if not use_nn_moe:
        return f"E={E},N={N},device_name={device_name}{dtype_selector}{block_shape_selector}.json"  # noqa: E501
    else:
        return f"E={E},N={N},device_name={device_name}{dtype_selector}{block_shape_selector}_nn.json"


# Adapted from: https://github.com/sgl-project/sglang/pull/2628
@functools.lru_cache
def get_moe_configs(
    E: int,
    N: int,
    dtype: Optional[str],
    block_n: Optional[int] = None,
    block_k: Optional[int] = None,
    use_nn_moe: Optional[bool] = False
) -> Optional[Dict[int, Any]]:
    """
    Return optimized configurations for the fused MoE kernel.

    The return value will be a dictionary that maps an irregular grid of
    batch sizes to configurations of the fused_moe kernel. To evaluate the
    kernel on a given batch size bs, the closest batch size in the grid should
    be picked and the associated configuration chosen to invoke the kernel.
    """

    # First look up if an optimized configuration is available in the configs
    # directory
    block_shape = [block_n, block_k] if block_n and block_k else None
    json_file_name = get_config_file_name(E, N, dtype, block_shape, use_nn_moe=use_nn_moe)

    config_file_path = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "configs", json_file_name)
    if os.path.exists(config_file_path):
        with open(config_file_path) as f:
            logger.info("Using configuration from %s for MoE layer.",
                        config_file_path)
            # If a configuration has been found, return it
            return {int(key): val for key, val in json.load(f).items()}

    # If no optimized configuration is available, we will use the default
    # configuration
    logger.warning(
        ("Using default MoE config. Performance might be sub-optimal! "
         "Config file not found at %s"), config_file_path)
    return None


def get_default_config(
    M: int,
    E: int,
    N: int,
    K: int,
    topk: int,
    dtype: Optional[str],
    is_marlin: bool,
    block_shape: Optional[List[int]] = None,
    use_nn_moe: Optional[bool]=False
) -> Dict[str, int]:
    if dtype == "fp8_w8a8" and block_shape is not None:
        # Block-wise quant: BLOCK_SIZE_N must be divisible by block_shape[0]
        # BLOCK_SIZE_K must be divisible by block_shape[1]
        config = {
            "BLOCK_SIZE_M": 64,
            "BLOCK_SIZE_N": block_shape[0],
            "BLOCK_SIZE_K": block_shape[1],
            "GROUP_SIZE_M": 32,
            "num_warps": 4,
            "num_stages": 3,
        }
    else:
        config = {
            "BLOCK_SIZE_M": 64,
            "BLOCK_SIZE_N": 64,
            "BLOCK_SIZE_K": 32,
            "GROUP_SIZE_M": 8,
        }
        # A heuristic: fused marlin works faster with this config for small M
        if M <= E or (is_marlin and M <= 32):
            config = {
                "BLOCK_SIZE_M": 16,
                "BLOCK_SIZE_N": 32,
                "BLOCK_SIZE_K": 64,
                "GROUP_SIZE_M": 1,
            }
    if use_nn_moe:
        config["num_ldmatrixes"] = 1
    return config


def try_get_optimal_moe_config(
    w1_shape: Tuple[int, ...],
    w2_shape: Tuple[int, ...],
    top_k: int,
    dtype: Optional[str],
    M: int,
    is_marlin: bool = False,
    block_shape: Optional[List[int]] = None,
    use_nn_moe: Optional[bool] = False
):
    from vllm.model_executor.layers.fused_moe import get_config
    override_config = get_config()
    if override_config:
        config = override_config
    else:
        # First try to load optimal config from the file
        if not use_nn_moe:
            E, _, N = w2_shape
        else:
            E, N, _  = w2_shape
        block_n = block_shape[0] if block_shape else 0
        block_k = block_shape[1] if block_shape else 0
        configs = get_moe_configs(E, N, dtype, block_n, block_k, use_nn_moe=use_nn_moe)

        if configs:
            # If an optimal configuration map has been found, look up the
            # optimal config
            config = configs[min(configs.keys(), key=lambda x: abs(x - M))]
        else:
            # Else use the default config
            config = get_default_config(M, E, N, w1_shape[2] if not use_nn_moe else w1_shape[1], top_k, dtype,
                                        is_marlin, block_shape, use_nn_moe=use_nn_moe)
    return config
def fused_topk(
    hidden_states: torch.Tensor,
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
):
    assert hidden_states.shape[0] == gating_output.shape[0], (
        "Number of tokens mismatch")

    M, _ = hidden_states.shape

    topk_weights = torch.empty(M,
                               topk,
                               dtype=torch.float32,
                               device=hidden_states.device)
    topk_ids = torch.empty(M,
                           topk,
                           dtype=torch.int32,
                           device=hidden_states.device)
    token_expert_indicies = torch.empty(M,
                                        topk,
                                        dtype=torch.int32,
                                        device=hidden_states.device)

    ops.topk_softmax(
        topk_weights,
        topk_ids,
        token_expert_indicies,
        gating_output.float(),  # TODO(woosuk): Optimize this.
    )
    del token_expert_indicies  # Not used. Will be used in the future.

    if renormalize:
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

    return topk_weights, topk_ids


# This is used by the Deepseek-V2 and Deepseek-V3 model
@torch.compile(dynamic=True, backend=current_platform.simple_compile_backend)
def grouped_topk(hidden_states: torch.Tensor,
                 gating_output: torch.Tensor,
                 topk: int,
                 renormalize: bool,
                 num_expert_group: int = 0,
                 topk_group: int = 0,
                 scoring_func: str = "softmax",
                 e_score_correction_bias: Optional[torch.Tensor] = None):

    assert hidden_states.shape[0] == gating_output.shape[0], (
        "Number of tokens mismatch")

    if scoring_func == "softmax":
        scores = torch.softmax(gating_output, dim=-1)
    elif scoring_func == "sigmoid":
        scores = gating_output.sigmoid()
    else:
        raise ValueError(f"Unsupported scoring function: {scoring_func}")

    if e_score_correction_bias is not None:
        # Store original scores before applying correction bias. We use biased
        # scores for expert selection but original scores for routing weights
        original_scores = scores
        scores = scores + e_score_correction_bias.unsqueeze(0)

    num_token = scores.shape[0]
    group_scores = scores.view(num_token, num_expert_group,
                               -1).max(dim=-1).values  # [n, n_group]
    group_idx = torch.topk(group_scores, k=topk_group, dim=-1,
                           sorted=False)[1]  # [n, top_k_group]
    group_mask = torch.zeros_like(group_scores)  # [n, n_group]
    group_mask.scatter_(1, group_idx, 1)  # [n, n_group]
    score_mask = group_mask.unsqueeze(-1).expand(
        num_token, num_expert_group,
        scores.shape[-1] // num_expert_group).reshape(num_token, -1)  # [n, e]
    tmp_scores = scores.masked_fill(~score_mask.bool(), 0.0)  # [n, e]

    if e_score_correction_bias is not None:
        topk_ids = torch.topk(tmp_scores, k=topk, dim=-1, sorted=False)[1]
        # Use original unbiased scores for the routing weights
        topk_weights = original_scores.gather(1, topk_ids)
    else:
        topk_weights, topk_ids = torch.topk(tmp_scores,
                                            k=topk,
                                            dim=-1,
                                            sorted=False)

    if renormalize:
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

    return topk_weights.to(torch.float32), topk_ids.to(torch.int32)

# def ConfigGen():
#     import itertools
#     # BLOCK_N = [64,128,256]
#     # BLOCK_K = [64,128]
#     # GROUP_SIZE_M = [1,2,4]
#     # num_stages = [0]
#     # num_warps = [4,8,16]
#     BLOCK_N = [64]
#     BLOCK_K = [128]
#     GROUP_SIZE_M = [1]
#     num_stages = [1]
#     num_warps = [4]
    
#     config_list = list(itertools.product(
#         BLOCK_N, BLOCK_K, GROUP_SIZE_M, num_stages, num_warps,
#     ))
#     configs = [
#         triton.Config({"BLOCK_SIZE_N": i[0], "BLOCK_SIZE_K" : i[1], 'GROUP_SIZE_M': i[2]}, num_stages=i[3], num_warps=i[4])
#         for i in config_list
#     ]
#     return configs

# @triton.autotune(
#     configs=ConfigGen(),
#     key=['N', 'K', 'EM', 'num_valid_tokens'],
# )

@triton.jit
def fused_moe_kernel(
    # Pointers to matrices
    a_ptr,
    b_ptr,
    c_ptr,
    a_scale_ptr,
    b_scale_ptr,
    topk_weights_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    num_tokens_post_padded_ptr,
    # Matrix dimensions
    N,
    K,
    EM,
    num_valid_tokens,
    # The stride variables represent how much to increase the ptr by when
    # moving by 1 element in a particular dimension. E.g. `stride_am` is
    # how much to increase `a_ptr` by to get the element one row down
    # (A has M rows).
    stride_am,
    stride_ak,
    stride_be,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    stride_asm,
    stride_ask,
    stride_bse,
    stride_bsk,
    stride_bsn,
    # Block size for block-wise quantization
    group_n: tl.constexpr,
    group_k: tl.constexpr,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    MUL_ROUTED_WEIGHT: tl.constexpr,
    top_k: tl.constexpr,
    compute_type: tl.constexpr,
    use_fp8_w8a8: tl.constexpr,
    use_int8_w8a8: tl.constexpr,
    use_int8_w8a16: tl.constexpr,
    per_channel_quant: tl.constexpr,
    delta: tl.constexpr,
):
    """
    Implements the fused computation for a Mixture of Experts (MOE) using
    token and expert matrices.

    Key Parameters:
    - A: The input tensor representing tokens with shape (*, K), where '*' can
        be any shape representing batches and K is the feature dimension of
        each token.
    - B: The stacked MOE weight tensor with shape (E, N, K), where E is
        the number of experts, K is the input feature dimension, and N is
        the output feature dimension.
    - C: The output cache tensor with shape (M, topk, N), where M is the
        total number of tokens post padding, topk is the number of times
        each token is repeated, and N is the output feature dimension.
    - sorted_token_ids: A tensor containing the sorted indices of tokens,
        repeated topk times and arranged by the expert index they are
        assigned to.
    - expert_ids: A tensor containing the indices of the expert for each
        block. It determines which expert matrix from B should be used for
        each block in A.
    This kernel performs the multiplication of a token by its corresponding
    expert matrix as determined by `expert_ids`. The sorting of
    `sorted_token_ids` by expert index and padding ensures divisibility by
    BLOCK_SIZE_M, which is necessary to maintain consistency in block matrix
    multiplication across different blocks processed by the same expert.
    """
    # -----------------------------------------------------------
    # Map program ids `pid` to the block of C it should compute.
    # This is done in a grouped ordering to promote L2 data reuse.
    pid = tl.program_id(axis=0)
    # num_pid_m = tl.cdiv(EM, BLOCK_SIZE_M)
    # num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    # num_pid_in_group = GROUP_SIZE_M * num_pid_n
    # group_id = pid // num_pid_in_group
    # first_pid_m = group_id * GROUP_SIZE_M
    # group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    # pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    # pid_n = (pid % num_pid_in_group) // group_size_m
    if GROUP_SIZE_M ==1:
        num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
        pid_m = pid // num_pid_n
        pid_n = pid % num_pid_n
    else:
        num_pid_m = tl.cdiv(EM, BLOCK_SIZE_M)
        num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
        num_pid_in_group = GROUP_SIZE_M * num_pid_n
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
        pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
        pid_n = (pid % num_pid_in_group) // group_size_m

    # ----------------------------------------------------------
    # Create pointers for the first blocks of A and B.
    # We will advance this pointer as we move in the K direction
    # and accumulate
    # `a_ptrs` is a block of [BLOCK_SIZE_M, BLOCK_SIZE_K] pointers
    # `b_ptrs` is a block of [BLOCK_SIZE_K, BLOCK_SIZE_N] pointers
    num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)
    if pid_m * BLOCK_SIZE_M >= num_tokens_post_padded:
        return
    offs_token_id = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M).to(
        tl.int64)
    offs_token = tl.load(sorted_token_ids_ptr + offs_token_id)
    token_mask = offs_token < num_valid_tokens

    offs_bn = (pid_n * BLOCK_SIZE_N +
               tl.arange(0, BLOCK_SIZE_N).to(tl.int64)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_token[:, None] // top_k * stride_am +
                      offs_k[None, :] * stride_ak)

    off_experts = tl.load(expert_ids_ptr + pid_m // delta).to(tl.int64)
    b_ptrs = b_ptr + off_experts * stride_be + (offs_k[:, None] * stride_bk +
                                                offs_bn[None, :] * stride_bn)
    if use_int8_w8a16:
        b_scale_ptrs = b_scale_ptr + off_experts * stride_bse + offs_bn[
            None, :] * stride_bsn
        b_scale = tl.load(b_scale_ptrs)

    if use_fp8_w8a8 or use_int8_w8a8:
        # block-wise
        if group_k > 0 and group_n > 0:
            a_scale_ptrs = a_scale_ptr + (offs_token // top_k) * stride_asm
            offs_bsn = offs_bn // group_n
            b_scale_ptrs = (b_scale_ptr + off_experts * stride_bse +
                            offs_bsn * stride_bsn)
        # channel-wise
        elif per_channel_quant:
            b_scale_ptrs = b_scale_ptr + off_experts * stride_bse + offs_bn[
                None, :] * stride_bsn
            b_scale = tl.load(b_scale_ptrs)
            # Load per-token scale for activations
            a_scale_ptrs = a_scale_ptr + (offs_token // top_k) * stride_asm
            a_scale = tl.load(a_scale_ptrs, mask=token_mask, other=0.0)[:,
                                                                        None]
        # tensor-wise
        else:
            a_scale = tl.load(a_scale_ptr)
            b_scale = tl.load(b_scale_ptr + off_experts)

    # -----------------------------------------------------------
    # Iterate to compute a block of the C matrix.
    # We accumulate into a `[BLOCK_SIZE_M, BLOCK_SIZE_N]` block
    # of fp32 values for higher accuracy.
    # `accumulator` will be converted back to fp16 after the loop.
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Load the next block of A and B, generate a mask by checking the
        # K dimension.
        a = tl.load(a_ptrs,
                    mask=token_mask[:, None] &
                    (offs_k[None, :] < K - k * BLOCK_SIZE_K),
                    other=0.0)
        b = tl.load(b_ptrs,
                    mask=offs_k[:, None] < K - k * BLOCK_SIZE_K,
                    other=0.0)
        # We accumulate along the K dimension.
        if use_int8_w8a16:
            accumulator = tl.dot(a, b.to(compute_type), acc=accumulator)
        elif use_fp8_w8a8 or use_int8_w8a8:
            if group_k > 0 and group_n > 0:
                k_start = k * BLOCK_SIZE_K
                offs_ks = k_start // group_k
                a_scale = tl.load(a_scale_ptrs + offs_ks * stride_ask,
                                  mask=token_mask,
                                  other=0.0)
                b_scale = tl.load(b_scale_ptrs + offs_ks * stride_bsk)

                accumulator += tl.dot(a, b) * a_scale[:,
                                                      None] * b_scale[None, :]
            else:
                if use_fp8_w8a8:
                    # acc used to enable fp8_fast_accum
                    accumulator = tl.dot(a, b, acc=accumulator)
                else:
                    accumulator += tl.dot(a, b)
        else:
            accumulator += tl.dot(a, b)
        # Advance the ptrs to the next K block.
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    if MUL_ROUTED_WEIGHT:
        moe_weight = tl.load(topk_weights_ptr + offs_token,
                             mask=token_mask,
                             other=0)
        accumulator = accumulator * moe_weight[:, None]
    if use_int8_w8a16:
        accumulator = (accumulator * b_scale).to(compute_type)
    elif use_fp8_w8a8 or use_int8_w8a8:
        if group_k > 0 and group_n > 0:
            accumulator = accumulator.to(compute_type)
        else:
            accumulator = (accumulator * a_scale * b_scale).to(compute_type)
    else:
        accumulator = accumulator.to(compute_type)
    # -----------------------------------------------------------
    # Write back the block of the output
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn[
        None, :]
    c_mask = token_mask[:, None] & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)

def ceil_div(a, b):
    return (a + b - 1) // b



@triton.jit
def moe_align_block_size_stage1(
    topk_ids_ptr,
    tokens_cnts_ptr,
    num_experts: tl.constexpr,
    numel: tl.constexpr,
    tokens_per_thread: tl.constexpr,
):
    pid = tl.program_id(0)

    start_idx = pid * tokens_per_thread

    off_c = (pid + 1) * num_experts

    for i in range(tokens_per_thread):
        if start_idx + i < numel:
            idx = tl.load(topk_ids_ptr + start_idx + i)
            token_cnt = tl.load(tokens_cnts_ptr + off_c + idx)
            tl.store(tokens_cnts_ptr + off_c + idx, token_cnt + 1)


@triton.jit
def moe_align_block_size_stage2(
    tokens_cnts_ptr,
    num_experts: tl.constexpr,
):
    pid = tl.program_id(0)

    last_cnt = 0
    for i in range(1, num_experts + 1):
        token_cnt = tl.load(tokens_cnts_ptr + i * num_experts + pid)
        last_cnt = last_cnt + token_cnt
        tl.store(tokens_cnts_ptr + i * num_experts + pid, last_cnt)


@triton.jit
def moe_align_block_size_stage3(
    total_tokens_post_pad_ptr,
    tokens_cnts_ptr,
    cumsum_ptr,
    num_experts: tl.constexpr,
    block_size: tl.constexpr,
):
    last_cumsum = 0
    off_cnt = num_experts * num_experts
    for i in range(1, num_experts + 1):
        token_cnt = tl.load(tokens_cnts_ptr + off_cnt + i - 1)
        last_cumsum = last_cumsum + tl.cdiv(token_cnt, block_size) * block_size
        tl.store(cumsum_ptr + i, last_cumsum)
    tl.store(total_tokens_post_pad_ptr, last_cumsum)


@triton.jit
def moe_align_block_size_stage4(
    topk_ids_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    tokens_cnts_ptr,
    cumsum_ptr,
    num_experts: tl.constexpr,
    block_size: tl.constexpr,
    numel: tl.constexpr,
    tokens_per_thread: tl.constexpr,
):
    pid = tl.program_id(0)
    start_idx = tl.load(cumsum_ptr + pid)
    end_idx = tl.load(cumsum_ptr + pid + 1)

    for i in range(start_idx, end_idx, block_size):
        tl.store(expert_ids_ptr + i // block_size, pid)

    start_idx = pid * tokens_per_thread
    off_t = pid * num_experts

    for i in range(start_idx, tl.minimum(start_idx + tokens_per_thread,
                                         numel)):
        expert_id = tl.load(topk_ids_ptr + i)
        token_cnt = tl.load(tokens_cnts_ptr + off_t + expert_id)
        rank_post_pad = token_cnt + tl.load(cumsum_ptr + expert_id)
        tl.store(sorted_token_ids_ptr + rank_post_pad, i)
        tl.store(tokens_cnts_ptr + off_t + expert_id, token_cnt + 1)


# Triton implementation based on:
# https://github.com/sgl-project/sglang/commit/ba5112ff691d791a9e38c6c71f59324a5fcb49d0
def moe_align_block_size_triton(
    topk_ids: torch.Tensor,
    num_experts: int,
    block_size: int,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_pad: torch.Tensor,
) -> None:
    numel = topk_ids.numel()
    grid = (num_experts, )
    tokens_cnts = torch.zeros((num_experts + 1, num_experts),
                              dtype=torch.int32,
                              device=topk_ids.device)
    cumsum = torch.zeros((num_experts + 1, ),
                         dtype=torch.int32,
                         device=topk_ids.device)
    tokens_per_thread = ceil_div(numel, num_experts)

    moe_align_block_size_stage1[grid](
        topk_ids,
        tokens_cnts,
        num_experts,
        numel,
        tokens_per_thread,
    )
    moe_align_block_size_stage2[grid](
        tokens_cnts,
        num_experts,
    )
    moe_align_block_size_stage3[(1, )](
        num_tokens_post_pad,
        tokens_cnts,
        cumsum,
        num_experts,
        block_size,
    )
    moe_align_block_size_stage4[grid](
        topk_ids,
        sorted_token_ids,
        expert_ids,
        tokens_cnts,
        cumsum,
        num_experts,
        block_size,
        numel,
        tokens_per_thread,
    )



def moe_ep_align_block_size(
        topk_ids: torch.Tensor, block_size: int,
        num_experts: int, start_expert: int,
        end_expert: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Aligns the token distribution across experts to be compatible with block
    size for matrix multiplication.

    Parameters:
    - topk_ids: A tensor of shape [total_tokens, top_k] representing the
        top-k expert indices for each token.
    - block_size: The block size used in block matrix multiplication.
    - num_experts: The total number of experts.

    Returns:
    - sorted_token_ids: A tensor containing the sorted token indices according
        to their allocated expert.
    - expert_ids: A tensor indicating the assigned expert index for each block.
    - num_tokens_post_padded: The total number of tokens after padding,
        ensuring divisibility by block_size.

    This function pads the number of tokens that each expert needs to process
    so that it is divisible by block_size.
    Padding ensures that during block matrix multiplication, the dimensions
    align correctly.

    Example:
    Given topk_ids = [[2, 3, 4], [1, 2, 4], [1, 3, 4], [1, 2, 3]],
    block_size = 4, and num_experts = 4:
    - We initially have 12 tokens (after repeating 'top_k' times) and 4 experts,
        with each expert needing to process 3 tokens.
    - As block_size is 4, we pad 1 token for each expert.
    - First, flatten topk_ids to [2, 3, 4, 1, 2, 4, 1, 3, 4, 1, 2, 3].
    - Then append padding tokens [12, 12, 12, 12] for each block.
    - After sorting by expert index, we obtain token_ids
        [3, 6, 9, 12, 0, 4, 10, 12, 1, 7, 11, 12, 2, 5, 8, 12].
        Tokens 12 are non-existent (padding) and are ignored in
        the subsequent matrix multiplication.
    - The padding ensures that the total number of tokens is now divisible
        by block_size for proper block matrix operations.
    """
    max_num_tokens_padded = topk_ids.numel() + num_experts * (block_size - 1)
    sorted_ids = torch.empty((max_num_tokens_padded, ),
                             dtype=torch.int32,
                             device=topk_ids.device)
    sorted_ids.fill_(topk_ids.numel())
    max_num_m_blocks = triton.cdiv(max_num_tokens_padded, block_size)
    expert_ids = torch.empty((max_num_m_blocks, ),
                             dtype=torch.int32,
                             device=topk_ids.device)
    num_tokens_post_pad = torch.empty((1),
                                      dtype=torch.int32,
                                      device=topk_ids.device)
    ops.ep_moe_align_block_size(topk_ids, num_experts, block_size, sorted_ids,
                                 expert_ids, num_tokens_post_pad, start_expert, 
                                 end_expert)
        
    return sorted_ids, expert_ids, num_tokens_post_pad


def torch_moe(a, w1, C, topk, topk_ids, topk_weight):
    if topk_weight is None:
        B, D = a.shape
        a = a.view(B, -1, D).repeat(1, topk, 1).reshape(-1, D)
        out = torch.zeros(B * topk, w1.shape[1], dtype=torch.int32)
        topk_ids = topk_ids.view(-1)
        for i in range(w1.shape[0]):
            mask = topk_ids == i
            if mask.sum():
                out[mask] = a[mask].cpu().to(torch.int32) @ w1[i].cpu().transpose(0, 1).to(torch.int32)
        ref_out = out
    else:
        B, D = a.shape
        mm, topkk = topk_weight.shape
        out = torch.zeros(B, w1.shape[1], dtype=torch.int32)
        topk_ids = topk_ids.view(-1)
        for i in range(w1.shape[0]):
            mask = topk_ids == i
            if mask.sum():
                out[mask] = a[mask].cpu().to(torch.int32) @ w1[i].cpu().transpose(0, 1).to(torch.int32)
        if topk_weight is not None:
            topk_weight = topk_weight.view(-1)
            ref_out = out.view(B, -1, w1.shape[1]) * topk_weight.cpu().view(B, -1, 1)
            # ref_out = out
        ref_out = ref_out.view(mm, topkk, w1.shape[1])
    return ref_out.to(C.dtype).to('cuda')


def get_config_dtype_str(dtype: torch.dtype,
                         use_int4_w4a16: Optional[bool] = False,
                         use_int8_w8a16: Optional[bool] = False,
                         use_fp8_w8a8: Optional[bool] = False,
                         use_int8_w8a8: Optional[bool] = False,):
    if use_fp8_w8a8:
        return "fp8_w8a8"
    elif use_int8_w8a8:
        return "int8_w8a8"
    elif use_int8_w8a16:
        return "int8_w8a16"
    elif use_int4_w4a16:
        return "int4_w4a16"
    elif dtype == torch.float:
        # avoiding cases where kernel fails when float32 MoE
        # use fp16/bfloat16 configs
        return "float32"
    return None

def invoke_fused_moe_kernel_int8(A: torch.Tensor,
                            B: torch.Tensor,
                            C: torch.Tensor,
                            A_scale: Optional[torch.Tensor],
                            B_scale: Optional[torch.Tensor],
                            B_zp: Optional[torch.Tensor],
                            topk_weights: Optional[torch.Tensor],
                            topk_ids: torch.Tensor,
                            sorted_token_ids: torch.Tensor,
                            expert_ids: torch.Tensor,
                            num_tokens_post_padded: torch.Tensor,
                            mul_routed_weight: bool,
                            top_k: int,
                            config: Dict[str, Any],
                            compute_type: tl.dtype,
                            use_fp8_w8a8: bool,
                            use_int8_w8a8: bool,
                            use_int8_w8a16: bool,
                            use_int4_w4a16: bool,
                            per_channel_quant: bool,
                            block_shape: Optional[List[int]] = None,
                            use_nn_moe: Optional[bool]=False) -> None:
    assert topk_weights is not None or not mul_routed_weight
    assert topk_weights is None or topk_weights.stride(1) == 1
    assert sorted_token_ids.stride(0) == 1
    assert use_int8_w8a8 is True
    
    # M = A.shape[0]
    # num_tokens = M * top_k

    # EM = sorted_token_ids.shape[0]
    # if A.shape[0] < config["BLOCK_SIZE_M"]:
    #     # optimize for small batch_size.
    #     # We assume that top_ids of each token is unique, so
    #     # so num_valid_experts <= batch_size <= BLOCK_SIZE_M,
    #     # and we can skip some invalid blocks.
    #     EM = min(sorted_token_ids.shape[0],
    #              A.shape[0] * top_k * config['BLOCK_SIZE_M'])
    # grid = lambda META: (triton.cdiv(EM, META['BLOCK_SIZE_M']) * triton.cdiv(
    #     B.shape[1] if not use_nn_moe else B.shape[2], META['BLOCK_SIZE_N']), )

    # fused_moe_kernel[grid](
    #         A,
    #         B,
    #         C,
    #         A_scale,
    #         B_scale,
    #         topk_weights,
    #         sorted_token_ids,
    #         expert_ids,
    #         num_tokens_post_padded,
    #         B.shape[1] if not use_nn_moe else B.shape[2],
    #         A.shape[1],
    #         EM,
    #         num_tokens,
    #         A.stride(0),
    #         A.stride(1),
    #         B.stride(0),
    #         B.stride(2) if not use_nn_moe else B.stride(1),
    #         B.stride(1) if not use_nn_moe else B.stride(2),
    #         C.stride(1),
    #         C.stride(2),
    #         A_scale.stride(0)
    #         if A_scale is not None and A_scale.ndim == 2 else 0,
    #         A_scale.stride(1)
    #         if A_scale is not None and A_scale.ndim == 2 else 0,
    #         B_scale.stride(0)
    #         if B_scale is not None and B_scale.ndim >= 2 else 0,
    #         B_scale.stride(2)
    #         if B_scale is not None and B_scale.ndim == 3 else 0,
    #         B_scale.stride(1)
    #         if B_scale is not None and B_scale.ndim >= 2 else 0,
    #         0 if block_shape is None else block_shape[0],
    #         0 if block_shape is None else block_shape[1],
    #         MUL_ROUTED_WEIGHT=mul_routed_weight,
    #         top_k=top_k,
    #         compute_type=compute_type,
    #         use_fp8_w8a8=use_fp8_w8a8,
    #         use_int8_w8a8=use_int8_w8a8,
    #         use_int8_w8a16=use_int8_w8a16,
    #         per_channel_quant=per_channel_quant,
    #         **config
    #     )

    moe_gemm_w8a8(A, 
                    B, 
                    C,
                    A_scale, 
                    B_scale,
                    topk_weights if mul_routed_weight else None,
                    sorted_token_ids, 
                    expert_ids,
                    num_tokens_post_padded, 
                    top_k,
                    config)

def _fused_moegemm_helper_test_first_stage(hidden_states: torch.Tensor,
                       w1: torch.Tensor,
                       w2: torch.Tensor,
                       topk_weights: torch.Tensor,
                       topk_ids: torch.Tensor,
                       inplace: bool = False,
                       activation: str = "silu",
                       apply_router_weight_on_input: bool = False,
                       use_fp8_w8a8: bool = False,
                       use_int8_w8a8: bool = False,
                       use_int8_w8a16: bool = False,
                       use_int4_w4a16: bool = False,
                       per_channel_quant: bool = False,
                       global_num_experts: int = -1,
                       expert_map: Optional[torch.Tensor] = None,
                       w1_scale: Optional[torch.Tensor] = None,
                       w2_scale: Optional[torch.Tensor] = None,
                       w1_zp: Optional[torch.Tensor] = None,
                       w2_zp: Optional[torch.Tensor] = None,
                       a1_scale: Optional[torch.Tensor] = None,
                       a2_scale: Optional[torch.Tensor] = None,
                       block_shape: Optional[List[int]] = None,
                       use_nn_moe: Optional[bool] = False,
                       config: Optional[Dict[str, Any]] = None,
                       mode: Optional[int]=0):
    # Check constraints. 
    assert use_int8_w8a8 is True
    assert hidden_states.shape[1] == w1.shape[2], "Hidden size mismatch"

    assert topk_weights.shape == topk_ids.shape, "topk shape mismatch"
    assert hidden_states.is_contiguous(), "Hidden_states must be contiguous"
    assert w1.stride(-1) == 1, "Stride of last dimension must be 1"
    assert w2.stride(-1) == 1, "Stride of last dimension must be 1"
    assert hidden_states.dtype in [
        torch.float32, torch.float16, torch.bfloat16
    ]

    num_tokens, _ = hidden_states.shape

    if use_nn_moe:
        E, _, N = w1.shape
    else:
        E, N, _ = w1.shape

    K = w2.shape[1]
    if global_num_experts == -1:
        global_num_experts = E
    top_k_num = topk_ids.shape[1]
    # We execute the fused_moe kernel in chunks to circumvent this issue:
    # https://github.com/vllm-project/vllm/issues/5938
    CHUNK_SIZE = envs.VLLM_FUSED_MOE_CHUNK_SIZE
    M = min(num_tokens, CHUNK_SIZE)
    # We can reuse the memory between these because by the time we need
    # cache3, we're done with cache1
    cache13 = torch.empty(M * top_k_num * max(N, K if not use_nn_moe else w2.shape[2]),
                          device=hidden_states.device,
                          dtype=hidden_states.dtype)
    intermediate_cache1 = cache13[:M * top_k_num * N].view(M, top_k_num, N)
    intermediate_cache3 = cache13[:M * top_k_num * (K if not use_nn_moe else w2.shape[2])].view(M, top_k_num, K if not use_nn_moe else w2.shape[2])

    # This needs separate memory since it's used concurrently with cache1
    intermediate_cache2 = torch.empty((M * top_k_num, N // 2),
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

    for chunk in range((num_tokens // CHUNK_SIZE) + 1):
        begin_chunk_idx, end_chunk_idx = (chunk * CHUNK_SIZE,
                                          min((chunk + 1) * CHUNK_SIZE,
                                              num_tokens))
        curr_hidden_states = hidden_states[begin_chunk_idx:end_chunk_idx]
        tokens_in_chunk, _ = curr_hidden_states.shape
    
        if tokens_in_chunk == 0:
            break

        if tokens_in_chunk < CHUNK_SIZE and chunk > 0:
            # Adjust the intermediate cache size and config for the last
            # chunk. Note that in most cases we only have one chunk
            # so the cache size and config are already set correctly and
            # do not need to be adjusted.
            intermediate_cache1 = intermediate_cache1[:tokens_in_chunk]
            intermediate_cache2 = intermediate_cache2[:tokens_in_chunk *
                                                      topk_ids.shape[1]]
            intermediate_cache3 = intermediate_cache3[:tokens_in_chunk]

        M_pattern = "MODE_DECODE"
        if M > M_THRESHOLD:
            M_pattern = "MODE_PREFILL"
        if use_int8_w8a8:
            config1 = {}
            config2 = {}
            M=curr_hidden_states.shape[0]
            #print(f'+++++++-------_{M}_--------+++++++')
            if M > M_THRESHOLD:
                config1['BLOCK_SIZE_M'] = blockM_configs[config[M_pattern]]
                config1['MODE']         = config[M_pattern]
                config1['DELTA']        = 1
                config['BLOCK_SIZE_M']  = blockM_configs[config[M_pattern]]
            else:
                config1['BLOCK_SIZE_M'] = blockM_configs_decode[config[M_pattern]]
                config1['MODE']         = config[M_pattern]
                config1['DELTA']        = 1
                config['BLOCK_SIZE_M']  = blockM_configs_decode[config[M_pattern]]
            
        curr_topk_ids = topk_ids[begin_chunk_idx:end_chunk_idx]
        curr_topk_weights = topk_weights[begin_chunk_idx:end_chunk_idx]

        qcurr_hidden_states, qa1_scale = moe_kernel_prepare_input(
            A=curr_hidden_states,
            B=w1,
            A_scale=a1_scale,
            B_scale=w1_scale,
            use_fp8_w8a8=use_fp8_w8a8,
            use_int8_w8a8=use_int8_w8a8,
            use_int8_w8a16=use_int8_w8a16,
            use_int4_w4a16=use_int4_w4a16,
            per_channel_quant=per_channel_quant,
            block_shape=block_shape)

        sorted_token_ids, expert_ids, num_tokens_post_padded = (
                moe_align_block_size(curr_topk_ids, config1['BLOCK_SIZE_M'],
                                    global_num_experts, expert_map))

        invoke_fused_moe_kernel_int8(qcurr_hidden_states,
                                w1,
                                intermediate_cache1,
                                qa1_scale,
                                w1_scale,
                                w1_zp,
                                curr_topk_weights,
                                curr_topk_ids,
                                sorted_token_ids,
                                expert_ids,
                                num_tokens_post_padded,
                                apply_router_weight_on_input,
                                top_k_num,
                                config1,
                                compute_type=compute_type,
                                use_fp8_w8a8=use_fp8_w8a8,
                                use_int8_w8a8=use_int8_w8a8,
                                use_int8_w8a16=use_int8_w8a16,
                                use_int4_w4a16=use_int4_w4a16,
                                per_channel_quant=per_channel_quant,
                                block_shape=block_shape,
                                use_nn_moe=use_nn_moe)
        
        quantiles = [0.5, 0.2, 0.8]
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: invoke_fused_moe_kernel_int8(qcurr_hidden_states,
                                                                                            w1,
                                                                                            intermediate_cache1,
                                                                                            qa1_scale,
                                                                                            w1_scale,
                                                                                            w1_zp,
                                                                                            curr_topk_weights,
                                                                                            curr_topk_ids,
                                                                                            sorted_token_ids,
                                                                                            expert_ids,
                                                                                            num_tokens_post_padded,
                                                                                            apply_router_weight_on_input,
                                                                                            top_k_num,
                                                                                            config1,
                                                                                            compute_type=compute_type,
                                                                                            use_fp8_w8a8=use_fp8_w8a8,
                                                                                            use_int8_w8a8=use_int8_w8a8,
                                                                                            use_int8_w8a16=use_int8_w8a16,
                                                                                            use_int4_w4a16=use_int4_w4a16,
                                                                                            per_channel_quant=per_channel_quant,
                                                                                            block_shape=block_shape,
                                                                                            use_nn_moe=use_nn_moe),quantiles=quantiles)
       
        elapsed_time = ms
        print("w13 _time:{} ms\n".format(elapsed_time))
        
        
    return elapsed_time   

def _fused_moegemm_helper_test_second_stage(hidden_states: torch.Tensor,
                       w1: torch.Tensor,
                       w2: torch.Tensor,
                       topk_weights: torch.Tensor,
                       topk_ids: torch.Tensor,
                       inplace: bool = False,
                       activation: str = "silu",
                       apply_router_weight_on_input: bool = False,
                       use_fp8_w8a8: bool = False,
                       use_int8_w8a8: bool = False,
                       use_int8_w8a16: bool = False,
                       use_int4_w4a16: bool = False,
                       per_channel_quant: bool = False,
                       global_num_experts: int = -1,
                       expert_map: Optional[torch.Tensor] = None,
                       w1_scale: Optional[torch.Tensor] = None,
                       w2_scale: Optional[torch.Tensor] = None,
                       w1_zp: Optional[torch.Tensor] = None,
                       w2_zp: Optional[torch.Tensor] = None,
                       a1_scale: Optional[torch.Tensor] = None,
                       a2_scale: Optional[torch.Tensor] = None,
                       block_shape: Optional[List[int]] = None,
                       use_nn_moe: Optional[bool] = False,
                       config: Optional[Dict[str, Any]] = None,
                       mode: Optional[int]=0):
    # Check constraints. 
    assert use_int8_w8a8 is True
    assert hidden_states.shape[1] == w1.shape[2], "Hidden size mismatch"

    assert topk_weights.shape == topk_ids.shape, "topk shape mismatch"
    assert hidden_states.is_contiguous(), "Hidden_states must be contiguous"
    assert w1.stride(-1) == 1, "Stride of last dimension must be 1"
    assert w2.stride(-1) == 1, "Stride of last dimension must be 1"
    assert hidden_states.dtype in [
        torch.float32, torch.float16, torch.bfloat16
    ]

    num_tokens, _ = hidden_states.shape

    if use_nn_moe:
        E, _, N = w1.shape
    else:
        E, N, _ = w1.shape

    K = w2.shape[1]
    if global_num_experts == -1:
        global_num_experts = E
    top_k_num = topk_ids.shape[1]
    # We execute the fused_moe kernel in chunks to circumvent this issue:
    # https://github.com/vllm-project/vllm/issues/5938
    CHUNK_SIZE = envs.VLLM_FUSED_MOE_CHUNK_SIZE
    M = min(num_tokens, CHUNK_SIZE)
    # We can reuse the memory between these because by the time we need
    # cache3, we're done with cache1
    cache13 = torch.empty(M * top_k_num * max(N, K if not use_nn_moe else w2.shape[2]),
                          device=hidden_states.device,
                          dtype=hidden_states.dtype)
    intermediate_cache1 = cache13[:M * top_k_num * N].view(M, top_k_num, N)
    intermediate_cache3 = cache13[:M * top_k_num * (K if not use_nn_moe else w2.shape[2])].view(M, top_k_num, K if not use_nn_moe else w2.shape[2])

    # This needs separate memory since it's used concurrently with cache1
    intermediate_cache2 = torch.empty((M * top_k_num, N // 2),
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

    for chunk in range((num_tokens // CHUNK_SIZE) + 1):
        begin_chunk_idx, end_chunk_idx = (chunk * CHUNK_SIZE,
                                          min((chunk + 1) * CHUNK_SIZE,
                                              num_tokens))
        curr_hidden_states = hidden_states[begin_chunk_idx:end_chunk_idx]
        tokens_in_chunk, _ = curr_hidden_states.shape
    
        if tokens_in_chunk == 0:
            break

        if tokens_in_chunk < CHUNK_SIZE and chunk > 0:
            # Adjust the intermediate cache size and config for the last
            # chunk. Note that in most cases we only have one chunk
            # so the cache size and config are already set correctly and
            # do not need to be adjusted.
            intermediate_cache1 = intermediate_cache1[:tokens_in_chunk]
            intermediate_cache2 = intermediate_cache2[:tokens_in_chunk *
                                                      topk_ids.shape[1]]
            intermediate_cache3 = intermediate_cache3[:tokens_in_chunk]

        config1={}
        config2={}
        M_pattern = "MODE_DECODE"
        if M > M_THRESHOLD:
            M_pattern = "MODE_PREFILL"  
        if use_int8_w8a8:
            M=curr_hidden_states.shape[0]
            #print(f'+++++++-------_{M}_--------+++++++')
            if M > M_THRESHOLD:
                config1['BLOCK_SIZE_M'] = blockM_configs[config[M_pattern]]
                config1['MODE']         = config[M_pattern]
                config1['DELTA']        = 1
                config2['BLOCK_SIZE_M'] = blockM_configs[config[M_pattern]]
                config2['MODE']         = config[M_pattern]
                config2['DELTA']        = 1
                config['BLOCK_SIZE_M']  = blockM_configs[config[M_pattern]]
            else:
                config1['BLOCK_SIZE_M'] = blockM_configs_decode[config[M_pattern]]
                config1['MODE']         = config[M_pattern]
                config1['DELTA']        = 1
                config2['BLOCK_SIZE_M'] = blockM_configs_decode[config[M_pattern]]
                config2['MODE']         = config[M_pattern]
                config2['DELTA']        = 1
                config['BLOCK_SIZE_M']  = blockM_configs_decode[config[M_pattern]]
            
        curr_topk_ids = topk_ids[begin_chunk_idx:end_chunk_idx]
        curr_topk_weights = topk_weights[begin_chunk_idx:end_chunk_idx]

        qcurr_hidden_states, qa1_scale = moe_kernel_prepare_input(
            A=curr_hidden_states,
            B=w1,
            A_scale=a1_scale,
            B_scale=w1_scale,
            use_fp8_w8a8=use_fp8_w8a8,
            use_int8_w8a8=use_int8_w8a8,
            use_int8_w8a16=use_int8_w8a16,
            use_int4_w4a16=use_int4_w4a16,
            per_channel_quant=per_channel_quant,
            block_shape=block_shape)

        sorted_token_ids, expert_ids, num_tokens_post_padded = (
                moe_align_block_size(curr_topk_ids, config1['BLOCK_SIZE_M'],
                                    global_num_experts, expert_map))

        invoke_fused_moe_kernel_int8(qcurr_hidden_states,
                                w1,
                                intermediate_cache1,
                                qa1_scale,
                                w1_scale,
                                w1_zp,
                                curr_topk_weights,
                                curr_topk_ids,
                                sorted_token_ids,
                                expert_ids,
                                num_tokens_post_padded,
                                apply_router_weight_on_input,
                                top_k_num,
                                config1,
                                compute_type=compute_type,
                                use_fp8_w8a8=use_fp8_w8a8,
                                use_int8_w8a8=use_int8_w8a8,
                                use_int8_w8a16=use_int8_w8a16,
                                use_int4_w4a16=use_int4_w4a16,
                                per_channel_quant=per_channel_quant,
                                block_shape=block_shape,
                                use_nn_moe=use_nn_moe)
        
        if activation == "silu":
            torch.ops._C.silu_and_mul(intermediate_cache2,
                                      intermediate_cache1.view(-1, N))
        elif activation == "gelu":
            torch.ops._C.gelu_and_mul(intermediate_cache2,
                                      intermediate_cache1.view(-1, N))
        else:
            raise ValueError(f"Unsupported FusedMoe activation: {activation}")

        qintermediate_cache2, qa2_scale = moe_kernel_prepare_input(
            A=intermediate_cache2,
            B=w2,
            A_scale=a2_scale,
            B_scale=w2_scale,
            use_fp8_w8a8=use_fp8_w8a8,
            use_int8_w8a8=use_int8_w8a8,
            use_int8_w8a16=use_int8_w8a16,
            use_int4_w4a16=use_int4_w4a16,
            per_channel_quant=per_channel_quant,
            block_shape=block_shape)

        invoke_fused_moe_kernel_int8(qintermediate_cache2,
                                w2,
                                intermediate_cache3,
                                qa2_scale,
                                w2_scale,
                                w2_zp,
                                curr_topk_weights,
                                curr_topk_ids,
                                sorted_token_ids,
                                expert_ids,
                                num_tokens_post_padded,
                                not apply_router_weight_on_input,
                                1,
                                config2,
                                compute_type=compute_type,
                                use_fp8_w8a8=use_fp8_w8a8,
                                use_int8_w8a8=use_int8_w8a8,
                                use_int8_w8a16=use_int8_w8a16,
                                use_int4_w4a16=use_int4_w4a16,
                                per_channel_quant=per_channel_quant,
                                block_shape=block_shape,
                                use_nn_moe=use_nn_moe)
        quantiles = [0.5, 0.2, 0.8]
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: invoke_fused_moe_kernel_int8(qintermediate_cache2,
                                                                                            w2,
                                                                                            intermediate_cache3,
                                                                                            qa2_scale,
                                                                                            w2_scale,
                                                                                            w2_zp,
                                                                                            curr_topk_weights,
                                                                                            curr_topk_ids,
                                                                                            sorted_token_ids,
                                                                                            expert_ids,
                                                                                            num_tokens_post_padded,
                                                                                            not apply_router_weight_on_input,
                                                                                            1,
                                                                                            config2,
                                                                                            compute_type=compute_type,
                                                                                            use_fp8_w8a8=use_fp8_w8a8,
                                                                                            use_int8_w8a8=use_int8_w8a8,
                                                                                            use_int8_w8a16=use_int8_w8a16,
                                                                                            use_int4_w4a16=use_int4_w4a16,
                                                                                            per_channel_quant=per_channel_quant,
                                                                                            block_shape=block_shape,
                                                                                            use_nn_moe=use_nn_moe),quantiles=quantiles)
       
        elapsed_time = ms
        print("w2 _time:{} ms\n".format(elapsed_time))
        torch.cuda.synchronize()
        ops.moe_sum(intermediate_cache3.view(*intermediate_cache3.shape),
                    out_hidden_states[begin_chunk_idx:end_chunk_idx])
        
        
    return elapsed_time   

def fused_moe_first_stage_tune(
                                hidden_states: torch.Tensor,
                                w1: torch.Tensor,
                                w2: torch.Tensor,
                                gating_output: torch.Tensor,
                                topk: int,
                                renormalize: bool,
                                inplace: bool = False,
                                use_grouped_topk: bool = False,
                                num_expert_group: Optional[int] = None,
                                topk_group: Optional[int] = None,
                                custom_routing_function: Optional[Callable] = None,
                                use_fp8_w8a8: bool = False,
                                use_int8_w8a8: bool = False,
                                use_int8_w8a16: bool = False,
                                use_int4_w4a16: bool = False,
                                w1_scale: Optional[torch.Tensor] = None,
                                w2_scale: Optional[torch.Tensor] = None,
                                w1_zp: Optional[torch.Tensor] = None,
                                w2_zp: Optional[torch.Tensor] = None,
                                a1_scale: Optional[torch.Tensor] = None,
                                a2_scale: Optional[torch.Tensor] = None,
                                block_shape: Optional[List[int]] = None,
                                use_nn_moe: Optional[bool] = False,
                                moe_ep_size: Optional[int] = 1,
                                start_expert: Optional[int] = None,
                                end_expert: Optional[int] = None,
                                config:  Dict[str, Any] = None,
) -> torch.Tensor:
    """
    This function computes a Mixture of Experts (MoE) layer using two sets of
    weights, w1 and w2, and top-k gating mechanism.

    Parameters:
    - hidden_states (torch.Tensor): The input tensor to the MoE layer.
    - w1 (torch.Tensor): The first set of expert weights.
    - w2 (torch.Tensor): The second set of expert weights.
    - gating_output (torch.Tensor): The output of the gating operation
        (before softmax).
    - topk (int): The number of top-k experts to select.
    - renormalize (bool): If True, renormalize the top-k weights to sum to 1.
    - inplace (bool): If True, perform the operation in-place.
        Defaults to False.
    - num_expert_group: Optional[int]: additional parameter for grouped_topk
    - topk_group: Optional[int]: additional parameter for grouped_topk
    - use_grouped_topk: If True, use grouped_topk instead of fused_topk
        note: Deepseekv2 model uses grouped_topk
    - use_fp8_w8a8 (bool): If True, use fp8 arithmetic to compute the inner
        products for w1 and w2. Defaults to False.
    - use_int8_w8a16 (bool): If True, use matmul of int8 weight and bf16/fp16
        activation to compute the inner products for w1 and w2.
        Defaults to False.
    - use_int4_w4a16 (bool): If True, use matmul of int4 weight and bf16/fp16
        activation to compute the inner products for w1 and w2.
        Defaults to False.
    - w1_scale (Optional[torch.Tensor]): Optional scale to be used for
        w1.
    - w2_scale (Optional[torch.Tensor]): Optional scale to be used for
        w2.
    - a1_scale (Optional[torch.Tensor]): Optional scale to be used for
        a1.
    - a2_scale (Optional[torch.Tensor]): Optional scale to be used for
        a2.
    - block_shape: (Optional[List[int]]): Optional block size for block-wise
        quantization.

    Returns:
    - torch.Tensor: The output tensor after applying the MoE layer.
    """
    # Check constraints.
    assert gating_output.shape[1] == w1.shape[0], "Number of experts mismatch"

    if use_grouped_topk:
        assert num_expert_group is not None and topk_group is not None
        topk_weights, topk_ids = grouped_topk(hidden_states, gating_output,
                                              topk, renormalize,
                                              num_expert_group, topk_group)
    elif custom_routing_function is None:
        topk_weights, topk_ids = fused_topk(hidden_states, gating_output, topk,
                                            renormalize)
    else:
        topk_weights, topk_ids = custom_routing_function(
            hidden_states, gating_output, topk, renormalize)
    return _fused_moegemm_helper_test_first_stage(hidden_states=hidden_states,
                                                    w1=w1,
                                                    w2=w2,
                                                    topk_weights=topk_weights,
                                                    topk_ids=topk_ids,
                                                    inplace=False,
                                                    use_int8_w8a8=True,
                                                    w1_scale=w1_scale,
                                                    w2_scale=w2_scale,
                                                    block_shape=block_shape,
                                                    config=config,
                                                    per_channel_quant=True)

def fused_moe_second_stage_tune(
                                hidden_states: torch.Tensor,
                                w1: torch.Tensor,
                                w2: torch.Tensor,
                                gating_output: torch.Tensor,
                                topk: int,
                                renormalize: bool,
                                inplace: bool = False,
                                use_grouped_topk: bool = False,
                                num_expert_group: Optional[int] = None,
                                topk_group: Optional[int] = None,
                                custom_routing_function: Optional[Callable] = None,
                                use_fp8_w8a8: bool = False,
                                use_int8_w8a8: bool = False,
                                use_int8_w8a16: bool = False,
                                use_int4_w4a16: bool = False,
                                w1_scale: Optional[torch.Tensor] = None,
                                w2_scale: Optional[torch.Tensor] = None,
                                w1_zp: Optional[torch.Tensor] = None,
                                w2_zp: Optional[torch.Tensor] = None,
                                a1_scale: Optional[torch.Tensor] = None,
                                a2_scale: Optional[torch.Tensor] = None,
                                block_shape: Optional[List[int]] = None,
                                use_nn_moe: Optional[bool] = False,
                                moe_ep_size: Optional[int] = 1,
                                start_expert: Optional[int] = None,
                                end_expert: Optional[int] = None,
                                config:  Dict[str, Any] = None,
) -> torch.Tensor:
    """
    This function computes a Mixture of Experts (MoE) layer using two sets of
    weights, w1 and w2, and top-k gating mechanism.

    Parameters:
    - hidden_states (torch.Tensor): The input tensor to the MoE layer.
    - w1 (torch.Tensor): The first set of expert weights.
    - w2 (torch.Tensor): The second set of expert weights.
    - gating_output (torch.Tensor): The output of the gating operation
        (before softmax).
    - topk (int): The number of top-k experts to select.
    - renormalize (bool): If True, renormalize the top-k weights to sum to 1.
    - inplace (bool): If True, perform the operation in-place.
        Defaults to False.
    - num_expert_group: Optional[int]: additional parameter for grouped_topk
    - topk_group: Optional[int]: additional parameter for grouped_topk
    - use_grouped_topk: If True, use grouped_topk instead of fused_topk
        note: Deepseekv2 model uses grouped_topk
    - use_fp8_w8a8 (bool): If True, use fp8 arithmetic to compute the inner
        products for w1 and w2. Defaults to False.
    - use_int8_w8a16 (bool): If True, use matmul of int8 weight and bf16/fp16
        activation to compute the inner products for w1 and w2.
        Defaults to False.
    - use_int4_w4a16 (bool): If True, use matmul of int4 weight and bf16/fp16
        activation to compute the inner products for w1 and w2.
        Defaults to False.
    - w1_scale (Optional[torch.Tensor]): Optional scale to be used for
        w1.
    - w2_scale (Optional[torch.Tensor]): Optional scale to be used for
        w2.
    - a1_scale (Optional[torch.Tensor]): Optional scale to be used for
        a1.
    - a2_scale (Optional[torch.Tensor]): Optional scale to be used for
        a2.
    - block_shape: (Optional[List[int]]): Optional block size for block-wise
        quantization.

    Returns:
    - torch.Tensor: The output tensor after applying the MoE layer.
    """
    # Check constraints.
    assert gating_output.shape[1] == w1.shape[0], "Number of experts mismatch"

    if use_grouped_topk:
        assert num_expert_group is not None and topk_group is not None
        topk_weights, topk_ids = grouped_topk(hidden_states, gating_output,
                                              topk, renormalize,
                                              num_expert_group, topk_group)
    elif custom_routing_function is None:
        topk_weights, topk_ids = fused_topk(hidden_states, gating_output, topk,
                                            renormalize)
    else:
        topk_weights, topk_ids = custom_routing_function(
            hidden_states, gating_output, topk, renormalize)
    return _fused_moegemm_helper_test_second_stage(hidden_states=hidden_states,
                                                    w1=w1,
                                                    w2=w2,
                                                    topk_weights=topk_weights,
                                                    topk_ids=topk_ids,
                                                    inplace=False,
                                                    use_int8_w8a8=True,
                                                    w1_scale=w1_scale,
                                                    w2_scale=w2_scale,
                                                    block_shape=block_shape,
                                                    config=config,
                                                    per_channel_quant=True)

@functools.lru_cache(maxsize=128)
def get_config_mode(num_token):
    # 这里修改tuning的config
    mode1 = 0
    mode2 = 0
    token_mode_config = {
        1: [26, 0],
        2: [26, 0],
        4: [3, 0],
        8: [0, 0],
        16: [0, 0],
        32: [0, 0],
        64: [3, 0],
        128: [0, 0],
        256: [0, 0],
        512: [33, 20],
        1024: [11, 20],
        2048: [21, 23],
        4096: [21, 23],
        6144: [21, 23],
        8192: [21, 23],
        16384: [21, 23],
        32768: [21, 21],
    }
    # 找到最接近的token数
    min_diff = float('inf')
    closest_token = None
    for token in token_mode_config.keys():
        diff = abs(token - num_token)
        if diff < min_diff:
            min_diff = diff
            closest_token = token
    # 获取对应的配置
    if closest_token is not None:
        mode1, mode2 = token_mode_config[closest_token]
    assert blockM_configs[mode1] == blockM_configs[mode2], "mode0 and mode1 blockm must be equal!"
    return mode1, mode2

def _fused_moegemm_helper_test_pure(hidden_states: torch.Tensor,
                       w1: torch.Tensor,
                       w2: torch.Tensor,
                       topk_weights: torch.Tensor,
                       topk_ids: torch.Tensor,
                       inplace: bool = False,
                       activation: str = "silu",
                       apply_router_weight_on_input: bool = False,
                       use_fp8_w8a8: bool = False,
                       use_int8_w8a8: bool = False,
                       use_int8_w8a16: bool = False,
                       use_int4_w4a16: bool = False,
                       per_channel_quant: bool = False,
                       global_num_experts: int = -1,
                       expert_map: Optional[torch.Tensor] = None,
                       w1_scale: Optional[torch.Tensor] = None,
                       w2_scale: Optional[torch.Tensor] = None,
                       w1_zp: Optional[torch.Tensor] = None,
                       w2_zp: Optional[torch.Tensor] = None,
                       a1_scale: Optional[torch.Tensor] = None,
                       a2_scale: Optional[torch.Tensor] = None,
                       block_shape: Optional[List[int]] = None,
                       use_nn_moe: Optional[bool] = False,
                       config: Optional[Dict[str, Any]] = None,
                       mode: Optional[int]=0):
    # Check constraints. 
    assert use_int8_w8a8 is True
    assert hidden_states.shape[1] == w1.shape[2], "Hidden size mismatch"

    assert topk_weights.shape == topk_ids.shape, "topk shape mismatch"
    assert hidden_states.is_contiguous(), "Hidden_states must be contiguous"
    assert w1.stride(-1) == 1, "Stride of last dimension must be 1"
    assert w2.stride(-1) == 1, "Stride of last dimension must be 1"
    assert hidden_states.dtype in [
        torch.float32, torch.float16, torch.bfloat16
    ]

    num_tokens, _ = hidden_states.shape

    if use_nn_moe:
        E, _, N = w1.shape
    else:
        E, N, _ = w1.shape

    K = w2.shape[1]
    if global_num_experts == -1:
        global_num_experts = E
    top_k_num = topk_ids.shape[1]
    # We execute the fused_moe kernel in chunks to circumvent this issue:
    # https://github.com/vllm-project/vllm/issues/5938
    CHUNK_SIZE = envs.VLLM_FUSED_MOE_CHUNK_SIZE
    M = min(num_tokens, CHUNK_SIZE)
    # We can reuse the memory between these because by the time we need
    # cache3, we're done with cache1
    cache13 = torch.empty(M * top_k_num * max(N, K if not use_nn_moe else w2.shape[2]),
                          device=hidden_states.device,
                          dtype=hidden_states.dtype)
    intermediate_cache1 = cache13[:M * top_k_num * N].view(M, top_k_num, N)
    intermediate_cache3 = cache13[:M * top_k_num * (K if not use_nn_moe else w2.shape[2])].view(M, top_k_num, K if not use_nn_moe else w2.shape[2])

    # This needs separate memory since it's used concurrently with cache1
    intermediate_cache2 = torch.empty((M * top_k_num, N // 2),
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

    for chunk in range((num_tokens // CHUNK_SIZE) + 1):
        begin_chunk_idx, end_chunk_idx = (chunk * CHUNK_SIZE,
                                          min((chunk + 1) * CHUNK_SIZE,
                                              num_tokens))
        curr_hidden_states = hidden_states[begin_chunk_idx:end_chunk_idx]
        tokens_in_chunk, _ = curr_hidden_states.shape
    
        if tokens_in_chunk == 0:
            break

        if tokens_in_chunk < CHUNK_SIZE and chunk > 0:
            # Adjust the intermediate cache size and config for the last
            # chunk. Note that in most cases we only have one chunk
            # so the cache size and config are already set correctly and
            # do not need to be adjusted.
            intermediate_cache1 = intermediate_cache1[:tokens_in_chunk]
            intermediate_cache2 = intermediate_cache2[:tokens_in_chunk *
                                                      topk_ids.shape[1]]
            intermediate_cache3 = intermediate_cache3[:tokens_in_chunk]

        # if use_int8_w8a8:
        #     M=curr_hidden_states.shape[0]
        #     #print(f'+++++++-------_{M}_--------+++++++')
        #     # config=get_moe_config(M,N,K)
        #     config,status=get_moe_cuda_config(E,M,N,K,device_name,num_cus,compute_type)
            
        if use_int8_w8a8:
            if use_nn_moe:
                _, _K, _N = w2.shape
            else:
                _, _N, _K = w2.shape
            # _M=curr_hidden_states.shape[0]
            # config=get_moe_config(_M,_N,_K)
            config1, config2, status=get_moe_cuda_config(E,M,N,K,_N,_K,top_k_num,device_name,num_cus,compute_type)
            
        curr_topk_ids = topk_ids[begin_chunk_idx:end_chunk_idx]
        curr_topk_weights = topk_weights[begin_chunk_idx:end_chunk_idx]

        qcurr_hidden_states, qa1_scale = moe_kernel_prepare_input(
            A=curr_hidden_states,
            B=w1,
            A_scale=a1_scale,
            B_scale=w1_scale,
            use_fp8_w8a8=use_fp8_w8a8,
            use_int8_w8a8=use_int8_w8a8,
            use_int8_w8a16=use_int8_w8a16,
            use_int4_w4a16=use_int4_w4a16,
            per_channel_quant=per_channel_quant,
            block_shape=block_shape)
        #这里
        align = 64
        config1['MODE'] = 1000
        sorted_token_ids, expert_ids, num_tokens_post_padded = (
                moe_align_block_size(curr_topk_ids, align,
                                    global_num_experts, expert_map))
        
        # sorted_token_ids_2, expert_ids_2, num_tokens_post_padded_2 = (
        #         moe_align_block_size(curr_topk_ids, config1['BLOCK_SIZE_M'],
        #                             global_num_experts, expert_map))
        
        # import pdb
        # pdb.set_trace()
        invoke_fused_moe_kernel_int8(qcurr_hidden_states,
                                w1,
                                intermediate_cache1,
                                qa1_scale,
                                w1_scale,
                                w1_zp,
                                curr_topk_weights,
                                curr_topk_ids,
                                sorted_token_ids,
                                expert_ids,
                                num_tokens_post_padded,
                                apply_router_weight_on_input,
                                top_k_num,
                                config1,
                                compute_type=compute_type,
                                use_fp8_w8a8=use_fp8_w8a8,
                                use_int8_w8a8=use_int8_w8a8,
                                use_int8_w8a16=use_int8_w8a16,
                                use_int4_w4a16=use_int4_w4a16,
                                per_channel_quant=per_channel_quant,
                                block_shape=block_shape,
                                use_nn_moe=use_nn_moe)
        '''
        if activation == "silu":
            torch.ops._C.silu_and_mul(intermediate_cache2,
                                      intermediate_cache1.view(-1, N))
        elif activation == "gelu":
            torch.ops._C.gelu_and_mul(intermediate_cache2,
                                      intermediate_cache1.view(-1, N))
        else:
            raise ValueError(f"Unsupported FusedMoe activation: {activation}")

        qintermediate_cache2, qa2_scale = moe_kernel_prepare_input(
            A=intermediate_cache2,
            B=w2,
            A_scale=a2_scale,
            B_scale=w2_scale,
            use_fp8_w8a8=use_fp8_w8a8,
            use_int8_w8a8=use_int8_w8a8,
            use_int8_w8a16=use_int8_w8a16,
            use_int4_w4a16=use_int4_w4a16,
            per_channel_quant=per_channel_quant,
            block_shape=block_shape)

        invoke_fused_moe_kernel_int8(qintermediate_cache2,
                                w2,
                                intermediate_cache3,
                                qa2_scale,
                                w2_scale,
                                w2_zp,
                                curr_topk_weights,
                                curr_topk_ids,
                                sorted_token_ids_2,
                                expert_ids_2,
                                num_tokens_post_padded_2,
                                not apply_router_weight_on_input,
                                1,
                                config2,
                                compute_type=compute_type,
                                use_fp8_w8a8=use_fp8_w8a8,
                                use_int8_w8a8=use_int8_w8a8,
                                use_int8_w8a16=use_int8_w8a16,
                                use_int4_w4a16=use_int4_w4a16,
                                per_channel_quant=per_channel_quant,
                                block_shape=block_shape,
                                use_nn_moe=use_nn_moe)
        
        # torch.cuda.synchronize()
        ops.moe_sum(intermediate_cache3.view(*intermediate_cache3.shape),
                    out_hidden_states[begin_chunk_idx:end_chunk_idx])
        '''    
        
        
    return out_hidden_states   

def fused_moe_impl( hidden_states: torch.Tensor,
                    w1: torch.Tensor,
                    w2: torch.Tensor,
                    gating_output: torch.Tensor,
                    topk: int,
                    renormalize: bool,
                    inplace: bool = False,
                    use_grouped_topk: bool = False,
                    num_expert_group: Optional[int] = None,
                    topk_group: Optional[int] = None,
                    custom_routing_function: Optional[Callable] = None,
                    use_fp8_w8a8: bool = False,
                    use_int8_w8a8: bool = False,
                    use_int8_w8a16: bool = False,
                    use_int4_w4a16: bool = False,
                    w1_scale: Optional[torch.Tensor] = None,
                    w2_scale: Optional[torch.Tensor] = None,
                    w1_zp: Optional[torch.Tensor] = None,
                    w2_zp: Optional[torch.Tensor] = None,
                    a1_scale: Optional[torch.Tensor] = None,
                    a2_scale: Optional[torch.Tensor] = None,
                    block_shape: Optional[List[int]] = None,
                    use_nn_moe: Optional[bool] = False,
                    moe_ep_size: Optional[int] = 1,
                    start_expert: Optional[int] = None,
                    end_expert: Optional[int] = None,
                    config:  Dict[str, Any] = None,
) -> torch.Tensor:
    """
    This function computes a Mixture of Experts (MoE) layer using two sets of
    weights, w1 and w2, and top-k gating mechanism.

    Parameters:
    - hidden_states (torch.Tensor): The input tensor to the MoE layer.
    - w1 (torch.Tensor): The first set of expert weights.
    - w2 (torch.Tensor): The second set of expert weights.
    - gating_output (torch.Tensor): The output of the gating operation
        (before softmax).
    - topk (int): The number of top-k experts to select.
    - renormalize (bool): If True, renormalize the top-k weights to sum to 1.
    - inplace (bool): If True, perform the operation in-place.
        Defaults to False.
    - num_expert_group: Optional[int]: additional parameter for grouped_topk
    - topk_group: Optional[int]: additional parameter for grouped_topk
    - use_grouped_topk: If True, use grouped_topk instead of fused_topk
        note: Deepseekv2 model uses grouped_topk
    - use_fp8_w8a8 (bool): If True, use fp8 arithmetic to compute the inner
        products for w1 and w2. Defaults to False.
    - use_int8_w8a16 (bool): If True, use matmul of int8 weight and bf16/fp16
        activation to compute the inner products for w1 and w2.
        Defaults to False.
    - use_int4_w4a16 (bool): If True, use matmul of int4 weight and bf16/fp16
        activation to compute the inner products for w1 and w2.
        Defaults to False.
    - w1_scale (Optional[torch.Tensor]): Optional scale to be used for
        w1.
    - w2_scale (Optional[torch.Tensor]): Optional scale to be used for
        w2.
    - a1_scale (Optional[torch.Tensor]): Optional scale to be used for
        a1.
    - a2_scale (Optional[torch.Tensor]): Optional scale to be used for
        a2.
    - block_shape: (Optional[List[int]]): Optional block size for block-wise
        quantization.

    Returns:
    - torch.Tensor: The output tensor after applying the MoE layer.
    """
    # Check constraints.
    assert gating_output.shape[1] == w1.shape[0], "Number of experts mismatch"

    if use_grouped_topk:
        assert num_expert_group is not None and topk_group is not None
        topk_weights, topk_ids = grouped_topk(hidden_states, gating_output,
                                              topk, renormalize,
                                              num_expert_group, topk_group)
    elif custom_routing_function is None:
        topk_weights, topk_ids = fused_topk(hidden_states, gating_output, topk,
                                            renormalize)
    else:
        topk_weights, topk_ids = custom_routing_function(
            hidden_states, gating_output, topk, renormalize)
    
    return _fused_moegemm_helper_test_pure(hidden_states=hidden_states,
                                                    w1=w1,
                                                    w2=w2,
                                                    topk_weights=topk_weights,
                                                    topk_ids=topk_ids,
                                                    inplace=False,
                                                    use_int8_w8a8=True,
                                                    w1_scale=w1_scale,
                                                    w2_scale=w2_scale,
                                                    block_shape=block_shape,
                                                    config=config,
                                                    per_channel_quant=True)
            
from vllm.model_executor.layers.fused_moe import fused_experts

def _fused_moegemm_helper_test_dev( num_tokens: int,
                                    num_experts: int,
                                    shard_intermediate_size: int,
                                    hidden_size: int,
                                    topk: int,
                                    dtype: torch.dtype,
                                    use_fp8_w8a8: bool,
                                    use_int8_w8a16: bool,
                                    use_int8_w8a8: bool,
                                    num_iters: int=100,
                                    block_quant_shape: List[int] = None,
                                    ):
    
    seed = 0
    # 设置 PyTorch 的随机数种子
    torch.manual_seed(seed)
    # 如果使用 GPU，设置 GPU 的随机数种子
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # 如果有多个 GPU
    
    device="cuda"  
    x = torch.randn(num_tokens, hidden_size, device=device, dtype=dtype)
    w1_scale = None
    w2_scale = None
    a1_scale = None
    a2_scale = None

    w1 = torch.randint(-127,
                        127, (
                            num_experts,
                            shard_intermediate_size, # n
                            hidden_size, # k
                        ),
                        device=device,
                        dtype=torch.int8)
    w2 = torch.randint(-127,
                        127, (
                            num_experts,
                            hidden_size, # n2
                            shard_intermediate_size // 2, # k2
                        ),
                        device=device,
                        dtype=torch.int8)
    w1_scale = (torch.randn((256,math.ceil(shard_intermediate_size/block_quant_shape[0]), math.ceil(hidden_size/block_quant_shape[1])),device=device,
                        dtype=torch.float32))
    w2_scale = (torch.randn((256,math.ceil(hidden_size/block_quant_shape[0]), math.ceil((shard_intermediate_size // 2)/block_quant_shape[1])),device=device,
                        dtype=torch.float32))    

    input_gating = torch.randn(num_tokens, num_experts, device=device, dtype=torch.float32)
    output = fused_moe_impl(x,
                        w1,
                        w2,
                        input_gating,
                        topk,
                        renormalize=True,
                        inplace=False,
                        use_fp8_w8a8=use_fp8_w8a8,
                        use_int8_w8a16=use_int8_w8a16,
                        w1_scale=w1_scale,
                        w2_scale=w2_scale,
                        a1_scale=a1_scale,
                        a2_scale=a2_scale,
                        block_shape=block_quant_shape,
                        config=None)
    
def invoke_fused_moe_kernel_int8_preprocess(A: torch.Tensor,
                                            B: torch.Tensor,
                                            C: torch.Tensor,
                                            A_scale: Optional[torch.Tensor],
                                            B_scale: Optional[torch.Tensor],
                                            B_zp: Optional[torch.Tensor],
                                            topk_weights: Optional[torch.Tensor],
                                            topk_ids: torch.Tensor,
                                            sorted_token_ids: torch.Tensor,
                                            expert_ids: torch.Tensor,
                                            num_tokens_post_padded: torch.Tensor,
                                            mul_routed_weight: bool,
                                            top_k: int,
                                            config: Dict[str, Any],
                                            compute_type: tl.dtype,
                                            use_fp8_w8a8: bool,
                                            use_int8_w8a8: bool,
                                            use_int8_w8a16: bool,
                                            use_int4_w4a16: bool,
                                            per_channel_quant: bool,
                                            block_shape: Optional[List[int]] = None,
                                            use_nn_moe: Optional[bool]=False,
                                            gemm_compare_list: List[int]=[]) -> None:
    assert topk_weights is not None or not mul_routed_weight
    assert topk_weights is None or topk_weights.stride(1) == 1
    assert sorted_token_ids.stride(0) == 1
    assert use_int8_w8a8 is True
    
    M = A.shape[0]
    num_tokens = M * top_k

    triton_config = {   "BLOCK_SIZE_M": 64,
        "BLOCK_SIZE_N": 64,
        "BLOCK_SIZE_K": 64,
        "GROUP_SIZE_M": 1,
        "kpack": 1,
        "num_stages": 0,
        "num_warps": 2
    }
    triton_config["BLOCK_SIZE_M"] = config["BLOCK_SIZE_M"]

    EM = sorted_token_ids.shape[0]
    if A.shape[0] < triton_config["BLOCK_SIZE_M"]:
        # optimize for small batch_size.
        # We assume that top_ids of each token is unique, so
        # so num_valid_experts <= batch_size <= BLOCK_SIZE_M,
        # and we can skip some invalid blocks.
        EM = min(sorted_token_ids.shape[0],
                 A.shape[0] * top_k * triton_config['BLOCK_SIZE_M'])
    grid = lambda META: (triton.cdiv(EM, META['BLOCK_SIZE_M']) * triton.cdiv(
        B.shape[1] if not use_nn_moe else B.shape[2], META['BLOCK_SIZE_N']), )

    fused_moe_kernel[grid](
            A,
            B,
            C,
            A_scale,
            B_scale,
            topk_weights,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            B.shape[1] if not use_nn_moe else B.shape[2],
            A.shape[1],
            EM,
            num_tokens,
            A.stride(0),
            A.stride(1),
            B.stride(0),
            B.stride(2) if not use_nn_moe else B.stride(1),
            B.stride(1) if not use_nn_moe else B.stride(2),
            C.stride(1),
            C.stride(2),
            A_scale.stride(0)
            if A_scale is not None and A_scale.ndim == 2 else 0,
            A_scale.stride(1)
            if A_scale is not None and A_scale.ndim == 2 else 0,
            B_scale.stride(0)
            if B_scale is not None and B_scale.ndim >= 2 else 0,
            B_scale.stride(2)
            if B_scale is not None and B_scale.ndim == 3 else 0,
            B_scale.stride(1)
            if B_scale is not None and B_scale.ndim >= 2 else 0,
            0 if block_shape is None else block_shape[0],
            0 if block_shape is None else block_shape[1],
            MUL_ROUTED_WEIGHT=mul_routed_weight,
            top_k=top_k,
            compute_type=compute_type,
            use_fp8_w8a8=use_fp8_w8a8,
            use_int8_w8a8=use_int8_w8a8,
            use_int8_w8a16=use_int8_w8a16,
            per_channel_quant=per_channel_quant,
            delta=int(config['DELTA']),
            **triton_config
        )
    out_cuda = torch.zeros_like(C)

    moe_gemm_w8a8(A, 
                    B, 
                    out_cuda,
                    A_scale, 
                    B_scale,
                    topk_weights if mul_routed_weight else None,
                    sorted_token_ids, 
                    expert_ids,
                    num_tokens_post_padded, 
                    top_k,
                    config)
    torch.cuda.synchronize()
    stage = 0
    if not mul_routed_weight:
        stage = 1
    else:
        stage = 2
    if not (torch.allclose(C, out_cuda, rtol=1e-2, atol=1e-2)):
        if not mul_routed_weight and stage == 1:
            gemm_compare_list.append(config['MODE'])
        elif stage == 2:
            gemm_compare_list.append(config['MODE'])    

def fused_experts_impl_int8_preprocess(hidden_states: torch.Tensor,
                                        w1: torch.Tensor,
                                        w2: torch.Tensor,
                                        topk_weights: torch.Tensor,
                                        topk_ids: torch.Tensor,
                                        inplace: bool = False,
                                        activation: str = "silu",
                                        apply_router_weight_on_input: bool = False,
                                        use_fp8_w8a8: bool = False,
                                        use_int8_w8a8: bool = False,
                                        use_int8_w8a16: bool = False,
                                        use_int4_w4a16: bool = False,
                                        per_channel_quant: bool = False,
                                        global_num_experts: int = -1,
                                        expert_map: Optional[torch.Tensor] = None,
                                        w1_scale: Optional[torch.Tensor] = None,
                                        w2_scale: Optional[torch.Tensor] = None,
                                        w1_zp: Optional[torch.Tensor] = None,
                                        w2_zp: Optional[torch.Tensor] = None,
                                        a1_scale: Optional[torch.Tensor] = None,
                                        a2_scale: Optional[torch.Tensor] = None,
                                        block_shape: Optional[List[int]] = None,
                                        use_nn_moe: Optional[bool] = False,
                                        config: Optional[Dict[str, Any]] = None,
                                        mode: Optional[List[int]]=None,
                                        gemm1_compare_list: List[int]=[],
                                        gemm2_compare_list: List[int]=[]):
    # Check constraints. 
    assert use_int8_w8a8 is True
    assert hidden_states.shape[1] == w1.shape[2], "Hidden size mismatch"

    assert topk_weights.shape == topk_ids.shape, "topk shape mismatch"
    assert hidden_states.is_contiguous(), "Hidden_states must be contiguous"
    assert w1.stride(-1) == 1, "Stride of last dimension must be 1"
    assert w2.stride(-1) == 1, "Stride of last dimension must be 1"
    assert hidden_states.dtype in [
        torch.float32, torch.float16, torch.bfloat16
    ]

    num_tokens, _ = hidden_states.shape

    if use_nn_moe:
        E, _, N = w1.shape
    else:
        E, N, _ = w1.shape

    K = w2.shape[1]
    if global_num_experts == -1:
        global_num_experts = E
    top_k_num = topk_ids.shape[1]
    # We execute the fused_moe kernel in chunks to circumvent this issue:
    # https://github.com/vllm-project/vllm/issues/5938
    CHUNK_SIZE = envs.VLLM_FUSED_MOE_CHUNK_SIZE
    M = min(num_tokens, CHUNK_SIZE)
    # We can reuse the memory between these because by the time we need
    # cache3, we're done with cache1
    cache13 = torch.empty(M * top_k_num * max(N, K if not use_nn_moe else w2.shape[2]),
                          device=hidden_states.device,
                          dtype=hidden_states.dtype)
    intermediate_cache1 = cache13[:M * top_k_num * N].view(M, top_k_num, N)
    intermediate_cache3 = cache13[:M * top_k_num * (K if not use_nn_moe else w2.shape[2])].view(M, top_k_num, K if not use_nn_moe else w2.shape[2])

    # This needs separate memory since it's used concurrently with cache1
    intermediate_cache2 = torch.empty((M * top_k_num, N // 2),
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


    for chunk in range((num_tokens // CHUNK_SIZE) + 1):
        begin_chunk_idx, end_chunk_idx = (chunk * CHUNK_SIZE,
                                          min((chunk + 1) * CHUNK_SIZE,
                                              num_tokens))
        curr_hidden_states = hidden_states[begin_chunk_idx:end_chunk_idx]
        tokens_in_chunk, _ = curr_hidden_states.shape
        # mode0, mode1 = 4, 68 # 44
        mode0, mode1 = mode[0], mode[1]
    
        if tokens_in_chunk == 0:
            break

        if tokens_in_chunk < CHUNK_SIZE and chunk > 0:
            # Adjust the intermediate cache size and config for the last
            # chunk. Note that in most cases we only have one chunk
            # so the cache size and config are already set correctly and
            # do not need to be adjusted.
            intermediate_cache1 = intermediate_cache1[:tokens_in_chunk]
            intermediate_cache2 = intermediate_cache2[:tokens_in_chunk *
                                                      topk_ids.shape[1]]
            intermediate_cache3 = intermediate_cache3[:tokens_in_chunk]

        curr_topk_ids = topk_ids[begin_chunk_idx:end_chunk_idx]
        curr_topk_weights = topk_weights[begin_chunk_idx:end_chunk_idx]

        # print("curr_hidden_states:",curr_hidden_states.shape)
        qcurr_hidden_states, qa1_scale = moe_kernel_prepare_input(
            A=curr_hidden_states,
            B=w1,
            A_scale=a1_scale,
            B_scale=w1_scale,
            use_fp8_w8a8=use_fp8_w8a8,
            use_int8_w8a8=use_int8_w8a8,
            use_int8_w8a16=use_int8_w8a16,
            use_int4_w4a16=use_int4_w4a16,
            per_channel_quant=per_channel_quant,
            block_shape=block_shape)

        config1={}
        config2={}
        if M > M_THRESHOLD:
            config1['BLOCK_SIZE_M'] = blockM_configs[mode0]
            config1['MODE']         = mode0
            config1['DELTA']        = 1
            config2['BLOCK_SIZE_M'] = blockM_configs[mode1]
            config2['MODE']         = mode1
            config2['DELTA']        = 1
        else:
            config1['BLOCK_SIZE_M'] = blockM_configs_decode[mode0]
            config1['MODE']         = mode0
            config1['DELTA']        = 1
            config2['BLOCK_SIZE_M'] = blockM_configs_decode[mode1]
            config2['MODE']         = mode1
            config2['DELTA']        = 1
        sorted_token_ids, expert_ids, num_tokens_post_padded = (
                moe_align_block_size(curr_topk_ids, config1['BLOCK_SIZE_M'],
                                    global_num_experts, expert_map))

        invoke_fused_moe_kernel_int8_preprocess(qcurr_hidden_states,
                                                w1,
                                                intermediate_cache1,
                                                qa1_scale,
                                                w1_scale,
                                                w1_zp,
                                                curr_topk_weights,
                                                curr_topk_ids,
                                                sorted_token_ids,
                                                expert_ids,
                                                num_tokens_post_padded,
                                                apply_router_weight_on_input,
                                                top_k_num,
                                                config1,
                                                compute_type=compute_type,
                                                use_fp8_w8a8=use_fp8_w8a8,
                                                use_int8_w8a8=use_int8_w8a8,
                                                use_int8_w8a16=use_int8_w8a16,
                                                use_int4_w4a16=use_int4_w4a16,
                                                per_channel_quant=per_channel_quant,
                                                block_shape=block_shape,
                                                use_nn_moe=use_nn_moe,
                                                gemm_compare_list=gemm1_compare_list)

        if activation == "silu":
            # d = intermediate_cache1.shape[-1] // 2
            # intermediate_cache2 = F.silu(intermediate_cache1[..., :d]) * intermediate_cache1[..., d:]
            torch.ops._C.silu_and_mul(intermediate_cache2,
                                      intermediate_cache1.view(-1, N))
        elif activation == "gelu":
            torch.ops._C.gelu_and_mul(intermediate_cache2,
                                      intermediate_cache1.view(-1, N))
        else:
            raise ValueError(f"Unsupported FusedMoe activation: {activation}")

        qintermediate_cache2, qa2_scale = moe_kernel_prepare_input(
            A=intermediate_cache2,
            B=w2,
            A_scale=a2_scale,
            B_scale=w2_scale,
            use_fp8_w8a8=use_fp8_w8a8,
            use_int8_w8a8=use_int8_w8a8,
            use_int8_w8a16=use_int8_w8a16,
            use_int4_w4a16=use_int4_w4a16,
            per_channel_quant=per_channel_quant,
            block_shape=block_shape)

        invoke_fused_moe_kernel_int8_preprocess(qintermediate_cache2,
                                                w2,
                                                intermediate_cache3,
                                                qa2_scale,
                                                w2_scale,
                                                w2_zp,
                                                curr_topk_weights,
                                                curr_topk_ids,
                                                sorted_token_ids,
                                                expert_ids,
                                                num_tokens_post_padded,
                                                not apply_router_weight_on_input,
                                                1,
                                                config2,
                                                compute_type=compute_type,
                                                use_fp8_w8a8=use_fp8_w8a8,
                                                use_int8_w8a8=use_int8_w8a8,
                                                use_int8_w8a16=use_int8_w8a16,
                                                use_int4_w4a16=use_int4_w4a16,
                                                per_channel_quant=per_channel_quant,
                                                block_shape=block_shape,
                                                use_nn_moe=use_nn_moe,
                                                gemm_compare_list=gemm2_compare_list)
        
        ops.moe_sum(intermediate_cache3.view(*intermediate_cache3.shape),
                    out_hidden_states[begin_chunk_idx:end_chunk_idx])

    return out_hidden_states
    
def main():
    # m_list=[1,2,4,8,16,32,64,128,256,512,1024,2048,4096,8192,16384,32768]
    # m_list=[1,2,4,8,16,32,64,128,256,512,1024,2048,4096,8192,16384]
    m_list=[1024]
    # m_list=[16]
    n_list_1=[256]
    # n_list_1=[256]
    k_list_1=[7168]
    # k_list_1=[1024]
    
    n_list_2=[7168]
    k_list_2=[128]
    
    block_size=[128,128]
    out_dtype=torch.bfloat16
    _w13n=[]
    _w13k=[]
    
    _w2n=[]
    _w2k=[]
    
    _m=[]
    cost_times1=[]
    cost_times2=[]
    topk=8
    num_experts=256
    best_config = None
    
    for i in range(0,len(k_list_1),1):
        for m in m_list:
            print("m:{} n:{} k:{} ".format(m,n_list_1[i],k_list_1[i]))
            _fused_moegemm_helper_test_dev(m, num_experts, n_list_1[i], k_list_1[i], topk, out_dtype, False, False, True, 1, block_size)

    print("表格已保存到 moegemmoutput.xlsx 文件中。")    

if __name__ == "__main__":
    main()







