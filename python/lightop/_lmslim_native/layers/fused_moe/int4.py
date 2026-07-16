import functools
import json
import os
import math
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import triton
import triton.language as tl

#import vllm.envs as envs
# from vllm import _custom_ops as ops
# from vllm.logger import init_logger
# from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
#     moe_align_block_size)
from ...vllm_compat.moe_align_block_size import moe_align_block_size
from ...vllm_compat.fused_moe_cache import get_moe_cache
from ... import envs as lsenvs
# from vllm.platforms import current_platform
# from vllm_hcu.platforms import HCUPlatform as current_platform
# logger = init_logger(__name__)
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
os.environ['W4A16_MOE_CUDA'] = os.environ.get('W4A16_MOE_CUDA', '0')
if os.environ['W4A16_MOE_CUDA'] == '1':
    from .int4_cuda import fused_experts_cuda

# device_name = current_platform.get_device_name().replace(" ", "_")
device_name = torch.cuda.get_device_properties(0).name

@triton.jit
def fused_moe_kernel_awq(
        # Pointers to matrices
        a_ptr, # [4, 7168] 
        b_ptr, # [256, 512, 3584]
        c_ptr, # (8, 8, 512)
        b_scale_ptr, # (256, 512, 56)
        b_zp_ptr, # (256, 256, 56)
        topk_weights_ptr, 
        sorted_token_ids_ptr, # [0, 1, 2, 3, 4]
        expert_ids_ptr,
        num_tokens_post_padded_ptr,
        # Matrix dimensions
        N: tl.constexpr,
        K: tl.constexpr,
        EM, # pading后的总索引长度
        num_valid_tokens, # 有效索引的上限
        # The stride variables represent how much to increase the ptr by when
        # moving by 1 element in a particular dimension. E.g. `stride_am` is
        # how much to increase `a_ptr` by to get the element one row down
        # (A has M rows).
        stride_am,
        stride_ak,
        stride_be,
        stride_bk, #1
        stride_bn,
        stride_cm,
        stride_cn,
        stride_bse,
        stride_bsk,#1
        stride_bsn,
        stride_bze,
        stride_bzk,
        stride_bzn,
        block_k_diviable: tl.constexpr,
        group_size: tl.constexpr, # 128
        # Meta-parameters
        BLOCK_SIZE_M: tl.constexpr,
        BLOCK_SIZE_N: tl.constexpr,
        BLOCK_SIZE_K: tl.constexpr,
        GROUP_SIZE_M: tl.constexpr,
        MUL_ROUTED_WEIGHT: tl.constexpr,
        top_k: tl.constexpr,
        compute_type: tl.constexpr,
        has_zp: tl.constexpr,
        use_int4_w4a16: tl.constexpr,
        use_int8_w8a16: tl.constexpr):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(EM, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)
    if pid_m * BLOCK_SIZE_M >= num_tokens_post_padded:
        return
    offs_token_id = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_token = tl.load(sorted_token_ids_ptr + offs_token_id) # [block_m]
    token_mask = offs_token < num_valid_tokens

    offs_bn = (pid_n * BLOCK_SIZE_N +
               tl.arange(0, BLOCK_SIZE_N)) % N # [block_n]
    offs_k = tl.arange(0, BLOCK_SIZE_K) # 0, 1, 2, ...... , 127 # # [block_k]
    offs_k2 = tl.arange(0, BLOCK_SIZE_K // 2) # 0, 1, 2, ...... , 127 # # [block_k]
    a_ptrs = a_ptr + (offs_token[:, None] // top_k * stride_am +
                      offs_k[None, :] * stride_ak) # [block_m, block_k] 

    off_experts = tl.load(expert_ids_ptr + pid_m)

    if use_int4_w4a16:
        # [0, 1, 2, ...... , 126, 127] --> [0, 0, 1, 1 ...... , 63, 63] 
        # [128, 129, 130, ...... , 254, 255] --> [64, 64, 65, 65 ...... , 127, 127] 
        
        # b_ptrs = b_ptr + off_experts * stride_be + \
        #     (offs_k[:, None] // 2) * stride_bk + offs_bn[None, :] * stride_bn
        b_ptrs = b_ptr + off_experts * stride_be + \
            offs_bn[:, None] * stride_bn + (offs_k2[None, :]) * stride_bk 
        # tl.device_print("stride_bn",stride_bsn)>1
        # tl.device_print("stride_bk",stride_bk)=1
        b_shifter = (offs_k[:, None] % 2) * 4  # 0, 4
    elif use_int8_w8a16:
        b_ptrs = b_ptr + off_experts * stride_be + \
            offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn

    if not has_zp and use_int4_w4a16:
        b_zp_num = 8
    if not has_zp and use_int8_w8a16:
        b_zp_num = 128
    elif has_zp and use_int4_w4a16:
        b_zp_shifter = (offs_bn[None, :] % 2) * 4 #  0, 4

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        if not block_k_diviable:
            k_mask = offs_k[:, None] < K - k * BLOCK_SIZE_K
            k_other = 0.0
        else:
            k_mask = None
            k_other = None

        a = tl.load(a_ptrs,
                    mask=token_mask[:, None] &
                    (offs_k[None, :] < K - k * BLOCK_SIZE_K),
                    other=0.0)
        b = tl.load(b_ptrs)
        if use_int4_w4a16:
            b = tl.interleave(b, b)
            b= b.trans()
            b = (b >> b_shifter) & 0xF

        b_scale_ptrs = b_scale_ptr + off_experts * stride_bse + \
            offs_bn[None, :] * stride_bsk + \
            ((offs_k[:, None] + BLOCK_SIZE_K * k) // group_size) * stride_bsn
        qzeros_scles = tl.load(b_scale_ptrs, mask=k_mask, other=k_other)

        scales_int16 = tl.cast(qzeros_scles,tl.uint16)
        b_scale = tl.cast(scales_int16,tl.float16,bitcast=True)
        # tl.device_print("b_scale dequant",b_scale)

        mid = qzeros_scles >> 16
        # b_zp = tl.cast(mid,tl.float16,bitcast=False)
        b_zp = tl.cast(mid,tl.float16)
        # b_zp = tl.cast(zeros_int16,tl.float16,bitcast=False)

        # tl.device_print("bzp",b_zp)

        # We accumulate along the K dimension.
        b = ((b - b_zp) * b_scale).to(tl.float16)
        accumulator = tl.dot(a, b, acc=accumulator)

        # Advance the ptrs to the next K block.
        a_ptrs += BLOCK_SIZE_K * stride_ak
        if use_int4_w4a16:
            b_ptrs += (BLOCK_SIZE_K // 2) * stride_bk
        else:
            b_ptrs += BLOCK_SIZE_K * stride_bk

    if MUL_ROUTED_WEIGHT:
        moe_weight = tl.load(topk_weights_ptr + offs_token,
                             mask=token_mask,
                             other=0)
        accumulator = accumulator * moe_weight[:, None]

    accumulator = accumulator.to(compute_type)
    # -----------------------------------------------------------
    # Write back the block of the output
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn[
        None, :]
    c_mask = token_mask[:, None] & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)


def invoke_fused_moe_kernel(A: torch.Tensor,
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

    M = A.shape[0]
    num_tokens = M * top_k

    EM = sorted_token_ids.shape[0]
    if A.shape[0] < config["BLOCK_SIZE_M"]:
        # optimize for small batch_size.
        # We assume that top_ids of each token is unique, so
        # so num_valid_experts <= batch_size <= BLOCK_SIZE_M,
        # and we can skip some invalid blocks.
        EM = min(sorted_token_ids.shape[0],
                 A.shape[0] * top_k * config['BLOCK_SIZE_M'])
    grid = lambda META: (triton.cdiv(EM, META['BLOCK_SIZE_M']) * triton.cdiv(
        B.shape[1] if not use_nn_moe else B.shape[2], META['BLOCK_SIZE_N']), )
        
    assert B_scale is not None and B_scale.ndim == 3
    assert B_zp is None or B_zp.ndim == 3

    fused_moe_kernel_awq[grid](
        A,
        B,
        C,
        B_scale,
        B_zp,
        topk_weights,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        B.shape[1],
        A.shape[1],
        EM,
        topk_ids.numel(),
        A.stride(0),
        A.stride(1),
        B.stride(0),
        B.stride(2),
        B.stride(1),
        C.stride(1),
        C.stride(2),
        B_scale.stride(0),
        B_scale.stride(2),
        B_scale.stride(1),
        B_zp.stride(0) if B_zp is not None else 0,
        B_zp.stride(2) if B_zp is not None else 0,
        B_zp.stride(1) if B_zp is not None else 0,
        block_k_diviable=A.shape[1] % config["BLOCK_SIZE_K"] == 0,
        group_size=block_shape[1],
        MUL_ROUTED_WEIGHT=mul_routed_weight,
        top_k=top_k,
        compute_type=compute_type,
        has_zp=B_zp is not None,
        use_int4_w4a16=use_int4_w4a16,
        use_int8_w8a16=use_int8_w8a16,
        **config,
    )
# Adapted from: https://github.com/sgl-project/sglang/pull/2628
def get_config_file_name(E: int,
                         N: int,
                         dtype: Optional[str],
                         block_shape: Optional[List[int]] = None, use_nn_moe: Optional[bool] = False) -> str:
    
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
    if torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count == 120:
        config_file_path_120 = config_file_path.replace(".json","_120.json")
        if os.path.exists(config_file_path_120):
            with open(config_file_path_120) as f:
                logger.info("Using configuration from %s for MoE layer.",
                            config_file_path_120)
                # If a configuration has been found, return it
                return {int(key): val for key, val in json.load(f).items()}

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

    # elif dtype in ["int4_w4a16", "int8_w8a16"] and block_shape is not None:
    #     # moe wna16 kernels
    #     # only set BLOCK_SIZE_M
    #     # BLOCK_SIZE_N and BLOCK_SIZE_K would be set later
    #     bit = 4 if dtype == "int4_w4a16" else 8
    #     use_moe_wna16_cuda = should_moe_wna16_use_cuda(M * topk,
    #                                                    block_shape[1], E, bit)
    #     if use_moe_wna16_cuda:
    #         config = {"BLOCK_SIZE_M": min(16, M)}
    #     elif M <= 20:
    #         config = {"BLOCK_SIZE_M": 16, "GROUP_SIZE_M": 1}
    #     elif M <= 40:
    #         config = {"BLOCK_SIZE_M": 32, "GROUP_SIZE_M": 1}
    #     else:
    #         config = {"BLOCK_SIZE_M": 64, "GROUP_SIZE_M": 1}
    elif is_marlin:
        for block_size_m in [8, 16, 32, 48, 64]:
            if M * topk / E / block_size_m < 0.9:
                break
        return {"BLOCK_SIZE_M": block_size_m}
    elif M <= E:
        config = {
            "BLOCK_SIZE_M": 16,
            "BLOCK_SIZE_N": 32,
            "BLOCK_SIZE_K": 64,
            "GROUP_SIZE_M": 1,
        }
    else:
        config = {
            "BLOCK_SIZE_M": 64,
            "BLOCK_SIZE_N": 64,
            "BLOCK_SIZE_K": 32,
            "GROUP_SIZE_M": 8,
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
    use_nn_moe: Optional[bool] = False,
):
    # from vllm.model_executor.layers.fused_moe import get_config
    # override_config = get_config()
    # if override_config:
    #     config = override_config
    # else:
        # First try to load optimal config from the file
    if not use_nn_moe:
        E, _, N = w2_shape
    else:
        E, N, _  = w2_shape
    # if dtype == "int4_w4a16":
    #     N = N * 2
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

def get_config_dtype_str(
        dtype: torch.dtype,
        use_int4_w4a16: Optional[bool] = False,
        use_int8_w8a16: Optional[bool] = False,
        use_fp8_w8a8: Optional[bool] = False,
        use_int8_w8a8: Optional[bool] = False) -> Optional[str]:
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
            # A, A_scale = ops.scaled_fp8_quant(
            #     A, A_scale, use_per_token_if_dynamic=per_channel_quant)
            # 这里使用的是lmslim.vllm_compat.scaled_fp8_quant.scaled_fp8_quant，替换了vllm中的ops.scaled_fp8_quant
            from ...vllm_compat.scaled_fp8_quant import scaled_fp8_quant
            A, A_scale = scaled_fp8_quant(A, A_scale, use_per_token_if_dynamic=per_channel_quant)
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

def fused_experts_impl_int4(hidden_states: torch.Tensor,
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
                       use_nn_moe: Optional[bool] = False, **_):

    assert hidden_states.shape[1] // 2 == w1.shape[
        2], "Hidden size mismatch"

    assert topk_weights.shape == topk_ids.shape, "topk shape mismatch"
    assert hidden_states.is_contiguous(), "Hidden_states must be contiguous"
    assert topk_weights.shape == topk_ids.shape, "topk shape mismatch"
    assert hidden_states.is_contiguous(), "Hidden_states must be contiguous"
    assert w1.stride(-1) == 1, "Stride of last dimension must be 1"
    assert w2.stride(-1) == 1, "Stride of last dimension must be 1"
    assert hidden_states.dtype in [
        torch.float32, torch.float16, torch.bfloat16
    ]
    num_tokens, _ = hidden_states.shape


    E, N, _ = w1.shape
    K = w2.shape[1]
    if global_num_experts == -1:
        global_num_experts = E
    top_k_num = topk_ids.shape[1]
    # We execute the fused_moe kernel in chunks to circumvent this issue:
    # https://github.com/vllm-project/vllm/issues/5938
    CHUNK_SIZE = lsenvs.LMSLIM_FUSED_MOE_CHUNK_SIZE
    M = min(num_tokens, CHUNK_SIZE)
    config_dtype = get_config_dtype_str(use_fp8_w8a8=use_fp8_w8a8,
                                        use_int8_w8a16=use_int8_w8a16,
                                        use_int4_w4a16=use_int4_w4a16,
                                        dtype=hidden_states.dtype)

    get_config_func = functools.partial(
        try_get_optimal_moe_config,
        w1.shape,
        w2.shape,
        top_k_num,
        config_dtype,
        block_shape=block_shape,
        use_nn_moe=use_nn_moe,
    )

    config = get_config_func(M)
    if lsenvs.LMSLIM_USE_GLOBAL_MOE_CACHE:
        cache13 = get_moe_cache(top_k_num, N, K if not use_nn_moe else w2.shape[2], device=hidden_states.device, dtype=hidden_states.dtype)
    else:
        cache13 = torch.empty(M * top_k_num * max(N, K if not use_nn_moe else w2.shape[2]), device=hidden_states.device, dtype=hidden_states.dtype)
    # We can reuse the memory between these because by the time we need
    # cache3, we're done with cache1
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
            config = get_config_func(tokens_in_chunk)

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
            moe_align_block_size(curr_topk_ids, config['BLOCK_SIZE_M'],
                                global_num_experts, expert_map, curr_hidden_states.shape[0]))

        invoke_fused_moe_kernel(qcurr_hidden_states,
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
                                config,
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
                
        invoke_fused_moe_kernel(qintermediate_cache2,
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
                                config,
                                compute_type=compute_type,
                                use_fp8_w8a8=use_fp8_w8a8,
                                use_int8_w8a8=use_int8_w8a8,
                                use_int8_w8a16=use_int8_w8a16,
                                use_int4_w4a16=use_int4_w4a16,
                                per_channel_quant=per_channel_quant,
                                block_shape=block_shape,
                                use_nn_moe=use_nn_moe)

        # ops.moe_sum(intermediate_cache3.view(*intermediate_cache3.shape),
        #             out_hidden_states[begin_chunk_idx:end_chunk_idx])
        torch.ops._moe_C.moe_sum(intermediate_cache3.view(*intermediate_cache3.shape), out_hidden_states[begin_chunk_idx:end_chunk_idx])

    return out_hidden_states

def fused_experts_w4a16(hidden_states: torch.Tensor,
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
                  allow_deep_gemm: bool = False,
                  use_nn_moe: Optional[bool] = False, **_) -> torch.Tensor:
    
    if os.environ['W4A16_MOE_CUDA'] == "1":          
        m = topk_ids.shape[0]
        if m <= 512:
            return fused_experts_cuda(hidden_states,
                        w1,
                        w2,
                        topk_weights,
                        topk_ids,
                        inplace=True,
                        activation=activation,                    
                        use_int4_w4a16=True,
                        w1_scale=w1_scale,
                        w2_scale=w2_scale,
                        block_shape=block_shape,
                        expert_map=expert_map,
                        )    
    return fused_experts_impl_int4(hidden_states,
            w1,
            w2,
            topk_weights,
            topk_ids=topk_ids,
            inplace=True,
            activation=activation,
            apply_router_weight_on_input=apply_router_weight_on_input,
            use_int4_w4a16=True,
            global_num_experts=global_num_experts,
            expert_map=expert_map,
            w1_scale=w1_scale,
            w2_scale=w2_scale,
            block_shape=block_shape,
            )