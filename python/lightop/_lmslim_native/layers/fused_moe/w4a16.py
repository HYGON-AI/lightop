import functools
import json
import os
import math
from typing import Any, Callable, Dict, Optional, List, Optional, Tuple
#import vllm.envs as envs
import torch

import triton.language as tl
import triton

# from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
#     moe_align_block_size)
from ...vllm_compat.moe_align_block_size import moe_align_block_size
# from vllm import _custom_ops as ops
from ... import envs as lsenvs
use_lightop = lsenvs.LMSLIM_USE_LIGHTOP

arch_name = torch.cuda.get_device_properties("cuda").gcnArchName.split(':')[0]
arch_cu = torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count

if use_lightop:
    from lightop import moe_gemm_w4a8,get_moe_cuda_config,get_moe_cuda_config_w4a8, fuse_silu_mul_quant
    from .... import op

# @torch.compile
# def moe_sum_reduce_torch_compile(x, out, routed_scaling_factor):
#     torch.sum(x, dim=1, out=out)
#     out.mul_(routed_scaling_factor)

# @triton.jit
# def _moe_sum_reduce_kernel(
#         input_ptr,
#         input_stride_0,
#         input_stride_1,
#         input_stride_2,
#         output_ptr,
#         output_stride_0,
#         output_stride_1,
#         token_num: int,
#         topk_num: int,
#         hidden_dim: int,
#         routed_scaling_factor: tl.constexpr,
#         BLOCK_M: tl.constexpr,
#         BLOCK_DIM: tl.constexpr,
#         NUM_STAGE: tl.constexpr,
# ):
#     input_stride_0 = tl.cast(input_stride_0, dtype=tl.int64)
#     input_stride_1 = tl.cast(input_stride_1, dtype=tl.int64)
#     output_stride_0 = tl.cast(output_stride_0, dtype=tl.int64)

#     token_block_id = tl.program_id(0)
#     dim_block_id = tl.program_id(1)

#     token_start = token_block_id * BLOCK_M
#     token_end = min((token_block_id + 1) * BLOCK_M, token_num)

#     dim_start = dim_block_id * BLOCK_DIM
#     dim_end = min((dim_block_id + 1) * BLOCK_DIM, hidden_dim)

#     offs_dim = dim_start + tl.arange(0, BLOCK_DIM)

#     for token_index in range(token_start, token_end):
#         accumulator = tl.zeros((BLOCK_DIM,), dtype=tl.float32)
#         input_t_ptr = input_ptr + token_index * input_stride_0 + offs_dim
#         for i in tl.range(0, topk_num, num_stages=NUM_STAGE):
#             tmp = tl.load(
#                 input_t_ptr + i * input_stride_1, mask=offs_dim < dim_end, other=0.0
#             )
#             accumulator += tmp
#         accumulator = accumulator * routed_scaling_factor
#         store_t_ptr = output_ptr + token_index * output_stride_0 + offs_dim
#         tl.store(
#             store_t_ptr,
#             accumulator.to(input_ptr.dtype.element_ty),
#             mask=offs_dim < dim_end,
#         )

# def moe_sum_reduce_triton(
#         input: torch.Tensor, output: torch.Tensor, routed_scaling_factor: float
# ):
#     assert input.is_contiguous()
#     assert output.is_contiguous()

#     token_num, topk_num, hidden_dim = input.shape
#     assert output.shape[0] == token_num and output.shape[1] == hidden_dim

#     if token_num <= 32:
#         BLOCK_M = 1
#         BLOCK_DIM = 512
#         NUM_STAGE = 2
#         num_warps = 4

#     elif token_num <= 128:
#         BLOCK_M = 1
#         BLOCK_DIM = 1024
#         NUM_STAGE = 0
#         num_warps = 2

#     elif token_num <= 4096:
#         BLOCK_M = 1
#         BLOCK_DIM = 2048
#         NUM_STAGE = 0
#         num_warps = 2
#     else:
#         BLOCK_M = 1
#         BLOCK_DIM = 2048
#         NUM_STAGE = 2
#         num_warps = 8

#     grid = (
#         triton.cdiv(token_num, BLOCK_M),
#         triton.cdiv(hidden_dim, BLOCK_DIM),
#     )

#     _moe_sum_reduce_kernel[grid](
#         input,
#         *input.stride(),
#         output,
#         *output.stride(),
#         token_num=token_num,
#         topk_num=topk_num,
#         hidden_dim=hidden_dim,
#         routed_scaling_factor=routed_scaling_factor,
#         BLOCK_M=BLOCK_M,
#         BLOCK_DIM=BLOCK_DIM,
#         NUM_STAGE=NUM_STAGE,
#         num_warps=num_warps,
#     )
#     return

# def moe_reduce_dispatch(
#     intermediate_cache3: torch.Tensor,
#     out_hidden_states: torch.Tensor,
#     begin_chunk_idx: int,
#     end_chunk_idx: int,
#     routed_scaling_factor: float,
#     shared_output: Optional[torch.Tensor] = None,
# ):
#     inter_cache_view = intermediate_cache3.view(*intermediate_cache3.shape)
#     n = intermediate_cache3.shape[0]

#     # 根据 n 大小选择不同的 reduce 实现
#     if 1 <= n <= 4:
#         moe_sum_reduce_torch_compile(inter_cache_view, out_hidden_states[begin_chunk_idx:end_chunk_idx], 1.0)
#     elif 4 < n <= 1024:
#         moe_sum_reduce_triton(inter_cache_view, out_hidden_states[begin_chunk_idx:end_chunk_idx], 1.0)
#     elif 1024 < n <= 32768:
#         ops.moe_sum_opt1(inter_cache_view, out_hidden_states[begin_chunk_idx:end_chunk_idx])
#     else:
#         ops.moe_sum(inter_cache_view, out_hidden_states[begin_chunk_idx:end_chunk_idx])

#     # 根据 shared_output 是否存在决定怎么更新
#     if shared_output is not None:
#         out_hidden_states[begin_chunk_idx:end_chunk_idx].mul_(routed_scaling_factor).add_(shared_output[begin_chunk_idx:end_chunk_idx])
#     else:
#         out_hidden_states[begin_chunk_idx:end_chunk_idx].mul_(routed_scaling_factor)

@functools.lru_cache
def get_moe_triton_config_w4a16(
    M: int, # 
    E: int,
    N1: int,
    N2: int,
    K: int,
    topk: int,
    group_size: int,
    dtype: torch.dtype,
):
    """
    Return optimized configurations for the fused MoE kernel.

    The return value will be a dictionary that maps an irregular grid of
    batch sizes to configurations of the fused_moe kernel. To evaluate the
    kernel on a given batch size bs, the closest batch size in the grid should
    be picked and the associated configuration chosen to invoke the kernel.
    """
    # First look up if an optimized configuration is available in the configs
    # directory
    config_file_name = None
    if(dtype == torch.half):
        config_file_name = "MOE_BLOCKW4A16_FP16_" + str(group_size) + "_E=" + str(E) + "_N1="+ str(N1) + "_N2=" + str(N2) + "_K=" + str(K) + "_TOPK=" + str(topk) + "_" + str(arch_name) + "_" + str(arch_cu) + ".json"
    elif(dtype == torch.bfloat16):
        config_file_name = "MOE_BLOCKW4A16_BF16_" + str(group_size) + "_E=" + str(E) + "_N1="+ str(N1) + "_N2=" + str(N2) + "_K=" + str(K) + "_TOPK=" + str(topk) + "_" + str(arch_name) + "_" + str(arch_cu) + ".json"
    else:
        raise ValueError("dtype only support bf16 and fp16!")
    
    config_file_path_name = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "configs", config_file_name)

    # 初始化默认值
    config1 = {}
    config2 = {}

    try:
        with open(config_file_path_name, 'r') as file:
            config_infos = json.load(file)
            
            # 拼接顶层key：512_7168_256
            top_key = f"{N1}_{N2}_{K}"
            
            if top_key in config_infos:
                config_info = config_infos[top_key]
                # print("config_info: ", config_info)
                # 把key转成整数，方便比较
                int_keys = [int(key) for key in config_info.keys()] 
                # print("int_keys: ", int_keys)
                # 找到 >= M 的最小 token
                greater_numbers = [num for num in int_keys if num >= M]
                num_token = min(greater_numbers) if greater_numbers else None
                
                # 如果没找到 >=M 的，就找最接近的
                if num_token is None:
                    min_diff = float('inf')
                    closest_token = None
                    for token in int_keys:
                        diff = abs(token - M)
                        if diff < min_diff:
                            min_diff = diff
                            closest_token = token
                    num_token = closest_token

                # 从对应 num_token 下提取 config1 和 config2
                token_key = str(num_token)
                if token_key in config_info:
                    token_data = config_info[token_key]
                    config1 = token_data.get("config1", {})
                    config2 = token_data.get("config2", {})

    except FileNotFoundError:
        # 文件不存在时保持空字典
        pass

    status = True    
    if not config1 or not config2: 
        # print("Config is not found! Use default config.")
        status = False    
    return config1, config2, status


@triton.jit
def fused_moe_kernel_gptq_awq(
        # Pointers to matrices
        a_ptr,
        b_ptr,
        c_ptr,
        b_scale_ptr,
        b_zp_ptr,
        topk_weights_ptr,
        sorted_token_ids_ptr,
        expert_ids_ptr,
        num_tokens_post_padded_ptr,
        # Matrix dimensions
        N: tl.constexpr,
        K: tl.constexpr,
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
        stride_bse,
        stride_bsk,
        stride_bsn,
        stride_bze,
        stride_bzk,
        stride_bzn,
        block_k_diviable: tl.constexpr,
        group_size: tl.constexpr,
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

    off_experts = tl.load(expert_ids_ptr + pid_m)

    if use_int4_w4a16:
        b_ptrs = b_ptr + off_experts * stride_be.to(tl.int64) + \
            (offs_k[:, None] // 2) * stride_bk + offs_bn[None, :] * stride_bn
        b_shifter = (offs_k[:, None] % 2) * 4
    elif use_int8_w8a16:
        b_ptrs = b_ptr + off_experts * stride_be + \
            offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn

    if not has_zp and use_int4_w4a16:
        b_zp_num = 8
    if not has_zp and use_int8_w8a16:
        b_zp_num = 128
    elif has_zp and use_int4_w4a16:
        b_zp_shifter = (offs_bn[None, :] % 2) * 4

    # -----------------------------------------------------------
    # Iterate to compute a block of the C matrix.
    # We accumulate into a `[BLOCK_SIZE_M, BLOCK_SIZE_N]` block
    # of fp32 values for higher accuracy.
    # `accumulator` will be converted back to fp16 after the loop.
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Load the next block of A and B, generate a mask by checking the
        # K dimension.

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
            b = (b >> b_shifter) & 0xF

        b_scale_ptrs = b_scale_ptr + off_experts * stride_bse + \
            offs_bn[None, :] * stride_bsn + \
            ((offs_k[:, None] + BLOCK_SIZE_K * k) // group_size) * stride_bsk
        b_scale = tl.load(b_scale_ptrs, mask=k_mask, other=k_other)
        b_scale = b_scale.to(tl.float32)

        if has_zp and use_int4_w4a16:
            offs_k_true = (offs_k[:, None] + BLOCK_SIZE_K * k) // group_size
            b_zp_ptrs = b_zp_ptr + off_experts * stride_bze + \
                (offs_bn[None, :] // 2) * stride_bzn + \
                offs_k_true * stride_bzk
            b_zp = tl.load(b_zp_ptrs, mask=k_mask, other=k_other)
            b_zp = ((b_zp >> b_zp_shifter) & 0xF)
            b_zp = b_zp.to(tl.float32)
        elif has_zp and use_int8_w8a16:
            offs_k_true = (offs_k[:, None] + BLOCK_SIZE_K * k) // group_size
            b_zp_ptrs = b_zp_ptr + off_experts * stride_bze + \
                offs_bn[None, :] * stride_bzn + \
                offs_k_true * stride_bzk
            b_zp = tl.load(b_zp_ptrs, mask=k_mask, other=k_other)
            b_zp = b_zp.to(tl.float32)

        # We accumulate along the K dimension.
        if has_zp:
            b = ((b.to(tl.float32) - b_zp) * b_scale).to(compute_type)
        else:
            b = ((b.to(tl.float32) - b_zp_num) * b_scale).to(compute_type)
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


def invoke_fused_moe_kernel_w4a16(
                                A: torch.Tensor,
                                B: torch.Tensor,
                                C: torch.Tensor,
                                B_scale: Optional[torch.Tensor],
                                B_zp: Optional[torch.Tensor],
                                topk_weights: Optional[torch.Tensor],
                                sorted_token_ids: torch.Tensor,
                                expert_ids: torch.Tensor,
                                num_tokens_post_padded: torch.Tensor,
                                mul_routed_weight: bool,
                                top_k: int,
                                config: dict[str, Any],
                                compute_type: tl.dtype,
                                group_size: int
                            ) -> None:
                            
    assert topk_weights is not None or not mul_routed_weight
    assert topk_weights is None or topk_weights.stride(1) == 1
    assert sorted_token_ids.stride(0) == 1

    assert B_scale is not None


    M = A.size(0)
    num_tokens = M * top_k

    EM = sorted_token_ids.size(0)
    if A.size(0) < config["BLOCK_SIZE_M"]:
        EM = min(sorted_token_ids.size(0),
                 A.size(0) * top_k * config['BLOCK_SIZE_M'])
    grid = lambda META: (triton.cdiv(EM, META['BLOCK_SIZE_M']) * triton.cdiv(
        B.size(1), META['BLOCK_SIZE_N']), )


    assert B_scale is not None and B_scale.ndim == 3
    assert B_zp is None or B_zp.ndim == 3
    fused_moe_kernel_gptq_awq[grid](
        A,
        B,
        C,
        B_scale,
        B_zp,
        topk_weights,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        B.size(1),
        A.size(1),
        EM,
        num_tokens,
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
        block_k_diviable=A.size(1) % config["BLOCK_SIZE_K"] == 0,
        group_size=group_size,
        MUL_ROUTED_WEIGHT=mul_routed_weight,
        top_k=top_k,
        compute_type=compute_type,
        has_zp=B_zp is not None,
        use_int4_w4a16=True,
        use_int8_w8a16=False,
        **config,
    )


def fused_experts_impl_w4a16(
                        hidden_states: torch.Tensor,
                        w1: torch.Tensor,
                        w2: torch.Tensor,
                        topk_weights: torch.Tensor,
                        topk_ids: torch.Tensor,
                        w1_scale: torch.Tensor,
                        w2_scale: torch.Tensor,
                        w1_zp: torch.Tensor,
                        w2_zp: torch.Tensor,
                        cache13: torch.Tensor,
                        inplace: bool = False,
                        activation: str = "silu",
                        apply_router_weight_on_input: bool = False,
                        global_num_experts: int = -1,
                        expert_map: Optional[torch.Tensor] = None,
                        group_size: int = 32,
                        routed_scaling_factor: Optional[float] = 1.0,
                        shared_output: Optional[torch.Tensor] = None, **_
                        ):
    # Check constraints. 
    assert hidden_states.shape[1] == w1.shape[2] * 2, "Hidden size mismatch"

    assert topk_weights.shape == topk_ids.shape, "topk shape mismatch"
    assert hidden_states.is_contiguous(), "Hidden_states must be contiguous"
    assert w1.stride(-1) == 1, "Stride of last dimension must be 1"
    assert w2.stride(-1) == 1, "Stride of last dimension must be 1"
    assert hidden_states.dtype in [
        torch.float16, torch.bfloat16
    ]

    num_tokens, _ = hidden_states.shape

    E, N, _ = w1.shape
    K = w2.shape[1]

    _, N1, _ = w1.shape
    _, N2, K2 = w2.shape

    if global_num_experts == -1:
        global_num_experts = E

    top_k_num = topk_ids.shape[1]
    # We execute the fused_moe kernel in chunks to circumvent this issue:
    # https://github.com/vllm-project/vllm/issues/5938
    CHUNK_SIZE = lsenvs.LMSLIM_FUSED_MOE_CHUNK_SIZE
    M = min(num_tokens, CHUNK_SIZE)
    # We can reuse the memory between these because by the time we need
    # cache3, we're done with cache1

    intermediate_cache1 = cache13[:M * top_k_num * N].view(M, top_k_num, N)
    intermediate_cache3 = cache13[:M * top_k_num * K].view(M, top_k_num, K)

    # This needs separate memory since it's used concurrently with cache1
    intermediate_cache2 = torch.empty((M * top_k_num, N // 2),
                                      device=hidden_states.device,
                                      dtype=hidden_states.dtype)

    compute_type = tl.bfloat16 if hidden_states.dtype == torch.bfloat16 else tl.float16

    if inplace:
        out_hidden_states = hidden_states
    else:
        out_hidden_states = torch.empty_like(hidden_states)

    for chunk in range((num_tokens // CHUNK_SIZE) + 1):
        begin_chunk_idx, end_chunk_idx = (chunk * CHUNK_SIZE,
                                          min((chunk + 1) * CHUNK_SIZE,
                                              num_tokens))
        curr_hidden_states = hidden_states[begin_chunk_idx:end_chunk_idx]
        tokens_in_chunk, K = curr_hidden_states.shape

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

        
        config1, config2, status = get_moe_triton_config_w4a16(
                                                                tokens_in_chunk, # M
                                                                E,  # E
                                                                N1,  # N1
                                                                N2,  # N2
                                                                K2 * 2,  # K
                                                                top_k_num,   # topk
                                                                group_size,   # group_size
                                                                hidden_states.dtype # 
                                                                )

        assert status, "config not found!!!"

        sorted_token_ids, expert_ids, num_tokens_post_padded = (
                moe_align_block_size(curr_topk_ids, config1['BLOCK_SIZE_M'],
                                    global_num_experts, None))

        invoke_fused_moe_kernel_w4a16(
                                      curr_hidden_states,
                                      w1,
                                      intermediate_cache1,
                                      w1_scale,
                                      w1_zp,
                                      topk_weights,
                                      sorted_token_ids,
                                      expert_ids,
                                      num_tokens_post_padded,
                                      False,
                                      top_k_num,
                                      config1,
                                      compute_type,
                                      group_size
        )
        
        if activation == "silu":
            # if envs.VLLM_USE_FUSE_SILU_AND_MUL and intermediate_cache1.dtype == intermediate_cache2.dtype == torch.float16:
            # WUFAN: 使用LMSLIM_USE_FUSE_SILU_AND_MUL代替VLLM_USE_FUSE_SILU_AND_MUL
            if lsenvs.LMSLIM_USE_FUSE_SILU_AND_MUL and intermediate_cache1.dtype == intermediate_cache2.dtype == torch.float16:
                from lightop import fuse_silu_and_mul
                fuse_silu_and_mul(intermediate_cache1.view(-1, N),intermediate_cache2)    
            else:
                torch.ops._C.silu_and_mul(intermediate_cache2,
                                        intermediate_cache1.view(-1, N))
        elif activation == "gelu":
            torch.ops._C.gelu_and_mul(intermediate_cache2,
                                      intermediate_cache1.view(-1, N))
        else:
            raise ValueError(f"Unsupported FusedMoe activation: {activation}")

        invoke_fused_moe_kernel_w4a16(
                                      intermediate_cache2,
                                      w2,
                                      intermediate_cache3,
                                      w2_scale,
                                      w2_zp,
                                      topk_weights,
                                      sorted_token_ids,
                                      expert_ids,
                                      num_tokens_post_padded,
                                      True,
                                      1,
                                      config2,
                                      compute_type,
                                      group_size
        )

        if use_lightop and shared_output is not None:
            op.moe_sum(input=intermediate_cache3.view(*intermediate_cache3.shape),
                       output=out_hidden_states[begin_chunk_idx:end_chunk_idx],
                       bias=shared_output[begin_chunk_idx:end_chunk_idx],
                       expert_mask=None,
                       num_local_tokens=None,
                       factor=routed_scaling_factor)
        # elif shared_output is not None:
        #     #vllm开启了融合，但是lmslim 禁用了lightop
        #     moe_reduce_dispatch(
        #         intermediate_cache3,
        #         out_hidden_states,
        #         begin_chunk_idx,
        #         end_chunk_idx,
        #         routed_scaling_factor,
        #         shared_output,
        #     )
        # else:
        #     #vllm 关闭了融合
        #     moe_reduce_dispatch(
        #         intermediate_cache3,
        #         out_hidden_states,
        #         begin_chunk_idx,
        #         end_chunk_idx,
        #         1.0,
        #         None,
        #     )   

    return out_hidden_states