import torch
torch.set_printoptions(precision=4, profile="default", sci_mode=False)
import time
from typing import Optional, Type,Any, Dict, List, Tuple
import pandas as pd
import os
import json
from torch.jit import ignore
import triton
import triton.language as tl
import pandas as pd
import math
import torch.nn.functional as F
import vllm.envs as envs
from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
    moe_align_block_size)
from lmslim.layers.fused_moe.fuse_moe_int8 import moe_kernel_prepare_input
from vllm import _custom_ops as ops

from lightop.moe import moe_gemm_w4a8


gemm1_compare_failed_list = []
gemm2_compare_failed_list = []

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
}

M_THRESHOLD = 128

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
    1000: 128,
}

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
    # Meta-parameters
    MUL_ROUTED_WEIGHT: tl.constexpr,
    top_k: tl.constexpr,
    compute_type: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
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
    if GROUP_SIZE_M ==1:
        num_pid_n = tl.cdiv(N,BLOCK_SIZE_N)
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
    offs_token_id = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_token = tl.load(sorted_token_ids_ptr + offs_token_id)
    token_mask = offs_token < num_valid_tokens
    
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    
    offs_ak = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (
        offs_token[:, None] // top_k * stride_am + offs_ak[None, :] * stride_ak
    )

    off_experts = tl.load(expert_ids_ptr + pid_m)
    offs_bk = tl.arange(0, BLOCK_SIZE_K // 2)
    #NN load copy in register by broadcast
    b_ptrs = (
        b_ptr
        + off_experts * stride_be
        #+ (offs_bk[None, :] * stride_bk + offs_bn[:, None] * stride_bn)
        + (offs_bn[:, None] * stride_bn + offs_bk[None, :] * stride_bk)
    )
    b_shifter = (offs_ak[:,None]%2)*4

    # Create reverse AWQ order as tensor: [0, 4, 1, 5, 2, 6, 3, 7]
    # that will map given indices to the correct order.
    # reverse_awq_order_tensor = ((tl.arange(0, 2) * 4)[None, :] +
    #                             tl.arange(0, 4)[:, None]).to(tl.uint8).reshape(8)
    

    a_scale_ptrs = a_scale_ptr + (offs_token // top_k) * stride_asm

    b_scale_ptrs = (
            b_scale_ptr + off_experts * stride_bse + offs_bn * stride_bsn
        )

    b_scale = tl.load(b_scale_ptrs)
    a_scale = tl.load(a_scale_ptrs, mask=token_mask, other=0.0)
    # -----------------------------------------------------------
    # Iterate to compute a block of the C matrix.
    # We accumulate into a `[BLOCK_SIZE_M, BLOCK_SIZE_N]` block
    # of fp32 values for higher accuracy.
    # `accumulator` will be converted back to fp16 after the loop.
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Load the next block of A and B, generate a mask by checking the
        # K dimension.
        
        a = tl.load(
            a_ptrs,
            mask=token_mask[:, None],
            other=0.0,
        )
        b = tl.load(b_ptrs,cache_modifier="")

        b = tl.interleave(b,b)
        b = b.trans()
        b = (b >> b_shifter) & 0xF
    
        # We accumulate along the K dimension.
        
        b = (b[:,:]-8).to(a_ptr.type.element_ty)
        accumulator += tl.dot(a, b) 

        # Advance the ptrs to the next K block.
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += (BLOCK_SIZE_K // 2) * stride_bk

    if MUL_ROUTED_WEIGHT:
        moe_weight = tl.load(topk_weights_ptr + offs_token, mask=token_mask, other=0)
        accumulator = accumulator * moe_weight[:, None]


    accumulator = (accumulator * a_scale[:, None] * b_scale[None,:]).to(compute_type)

    # -----------------------------------------------------------
    # Write back the block of the output
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn[None, :]
    c_mask = token_mask[:, None] & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)


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
                            use_int8_w4a8: bool,
                            use_int8_w8a16: bool,
                            use_int4_w4a16: bool,
                            per_channel_quant: bool,
                            block_shape: Optional[List[int]] = None,
                            use_nn_moe: Optional[bool]=False) -> None:
    assert topk_weights is not None or not mul_routed_weight
    assert topk_weights is None or topk_weights.stride(1) == 1
    assert sorted_token_ids.stride(0) == 1
    assert use_int8_w4a8 is True
    
    M = A.shape[0]
    num_tokens = M * top_k

    triton_config = {   "BLOCK_SIZE_M": 16,
        "BLOCK_SIZE_N": 64,
        "BLOCK_SIZE_K": 64,
        "GROUP_SIZE_M": 1,
        # "kpack": 1,
        # "num_stages": 0,
        # "num_warps": 2
    }
    triton_config["BLOCK_SIZE_M"] = config["BLOCK_SIZE_M"]
    print("BLOCK_SIZE_M",config["BLOCK_SIZE_M"])
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

    # torch.manual_seed(0)
    # begin, end, expert_id = 0, 1, 0
    # A = torch.randint(-10, 10, A.shape, dtype=torch.int8, device=B.device)
    # B = torch.randint(-10, 10, B.shape, dtype=torch.int8, device=B.device)
    # res = torch.matmul(A[sorted_token_ids[begin:end] // top_k].cpu().to(torch.int32), B[expert_id, :, :].t().cpu().to(torch.int32))
    # out = res.to(A.device) * A_scale[sorted_token_ids[begin:end] // top_k] * B_scale[expert_id].t()
    # '''
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
            A.stride(0),#strideam
            A.stride(1),#strideak
            B.stride(0),#stridebe
            B.stride(2) if not use_nn_moe else B.stride(1),#stridebk
            B.stride(1) if not use_nn_moe else B.stride(2),#stridebn
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

            MUL_ROUTED_WEIGHT=mul_routed_weight,
            top_k=top_k,
            compute_type=compute_type,   
            **triton_config, 
        )
    #'''
    # out_cuda = torch.zeros_like(C)
    _, n, k = B.shape
    num_size = 1024 * 1024 * 1024
    out_cuda_size = M * top_k * n
    str_offset = 100 * 1024 * 1024
    end_offset = str_offset + out_cuda_size
    buff = torch.ones(num_size, dtype=C.dtype, device=C.device)
    out_cuda = buff[str_offset:end_offset].reshape(C.shape)
    print(f"num_size:{num_size}, out_cuda_size:{out_cuda_size}, start offset:{str_offset}, end offset:{end_offset}")

    
    print("A",A.size())
    print("B",B.size())
    print("out_cuda",out_cuda.size())
    print("A_scale",A_scale.size())
    print("B_scale",B_scale.size())
    print("sorted_token_ids",sorted_token_ids.size())
    print("expert_ids",expert_ids.size())
    print("num_tokens_post_padded",num_tokens_post_padded)
    print("top_k",top_k)
    print("config",config)
    # print("end offset:", m*top*n)
    moe_gemm_w4a8(A, 
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
    # print("C",C)
    # print("out_cuda",out_cuda)
    if not (torch.mean(torch.abs(C.to(torch.float32) - out_cuda.to(torch.float32)))
            / torch.mean(torch.abs(C.to(torch.float32)))
            < 0.05):
        print(f"kernel{stage} torch.mean 精度检查不合格!!!")
    else:
        print(f"kernel{stage} torch.mean 精度检查合格.")
    # if not (torch.allclose(C.to(torch.bfloat16), out_cuda.to(torch.bfloat16), rtol=1e-3, atol=1e-5)):
    if not (torch.allclose(C, out_cuda, rtol=1e-3, atol=1e-5)):
        if not mul_routed_weight and stage == 1:
            gemm1_compare_failed_list.append(config['MODE'])
        elif stage == 2:
            gemm2_compare_failed_list.append(config['MODE'])
        
        print(f"kernel{stage} 精度检查不合格!!!")
    else:
        print(f"kernel{stage} 精度检查合格.")
    
    print(torch.nonzero(buff != 1))
    '''
    def unpack_int8_to_int4(weight):
        assert weight.dim() == 2
        assert weight.dtype == torch.int8
        return torch.stack((weight & 0xF, (weight >> 4) & 0xF), dim=2).view(
            weight.shape[0], 2 * weight.shape[1]
        )
    bint8 = unpack_int8_to_int4(B.view(-1, B.shape[-1])) - 8
    bint8 = bint8.reshape(B.shape[0], B.shape[1], -1)
    torch.stack((B.view(-1, B.shape[-1]) & 0xF, (B.view(-1, B.shape[-1]) >> 4) & 0xF), dim=2)
    torch.nonzero((C - out_cuda).abs() > 0.001)
    torch.nonzero((C.to(torch.bfloat16) - out_cuda.to(torch.bfloat16)).abs() > 2)
    torch.nonzero((C.view(-1, 256) - out_cuda.view(-1, 256)).abs() > 1)
    torch.nonzero(sorted_token_ids == 0)
    
    begin, end, expert_id = 6918, 6919, 54
    res = torch.matmul(A[sorted_token_ids[begin:end] // top_k].cpu().to(torch.int32), bint8[expert_id, :, :].t().cpu().to(torch.int32))
    scale = A_scale[sorted_token_ids[begin:end] // top_k] * B_scale[expert_id].t()
    # out = res.to(A.device) * scale
    out = res.to(A.device) * A_scale[sorted_token_ids[begin:end] // top_k] * B_scale[expert_id].t()
    out_bf16 =  out.to(torch.bfloat16)
    import pdb
    pdb.set_trace()
    # '''
    return out_cuda, C


def fused_experts_impl_int8(hidden_states: torch.Tensor,
                       w1: torch.Tensor,
                       w2: torch.Tensor,
                       topk_weights: torch.Tensor,
                       topk_ids: torch.Tensor,
                       inplace: bool = False,
                       activation: str = "silu",
                       apply_router_weight_on_input: bool = False,
                       use_fp8_w8a8: bool = False,
                       use_int8_w4a8: bool = False,
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
                       mode: Optional[List[int]]=None):
    # Check constraints. 
    assert use_int8_w4a8 is True
    assert hidden_states.shape[1] == w1.shape[2] * 2, "Hidden size mismatch"

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
    # print("E,N,K",E,N,K)
    if global_num_experts == -1:
        global_num_experts = E
    top_k_num = topk_ids.shape[1]
    # We execute the fused_moe kernel in chunks to circumvent this issue:
    # https://github.com/vllm-project/vllm/issues/5938
    CHUNK_SIZE = envs.VLLM_FUSED_MOE_CHUNK_SIZE
    M = min(num_tokens, CHUNK_SIZE)
    # print("M",M)
    # We can reuse the memory between these because by the time we need
    # cache3, we're done with cache1

    cache13 = torch.empty(M * top_k_num * max(N, K if not use_nn_moe else w2.shape[2]),
                          device=hidden_states.device,
                          dtype=hidden_states.dtype)
    intermediate_cache1 = cache13[:M * top_k_num * N].view(M, top_k_num, N)
    # print("intermediate_cache1",intermediate_cache1.size())
    intermediate_cache3 = cache13[:M * top_k_num * (K if not use_nn_moe else w2.shape[2])].view(M, top_k_num, K if not use_nn_moe else w2.shape[2])
    # print("intermediate_cache3",intermediate_cache3.size())
    # This needs separate memory since it's used concurrently with cache1
    intermediate_cache2 = torch.empty((M * top_k_num, N // 2),
                                      device=hidden_states.device,
                                      dtype=hidden_states.dtype)
    # print("intermediate_cache2",intermediate_cache2.size())
    if hidden_states.dtype == torch.bfloat16:
        compute_type = tl.bfloat16
    elif hidden_states.dtype == torch.float16:
        compute_type = tl.float16
    elif hidden_states.dtype == torch.float32:
        compute_type = tl.float32
    else:
        raise ValueError(f"Unsupported compute_type: {hidden_states.dtype}")

    # debug_output = torch.zeros([16512, 7680], dtype=hidden_states.dtype, device=hidden_states.device)
    if inplace:
        out_hidden_states = hidden_states
    else:
        # print("hidden_states:", hidden_states.shape)
        out_hidden_states = torch.empty_like(hidden_states)
        # row, col = hidden_states.shape
        # out_hidden_states = debug_output[:row, :col]
    # torch.nonzero(out_hidden_states==0)
    # torch.nonzero(debug_output[256:, 7168:]!=0)

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
            use_int8_w8a8=use_int8_w4a8,
            use_int8_w8a16=use_int8_w8a16,
            use_int4_w4a16=use_int4_w4a16,
            per_channel_quant=per_channel_quant,
            block_shape=block_shape)

        # config['BLOCK_SIZE_M'] = blockM_configs[mode0]
        config1 = {}
        config2 = {}
        if M > M_THRESHOLD:
            config1['BLOCK_SIZE_M'] = blockM_configs[mode0]
            config1['MODE']         = mode0
            config1['DELTA']        = 1
            config2['BLOCK_SIZE_M'] = blockM_configs[mode1]
            config2['MODE']         = mode1
            config2['DELTA']        = 1
        else:
            config1['BLOCK_SIZE_M'] = blockM_configs_decode[mode0]
            config1['MODE'] = mode0
            config1['DELTA'] = 1
            config2['BLOCK_SIZE_M'] = blockM_configs_decode[mode1]
            config2['MODE'] = mode1
            config2['DELTA'] = 1
        
        '''
        align = 128
        config1['BLOCK_SIZE_M'] = align
        config1['MODE'] = 1000
        # '''
        sorted_token_ids, expert_ids, num_tokens_post_padded = (
                moe_align_block_size(curr_topk_ids, config1['BLOCK_SIZE_M'],
                                    global_num_experts, expert_map))
        # '''
        # sorted_token_ids[:128] = torch.tensor([20375, 20444, 40787,  6104, 61222, 61244, 24493, 26563, 26412, 17653,
        # 46929, 46945, 44895, 44441, 44353, 44411, 64849, 22648,  6940, 45239,
        # 43188, 43162, 43011, 43078, 27564, 30711, 14325, 34653, 19490, 39950,
        # 53233, 60612, 42139,  6305, 55230, 60539, 42237, 26709, 62640,  7779,
        # 49381, 49402,  7817,  7867, 28337, 26189, 26282, 48722, 48740,  9009,
        #  1839,  1895,  1901, 29597, 22437, 31736, 63363, 51971,  3661, 13135,
        # 22095, 24130, 24146, 24205, 13263, 42684, 44571, 42603, 35685, 26073,
        # 56318, 25895, 54050, 48426, 46415, 65045, 51485, 51526, 51545, 65145,
        # 51532, 15411, 15386, 30951, 30952, 30968, 30843, 64161, 51233, 25504,
        # 25575, 13877, 14024, 34546, 36352, 34315, 34345, 54972, 54841, 56903,
        # 23463, 23481, 23353, 11320, 11371, 13415, 41935, 11512, 31751, 43838,
        # 33910, 31815, 31889, 12537, 54480, 12380, 14491, 34881, 32952, 34820,
        # 62423, 53424, 62445, 53281, 27033, 45492, 16208, 18295],
        # device='cuda:0', dtype=torch.int32)
        align = 128
        print("sorted_token_ids:", sorted_token_ids[0:align])
        print("sorted_token_ids // topk:", sorted_token_ids[0:align] // top_k_num)
        print("sorted_token_ids % topk:", sorted_token_ids[0:align] % top_k_num)
        print("expert_ids:", expert_ids[:10])
        # pdb.set_trace()
        # torch.nonzero(sorted_token_ids == 0)
        # '''
        out_cuda, out_triton = invoke_fused_moe_kernel_int8(qcurr_hidden_states,
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
                                use_int8_w4a8=use_int8_w4a8,
                                use_int8_w8a16=use_int8_w8a16,
                                use_int4_w4a16=use_int4_w4a16,
                                per_channel_quant=per_channel_quant,
                                block_shape=block_shape,
                                use_nn_moe=use_nn_moe)
        '''
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
            use_int8_w8a8=use_int8_w4a8,
            use_int8_w8a16=use_int8_w8a16,
            use_int4_w4a16=use_int4_w4a16,
            per_channel_quant=per_channel_quant,
            block_shape=block_shape)
        # print("qintermediate_cache2, qa2_scale",qintermediate_cache2.size(), qa2_scale.size())
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
                                use_int8_w4a8=use_int8_w4a8,
                                use_int8_w8a16=use_int8_w8a16,
                                use_int4_w4a16=use_int4_w4a16,
                                per_channel_quant=per_channel_quant,
                                block_shape=block_shape,
                                use_nn_moe=use_nn_moe)

        ops.moe_sum(intermediate_cache3.view(*intermediate_cache3.shape),
                    out_hidden_states[begin_chunk_idx:end_chunk_idx])
        # '''
        # import pdb
        # pdb.set_trace()
    return out_hidden_states

def _fused_moe_helper(m: int,
                             n: int,
                             k: int,
                             n2:int,
                             k2:int,
                             topk:int,
                             E:int,
                             out_dtype: Type[torch.dtype] = torch.bfloat16,
                             device: str = "cuda",
                             block_size: List[int]=[128,128],
                             best_config:Optional[list] = None):

    input = (torch.randn((m, n2), device=device) * 5).to(dtype=out_dtype)
    weight13 = to_int8(torch.randn((E,n ,k), device=device) * 5)
    weight2 = to_int8(torch.randn((E,n2 ,k2), device=device) * 5)

    weight13_scale = (torch.randn((E, n ,1), device=device,
                           dtype=torch.float32))
    weight2_scale = (torch.randn((E, n2, 1), device=device,
                           dtype=torch.float32))   

    tensor = torch.rand((m, topk),device=device)
    topk_weights = F.softmax(tensor, dim=1)

    topk_ids=generate_unique_int_tensor(m=m, topk=topk, low=0, high=E).cuda()

    
    return input,weight13,weight2,weight13_scale,weight2_scale,topk_weights,topk_ids






def run_perchannel_vaild():
    # m_list=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,32,64,128,256,512,1024,2048,4096,8192,16384,32768]
    # m_list=[1, 8, 16, 32, 33, 64, 65, 128] #decode
    # m_list=[128, 256,512,1024,2048,4096,8192,16384] #prefill
    # m_list=[1,2,4,8,16,32,64,128]   
    m_list = [4096] 
    E=256
    n_list_1=[512]
    k_list_1=[3584]
    n_list_2=[7168]
    k_list_2=[128]
    out_dtype=torch.bfloat16
    topk=8
    best_config = None
    triton_json_dir="./cache/moe/"
    os.makedirs(triton_json_dir, exist_ok=True)
    
    device_name=torch.cuda.get_device_properties('cuda').name.replace(" ","_")

    # mode_ranges = [i for i in range(len(blockM_configs))]
    # mode_ranges = [mode for mode in blockM_configs.keys()]
    mode_ranges = [1000]
    # compare_fail = [42,43,46,52,54,60]
    # compare_fail = [64,66,67,69,70,71,77,78,79,80,81,82,83,84,85,94]
    # compare_fail = [6,11,13,42,43,46,52,54,60,64,66,67,69,70,71,77,78,79,80,81,82,83,84,85,94]
    #deepseek-471b-tp16

    # # decode corrtnese
    # mode_ranges = [i for i in range(len(blockM_configs_decode))]
    # exclude_modes = [13, 32, 33, 34, 35]
    # mode_ranges = [3,4,6,14,22,26,28,29]
    # #exclude_modes = [13] # nk 4096 192 gemm2<16
    
    
    for i in range(0,len(k_list_1),1):
        for m in m_list:
            for mode in mode_ranges:
                mode_list=[mode,mode]
                # mode_list=[11,27]
                print("\033[32mmode:{} m:{} n:{} k:{}\033[0m ".format(mode_list, m,n_list_1[i],k_list_1[i]))
                input,weight13,weight2,weight13_scale,weight2_scale,topk_weights,topk_ids=_fused_moe_helper(m=m,n=n_list_1[i],k=k_list_1[i],n2=n_list_2[i],k2=k_list_2[i],topk=topk,E=E,block_size=None,out_dtype=out_dtype)
                # input = input.to(torch.float)
                print("input:", input.shape, input)
                # print("weight13:", weight13.shape, weight13)
                # print("weight2:", weight2.shape, weight2)
                # triton moe: output
                output=fused_experts_impl_int8(hidden_states=input,
                                                w1=weight13,
                                                w2=weight2,
                                                topk_weights=topk_weights,
                                                topk_ids=topk_ids,
                                                inplace=False,
                                                use_int8_w4a8=True,
                                                w1_scale=weight13_scale,
                                                w2_scale=weight2_scale,
                                                block_shape=None,
                                                config={"BLOCK_SIZE_M": 16},
                                                per_channel_quant=True,
                                                mode=mode_list)

                # print("output:", output)
                # if not (torch.mean(torch.abs(output.to(torch.float32) - out_native.to(torch.float32)))
                # / torch.mean(torch.abs(out_native.to(torch.float32)))
                # < 0.05):
                #     print("triton 精度检查不合格！！！") 
                #     # print("output:",output)
                #     # print("out_native:",out_native)
                # else:
                #     print("triton 精度检查合格")
                     
    print("gemm1_compare_failed_list: ", gemm1_compare_failed_list)
    print("gemm2_compare_failed_list: ", gemm2_compare_failed_list)
            
def main():
    # run_tuning()
    run_perchannel_vaild()

if __name__ == "__main__":
    main()