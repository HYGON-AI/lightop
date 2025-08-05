import pdb
import torch
torch.manual_seed(0)
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
# from test_int8_kernel import torch_w8a8_per_column_moe
# import _custom_ops as ops
from lightop import moe_gemm_w8a8
from lmslim.layers.gemm.int8_utils import (
    per_token_group_quant_int8, per_token_quant_int8)

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
    1001: 64,
    1002: 32,
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
    36: 16,
    37: 16,
    38: 16,
    39: 16,
    40: 16,
    41: 16,
    42: 16,
    43: 16
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

    # import pdb
    # pdb.set_trace()
    # A = torch.ones_like(A)
    # B = torch.ones_like(B)
    # A_scale = torch.ones_like(A_scale)
    # B_scale = torch.ones_like(B_scale)
    # topk_weights = torch.ones_like(topk_weights)
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
    #'''
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

    # import pdb
    # pdb.set_trace()
    stage = 0
    if not mul_routed_weight:
        stage = 1
    else:
        stage = 2
    if not (torch.allclose(C, out_cuda, rtol=1e-2, atol=1e-2)):
        if not mul_routed_weight and stage == 1:
            gemm1_compare_failed_list.append(config['MODE'])
        elif stage == 2:
            gemm2_compare_failed_list.append(config['MODE'])
        
        print(f"kernel{stage} 精度检查不合格!!!")
    else:
        print(f"kernel{stage} 精度检查合格.")
    
    '''
    def torch_moe(a, b, sa, sb, topk_ids, out_dtype):
        B, D = a.shape
        device = a.device
        topk = topk_ids.shape[1]
        topk_ids = topk_ids.view(-1)
        out = torch.zeros(B * topk, b.shape[1], dtype=torch.float, device=device)
        a = a.view(B, -1, D).repeat(1, topk, 1).reshape(-1, D).cpu().to(torch.int32)
        b = b.cpu().to(torch.int32)
        sa = sa.view(B, -1, 1).repeat(1, topk, 1).reshape(-1, 1)
        for i in range(b.shape[0]):
            mask = topk_ids == i
            if mask.sum():
                scale = sa[mask].mul(sb[i].t())
                axb = a[mask.cpu()] @ b[i].transpose(0, 1)
                out[mask] = axb.to(device).to(torch.float) * scale
        
        return out.reshape(B, topk, -1).to(out_dtype)
        
    pdb.set_trace()
    torch.nonzero((C - out_cuda).abs() > 1)
    torch.nonzero((C.view(-1, 256) - out_cuda.view(-1, 256)).abs() > 1)
    torch.nonzero((C[404, 0] - out_cuda[404, 0]).abs() > 1)
    torch.nonzero(sorted_token_ids == 135)
    torch_c = torch.zeros_like(C)
    out_dtype = torch_c.dtype
    torch_c = torch_moe(A, B, A_scale, B_scale, topk_ids, out_dtype)
    assert torch.allclose(torch_c, C, rtol=1e-2, atol=1e-2)
    assert torch.allclose(torch_c, out_cuda, rtol=1e-2, atol=1e-2)
    #'''
    # torch.cuda.synchronize()
    


def fused_experts_impl_int8(hidden_states: torch.Tensor,
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
                       mode: Optional[List[int]]=None):
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

        # config['BLOCK_SIZE_M'] = blockM_configs[mode0]
        config1 = {}
        config2 = {}
        if 1:
            config1['BLOCK_SIZE_M'] = blockM_configs[mode0]
            config1['MODE']         = mode0
            config1['DELTA']        = 1
            config2['BLOCK_SIZE_M'] = blockM_configs[mode1]
            config2['MODE']         = mode1
            config2['DELTA']        = int(blockM_configs[mode0]/blockM_configs[mode1])
        else:
            config1['BLOCK_SIZE_M'] = blockM_configs_decode[mode0]
            config1['MODE']         = mode0
            config1['DELTA']        = 1
            config2['BLOCK_SIZE_M'] = blockM_configs_decode[mode1]
            config2['MODE']         = mode1
            config2['DELTA']        = int(blockM_configs[mode0]/blockM_configs[mode1])
            
    
        align = 16
        config1['BLOCK_SIZE_M'] = align
        config1['MODE'] = 1004
        sorted_token_ids, expert_ids, num_tokens_post_padded = (
                moe_align_block_size(curr_topk_ids, config1['BLOCK_SIZE_M'],
                                    global_num_experts, expert_map))
        
        invoke_fused_moe_kernel_int8(qcurr_hidden_states,
                                w1,
                                intermediate_cache1,
                                qa1_scale,
                                w1_scale,
                                w1_zp, #none
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
    
        # import pdb
        # pdb.set_trace()
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

        # mode1=2
        # config1 = {}
        # config2 = {}
        # if M > M_THRESHOLD:
        #     config1['BLOCK_SIZE_M'] = blockM_configs[mode0]
        #     config1['MODE']         = mode0
        #     config1['DELTA']        = 1
        #     config2['BLOCK_SIZE_M'] = blockM_configs[mode1]
        #     config2['MODE']         = mode1
        #     config2['DELTA']        = blockM_configs[mode0]//blockM_configs[mode1]
        # else:
        #     config1['BLOCK_SIZE_M'] = blockM_configs_decode[mode0]
        #     config1['BLOCK_SIZE_M'] = mode0
        #     config1['BLOCK_SIZE_M'] = 1
        #     config2['BLOCK_SIZE_M'] = blockM_configs_decode[mode1]
        #     config2['BLOCK_SIZE_M'] = mode1
        #     config2['BLOCK_SIZE_M'] = blockM_configs[mode0]//blockM_configs[mode1]
            
        # intermediate_cache3_copy = torch.zeros_like(intermediate_cache3)
        # invoke_fused_moe_kernel_int8(qintermediate_cache2,
        #                         w2,
        #                         intermediate_cache3_copy,
        #                         qa2_scale,
        #                         w2_scale,
        #                         w2_zp,
        #                         curr_topk_weights,
        #                         curr_topk_ids,
        #                         sorted_token_ids,
        #                         expert_ids,
        #                         num_tokens_post_padded,
        #                         not apply_router_weight_on_input,
        #                         1,
        #                         config2,
        #                         compute_type=compute_type,
        #                         use_fp8_w8a8=use_fp8_w8a8,
        #                         use_int8_w8a8=use_int8_w8a8,
        #                         use_int8_w8a16=use_int8_w8a16,
        #                         use_int4_w4a16=use_int4_w4a16,
        #                         per_channel_quant=per_channel_quant,
        #                         block_shape=block_shape,
        #                         use_nn_moe=use_nn_moe)
        
        # if not (torch.allclose(intermediate_cache3, intermediate_cache3_copy, rtol=1e-2, atol=1e-2)):            
        #     print(f"block {config2['BLOCK_SIZE_M']} kernel 精度检查不合格!!!")
        #     # torch.set_printoptions(profile="full")  
        #     # ftriton = open('cuda_output.txt', 'w')   
        #     # print(f'cuda_output: {out_cuda}', file=ftriton)
        #     # ftriton.close
        #     # ftorch = open('bench_output.txt', 'w')   
        #     # print(f'bench_output: {C}', file=ftorch)
        #     # ftorch.close
        # else:
        #     print(f"block {config2['BLOCK_SIZE_M']} kernel 精度检查合格.")

        ops.moe_sum(intermediate_cache3.view(*intermediate_cache3.shape),
                    out_hidden_states[begin_chunk_idx:end_chunk_idx])
        '''
    return out_hidden_states

def _fused_moe_helper(m: int,
                             n: int,
                             k: int,
                             n2:int,
                             k2:int,
                             topk:int,
                             E:int,
                             out_dtype: Type[torch.dtype] = torch.float16,
                             device: str = "cuda",
                             block_size: List[int]=[128,128],
                             best_config:Optional[list] = None):
    # Test for a cutlass kernel with per-token activation quantization
    # and per-output channel weight quantization.
    torch.manual_seed(0)
    input = (torch.randn((m, k), device=device) * 5).to(dtype=out_dtype)
    weight13 = to_int8(torch.randn((E,n ,k), device=device) * 5)
    weight2 = to_int8(torch.randn((E,n2 ,k2), device=device) * 5)

    weight13_scale = (torch.randn((E, n ,1), device=device,
                           dtype=torch.float32))
    weight2_scale = (torch.randn((E, n2, 1), device=device,
                           dtype=torch.float32))   
    # weight13_scale = (torch.randn((E,math.ceil(n/block_size[0]), math.ceil(k/block_size[1])), device=device,
    #                        dtype=torch.float32))
    # weight2_scale = (torch.randn((E,math.ceil(n2/block_size[0]), math.ceil(k2/block_size[1])), device=device,
    #                        dtype=torch.float32))    
    
    # print("input.dtype:",input.dtype)

    #q_input, x_scale = per_token_group_quant_int8(input_2d, block_size[1])
    
    #topk_weights [m,topk]
    tensor = torch.rand((m, topk),device=device)
    topk_weights = F.softmax(tensor, dim=1)
    #print("topk_weights:",topk_weights)
    
    #topk_ids [m,topk]
    topk_ids=generate_unique_int_tensor(m=m, topk=topk, low=0, high=E).cuda()
    # topk_ids=generate_unique_int_tensor(m, topk).cuda()
    #print("topk_ids:",topk_ids)
    
    return input,weight13,weight2,weight13_scale,weight2_scale,topk_weights,topk_ids

def native_w8a8_per_token_matmul(A, B, As, Bs, output_dtype=torch.float16):
    """Matrix multiplication function that supports per-token input quantization and per-column weight quantization"""
    A = A.to(torch.float32)
    B = B.to(torch.float32)

    assert A.shape[-1] == B.shape[-1], "Dimension mismatch"
    assert B.ndim == 2 and B.is_contiguous(), "B must be a 2D contiguous tensor"

    # Reshape input
    M = A.numel() // A.shape[-1]
    B = B.t()  # Transpose weight matrix
    N, K = B.shape
    origin_C_shape = A.shape[:-1] + (K,)
    A = A.reshape(M, N)

    # As is per-token [M, 1], Bs is per-column [1, K]
    C = torch.matmul(A, B)  # [M, K]
    C = As * C * Bs.view(1, -1)  # Broadcast per-column scale

    return C.reshape(origin_C_shape).to(output_dtype)


def torch_w8a8_per_column_moe(a, w1, w2, w1_s, w2_s, topk_weight, topk_ids,topk):
    """This function performs fused moe with per-column int8 quantization using native torch."""

    B, D = a.shape
    # Perform per-token quantization
    a_q, a_s = per_token_quant_int8(a)
    # Repeat tokens to match topk
    a_q = a_q.view(B, -1, D).repeat(1, topk, 1).reshape(-1, D)
    # Also repeat the scale
    a_s = a_s.view(B, -1, 1).repeat(1, topk, 1).reshape(-1, 1)  # [B*topk, 1]

    out = torch.zeros(B * topk, w2.shape[1], dtype=a.dtype, device=a.device)

    # Calculate routing
    # score = torch.softmax(score, dim=-1, dtype=torch.float32)
    # topk_weight, topk_ids = torch.topk(score, topk)
    topk_weight = topk_weight.view(-1)
    topk_ids = topk_ids.view(-1)
    # Process each expert
    for i in range(w1.shape[0]):
        mask = topk_ids == i
        if mask.sum():
            # First MLP layer: note that a_s is now per-token
            inter_out = native_w8a8_per_token_matmul(
                a_q[mask], w1[i], a_s[mask], w1_s[i], output_dtype=a.dtype
            )
            d = inter_out.shape[-1] // 2
            act_out = F.silu(inter_out[..., :d]) * inter_out[..., d:]
            # Quantize activation output with per-token
            act_out_q, act_out_s = per_token_quant_int8(act_out)

            # Second MLP layer
            out[mask] = native_w8a8_per_token_matmul(
                act_out_q, w2[i], act_out_s, w2_s[i], output_dtype=a.dtype
            )
    # Apply routing weights and sum
    return (
        out.view(B, -1, w2.shape[1]) * topk_weight.view(B, -1, 1).to(out.dtype)
    ).sum(dim=1)


def run_perchannel_vaild():
    # m_list=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,32,64,128,256,512,1024,2048,4096,8192,16384,32768]
    # m_list=[1, 8, 16, 32, 33, 64, 65, 128] #decode
    # m_list=[256,512,1024,2048,4096,8192,16384] #prefill
    m_list=[
            144,160,176,192,208,224,240,240,256,288,320,
            # 352,384,416,448,480,512,640,768,896,1024,1152,
            # 1280,1408,1536,1664,1792,192,1920, 2048,2176,
            # 2304, 2432,2560,2688,2816, 2944,3072,
            # 3200,3328,3456,3584,3712,3840,3968,4096,
            # 6144, 8192, 10240, 12288, 14336, 16384, 
            # 18432, 20480,22528,24576,26624,28672,32768
            ]
    # m_list=[8192,]
    # m_list = [16384]
    # n_list_1=[512,]
    # k_list_1=[7168,]
    
    # ##deepseek-671b-tp16
    # E=256
    # topk=8
    # n_list_1=[256]
    # k_list_1=[7168]
    # n_list_2=[7168]
    # k_list_2=[128]
    
    # #deepseek-671b-tp8
    # E=256
    # topk=8
    # n_list_1=[512]
    # k_list_1=[7168]
    
    #qwen3
    # E=128
    #qwen3-30b-A3B-tp1
    # n_list_1=[1536]
    # k_list_1=[2048]
    
    ##qwen3-30b-A3B-tp2
    # n_list_1=[768]
    # k_list_1=[2048]
    
    # ##qwen3-235b-A22B-tp8
    # n_list_1=[384]
    # k_list_1=[4096]
    
    #deepseek-671b-tp16
    # n_list_2=[7168]
    # k_list_2=[128]
    # #deepseek-671b-tp8
    # n_list_2=[7168]
    # k_list_2=[256]
    
    # #qwen3
    # #qwen3-30b-A3B-tp1
    # E=128
    # topk=8
    # n_list_1=[1536]
    # k_list_1=[2048]
    # n_list_2=[2048]
    # k_list_2=[768]
    
    # ##qwen3-30b-A3B-tp2
    # E=128
    # topk=8
    # n_list_1=[768]
    # k_list_1=[2048]
    # n_list_2=[2048]
    # k_list_2=[384]
    
    ##qwen3-235b-A22B-tp8
    # E=128
    # topk=8
    # n_list_1=[384]
    # k_list_1=[4096]
    # n_list_2=[4096]
    # k_list_2=[192]
    
    ##zijie
    E=64
    topk=4
    n_list_1=[2048]
    k_list_1=[4096]
    n_list_2=[4096]
    k_list_2=[1024]
    
    out_dtype=torch.bfloat16
    # out_dtype = torch.float16


    # configs=BlockSizeMConfigGen()
    best_config = None
    triton_json_dir="./cache/moe/"
    os.makedirs(triton_json_dir, exist_ok=True)
    
    device_name=torch.cuda.get_device_properties('cuda').name.replace(" ","_")

    # mode_ranges = [i for i in range(len(blockM_configs))]
    mode_ranges = [mode for mode in blockM_configs.keys()]
    # mode_ranges = [61]
    # compare_fail = [42,43,46,52,54,60]
    # compare_fail = [64,66,67,69,70,71,77,78,79,80,81,82,83,84,85,94]
    # compare_fail = [6,11,13,42,43,46,52,54,60,64,66,67,69,70,71,77,78,79,80,81,82,83,84,85,94]
    #deepseek-471b-tp16

    # # decode corrtnese
    mode_ranges = [i for i in range(len(blockM_configs_decode))]
    # exclude_modes = [13, 32, 33, 34, 35]
    # #mode_ranges = [3,4,6,8,10,12,14]
    # #exclude_modes = [13] # nk 4096 192 gemm2<16

    #mode_ranges = [3]

    
    for i in range(0,len(k_list_1),1):
        for m in m_list:
            for mode in mode_ranges:
                mode_list=[mode,mode]
                print("mode:{} m:{} n:{} k:{} ".format(mode_list, m,n_list_1[i],k_list_1[i]))
                input,weight13,weight2,weight13_scale,weight2_scale,topk_weights,topk_ids=_fused_moe_helper(m=m,n=n_list_1[i],k=k_list_1[i],n2=n_list_2[i],k2=k_list_2[i],topk=topk,E=E,block_size=None,out_dtype=out_dtype)
                # triton moe: output
                output=fused_experts_impl_int8(hidden_states=input,
                                                w1=weight13,
                                                w2=weight2,
                                                topk_weights=topk_weights,
                                                topk_ids=topk_ids,
                                                inplace=False,
                                                use_int8_w8a8=True,
                                                w1_scale=weight13_scale,
                                                w2_scale=weight2_scale,
                                                block_shape=None,
                                                config={"BLOCK_SIZE_M": 32},
                                                per_channel_quant=True,
                                                mode=mode_list)
                # pdb.set_trace()
                '''
                #torch native moe: out_native
                out_native = torch_w8a8_per_column_moe(input, weight13, weight2, weight13_scale, weight2_scale, topk_weights, topk_ids,topk)
                
                if not (torch.mean(torch.abs(output.to(torch.float32) - out_native.to(torch.float32)))
                / torch.mean(torch.abs(out_native.to(torch.float32)))
                < 0.05):
                    print("triton 精度检查不合格！！！") 
                    # print("output:",output)
                    # print("out_native:",out_native)
                else:
                    print("triton 精度检查合格")
                '''
                     
    print("gemm1_compare_failed_list: ", gemm1_compare_failed_list)
    print("gemm2_compare_failed_list: ", gemm2_compare_failed_list)
            
def main():
    # run_tuning()
    run_perchannel_vaild()

if __name__ == "__main__":
    main()