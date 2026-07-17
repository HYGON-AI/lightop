import functools
import json
import os
from typing import Any, Callable, Dict, List, Optional, Tuple
import vllm.envs as envs
from vllm import _custom_ops as ops
from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
    moe_align_block_size)
import torch
import triton
import triton.language as tl
from lightop._lmslim_native import envs as lsenvs
from vllm.utils import W8a8GetCacheJSON

use_lightop = lsenvs.LMSLIM_USE_LIGHTOP
device_name = lsenvs.LMSLIM_GPU_NAME

from lightop.moe import (
   per_token_group_quant_int8, per_token_quant_int8)

def to_int8(tensor: torch.Tensor):
    return torch.round(tensor.clamp(min=-128,max=127)).to(dtype=torch.int8)

@triton.jit
def write_zeros_to_output(c_ptr, stride_cm, stride_cn, pid_n, N, offs_token,
                          token_mask, BLOCK_SIZE_M, BLOCK_SIZE_N,
                          compute_type):
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=compute_type)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn[
                                                                   None, :]
    c_mask = token_mask[:, None] & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)
    
def grouped_topk(
    hidden_states: torch.Tensor,
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
    num_expert_group: int = 0,
    topk_group: int = 0,
):
    # assert hidden_states.shape[0] == gating_output.shape[0], "Number of tokens mismatch"

    scores = torch.softmax(gating_output, dim=-1)
    num_token = scores.shape[0]
    group_scores = (
        scores.view(num_token, num_expert_group, -1).max(dim=-1).values
    )  # [n, n_group]
    group_idx = torch.topk(group_scores, k=topk_group, dim=-1, sorted=False)[
        1
    ]  # [n, top_k_group]
    group_mask = torch.zeros_like(group_scores)  # [n, n_group]
    group_mask.scatter_(1, group_idx, 1)  # [n, n_group]
    score_mask = (
        group_mask.unsqueeze(-1)
        .expand(num_token, num_expert_group, scores.shape[-1] // num_expert_group)
        .reshape(num_token, -1)
    )  # [n, e]
    tmp_scores = scores.masked_fill(~score_mask.bool(), 0.0)  # [n, e]
    topk_weights, topk_ids = torch.topk(tmp_scores, k=topk, dim=-1, sorted=False)

    if renormalize:
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

    return topk_weights.to(torch.float32), topk_ids.to(torch.int32)


def get_w8a8moe_json(E:int,m:int,n1:int,n2:int,k:int):
    #接续json
    configs=None
    moe_dict= W8a8GetCacheJSON().triton_moejson_dict
    
    if moe_dict is not None:
        if len(moe_dict)==0:
            configs=None
        elif f"1_{n1}_{n2}_{k}" in moe_dict:
            if m<=16:
                m_=m
            elif m<=32:
                m_= 16 #取值到最近的4的倍数
            elif m<=64:
                m_=256
            elif m<1024: #1024
                m_=1024
            elif m<8192:
                m_=8192
            else:
                m_=16384
            configs=moe_dict[f"{m_}_{n1}_{n2}_{k}"]
            config1=configs['config1']
            config2=configs['config2']

    if configs is None:
        config1=config2={"BLOCK_SIZE_M": 16,
                "BLOCK_SIZE_N": 64,
                "BLOCK_SIZE_K": 64,
                "GROUP_SIZE_M": 1,
                "kpack": 1,
                "num_stages": 0,
                "num_warps": 4}
    return config1,config2

def moe_kernel_prepare_input(
    A: torch.Tensor,
    B: torch.Tensor,
    A_scale: Optional[torch.Tensor],
    B_scale: Optional[torch.Tensor],
    use_fp8_w8a8: bool,
    use_int8_w8a8: bool,
    use_int8_w8a16: bool,
    use_int4_w4a16: bool,
    use_int4_w4a8: bool,
    per_channel_quant: bool,
    block_shape: Optional[List[int]] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    if use_int8_w8a8 or use_int4_w4a8: # 进入此分支
        assert B_scale is not None
        if block_shape is None:
            # activation channel-wise int8 quantization
            assert (per_channel_quant
                    ), "int8 or int4 quantization only supports block or channel-wise"
            A, A_scale = per_token_quant_int8(A)
        else:
            # activation block-wise int8 quantization
            assert len(block_shape) == 2
            _, block_k = block_shape[0], block_shape[1]
            A, A_scale = per_token_group_quant_int8(A, block_k)
            assert triton.cdiv(A.shape[-1], block_k) == A_scale.shape[-1]
    elif use_int8_w8a16 or use_int4_w4a16:
        assert B_scale is not None
        assert block_shape is None or block_shape[0] == 0
    else:
        assert A_scale is None
        assert B_scale is None

    return A, A_scale

@triton.jit
def fused_moe_kernel_int4_w4a8(
    # Pointers to matrices
    a_ptr,
    b_ptr,
    c_ptr,
    a_scale_ptr,
    b_scale_ptr,
    b_qzero_ptr,
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
    stride_bze,
    stride_bzk,
    stride_bzn,
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

):
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
    if off_experts == -1:
        # -----------------------------------------------------------
        # Write back zeros to the output when the expert is not
        # in the current expert parallel rank.
        write_zeros_to_output(c_ptr, stride_cm, stride_cn, pid_n, N,
                              offs_token, token_mask, BLOCK_SIZE_M,
                              BLOCK_SIZE_N, compute_type)
        return
    offs_bk = tl.arange(0, BLOCK_SIZE_K // 2)
    #NN load copy in register by broadcast
    b_ptrs = (
        b_ptr
        + off_experts * stride_be
        #+ (offs_bk[None, :] * stride_bk + offs_bn[:, None] * stride_bn)
        + (offs_bn[:, None] * stride_bn + offs_bk[None, :] * stride_bk)
    )
    b_shifter = (offs_ak[:,None]%2) * 4

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
    # accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.int32)
    # accumulator_int32 = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.int32)

    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
    # for k in range(0, 1):
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
        
        # 同一左移4bit，相当于 * 16 
        b = b << 4

        # We accumulate along the K dimension.
        b = b.to(a_ptr.type.element_ty)
        accumulator += tl.dot(a, b)  #* a_scale[:, None] * b_scale[None,:]
        # Advance the ptrs to the next K block.
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += (BLOCK_SIZE_K // 2) * stride_bk

    if MUL_ROUTED_WEIGHT:
        moe_weight = tl.load(topk_weights_ptr + offs_token, mask=token_mask, other=0)
        accumulator = accumulator * moe_weight[:, None]

    accumulator = (accumulator * a_scale[:, None] * b_scale[None,:]).to(compute_type)

    # accumulator = (accumulator * 1.0).to(compute_type)

    # ========================== 计算结果直接回写到结果里面 ========================

    # -----------------------------------------------------------
    # Write back the block of the output
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn[None, :]
    c_mask = token_mask[:, None] & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)


def invoke_fused_moe_kernel_w4a8(A: torch.Tensor,
                                    B: torch.Tensor,
                                    C: torch.Tensor,
                                    A_scale: Optional[torch.Tensor],
                                    B_scale: Optional[torch.Tensor],
                                    B_zp: Optional[torch.Tensor],
                                    topk_weights: torch.Tensor,
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
                                    use_int4_w4a8: bool,
                                    block_shape: Optional[List[int]] = None,
                                ) -> None:
    
    assert topk_weights is not None or not mul_routed_weight
    assert topk_weights is None or topk_weights.stride(1) == 1
    assert sorted_token_ids.stride(0) == 1
    assert use_int4_w4a8 is True
    
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
        B.shape[1], META['BLOCK_SIZE_N']), )

    fused_moe_kernel_int4_w4a8[grid](
        A,
        B,
        C,
        A_scale,
        B_scale,
        B_zp,
        topk_weights,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        B.shape[1],     
        A.shape[1],   
        EM,
        num_tokens,
        A.stride(0),
        A.stride(1),
        B.stride(0), 
        B.stride(2), 
        B.stride(1), 
        C.stride(1),
        C.stride(2),
        A_scale.stride(0) if A_scale is not None and A_scale.ndim == 2 else 0,
        A_scale.stride(1) if A_scale is not None and A_scale.ndim == 2 else 0,
        B_scale.stride(0) if B_scale is not None and B_scale.ndim >= 2 else 0,
        B_scale.stride(1) if B_scale is not None and B_scale.ndim == 3 else 0,
        B_scale.stride(2) if B_scale is not None and B_scale.ndim >= 2 else 0,
        B_zp.stride(0) if B_zp is not None and B_zp.ndim >= 2 else 0,
        B_zp.stride(1) if B_zp is not None and B_zp.ndim == 3 else 0,
        B_zp.stride(2) if B_zp is not None and B_zp.ndim >= 2 else 0,
        0 if block_shape is None else block_shape[0],
        0 if block_shape is None else block_shape[1],
        MUL_ROUTED_WEIGHT=mul_routed_weight,
        top_k=top_k,
        compute_type=compute_type,
        **config,
    )


