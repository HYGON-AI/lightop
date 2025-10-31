import torch
import time
import pandas as pd
import functools
import json
import os
import math
import argparse
import ast
from typing import Any, Callable, Dict, List, Optional, Tuple, Type
import vllm.envs as envs
from vllm import _custom_ops as ops
from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
    moe_align_block_size)
import triton
import triton.language as tl
import lmslim.envs as lsenvs
from vllm.utils import W8a8GetCacheJSON
import torch.nn.functional as F
from lmslim.layers.gemm.int8_utils import (
    per_token_group_quant_int8, per_token_quant_int8)

use_lightop = lsenvs.LMSLIM_USE_LIGHTOP
device_name = lsenvs.LMSLIM_GPU_NAME

# device_name = torch.cuda.get_device_properties(torch.cuda.current_device()).name


if use_lightop:
    from lightop import moe_gemm_w8a8, get_moe_cuda_config
    try:
        from moe_w8a8_prefill_gemm import backend
        from grouped_gemm import moe_gemm_decode_prof
    except ImportError:
        print("Warning: Could not import moe_w8a8_prefill_gemm or grouped_gemm")
        backend = None
        moe_gemm_decode_prof = None

# 全局变量用于存储失败的配置
gemm1_compare_failed_list = []
gemm2_compare_failed_list = []

# BlockM配置
blockM_configs = {
    0: 16, 1: 32, 2: 64, 3: 16, 4: 32, 5: 64, 6: 16, 7: 32, 8: 64,
    9: 32, 10: 64, 11: 32, 12: 64, 13: 64, 14: 16, 15: 32, 16: 64,
    17: 16, 18: 32, 19: 64, 20: 16, 21: 32, 22: 64,
    1000: 128, 1001: 64
}

blockM_configs_decode = {i: 16 for i in range(44)}

M_THRESHOLD = 128


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
    """
    # Map program ids `pid` to the block of C it should compute.
    pid = tl.program_id(axis=0)
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

    # Create pointers for the first blocks of A and B.
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

    # Iterate to compute a block of the C matrix.
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


def to_int8(tensor: torch.Tensor):
    """将张量转换为int8格式"""
    return torch.round(tensor.clamp(min=-128, max=127)).to(dtype=torch.int8)


def generate_unique_int_tensor(m, topk, low=0, high=256):
    """生成[m, topk]的整数张量，每行元素不重复"""
    if topk > (high - low):
        raise ValueError(f"topk ({topk}) 不能超过范围 ({high - low})")
    
    tensor = torch.zeros((m, topk), dtype=torch.int)
    for i in range(m):
        row = torch.randperm(high - low)[:topk] + low
        tensor[i] = row
    return tensor


# Torch参考实现已移除，现在使用Triton作为参考实现进行精度验证


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
    """准备MOE kernel的输入"""
    if use_int8_w8a8:
        assert B_scale is not None
        if block_shape is None:
            # 激活channel-wise int8量化
            assert (per_channel_quant), "int8 quantization only supports block or channel-wise"
            A, A_scale = per_token_quant_int8(A)
        else:
            # 激活block-wise int8量化
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


def get_w8a8moe_json(E: int, m: int, n1: int, n2: int, k: int):
    """获取配置JSON"""
    moe_dict = W8a8GetCacheJSON().triton_moejson_dict if W8a8GetCacheJSON().triton_moejson_dict else {}
    if len(moe_dict) == 0:
        configs = None
    elif f"1_{n1}_{n2}_{k}" in moe_dict:
        if m <= 16:
            m_ = m
        elif m <= 32:
            m_ = 16
        elif m <= 64:
            m_ = 256
        elif m < 1024:
            m_ = 1024
        elif m < 8192:
            m_ = 8192
        else:
            m_ = 16384
        configs = moe_dict[f"{m_}_{n1}_{n2}_{k}"]
        config1 = configs['config1']
        config2 = configs['config2']
    else:
        configs = None
         
    if configs is None:
        if m < E:
            config1 = config2 = {
                "BLOCK_SIZE_M": 16,
                "BLOCK_SIZE_N": 32,
                "BLOCK_SIZE_K": 64,
                "GROUP_SIZE_M": 1,
            }
        else:
            config1 = config2 = {
                "BLOCK_SIZE_M": 64,
                "BLOCK_SIZE_N": 64,
                "BLOCK_SIZE_K": 32,
                "GROUP_SIZE_M": 8,
            }
    return config1, config2


def test_single_gemm_performance_accuracy(
    input_tensor: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: Optional[torch.Tensor],
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    top_k: int,
    cuda_config: Dict[str, Any],
    triton_config: Dict[str, Any],
    gemm_type: str,  # "gemm1" or "gemm2"
    is_first_stage: bool,
    per_channel_quant: bool = True,
    compute_type: tl.dtype = tl.bfloat16
) -> Tuple[float, bool]:
    """
    测试单个GEMM的性能和精度
    
    Returns:
        Tuple[float, bool]: (耗时(微秒), 精度验证是否通过)
    """
    # 准备输出张量
    if gemm_type == "gemm1":
        output_shape = (input_tensor.shape[0], top_k, weight.shape[1])
    else:  # gemm2
        output_shape = (input_tensor.shape[0] // 8, 8, weight.shape[1])
    
    output_cuda = torch.zeros(output_shape, device=input_tensor.device, dtype=input_tensor.dtype)
    output_triton = torch.zeros(output_shape, device=input_tensor.device, dtype=input_tensor.dtype)
    
    # 准备输入量化
    qinput, qa_scale = moe_kernel_prepare_input(
        A=input_tensor,
        B=weight,
        A_scale=None,
        B_scale=weight_scale,
        use_fp8_w8a8=False,
        use_int8_w8a8=True,
        use_int8_w8a16=False,
        use_int4_w4a16=False,
        per_channel_quant=per_channel_quant,
        block_shape=None
    )
    
    # CUDA实现性能测试
    def run_cuda_gemm():
        moe_gemm_w8a8(
            qinput,
            weight,
            output_cuda,
            qa_scale,
            weight_scale,
            topk_weights if not is_first_stage else None,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            top_k if is_first_stage else 1,
            cuda_config
        )
    
    # 预热
    for _ in range(5):
        run_cuda_gemm()
    torch.cuda.synchronize()
    

    # CUDA性能测试
    quantiles = [0.5, 0.2, 0.8]
    #ms, min_ms, max_ms = triton.testing.do_bench(run_cuda_gemm, quantiles=quantiles)
    ms, min_ms, max_ms = triton.testing.do_bench(lambda:  moe_gemm_w8a8(
                                                                qinput,
                                                                weight,
                                                                output_cuda,
                                                                qa_scale,
                                                                weight_scale,
                                                                topk_weights if not is_first_stage else None,
                                                                sorted_token_ids,
                                                                expert_ids,
                                                                num_tokens_post_padded,
                                                                top_k if is_first_stage else 1,
                                                                cuda_config), quantiles=quantiles)
    elapsed_time = ms * 1000  # 转换为微秒

    torch.cuda.synchronize()

    # return elapsed_time, True

    # import pdb
    # pdb.set_trace()
    # Triton参考实现
    invoke_fused_moe_kernel_int8(
        qinput,
        weight,
        output_triton,
        qa_scale,
        weight_scale,
        None,  # B_zp
        topk_weights,
        topk_ids,  # topk_ids (不需要)
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        not is_first_stage,  # mul_routed_weight
        top_k if is_first_stage else 1,
        cuda_config,
        compute_type,
        False,  # use_fp8_w8a8
        True,   # use_int8_w8a8
        False,  # use_int8_w8a16
        False,  # use_int4_w4a16
        per_channel_quant,
        None,   # block_shape
        False   # use_nn_moe
    )
    
    # 精度验证：对比CUDA输出和Triton输出
    accuracy_passed = torch.allclose(output_cuda, output_triton, rtol=1e-2, atol=1e-2)
    if not accuracy_passed:
        if is_first_stage:
            gemm1_compare_failed_list.append(cuda_config.get('MODE', -1))
        else:
            gemm2_compare_failed_list.append(cuda_config.get('MODE', -1))
    
    return elapsed_time, accuracy_passed


def _fused_moe_helper(
    m: int, n: int, k: int, n2: int, k2: int, topk: int, E: int,
    out_dtype: Type[torch.dtype] = torch.bfloat16,
    device: str = "cuda",
    per_channel_quant: bool = True
):
    """生成测试数据"""
    input_data = (torch.randn((m, k), device=device) * 5).to(dtype=out_dtype)
    weight1 = to_int8(torch.randn((E, n, k), device=device) * 5)
    weight2 = to_int8(torch.randn((E, n2, k2), device=device) * 5)

    if per_channel_quant:
        weight1_scale = torch.randn((E, n, 1), device=device, dtype=torch.float32)
        weight2_scale = torch.randn((E, n2, 1), device=device, dtype=torch.float32)
    else:
        # 这里可以添加block-wise量化的支持
        weight1_scale = torch.randn((E, n, 1), device=device, dtype=torch.float32)
        weight2_scale = torch.randn((E, n2, 1), device=device, dtype=torch.float32)
    
    # topk_weights [m, topk]
    tensor = torch.rand((m, topk), device=device)
    topk_weights = F.softmax(tensor, dim=1)
    
    # topk_ids [m, topk]
    topk_ids = generate_unique_int_tensor(m=m, topk=topk, low=0, high=E).cuda()
    
    return input_data, weight1, weight2, weight1_scale, weight2_scale, topk_weights, topk_ids


def test_moe_performance_accuracy(
    m: int, n1: int, k1: int, n2: int, k2: int, topk: int, E: int,
    out_dtype: Type[torch.dtype] = torch.bfloat16,
    per_channel_quant: bool = True
) -> Dict[str, Any]:
    """
    测试完整MOE的性能和精度
    
    Returns:
        Dict包含所有测试结果
    """
    # 生成测试数据
    input_data, weight1, weight2, weight1_scale, weight2_scale, topk_weights, topk_ids = _fused_moe_helper(
        m, n1, k1, n2, k2, topk, E, out_dtype, per_channel_quant=per_channel_quant
    )
    
    # 设置compute_type
    if out_dtype == torch.bfloat16:
        compute_type = tl.bfloat16
    elif out_dtype == torch.float16:
        compute_type = tl.float16
    elif out_dtype == torch.float32:
        compute_type = tl.float32
    else:
        raise ValueError(f"Unsupported compute_type: {out_dtype}")
    
    # 获取CUDA配置
    cuda_config1 = None
    cuda_config2 = None
    status = False
    
    if use_lightop and per_channel_quant:
        # 使用get_moe_cuda_config获取配置
        
        # import pdb
        # pdb.set_trace()
        num_cus = torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count
        try:
            cuda_config1, cuda_config2, status = get_moe_cuda_config(
                E, m, n1, k1, n2, k2, topk, device_name, num_cus, compute_type
            )
        except Exception as e:
            print(f"Warning: get_moe_cuda_config failed: {e}")
            status = False
    
    # 如果get_moe_cuda_config失败，使用默认配置
    if not status:
        cuda_config1, cuda_config2 = get_w8a8moe_json(E, m, n1, n2, k2)
        
        # 如果没有找到配置，根据数据大小设置默认值
        if cuda_config1 is None:
            if m > M_THRESHOLD:
                cuda_config1 = {'BLOCK_SIZE_M': blockM_configs.get(0, 16), 'MODE': 0, 'DELTA': 1}
                cuda_config2 = {'BLOCK_SIZE_M': blockM_configs.get(0, 16), 'MODE': 0, 'DELTA': 1}
            else:
                cuda_config1 = {'BLOCK_SIZE_M': blockM_configs_decode.get(0, 16), 'MODE': 0, 'DELTA': 1}
                cuda_config2 = {'BLOCK_SIZE_M': blockM_configs_decode.get(0, 16), 'MODE': 0, 'DELTA': 1}
    
    # 为Triton设置不同的配置（使用默认blockM配置）
    triton_config1 = {
        'BLOCK_SIZE_M': blockM_configs.get(0, 64) if m > M_THRESHOLD else blockM_configs_decode.get(0, 16),
        'MODE': 1000,  # 使用不同的MODE
        'DELTA': 1
    }
    triton_config2 = {
        'BLOCK_SIZE_M': blockM_configs.get(1, 64) if m > M_THRESHOLD else blockM_configs_decode.get(1, 16),
        'MODE': 1001,  # 使用不同的MODE
        'DELTA': 1
    }
    
    # 准备MOE对齐
    sorted_token_ids, expert_ids, num_tokens_post_padded = (
        moe_align_block_size(topk_ids, cuda_config1['BLOCK_SIZE_M'], E, None)
    )
    
    # 准备第一阶段的输入
    qinput1, qa1_scale = moe_kernel_prepare_input(
        A=input_data,
        B=weight1,
        A_scale=None,
        B_scale=weight1_scale,
        use_fp8_w8a8=False,
        use_int8_w8a8=True,
        use_int8_w8a16=False,
        use_int4_w4a16=False,
        per_channel_quant=per_channel_quant,
        block_shape=None
    )
    
   

    # 测试GEMM1
    gemm1_time, gemm1_accuracy = test_single_gemm_performance_accuracy(
        input_data, weight1, weight1_scale, topk_ids, topk_weights, sorted_token_ids, expert_ids,
        num_tokens_post_padded, topk, cuda_config1, triton_config1, "gemm1", True, per_channel_quant, compute_type
    )
    
    # 执行CUDA GEMM1获取中间结果
    intermediate_cache1 = torch.zeros((m, topk, n1), device=input_data.device, dtype=input_data.dtype)
    moe_gemm_w8a8(qinput1, weight1, intermediate_cache1, qa1_scale, weight1_scale,
                  None, sorted_token_ids, expert_ids, num_tokens_post_padded, topk, cuda_config1)
    
    # 执行激活函数
    intermediate_cache2 = torch.empty((m * topk, n1 // 2), device=input_data.device, dtype=input_data.dtype)
    torch.ops._C.silu_and_mul(intermediate_cache2, intermediate_cache1.view(-1, n1))
    
    # 准备第二阶段的输入
    qinput2, qa2_scale = moe_kernel_prepare_input(
        A=intermediate_cache2,
        B=weight2,
        A_scale=None,
        B_scale=weight2_scale,
        use_fp8_w8a8=False,
        use_int8_w8a8=True,
        use_int8_w8a16=False,
        use_int4_w4a16=False,
        per_channel_quant=per_channel_quant,
        block_shape=None
    )
    

    # 测试GEMM2
    gemm2_time, gemm2_accuracy = test_single_gemm_performance_accuracy(
        intermediate_cache2, weight2, weight2_scale, topk_ids, topk_weights, sorted_token_ids, expert_ids,
        num_tokens_post_padded, 1, cuda_config2, triton_config2, "gemm2", False, per_channel_quant, compute_type
    )
    
    # 整体精度验证已在单个GEMM测试中完成
    # 这里我们基于单个GEMM的结果来判断整体精度
    overall_accuracy = gemm1_accuracy and gemm2_accuracy
    
    #GEMM1+GEMM2的总耗时
    total_time = gemm1_time + gemm2_time # 转换为微秒
    
    return {
        'M': m,
        'N1': n1,
        'K1': k1,
        'N2': n2,
        'K2': k2,
        'E': E,
        'TopK': topk,
        'GEMM1_Time_us': gemm1_time,
        'GEMM2_Time_us': gemm2_time,
        'Total_Time_us': total_time,
        'GEMM1_Accuracy_Pass': gemm1_accuracy,
        'GEMM2_Accuracy_Pass': gemm2_accuracy,
        'Overall_Accuracy_Pass': overall_accuracy
    }


def main():
    parser = argparse.ArgumentParser(description='MOE GEMM性能和精度测试脚本')
    
    # 添加参数
    parser.add_argument('--E', type=str, default='[256, 256, 128, 128, 128]', 
                        help='专家数量,如 "[256]"')
    parser.add_argument('--n1-list', type=str, required=False, default='[256, 512, 1536,768,384]',
                       help='N1参数列表，如 "[256]"')
    parser.add_argument('--n2-list', type=str, required=False, default='[7168, 7168, 2048,2048,4096]',
                       help='N2参数列表，如 "[7168]"')
    parser.add_argument('--k-list', type=str, required=False, default='[128, 256, 768,384,192]',
                       help='K2参数列表，如 "[128]"')
    parser.add_argument('--m-list', type=str, default='[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,32,64,128,256,512,1024,2048,4096,8192,16384]',
                       help='M参数列表，如 "[32,128,512,1024]"')
    parser.add_argument('--topk', type=int, default=8, 
                       help='topk参数')
    parser.add_argument('--out-dtype', type=str, default='bfloat16',
                       choices=['float16', 'bfloat16', 'float32'],
                       help='输出数据类型')
    parser.add_argument('--use-channel-wise', action='store_true', default=True,
                       help='使用channel-wise量化')
    parser.add_argument('--output-file', type=str, default='moe_w8a8_gemm_channel-wise.xlsx',
                       help='输出Excel文件名')
    
    # 解析参数
    args = parser.parse_args()
    
    def parse_list(list_str):
        try:
            return ast.literal_eval(list_str)
        except (ValueError, SyntaxError) as e:
            raise ValueError(f"无法解析列表参数: {list_str}。错误: {str(e)}")
    
    # 解析参数
    E_list = parse_list(args.E)
    n1_list = parse_list(args.n1_list)
    n2_list = parse_list(args.n2_list)  
    k_list = parse_list(args.k_list)
    m_list = parse_list(args.m_list)
    topk = args.topk
    per_channel_quant = args.use_channel_wise
    
    # 设置数据类型
    dtype_map = {
        'float16': torch.float16,
        'bfloat16': torch.bfloat16,
        'float32': torch.float32
    }
    out_dtype = dtype_map[args.out_dtype]
    
    # 初始化配置
    tritonsingleton = W8a8GetCacheJSON()
    triton_json_dir = tritonsingleton.triton_json_dir
    os.makedirs(triton_json_dir, exist_ok=True)
    
    print(f"GPU设备: {device_name}")
    print(f"量化方式: {'channel-wise' if per_channel_quant else 'block-wise'}")
    print(f"输出数据类型: {args.out_dtype}")
    
    # 存储所有测试结果
    all_results = []
    
    # 测试所有配置组合
    for i in range(len(n1_list)):
        E = E_list[i]
        n1, n2, k2 = n1_list[i], n2_list[i], k_list[i]
        k1 = n2  # 根据MOE结构，第一个GEMM的K维度等于第二个GEMM的N维度
        
        print(f"\n测试配置: N1={n1}, N2={n2}, K1={k1}, K2={k2}")
        
        for m in m_list:
            print(f"  测试 M={m}")
            try:
                # 调用测试
                result = test_moe_performance_accuracy(
                    m, n1, k1, n2, k2, topk, E, out_dtype, per_channel_quant
                )
                all_results.append(result)
                
                print(f"    GEMM1耗时: {result['GEMM1_Time_us']:.2f}us, 精度: {'通过' if result['GEMM1_Accuracy_Pass'] else '失败'}")
                print(f"    GEMM2耗时: {result['GEMM2_Time_us']:.2f}us, 精度: {'通过' if result['GEMM2_Accuracy_Pass'] else '失败'}")
                print(f"    总耗时: {result['Total_Time_us']:.2f}us, 整体精度: {'通过' if result['Overall_Accuracy_Pass'] else '失败'}")
                
            except Exception as e:
                print(f"    测试失败: {str(e)}")
                # 添加失败的结果
                all_results.append({
                    'M': m, 'N1': n1, 'K1': k1, 'N2': n2, 'K2': k2, 'E': E, 'TopK': topk,
                    'GEMM1_Time_us': -1, 'GEMM2_Time_us': -1, 'Total_Time_us': -1,
                    'GEMM1_Accuracy_Pass': False, 'GEMM2_Accuracy_Pass': False,
                    'Overall_Accuracy_Pass': False
                })
    
    # 创建DataFrame并保存结果
    df = pd.DataFrame(all_results)
    df.to_excel(args.output_file, index=False)
    
    print(f"\n测试完成！结果已保存到 {args.output_file}")
    print(f"GEMM1精度失败的配置: {gemm1_compare_failed_list}")
    print(f"GEMM2精度失败的配置: {gemm2_compare_failed_list}")
    
    # 打印统计信息
    total_tests = len(all_results)
    gemm1_pass = sum(1 for r in all_results if r['GEMM1_Accuracy_Pass'])
    gemm2_pass = sum(1 for r in all_results if r['GEMM2_Accuracy_Pass'])
    overall_pass = sum(1 for r in all_results if r['Overall_Accuracy_Pass'])
    
    print(f"\n统计信息:")
    print(f"总测试数: {total_tests}")
    print(f"GEMM1精度通过率: {gemm1_pass}/{total_tests} ({100*gemm1_pass/total_tests:.1f}%)")
    print(f"GEMM2精度通过率: {gemm2_pass}/{total_tests} ({100*gemm2_pass/total_tests:.1f}%)")
    print(f"整体精度通过率: {overall_pass}/{total_tests} ({100*overall_pass/total_tests:.1f}%)")


if __name__ == "__main__":
    main()
    # 运行所有shape: HIP_VISIBLE_DEVICES=3 python benchmark_moe_w8a8.py 
    # 单侧某一个
    # HIP_VISIBLE_DEVICES=3 python benchmark_moe_w8a8.py \
    #     --E [256] \
    #     --n1-list '[256]' \
    #     --n2-list '[7168]' \
    #     --k-list '[128]' \
    #     --m-list '[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,32,64,128,256,512,1024,2048,4096,8192,16384]' \
    #     --topk 8 \
    #     --out-dtype bfloat16 \
    #     --use-channel-wise \
    #     --output-file deepseek_671b_tp16_results.xlsx