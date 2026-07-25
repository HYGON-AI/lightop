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
from lightop.quant import per_token_quant_int8
from lightop import get_batched_gemm_w8a8_config
from lightop.attention import fuse_rmsnorm_rope_quant_qkv, fused_rms_norm_rope_contiguous
from lightop.gemm_ops import fused_bmm

from test_fuse_rmsnorm_rope_quant_qkv import _compute_cos_sin_cache, _create_mla_cache

fused_bmm_compare_failed_list = []

def to_int8(tensor: torch.Tensor):
    return torch.round(tensor.clamp(min=-128, max=127)).to(dtype=torch.int8)

def batch_rocblas_scaled_mm(a: torch.Tensor,        # [n, b, p]
                            b: torch.Tensor,        # [n, l, p]
                            scale_a: torch.Tensor,  # [n, b, 1] 
                            scale_b: torch.Tensor,  # [n, l, 1]
                            out_dtype: torch.dtype = torch.float8_e4m3fn,
                            bias: Optional[torch.Tensor] = None) -> torch.Tensor: # [n, l]
    #scale_a.shape=[n,b,1]   fp32
    #scale_b.shape=[n,l,1]   fp32
    #a.shape=[n,b,p]         int8
    #b.shape=[n,l,p]         int8
    #output.shape=[n,b,l]    bf16/fp16
    #bias [n,l]              bf16/fp16

    batch_size = a.shape[0]
    output = torch.empty(batch_size, a.shape[1], b.shape[1], dtype=out_dtype)
    in_dtype = a.dtype
    dst_dtype = torch.float32
    if in_dtype == torch.int8:
        dst_dtype = torch.int32
    elif in_dtype == torch.float16:
        dst_dtype = torch.float16
    elif in_dtype == torch.float32:
        dst_dtype = torch.float32
    elif in_dtype == torch.float8_e4m3fn or in_dtype == torch.float8_e5m2:
        dst_dtype = torch.float32
    else:
        raise ValueError(f"Unsupported input dtype: {in_dtype}")
    
    for i in range(batch_size):
        scales = scale_a[i] * scale_b[i].T   # [1, b, l]
        # 使用 torch 实现 int8 矩阵乘法，输出 int32
        # a[i]: [b, p] int8, b[i]: [l, p] int8
        # 计算 a[i] @ b[i].T，结果: [b, l] int32
        gemmoutput = torch.matmul(a[i].cpu().to(dtype=dst_dtype), b[i].T.cpu().to(dtype=dst_dtype))  # [b, l]
        output[i] = (scales.cpu() * gemmoutput.to(dtype=torch.float32)).to(out_dtype)
        # import pdb; pdb.set_trace()
        # do not support bias!
        # if bias is not None:
        #     output[i] = output[i] + bias[i].cpu()
    return output.cuda()

def calc_diff(x: torch.Tensor, y: torch.Tensor):
    x, y = x.double(), y.double()
    denominator = (x * x + y * y).sum()
    if denominator == 0:    # Which means that all elements in x and y are 0
        return 0.0
    sim = 2 * (x * y).sum() / denominator
    return 1 - sim

def fused_bmm_trans_impl_int8(  torch_input: torch.Tensor,                      # b，n, p
                                input: torch.Tensor,                            # b，n, p
                                w_uk: torch.Tensor,                             # n, l, p
                                input_scale: Optional[torch.Tensor] = None,     # b, n, 1
                                w_uk_scale: Optional[torch.Tensor] = None,      # n, l, 1
                                bias: Optional[torch.Tensor] = None,            # n, l
                                dtype: Optional[torch.dtype] = torch.float8_e4m3fn,
                                transpose_bm: Optional[bool] = False,
                                transpose_bm_in: Optional[bool] = False,
                                mode: Optional[List[int]]=None,
                                is_tuning: bool = False):
    
    # input = torch.ones_like(input)
    # torch_input = torch.ones_like(torch_input)
    # w_uk = torch.ones_like(w_uk)
    # input_scale = torch.ones_like(input_scale)
    # w_uk_scale = torch.ones_like(w_uk_scale)

    input_clone = torch_input.clone()
    torch_uk = w_uk.clone()
    torch_input_scale = input_scale.clone()
    torch_w_uk_scale = w_uk_scale.clone()
    torch_bias = bias.clone() if bias is not None else None
    
    config = {"MODE": mode}
    
    # fused_bmm_output 应该是 [n, b, l] 的形状
    if not transpose_bm_in:
        torch_input_contiguous = input_clone.transpose(0,1).contiguous()  # [b, n, p] -> [n, b, p]
        torch_input_scale = torch_input_scale.transpose(0,1).contiguous()  # [b, n, 1] -> [n, b, 1]
    else:
        torch_input_contiguous = input_clone.contiguous()  # [b, n, p] -> [b, n, p]
    # 使用bfloat16作为基准输出类型，避免FP8类型比较时的运算问题
    fused_torch_output = batch_rocblas_scaled_mm(torch_input_contiguous, torch_uk, torch_input_scale, torch_w_uk_scale, out_dtype=dtype, bias=torch_bias)   # [n, b, l]
    if not transpose_bm:
        torch_output = fused_torch_output.transpose(0,1).contiguous() # [b, n, l]
    else:
        torch_output = fused_torch_output.contiguous() # [n, b, l]
    # 验证lightop kernel
    # torch.set_printoptions(profile="full") 
    cuda_output = torch.zeros_like(torch_output, dtype=dtype)
    fused_bmm(input, w_uk, input_scale, w_uk_scale, bias, dtype, cuda_output, transpose_bm, transpose_bm_in, config)
    torch.cuda.synchronize()
    # import pdb; pdb.set_trace()
    mode_status = True

    if dtype == torch.float8_e4m3fn:   
        torch_output_uint8 = torch_output.view(torch.uint8)
        cuda_output_uint8 = cuda_output.view(torch.uint8)

        torch_output_fp32 = torch_output_uint8.to(torch.float32)
        cuda_output_fp32 = cuda_output_uint8.to(torch.float32)
        
        rtol = 1e-2
        atol = 1e-2
        
        if is_tuning:
            rtol = 1e-1
            atol = 1e-1
        else:
            rtol = 1e-2
            atol = 1e-2

        if not (torch.allclose(cuda_output_fp32, torch_output_fp32, rtol=rtol, atol=atol, equal_nan=True)):
            mode_status = False
            if not is_tuning:
                print(f"kernel 精度检查不合格!!!")
                torch.set_printoptions(profile="full")  
                ftriton = open('cuda_output.txt', 'w')   
                print(f'cuda_output: {cuda_output}', file=ftriton)
                ftriton.close
                ftorch = open('bench_output.txt', 'w')   
                print(f'bench_output: {torch_output}', file=ftorch)
                ftorch.close
        else:
            if not is_tuning:
                print(f"kernel 精度检查合格.")
            mode_status = True
    elif dtype == torch.bfloat16:
        if not (torch.allclose(cuda_output, torch_output, rtol=1e-2, atol=1e-2, equal_nan=True)):
            mode_status = False
            if not is_tuning:
                print(f"kernel 精度检查不合格!!!")
                torch.set_printoptions(profile="full")  
                ftriton = open('cuda_output.txt', 'w')   
                print(f'cuda_output: {cuda_output}', file=ftriton)
                ftriton.close
                ftorch = open('bench_output.txt', 'w')   
                print(f'bench_output: {torch_output}', file=ftorch)
                ftorch.close
        else:
            if not is_tuning:
                print(f"kernel 精度检查合格.")
            mode_status = True

    return mode_status

def fused_bmm_trans_impl_int8_tuning(   torch_input: torch.Tensor,                      # b，n, p
                                        input: torch.Tensor,                            # b，n, p
                                        w_uk: torch.Tensor,                             # n, l, p
                                        input_scale: Optional[torch.Tensor] = None,     # b, n, 1
                                        w_uk_scale: Optional[torch.Tensor] = None,      # n, l, 1
                                        bias: Optional[torch.Tensor] = None,            # n, l
                                        mode: Optional[List[int]]=None,
                                        dtype: Optional[torch.dtype] = torch.float8_e4m3fn,
                                        transpose_bm: Optional[bool] = False,
                                        transpose_bm_in: Optional[bool] = False):
    
    # input = torch.ones_like(input)
    # torch_input = torch.ones_like(torch_input)
    # w_uk = torch.ones_like(w_uk)
    # input_scale = torch.ones_like(input_scale)
    # w_uk_scale = torch.ones_like(w_uk_scale)

    input_clone = torch_input.clone()
    torch_uk = w_uk.clone()
    torch_input_scale = input_scale.clone()
    torch_w_uk_scale = w_uk_scale.clone()
    torch_bias = bias.clone() if bias is not None else None
    
    config = {"MODE": mode}
    # fused_bmm_output 应该是 [n, b, l] 的形状
    if not transpose_bm_in:
        torch_input = input_clone.transpose(0,1).contiguous()  # [b, n, p] -> [n, b, p]
        torch_input_scale = torch_input_scale.transpose(0,1).contiguous()  # [b, n, 1] -> [n, b, 1]
    else:
        torch_input = input_clone.contiguous()  # [b, n, p] -> [b, n, p]
    # 使用bfloat16作为基准输出类型，避免FP8类型比较时的运算问题
    fused_torch_output = batch_rocblas_scaled_mm(torch_input, torch_uk, torch_input_scale, torch_w_uk_scale, out_dtype=dtype, bias=torch_bias)   # [n, b, l]

    if not transpose_bm:
        torch_output = fused_torch_output.transpose(0,1).contiguous() # [b, n, l]
    else:
        torch_output = fused_torch_output.contiguous() # [n, b, l]
    # 验证lightop kernel
    # torch.set_printoptions(profile="full") 
    cuda_output = torch.zeros_like(torch_output, dtype=dtype)
    fused_bmm(input, w_uk, input_scale, w_uk_scale, bias, dtype, cuda_output, transpose_bm, transpose_bm_in, config)
    torch.cuda.synchronize()
    # import pdb; pdb.set_trace()
    mode_status = True
    
    # diff = calc_diff(torch_output, cuda_output)

    # max_diff = 0.001
    # if diff > max_diff:
    #     # print(f"kernel 精度检查不合格!!!")
    #     mode_status = False
    # else:
    #     # print(f"kernel 精度检查合格.")
    #     mode_status = True    
    
    if dtype == torch.float8_e4m3fn:   

        torch_output_uint8 = torch_output.view(torch.uint8)
        cuda_output_uint8 = cuda_output.view(torch.uint8)

        torch_output_fp32 = torch_output_uint8.to(torch.float32)
        cuda_output_fp32 = cuda_output_uint8.to(torch.float32)
        
        if not (torch.allclose(torch_output_fp32, cuda_output_fp32, rtol=1e-2, atol=1e-2, equal_nan=True)):
            # print(f"kernel 精度检查不合格!!!")
            mode_status = False
            
            # torch.set_printoptions(profile="full")  
            # ftriton = open('cuda_output.txt', 'w')   
            # print(f'cuda_output: {cuda_output}', file=ftriton)
            # ftriton.close
            # ftorch = open('bench_output.txt', 'w')   
            # print(f'bench_output: {torch_output}', file=ftorch)
            # ftorch.close
        else:
            # print(f"kernel 精度检查合格.")
            mode_status = True
    elif dtype == torch.bfloat16:
        if not (torch.allclose(torch_output, cuda_output, rtol=1e-2, atol=1e-2, equal_nan=True)):
            mode_status = False
            # print(f"kernel 精度检查不合格!!!")
            # torch.set_printoptions(profile="full")  
            # ftriton = open('cuda_output.txt', 'w')   
            # print(f'cuda_output: {cuda_output}', file=ftriton)
            # ftriton.close
            # ftorch = open('bench_output.txt', 'w')   
            # print(f'bench_output: {torch_output}', file=ftorch)
            # ftorch.close
        else:
            print(f"kernel 精度检查合格.")
            mode_status = True


    warm_up_num = 20
    for i in range(warm_up_num):
        fused_bmm(input, w_uk, input_scale, w_uk_scale, bias, dtype, cuda_output, transpose_bm, transpose_bm_in, config)
        torch.cuda.synchronize()
    
    quantiles = [0.5, 0.2, 0.8]
    avg_ms, min_ms, max_ms = triton.testing.do_bench(lambda: fused_bmm(input, 
                                                                       w_uk, 
                                                                       input_scale, 
                                                                       w_uk_scale, 
                                                                       bias, 
                                                                       dtype, 
                                                                       cuda_output, 
                                                                       transpose_bm, 
                                                                       transpose_bm_in, 
                                                                       config), quantiles=quantiles)
    fused_bmm_kernel_time_us = avg_ms * 1000
        
        
    return mode_status, fused_bmm_kernel_time_us

def torch_benchmark_transpose_bmm_transpose_bf16(decode_q_nope: torch.Tensor,
                                                 w_uk: torch.Tensor) -> torch.Tensor:
    # decode_q_nope.shape = [b, n, p] bf16
    # w_uk.shape = [n, p, l] bf16
    # return: [b, n, l] bf16
    decode_q_nope = decode_q_nope.transpose(0,1)  # [b, n, p] -> [n, b, p]
    decode_ql_nope = torch.bmm(decode_q_nope.float(), w_uk.float())  # [n, b, l]
    decode_ql_nope = decode_ql_nope.transpose(0,1)  # [b, n, l]
    return decode_ql_nope

def torch_benchmark_fuse_bmm_bf16(torch_q_bf16: torch.Tensor,
                                  torch_w_uk_bf16: torch.Tensor) -> Tuple[torch.Tensor, float]:
    torch.manual_seed(0)
    torch_w_uk_bf16_t = torch_w_uk_bf16.transpose(1,2).contiguous()  # [n, l, p] -> [n, p, l]
 
    ref_output = torch_benchmark_transpose_bmm_transpose_bf16(torch_q_bf16, torch_w_uk_bf16_t)
    
    warm_up_num = 10
    # function:
    for i in range(warm_up_num):
        torch_benchmark_transpose_bmm_transpose_bf16(torch_q_bf16, torch_w_uk_bf16_t)
        torch.cuda.synchronize()
    
    quantiles = [0.5, 0.2, 0.8]
    avg_ms, min_ms, max_ms = triton.testing.do_bench(lambda: torch_benchmark_transpose_bmm_transpose_bf16(torch_q_bf16, torch_w_uk_bf16_t), 
                                                            quantiles=quantiles)
    
    return ref_output, avg_ms * 1000

def rms_rope_q_bf16_single_test(kv_cache_dtype_str,batch_size,num_head,q_c_dim,kv_c_dim,rot_dim,num_kv_cache_blocks,kv_cache_blocks_size,k_bs_stride,test_speed = False):
    DEVICE = "cuda"
    variance_epsilon = 1e-6
    DTYPE = torch.bfloat16    
    is_neox_style = False
    max_position = 163840
    base = 1e4
    if kv_cache_dtype_str == "fp8_e4m3":
        KV_CACHE_DTYPE = torch.float8_e4m3fn
    elif kv_cache_dtype_str == "fp8_e5m2":
        KV_CACHE_DTYPE = torch.float8_e5m2
    elif kv_cache_dtype_str == "fp16":
        KV_CACHE_DTYPE = torch.float16
    elif kv_cache_dtype_str == "bf16":
        KV_CACHE_DTYPE = torch.bfloat16
    else:
        raise ValueError(f"Unsupported kv_cache_dtype_str: {kv_cache_dtype_str}")
    kv_cache_scale = 1.0
    # rope(Q)的相关tensor创建
    q_tensor = torch.randn(batch_size, num_head,q_c_dim + rot_dim, dtype=DTYPE, device=DEVICE)
    cos_sin_cache = _compute_cos_sin_cache(max_position,base,rot_dim, device=DEVICE)
    if cos_sin_cache.dtype != DTYPE:
        cos_sin_cache = cos_sin_cache.to(DTYPE)
    positions = torch.randint(0,max_position, (batch_size,), device=DEVICE)
    q_pe = q_tensor[...,q_c_dim:]
    q_pe_no_fuse = q_pe.clone()
    q_pe_ori = q_pe.clone()
    # rope(K)的相关tensor创建
    k_tensor = torch.randn(batch_size, kv_c_dim + rot_dim, dtype=DTYPE, device=DEVICE)
    k_pe = k_tensor[...,kv_c_dim:]
    k_pe_no_fuse = k_pe.clone()
    kv_c = k_tensor[...,:kv_c_dim]

    k_tensor_big= torch.randn(batch_size, k_bs_stride, dtype=DTYPE, device=DEVICE)
    kv_c_big = k_tensor_big[...,:kv_c_dim]
    kv_c_big[:]=kv_c
    kv_c = kv_c_big

    k_pe_big = k_tensor_big[...,kv_c_dim:kv_c_dim+rot_dim]
    k_pe_big[:]=k_pe
    k_pe = k_pe_big


    kv_c_normed = torch.empty(kv_c.shape, dtype=DTYPE, device=DEVICE).contiguous()

    weight = torch.randn(kv_c_dim, dtype=DTYPE, device=DEVICE)
    kv_c_no_fuse = kv_c.clone()
    kv_c_no_fuse_out = torch.empty_like(kv_c_no_fuse).contiguous()
    # kv_cache相关
    kv_cache = _create_mla_cache(num_kv_cache_blocks, kv_cache_blocks_size, kv_c_dim+rot_dim, DTYPE,
                                  kv_cache_dtype_str, DEVICE)
    kv_cache_no_fuse = kv_cache.clone()
    kv_scale_tensor = torch.tensor(1., dtype=torch.float32, device=DEVICE)
    slot_mapping = torch.randperm(num_kv_cache_blocks * kv_cache_blocks_size, device=DEVICE)[:batch_size]

    if KV_CACHE_DTYPE == torch.float16:
        kv_cache = kv_cache.half()
        kv_cache_no_fuse = kv_cache_no_fuse.half()
    elif KV_CACHE_DTYPE == torch.bfloat16:
        kv_cache = kv_cache.bfloat16()
        kv_cache_no_fuse = kv_cache_no_fuse.bfloat16()

    def run_fuse_q_bf16_opt():
        fused_rms_norm_rope_contiguous(positions,q_pe,k_pe,kv_c,kv_c_normed,weight,cos_sin_cache,slot_mapping,kv_cache,kv_cache_dtype_str,kv_cache_scale,is_neox_style,variance_epsilon)
        
    warm_up_num = 10
    # function:
    for i in range(warm_up_num):
        run_fuse_q_bf16_opt()
        torch.cuda.synchronize()
    
    quantiles = [0.5, 0.2, 0.8]
    avg_ms, min_ms, max_ms = triton.testing.do_bench(lambda: run_fuse_q_bf16_opt(), 
                                                            quantiles=quantiles)
    return avg_ms * 1000

def rms_rope_q_fp8_single_test(kv_cache_dtype_str,batch_size,num_head,q_c_dim,kv_c_dim,rot_dim,num_kv_cache_blocks,kv_cache_blocks_size,test_speed = False):
    DEVICE = "cuda"
    variance_epsilon = 1e-6
    DTYPE = torch.bfloat16    
    is_neox_style = False
    max_position = 163840
    base = 1e4
    dtype_mapping = {
        "fp8_e4m3": torch.float8_e4m3fn,
        "fp8_e5m2": torch.float8_e5m2,
        "fp16": torch.float16,
        "bf16": torch.bfloat16
    }
    KV_CACHE_DTYPE = dtype_mapping.get(kv_cache_dtype_str)
    if KV_CACHE_DTYPE is None:
        raise ValueError(f"Unsupported kv_cache_dtype_str: {kv_cache_dtype_str}")
    kv_cache_scale = 1.0
    # rope(Q)的相关tensor创建
    q_tensor = torch.randn(batch_size, num_head,q_c_dim + rot_dim, dtype=DTYPE, device=DEVICE)
    q_quant_gt = torch.empty_like(q_tensor[:,:,:q_c_dim], dtype=KV_CACHE_DTYPE, device=DEVICE)
    q_quant = torch.empty_like(q_tensor, dtype=KV_CACHE_DTYPE, device=DEVICE)
    cos_sin_cache = _compute_cos_sin_cache(max_position,base,rot_dim, device=DEVICE)
    if cos_sin_cache.dtype != DTYPE:
        cos_sin_cache = cos_sin_cache.to(DTYPE)
    positions = torch.randperm(max_position, device=DEVICE)[:batch_size]
    q_c = q_tensor[...,:q_c_dim]
    q_pe = q_tensor[...,q_c_dim:]
    q_pe_no_fuse = q_pe.clone()
    q_pe_ori = q_pe.clone()
    q_scale = torch.empty((batch_size, num_head, 1), dtype=torch.float32, device=DEVICE)
    q_scale_gt = torch.empty((batch_size, num_head, 1), dtype=torch.float32, device=DEVICE)
    # rope(K)的相关tensor创建
    k_tensor = torch.randn(batch_size, kv_c_dim + rot_dim, dtype=DTYPE, device=DEVICE)
    k_pe = k_tensor[...,kv_c_dim:]
    k_pe_no_fuse = k_pe.clone()
    kv_c = k_tensor[...,:kv_c_dim]
    kv_c_normed = torch.empty(kv_c.shape, dtype=DTYPE, device=DEVICE)
    weight = torch.randn(kv_c_dim, dtype=DTYPE, device=DEVICE)
    kv_c_no_fuse = kv_c.clone()
    kv_c_no_fuse_out = torch.empty_like(kv_c_no_fuse).contiguous()
    # kv_cache相关
    kv_cache = _create_mla_cache(num_kv_cache_blocks, kv_cache_blocks_size, kv_c_dim+rot_dim, DTYPE,
                                  kv_cache_dtype_str, DEVICE)
    kv_cache_no_fuse = kv_cache.clone()
    kv_scale_tensor = torch.tensor(1., dtype=torch.float32, device=DEVICE)
    total_slots = num_kv_cache_blocks * kv_cache_blocks_size
    slot_mapping = torch.randperm(total_slots, device=DEVICE)[:batch_size]

    if KV_CACHE_DTYPE == torch.float16:
        kv_cache = kv_cache.half()
        kv_cache_no_fuse = kv_cache_no_fuse.half()
    elif KV_CACHE_DTYPE == torch.bfloat16:
        kv_cache = kv_cache.bfloat16()
        kv_cache_no_fuse = kv_cache_no_fuse.bfloat16()

    def run_fuse_q_fp8_opt():
        fuse_rmsnorm_rope_quant_qkv(positions,q_c,q_pe,q_quant,q_scale,k_pe,kv_c,kv_c_normed,weight,cos_sin_cache,slot_mapping,kv_cache,kv_cache_dtype_str,kv_cache_scale,is_neox_style,variance_epsilon)

    warm_up_num = 10
    # function:
    for i in range(warm_up_num):
        run_fuse_q_fp8_opt()
        torch.cuda.synchronize()
    
    quantiles = [0.5, 0.2, 0.8]
    avg_ms, min_ms, max_ms = triton.testing.do_bench(lambda: run_fuse_q_fp8_opt(), 
                                                            quantiles=quantiles)
    return avg_ms * 1000

def fuse_rmsnorm_rope_quant_q_benchmark(kv_cache_dtype_str, 
                                        batch_size,
                                        num_head,
                                        q_c_dim):
    
    q_bf16_time = rms_rope_q_bf16_single_test(kv_cache_dtype_str = "fp8_e4m3", batch_size=batch_size,num_head=num_head,q_c_dim=q_c_dim, kv_c_dim=512, rot_dim=64, num_kv_cache_blocks=1000, 
                           kv_cache_blocks_size=16, k_bs_stride=2112, test_speed = True)
    q_fp8_time = rms_rope_q_fp8_single_test(kv_cache_dtype_str = "fp8_e4m3", batch_size=batch_size,num_head=num_head,q_c_dim=q_c_dim, kv_c_dim=512, rot_dim=64, num_kv_cache_blocks=1000,    
                           kv_cache_blocks_size=16, test_speed = True)
    
    diff_time = q_fp8_time - q_bf16_time
    if diff_time > 0.001:
        print(f"q_bf16_time: {q_bf16_time}, q_fp8_time: {q_fp8_time}, diff_time: {diff_time}")
    else:
        print(f"q_bf16_time: {q_bf16_time}, q_fp8_time: {q_fp8_time}, diff_time: {diff_time}")
    
    if diff_time < 0:
        diff_time = 0.0

    return diff_time 

def fused_bmm_trans_impl_int8_benchmark(torch_q_bf16: torch.Tensor,
                                        torch_w_uk_bf16: torch.Tensor,
                                        torch_input: torch.Tensor,                      # b，n, p
                                        input: torch.Tensor,                            # b，n, p
                                        w_uk: torch.Tensor,                             # n, l, p
                                        input_scale: Optional[torch.Tensor] = None,     # n, b, 1
                                        w_uk_scale: Optional[torch.Tensor] = None,      # n, l, 1
                                        bias: Optional[torch.Tensor] = None,            # n, l
                                        dtype: Optional[torch.dtype] = torch.float8_e4m3fn,
                                        transpose_bm: Optional[bool] = False,
                                        transpose_bm_in: Optional[bool] = False):           

    # 用实际 shape 申请输出空间，避免依赖 zeros_like
    b, n, p = torch_input.shape
    if transpose_bm_in:
        n, b, p = torch_input.shape

    _, l, _ = w_uk.shape
    
    #input量化前后，kernel耗时比较
    # q_quant_time = fuse_rmsnorm_rope_quant_q_benchmark("fp8_e4m3", b, n, p)
    q_quant_time = 0.0
    
    # torch benchmark:
    ref_output, torch_benchmark_time = torch_benchmark_fuse_bmm_bf16(torch_q_bf16, torch_w_uk_bf16)
    
    input_clone = torch_input.clone()
    torch_uk = w_uk.clone()
    torch_input_scale = input_scale.clone()
    torch_w_uk_scale = w_uk_scale.clone()
    torch_bias = bias.clone() if bias is not None else None
    
    # fused_bmm_output 应该是 [n, b, l] 的形状
    if not transpose_bm_in:
        torch_input = input_clone.transpose(0,1).contiguous()  # [b, n, p] -> [n, b, p]
        torch_input_scale = torch_input_scale.transpose(0,1).contiguous()  # [b, n, 1] -> [n, b, 1]
    else:
        torch_input = input_clone.contiguous()  # [n, b, p]

    # 使用bfloat16作为基准输出类型，避免FP8类型比较时的运算问题
    fused_torch_output = batch_rocblas_scaled_mm(torch_input, torch_uk, torch_input_scale, torch_w_uk_scale, out_dtype=dtype, bias=torch_bias)   # [n, b, l]
    if not transpose_bm:
        torch_output = fused_torch_output.transpose(0,1).contiguous() # [b, n, l]
    else:
        torch_output = fused_torch_output.contiguous() # [n, b, l]
    
    # 验证lightop kernel
    config, status = get_batched_gemm_w8a8_config(b, n, p)
    gfx_version = torch.cuda.get_device_properties(0).gcnArchName.split(':')[0]
    cu_number = torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count
    print(f"config: {config}, status: {status}, gfx_version: {gfx_version}, cu_number: {cu_number}")
    # torch.set_printoptions(profile="full") 
    cuda_output = torch.zeros_like(torch_output, dtype=dtype)
    fused_bmm(input, w_uk, input_scale, w_uk_scale, bias, dtype, cuda_output, transpose_bm, transpose_bm_in, config)
    torch.cuda.synchronize()
    mode_status = True
    
    diff = calc_diff(ref_output, cuda_output)
    max_diff = 0.001

    if dtype == torch.float8_e4m3fn:   
        max_diff = 0.002
    elif dtype == torch.bfloat16:
        max_diff = 0.001

    if diff < max_diff:
        print(f"kernel 精度检查合格.")
        mode_status = True 
        # torch.set_printoptions(profile="full")  
        # ftriton = open('cuda_output.txt', 'w')   
        # print(f'cuda_output: {cuda_output}', file=ftriton)
        # ftriton.close
        # ftorch = open('bench_output.txt', 'w')   
        # print(f'bench_output: {torch_output}', file=ftorch)
        # ftorch.close
    else:
        print(f"kernel 精度检查不合格!!!")
        mode_status = False

    if dtype == torch.bfloat16:
        if not (torch.allclose(cuda_output, torch_output, rtol=1e-2, atol=1e-2, equal_nan=True)):
            print(f"torch check kernel 精度检查不合格!!!")
            # torch.set_printoptions(profile="full")  
            # ftriton = open('cuda_output.txt', 'w')   
            # print(f'cuda_output: {cuda_output}', file=ftriton)
            # ftriton.close
            # ftorch = open('bench_output.txt', 'w')   
            # print(f'bench_output: {torch_output}', file=ftorch)
            # ftorch.close
        else:
            print(f"torch check kernel 精度检查合格.")
    
    warm_up_num = 15
    for i in range(warm_up_num):
        fused_bmm(input, w_uk, input_scale, w_uk_scale, bias, dtype, cuda_output, transpose_bm, transpose_bm_in, config)
        torch.cuda.synchronize()
    
    quantiles = [0.5, 0.2, 0.8]
    avg_ms, min_ms, max_ms = triton.testing.do_bench(lambda: fused_bmm(input, 
                                                                       w_uk, 
                                                                       input_scale, 
                                                                       w_uk_scale, 
                                                                       bias, 
                                                                       dtype,
                                                                       cuda_output, 
                                                                       transpose_bm,
                                                                       transpose_bm_in,
                                                                       config), quantiles=quantiles)
    fused_bmm_kernel_time_us = avg_ms * 1000
    # 计算peak和bw
    # 对于每个 n，执行 [b, p] @ [l, p].T = [b, l] 的 GEMM
    # 每个 GEMM 的 FLOPS = b * p * l * 2 (乘加算两次操作)
    # 总共有 n 个这样的 GEMM，所以总 peak = n * b * p * l * 2
    peak = n * b * p * l * 2
    
    output_size_bytes = torch.finfo(cuda_output.dtype).bits // 8
    if input.dtype == torch.int8:
        input_size_bytes = torch.iinfo(input.dtype).bits // 8
        w_uk_size_bytes = torch.iinfo(w_uk.dtype).bits // 8
    elif input.dtype == torch.float8_e4m3fn or input.dtype == torch.float8_e5m2:
        input_size_bytes = torch.finfo(input.dtype).bits // 8
        w_uk_size_bytes = torch.finfo(w_uk.dtype).bits // 8
    
    scale_size_bytes = torch.finfo(input_scale.dtype).bits // 8
    if bias is not None:
        bias_size_bytes = torch.finfo(bias.dtype).bits // 8
    
    # 计算数据传输量（字节）
    # input: [b, n, p] int8
    input_bytes = b * n * p * input_size_bytes
    # w_uk: [n, l, p] int8
    w_uk_bytes = n * l * p * input_size_bytes
    # input_scale: [n, b, 1] float32
    input_scale_bytes = n * b * 1 * scale_size_bytes
    # w_uk_scale: [n, l, 1] float32
    w_uk_scale_bytes = n * l * 1 * scale_size_bytes
    # cuda_output: [b, n, l] bfloat16
    output_bytes = b * n * l * output_size_bytes
    # bias (如果存在): [b, l] bfloat16
    bias_bytes = b * l * bias_size_bytes if bias is not None else 0
    
    total_bytes = input_bytes + w_uk_bytes + input_scale_bytes + w_uk_scale_bytes + output_bytes + bias_bytes
    total_gb = total_bytes / 1024 / 1024 / 1024  # GB
    
    # avg_ms 是毫秒，转换为秒
    time_sec = avg_ms / 1000
    tflops = (peak / time_sec) / 1e12
    bw = total_gb / time_sec
    
    results = {
        'B': b,
        'N': n,
        'L': l,
        'P': p,
        'Passed': mode_status,
        'Torch_bf16_bmm_time_us': torch_benchmark_time,
        'Q_quant_time_us': q_quant_time,
        'Fused_fp8_bmm_Time_us': fused_bmm_kernel_time_us,
        'Q_quant+Fused_fp8_bmm_Time_us': q_quant_time + fused_bmm_kernel_time_us,
        'Fused_fp8_bmm_TFlops': tflops,
        'Fused_fp8_bmm_GB/s': bw,
    }
    
    return results

@triton.jit
def _per_token_quant_fp8(
    x_ptr,
    xq_ptr,
    scale_ptr,
    stride_x,
    stride_xq,
    N,
    BLOCK: tl.constexpr,
    fp8_min,
    fp8_max
):
    row_id = tl.program_id(0)

    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(x_ptr + row_id * stride_x + cols, mask=mask,
                other=0.0).to(tl.float32)
    absmax = tl.maximum(tl.max(tl.abs(x)), 1e-10)
    scale_x = absmax / fp8_max
    x_q = tl.clamp(x / scale_x, min=fp8_min, max=fp8_max).to(xq_ptr.dtype.element_ty)
    tl.store(xq_ptr + row_id * stride_xq + cols, x_q, mask=mask)
    tl.store(scale_ptr + row_id, scale_x)


def per_token_quant_fp8(x):
    M = x.numel() // x.shape[-1]
    N = x.shape[-1]
    x_q = torch.empty_like(x, device=x.device, dtype=torch.float8_e4m3fn)
    scales = torch.empty(x.shape[:-1] + (1, ),
                         device=x.device,
                         dtype=torch.float32)
    BLOCK = triton.next_power_of_2(N)
    # heuristics for number of warps
    num_warps = min(max(BLOCK // 256, 1), 8)
    finfo = torch.finfo(x_q.dtype)
    fp8_min = finfo.min
    fp8_max = finfo.max
    #assert x.is_contiguous()
    _per_token_quant_fp8[(M, )](
        x,
        x_q,
        scales,
        stride_x=x.stride(-2),
        stride_xq=x_q.stride(-2),
        N=N,
        BLOCK=BLOCK,
        num_warps=num_warps,
        num_stages=1,
        fp8_min=fp8_min,
        fp8_max=fp8_max
    )
    return x_q, scales

def _fused_bmm_helper(batch: int,
                        N: int,
                        P: int,
                        L: int,
                        in_dtype: Type[torch.dtype] = torch.float8_e4m3fn,
                        out_dtype: Type[torch.dtype] = torch.float16,
                        transpose_bm: Optional[bool] = False,
                        transpose_bm_in: Optional[bool] = False,
                        device: str = "cuda",
                        best_config:Optional[list] = None,
                        is_tuning:bool = False):
    # Test for a cutlass kernel with per-token activation quantization
    # and per-output channel weight quantization.
    torch.manual_seed(0)

    torch_input_bf16 = torch.randn((batch, N, P), device=device, dtype=torch.bfloat16)           # [b, n, p]
    torch_w_uk_bf16 = torch.randn((N, L, P), device=device, dtype=torch.bfloat16)           # [n, l, p]
    if in_dtype == torch.float8_e4m3fn or in_dtype == torch.float8_e5m2:
        q_quant, q_scale = per_token_quant_fp8(torch_input_bf16)
        w_uk, w_uk_scale = per_token_quant_fp8(torch_w_uk_bf16)
    elif in_dtype == torch.int8:
        q_quant, q_scale = per_token_quant_int8(torch_input_bf16)
        w_uk, w_uk_scale = per_token_quant_int8(torch_w_uk_bf16)
    # print(f"before w_uk_scale.shape: {w_uk_scale.shape}, w_uk_scale.stride: {w_uk_scale.stride()}")
    # # import pdb; pdb.set_trace()
    # w_uk_scale = w_uk_scale.reshape(N, 1, L).permute(0, 2, 1)
    # print(f"after w_uk_scale.shape: {w_uk_scale.shape}, w_uk_scale.stride: {w_uk_scale.stride()}")
    bias = None

    if transpose_bm_in:
        q_quant = q_quant.transpose(0,1).contiguous()      # [n, b, p]
        q_scale = q_scale.transpose(0,1).contiguous()
    else:
        q_quant = q_quant      
        q_scale = q_scale

    return q_quant.clone().contiguous(), q_quant, w_uk, q_scale, w_uk_scale, bias, torch_input_bf16, torch_w_uk_bf16

def run_trans_bmm_trans_tuning():
    
    batch_list=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,32]
    # batch_list=[32,]
    ##deepseek-671b-tp8
    n_list=[16,32,64,128]
    # bmm1
    # p_list=[128,128,128,128]
    # l_list=[512,512,512,512]
    # bmm2
    p_list=[512,512,512,512]
    l_list=[128,128,128,128]
    
    # in_dtype=torch.float8_e4m3fn
    in_dtype=torch.int8
    out_dtype = torch.bfloat16
    # out_dtype = torch.float8_e4m3fn
    transpose_bm = False
    transpose_bm_in = False
    is_tuning = False

    best_config_json_dir="./"
    os.makedirs(best_config_json_dir, exist_ok=True)
    
    gfx_version = torch.cuda.get_device_properties(0).gcnArchName.split(':')[0]
    cu_number = torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count

    # mode_ranges = [0,1,2,3,4,5,6,7,8,9]
    # mode_ranges = [10,11,12,13,14,15,16,17,18,19]
    
    # bmm1
    # mode_ranges = [0,1,2,3,4,10,11,12,13,14,50,51,52,53,54,60,61,62,63,64]
    # bmm2
    mode_ranges = [0,1,3,10,11,13,50,51,53,60,61,63]
    
    for i in range(0,len(p_list),1):
        best_config_list = {}
        for batch_size in batch_list:
            max_time = torch.inf
            best_config = None
            compare_failed_list = []
            print("tuning for batch:{} n:{} p:{} l:{} ".format(batch_size, n_list[i],p_list[i],l_list[i]))
            
            # 先跑一遍，把失败的mode记录下来
            for mode in mode_ranges:
                print(f"tuning for batch:{batch_size} n:{n_list[i]} p:{p_list[i]} l:{l_list[i]} mode:{mode} ")
                torch_input, input, w_uk_t, input_scale, w_uk_t_scale, bias, _, _ =_fused_bmm_helper(batch_size, n_list[i], p_list[i], l_list[i], in_dtype=in_dtype, is_tuning=is_tuning, transpose_bm=transpose_bm, transpose_bm_in=transpose_bm_in)
                status=fused_bmm_trans_impl_int8(torch_input, input, w_uk_t, input_scale, w_uk_t_scale, bias=bias, mode=mode, transpose_bm=transpose_bm, transpose_bm_in=transpose_bm_in, is_tuning=True, dtype=out_dtype)
                if not status:
                    compare_failed_list.append(mode)
            print(f"batch_size: {batch_size}, n: {n_list[i]}, p: {p_list[i]}, l: {l_list[i]} compare_failed_list: {compare_failed_list}")
            
            for mode in mode_ranges:
                if mode in compare_failed_list:
                    continue
                torch_input, input, w_uk_t, input_scale, w_uk_t_scale, bias, _, _ =_fused_bmm_helper(batch_size, n_list[i], p_list[i], l_list[i], in_dtype=in_dtype, is_tuning=True, transpose_bm=transpose_bm, transpose_bm_in=transpose_bm_in)
                status, kernel_time_us=fused_bmm_trans_impl_int8_tuning(torch_input, input, w_uk_t, input_scale, w_uk_t_scale, bias=bias, mode=mode, transpose_bm=transpose_bm, transpose_bm_in=transpose_bm_in, dtype=out_dtype)
                if status and kernel_time_us < max_time:
                    max_time = kernel_time_us
                    best_config = mode

            
            if best_config is not None:
                print(f"best_config: {best_config} for batch_size: {batch_size}, n: {n_list[i]}, p: {p_list[i]}, l: {l_list[i]}")
                # 保存最优配置，键格式为“{batch_size}_{num_head}”
                key = f"{batch_size}"
                best_config_list[key] = {
                    "MODE": best_config,
                }
            else:
                print(f"best_config not found!!! for batch_size: {batch_size}, n: {n_list[i]}, p: {p_list[i]}, l: {l_list[i]}")
        
        print(f"best_config_list: {best_config_list}")
        filename = f"BATCHED_GEMM_W8A8_{n_list[i]}_{p_list[i]}_{gfx_version.upper()}_CU{cu_number}.json"

        # 保存最优配置到文件
        config_file = os.path.join(best_config_json_dir, filename)
        with open(config_file, 'w') as f:
            json.dump(best_config_list, f, indent=4)
        print(f"最优配置已保存到: {config_file}")


def run_trans_bmm_trans_vaild():
    
    # batch_list=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32]
    batch_list=[24,]
    ##deepseek-671b-tp8
    # bmm1
    # n_list=[16,]
    # p_list=[128,]
    # l_list=[512,]
    
    # n_list=[32,]
    # p_list=[128,]
    # l_list=[512,]
    
    # n_list=[64,]
    # p_list=[128,]
    # l_list=[512,]
    
    # n_list=[128]
    # p_list=[128]
    # l_list=[512]

    # bmm2
    n_list=[16,]
    p_list=[512,]
    l_list=[128,]
    
    # n_list=[32,]
    # p_list=[512,]
    # l_list=[128,]
    
    # n_list=[64,]
    # p_list=[512,]
    # l_list=[128,]
    
    # n_list=[128]
    # p_list=[512]
    # l_list=[128]
    
    in_dtype=torch.float8_e4m3fn
    out_dtype = torch.bfloat16
    # out_dtype = torch.float8_e4m3fn
    transpose_bm = False
    transpose_bm_in = False

    best_config = None
    triton_json_dir="./cache/moe/"
    os.makedirs(triton_json_dir, exist_ok=True)
    
    device_name=torch.cuda.get_device_properties('cuda').name.replace(" ","_")

    # mode_ranges = [i for i in range(len(blockM_configs))]
    # mode_ranges = [mode for mode in blockM_configs.keys()]
    # mode_ranges = [0,1,2,3,4,5,6,7,8,9]
    # mode_ranges = [10,11,12,13,14,15,16,17,18,19]
    # mode_ranges = [0,1,2,3,4,10,11,12,13,14,50,51,52,53,54,60,61,62,63,64]
    mode_ranges = [0]
    
    for i in range(0,len(p_list),1):
        for batch_size in batch_list:
            for mode in mode_ranges:

                print("mode:{} batch:{} n:{} p:{} l:{} ".format(mode, batch_size, n_list[i],p_list[i],l_list[i]))
                torch_input, input, w_uk_t, input_scale, w_uk_t_scale, bias, _, _ =_fused_bmm_helper(batch_size, n_list[i], p_list[i], l_list[i], in_dtype=in_dtype, transpose_bm=transpose_bm, transpose_bm_in=transpose_bm_in)
                # triton moe: output
                status=fused_bmm_trans_impl_int8(torch_input, input, w_uk_t, input_scale, w_uk_t_scale, bias=bias, mode=mode, transpose_bm=transpose_bm, transpose_bm_in=transpose_bm_in, dtype=out_dtype)
                if not status:
                    fused_bmm_compare_failed_list.append(mode)
                     
    print("fused_bmm_compare_failed_list: ", fused_bmm_compare_failed_list)
     
    
def run_benchmark_trans_bmm_trans():
    batch_list=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32]
    # batch_list = [32]
    
    ##deepseek-671b-tp8
    # n_list=[16,32,64,128]
    n_list=[16,16,16,16]
    # bmm1
    # p_list=[128,128,128,128]
    # l_list=[512,512,512,512]
    # bmm2
    p_list=[512,512,512,512]
    l_list=[128,128,128,128]
    
    # in_dtype=torch.float8_e4m3fn
    in_dtype=torch.int8
    out_dtype = torch.bfloat16
    # out_dtype = torch.float8_e4m3fn
    transpose_bm = False
    transpose_bm_in = False

    results_list = []
    for i in range(0,len(p_list),1):
        for batch_size in batch_list:
            # for mode in mode_ranges:
            torch_input, input, w_uk_t, input_scale, w_uk_t_scale, bias, torch_q_bf16, torch_w_uk_bf16 =_fused_bmm_helper(batch_size, n_list[i], p_list[i], l_list[i], in_dtype=in_dtype, transpose_bm=transpose_bm, transpose_bm_in=transpose_bm_in)
            # triton moe: output
            result=fused_bmm_trans_impl_int8_benchmark(torch_q_bf16, torch_w_uk_bf16, torch_input, input, w_uk_t, input_scale, w_uk_t_scale, bias=bias, dtype=out_dtype, transpose_bm=transpose_bm, transpose_bm_in=transpose_bm_in)
            results_list.append(result)
    
    # 计算通过率
    total_count = len(results_list)
    passed_count = sum(1 for r in results_list if r.get("Passed", False))
    pass_rate = (passed_count / total_count * 100) if total_count > 0 else 0.0
    # 收集不通过的 shape (B, N, L, P)
    failed_shapes = [
        (r["B"], r["N"], r["L"], r["P"])
        for r in results_list
        if not r.get("Passed", False)
    ]
                
    output_file = "fused_bmm_benchmark.xlsx"
    df = pd.DataFrame(results_list)
    df.to_excel(output_file, index=False)
    print(f"通过率: {passed_count}/{total_count} = {pass_rate:.2f}%")
    if failed_shapes:
        print("不通过的 shape (B, N, L, P):")
        for shape in failed_shapes:
            print(f"  {shape}")                  
            
def main(mode: str = "debug"):
    """
    mode 支持三种：
      - "tune"      : 跑 kernel tuning, 生成/更新最优 config json
      - "debug"     : 做精度校验
      - "benchmark" : 使用最优 config 跑性能 benchmark
    """
    if mode == "tune":
        run_trans_bmm_trans_tuning()
    elif mode == "debug":
        run_trans_bmm_trans_vaild()
    elif mode == "benchmark":
        run_benchmark_trans_bmm_trans()
    else:
        print(f"未知模式: {mode}，请使用 'tune' / 'debug' / 'benchmark'")


if __name__ == "__main__":
    import sys
    # 命令行：python test_fused_trans_bmm_trans.py [tune|debug|benchmark]
    cli_mode = sys.argv[1] if len(sys.argv) > 1 else "debug"
    main(cli_mode)
    # 示例：
    #   HIP_VISIBLE_DEVICES=1 python3 test_fused_trans_bmm_trans.py tune
    #   HIP_VISIBLE_DEVICES=1 python3 test_fused_trans_bmm_trans.py debug
    #   HIP_VISIBLE_DEVICES=1 python3 test_fused_trans_bmm_trans.py benchmark
