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
from vllm import _custom_ops as ops
# from test_int8_kernel import torch_w8a8_per_column_moe
# import _custom_ops as ops
from lightop import fused_bmm
from lmslim.layers.gemm.int8_utils import (
    per_token_group_quant_int8, per_token_quant_int8)

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
    
    for i in range(batch_size):
        scales = scale_a[i] * scale_b[i].T   # [b, l]
        # 使用 torch 实现 int8 矩阵乘法，输出 int32
        # a[i]: [b, p] int8, b[i]: [l, p] int8
        # 计算 a[i] @ b[i].T，结果: [b, l] int32
        gemmoutput = torch.matmul(a[i].cpu().to(dtype=torch.int32), b[i].T.cpu().to(dtype=torch.int32))  # [b, l]
        output[i] = (scales.cpu() * gemmoutput.to(dtype=torch.float32)).to(out_dtype)
        # import pdb; pdb.set_trace()
        # do not support bias!
        # if bias is not None:
        #     output[i] = output[i] + bias[i].cpu()
    return output.cuda()

def fused_bmm_trans_impl_int8(input: torch.Tensor,                              # b，n, p
                                w_uk: torch.Tensor,                             # n, l, p
                                input_scale: Optional[torch.Tensor] = None,     # n, b, 1
                                w_uk_scale: Optional[torch.Tensor] = None,      # n, l, 1
                                bias: Optional[torch.Tensor] = None,            # n, l
                                config: Optional[Dict[str, Any]] = None,
                                mode: Optional[List[int]]=None):
    
    # input = torch.ones_like(input)
    # w_uk的shape是[n,l,p]，第0,1维保持不变，第2维从0-p-1增长
    # w_uk = torch.arange(w_uk.shape[2], dtype=w_uk.dtype, device=w_uk.device).view(1, 1, -1).expand_as(w_uk)
    # w_uk = torch.ones_like(w_uk)*2
    # input_scale = torch.ones_like(input_scale)
    # w_uk_scale = torch.ones_like(w_uk_scale)

    inputc_clone = input.clone()
    torch_uk = w_uk.clone()
    torch_input_scale = input_scale.clone()
    torch_w_uk_scale = w_uk_scale.clone()
    torch_bias = bias.clone() if bias is not None else None
    
    
    # fused_bmm_output 应该是 [n, b, l] 的形状
    torch_input = inputc_clone.transpose(0,1).contiguous()  # [b, n, p] -> [n, b, p]
    # 使用bfloat16作为基准输出类型，避免FP8类型比较时的运算问题
    fused_torch_output = batch_rocblas_scaled_mm(torch_input, torch_uk, torch_input_scale, torch_w_uk_scale, out_dtype=torch.float8_e4m3fn, bias=torch_bias)   # [n, b, l]
    torch_output = fused_torch_output.transpose(0,1).contiguous() # [b, n, l]
    # import pdb; pdb.set_trace()
    # 验证lightop kernel
    # torch.set_printoptions(profile="full") 
    cuda_output = torch.zeros_like(torch_output, dtype=torch.float8_e4m3fn)
    fused_bmm(input, w_uk, cuda_output, input_scale, w_uk_scale, bias, mode)
    torch.cuda.synchronize()
    # import pdb; pdb.set_trace()
    mode_status = True
    
    # 将FP8输出转换为bfloat16进行比较
    # PyTorch的FP8类型可能不支持直接.to()转换，需要通过view转换为uint8
    # 然后手动实现FP8到float的转换，或者使用torch的dequantize功能
    # 这里使用一个简单的方法：将FP8视为uint8，然后转换为float32再转bfloat16
    # 注意：这不是正确的FP8解码，但可以作为临时方案
    # 如果需要精确的FP8解码，应该使用lightop的转换函数或实现正确的FP8解码逻辑
    torch_output_uint8 = torch_output.view(torch.uint8)
    cuda_output_uint8 = cuda_output.view(torch.uint8)
    # 将uint8转换为float32（这里假设FP8值范围在合理范围内）
    # 正确的做法应该是实现FP8 E4M3格式的解码
    torch_output_fp32 = torch_output_uint8.to(torch.float32)
    cuda_output_fp32 = cuda_output_uint8.to(torch.float32)
    # import pdb; pdb.set_trace()
    if not (torch.allclose(torch_output_fp32, cuda_output_fp32, rtol=1e-2, atol=1e-2, equal_nan=True)):
        print(f"kernel 精度检查不合格!!!")
        mode_status = False
        
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
        
    return mode_status

def torch_benchmark_transpose_bmm_transpose_bf16(decode_q_nope: torch.Tensor,
                                                 w_uk_t: torch.Tensor) -> torch.Tensor:
    # decode_q_nope.shape = [b, n, p]
    # w_uk.shape = [n, p, l]
    # return: [b, n, l]
    decode_q_nope = decode_q_nope.transpose(0,1)  # [b, n, p] -> [n, b, p]
    decode_ql_nope = torch.bmm(decode_q_nope, w_uk_t)  # [n, b, l]
    decode_ql_nope = decode_ql_nope.transpose(0,1)  # [b, n, l]
    return decode_ql_nope

def torch_benchmark_fuse_bmm_bf16(B: int,
                                  N: int,
                                  P: int,
                                  L: int):
    torch.manual_seed(0)
    decode_q_nope = torch.randn(B, N, P, dtype=torch.bfloat16, device='cuda')  # [b, n, p]
    w_uk_t = torch.randn(N, P, L, dtype=torch.bfloat16, device='cuda')  # [n, p, l]
    
    warm_up_num = 10
    # function:
    for i in range(warm_up_num):
        torch_benchmark_transpose_bmm_transpose_bf16(decode_q_nope, w_uk_t)
        torch.cuda.synchronize()
    
    quantiles = [0.5, 0.2, 0.8]
    avg_ms, min_ms, max_ms = triton.testing.do_bench(lambda: torch_benchmark_transpose_bmm_transpose_bf16(decode_q_nope, w_uk_t), 
                                                            quantiles=quantiles)
    
    return avg_ms * 1000
    
    

def fused_bmm_trans_impl_int8_benchmark(input: torch.Tensor,                            # b，n, p
                                        w_uk: torch.Tensor,                             # n, l, p
                                        input_scale: Optional[torch.Tensor] = None,     # n, b, 1
                                        w_uk_scale: Optional[torch.Tensor] = None,      # n, l, 1
                                        bias: Optional[torch.Tensor] = None,            # b, l
                                        config: Optional[Dict[str, Any]] = None,
                                        mode: Optional[List[int]]=None):

    # 用实际 shape 申请输出空间，避免依赖 zeros_like
    b, n, p = input.shape
    _, l, _ = w_uk.shape
    cuda_output = torch.empty((b, n, l), dtype=torch.float8_e4m3fn, device=input.device)
    
    # torch benchmark:
    torch_benchmark_time = torch_benchmark_fuse_bmm_bf16(b, n, p, l)
    
    warm_up_num = 10
    for i in range(warm_up_num):
        fused_bmm(input, w_uk, cuda_output, input_scale, w_uk_scale, bias, mode)
        torch.cuda.synchronize()
    
    fused_bmm(input, w_uk, cuda_output, input_scale, w_uk_scale, bias, mode)
    torch.cuda.synchronize()
    
    quantiles = [0.5, 0.2, 0.8]
    avg_ms, min_ms, max_ms = triton.testing.do_bench(lambda: fused_bmm(input, 
                                                                       w_uk, 
                                                                       cuda_output, 
                                                                       input_scale, 
                                                                       w_uk_scale, 
                                                                       bias, 
                                                                       mode), quantiles=quantiles)
    fused_bmm_kernel_time_us = avg_ms * 1000
    # 计算peak和bw
    # 对于每个 n，执行 [b, p] @ [l, p].T = [b, l] 的 GEMM
    # 每个 GEMM 的 FLOPS = b * p * l * 2 (乘加算两次操作)
    # 总共有 n 个这样的 GEMM，所以总 peak = n * b * p * l * 2
    peak = n * b * p * l * 2
    
    # 计算数据传输量（字节）
    # input: [b, n, p] int8
    input_bytes = b * n * p * 1
    # w_uk: [n, l, p] int8
    w_uk_bytes = n * l * p * 1
    # input_scale: [n, b, 1] float32
    input_scale_bytes = n * b * 1 * 4
    # w_uk_scale: [n, l, 1] float32
    w_uk_scale_bytes = n * l * 1 * 4
    # cuda_output: [b, n, l] bfloat16
    output_bytes = b * n * l * 2
    # bias (如果存在): [b, l] bfloat16
    bias_bytes = b * l * 2 if bias is not None else 0
    
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
        'Torch_time_us': torch_benchmark_time,
        'GEMM_Time_us': fused_bmm_kernel_time_us,
        'GEMM_TFlops': tflops,
        'GEMM_GB/s': bw,
    }
    
    return results

def _fused_bmm_helper(batch: int,
                        N: int,
                        P: int,
                        L: int,
                        out_dtype: Type[torch.dtype] = torch.float16,
                        device: str = "cuda",
                        best_config:Optional[list] = None):
    # Test for a cutlass kernel with per-token activation quantization
    # and per-output channel weight quantization.
    torch.manual_seed(0)
    input = torch.randint(-128, 127, (batch, N, P), device=device, dtype=torch.int8)      # [b, n, p]
    # w_uk_t = torch.randint(-128, 127, (N, P ,L), device=device, dtype=torch.int8)       # [n, p, l]
    w_uk = torch.randint(-128, 127, (N, L, P), device=device, dtype=torch.int8)           # [n, l, p]
    # 生成0.001到0.1之间的随机scale值，保留三位小数,防止fp8算出来都是些nan值
    input_scale = (torch.rand((N, batch, 1), device=device, dtype=torch.float32) * 0.099 + 0.001).round(decimals=3)  # [n, b, 1]
    w_uk_scale = (torch.rand((N, L, 1), device=device, dtype=torch.float32) * 0.099 + 0.001).round(decimals=3)  # [n, l, 1]
    bias = torch.randn((N, L), device=device, dtype=torch.bfloat16)                       # [n, l]
    
    return input, w_uk, input_scale, w_uk_scale, bias

def run_trans_bmm_trans_vaild():
    
    # batch_list=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,32]
    batch_list=[32]
    ##deepseek-671b-tp8
    n_list=[16]
    p_list=[128]
    l_list=[512]
    
    out_dtype=torch.bfloat16
    # out_dtype = torch.float16

    best_config = None
    triton_json_dir="./cache/moe/"
    os.makedirs(triton_json_dir, exist_ok=True)
    
    device_name=torch.cuda.get_device_properties('cuda').name.replace(" ","_")

    # mode_ranges = [i for i in range(len(blockM_configs))]
    # mode_ranges = [mode for mode in blockM_configs.keys()]
    # mode_ranges = [0,1,2,3,4,5,6,7,8,9]
    # mode_ranges = [0,1,2]
    mode_ranges = [0]
    
    for i in range(0,len(p_list),1):
        for batch_size in batch_list:
            for mode in mode_ranges:

                print("mode:{} batch:{} n:{} p:{} l:{} ".format(mode, batch_size, n_list[i],p_list[i],l_list[i]))
                input, w_uk_t, input_scale, w_uk_t_scale, bias=_fused_bmm_helper(batch_size, n_list[i], p_list[i], l_list[i])
                # triton moe: output
                status=fused_bmm_trans_impl_int8(input, w_uk_t, input_scale, w_uk_t_scale, bias=bias, mode=mode)
                if not status:
                    fused_bmm_compare_failed_list.append(mode)
                     
    print("fused_bmm_compare_failed_list: ", fused_bmm_compare_failed_list)
     
    
def run_benchmark_trans_bmm_trans():
    batch_list=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,32]
    # batch_list = [32]
    
    ##deepseek-671b-tp8
    n_list=[16]
    p_list=[128]
    l_list=[512]
    
    out_dtype=torch.bfloat16
    # out_dtype = torch.float16


    # configs=BlockSizeMConfigGen()
    best_config = None
    triton_json_dir="./cache/moe/"
    os.makedirs(triton_json_dir, exist_ok=True)
    
    device_name=torch.cuda.get_device_properties('cuda').name.replace(" ","_")

    mode_ranges = [0]

    results_list = []
    for i in range(0,len(p_list),1):
        for batch_size in batch_list:
            for mode in mode_ranges:

                print("mode:{} batch:{} n:{} p:{} l:{} ".format(mode, batch_size, n_list[i],p_list[i],l_list[i]))
                input, w_uk_t, input_scale, w_uk_t_scale, bias=_fused_bmm_helper(batch_size, n_list[i], p_list[i], l_list[i])
                # triton moe: output
                result=fused_bmm_trans_impl_int8_benchmark(input, w_uk_t, input_scale, w_uk_t_scale, bias=bias, mode=mode)
                results_list.append(result)
                
    output_file = "fused_bmm_benchmark.xlsx"
    df = pd.DataFrame(results_list)
    df.to_excel(output_file, index=False)                  
            
def main():
    # run_tuning()
    # run_trans_bmm_trans_vaild()
    run_benchmark_trans_bmm_trans()

if __name__ == "__main__":
    main()
    # HIP_VISIBLE_DEVICES=1 python3 test_fused_trans_bmm_trans.py