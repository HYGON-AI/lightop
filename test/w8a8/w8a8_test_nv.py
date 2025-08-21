import torch
import torch.nn as nn
import time
from typing import Optional, Type
import pandas as pd
import os
import json
import triton
from vllm import _custom_ops as ops


def to_int8(tensor: torch.Tensor):
    return torch.round(tensor.clamp(min=-128, max=127)).to(dtype=torch.int8)

def baseline_scaled_mm(a: torch.Tensor,
                      b: torch.Tensor,
                      scale_a: torch.Tensor,
                      scale_b: torch.Tensor,
                      out_dtype: torch.dtype,
                      bias: Optional[torch.Tensor] = None) -> torch.Tensor:

    scales= scale_a* scale_b.T
    gemmout= torch.mm(
        a.to(dtype=torch.float32), b.to(dtype=torch.float32))
    output = (scales *gemmout).to(out_dtype)
    if bias is not None:
        output = output + bias
    return output


def _cutlass_scaled_mm(a: torch.Tensor,   #input
                            b: torch.Tensor,  #weight
                            scale_a: torch.Tensor,
                            scale_b: torch.Tensor,
                            out_dtype: torch.dtype,
                            bias: Optional[torch.Tensor] = None) -> torch.Tensor:

    return ops.cutlass_scaled_mm(a,
                                     b,
                                     scale_a=scale_a,
                                     scale_b=scale_b,
                                     out_dtype=out_dtype,
                                     bias=bias)

def dq_cutlass_scaled_mm(a: torch.Tensor,   #input
                            b: torch.Tensor,  #weight
                            scale_a: torch.Tensor,
                            scale_b: torch.Tensor,
                            out_dtype: torch.dtype,
                            bias: Optional[torch.Tensor] = None) -> torch.Tensor:
    x_q, x_scale, x_zp = ops.scaled_int8_quant(a,
                                               None,
                                               None,
                                               symmetric=True)  
    return ops.cutlass_scaled_mm(x_q,
                                     b,
                                     scale_a=x_scale,
                                     scale_b=scale_b,
                                     out_dtype=out_dtype,
                                     bias=bias)

def _fp16_gemm(a: torch.Tensor,   #input
                            b: torch.Tensor,  #weight
                            out_dtype: torch.dtype,
                            bias: Optional[torch.Tensor] = None) -> torch.Tensor:

    output=torch.matmul(a,b).to(out_dtype)
    if bias:
        output=output+bias

    return output

def _int8_gemm_helper(m: int,
                             n: int,
                             k: int,
                             per_token_act_quant: bool,
                             per_out_channel_weight_quant: bool,
                             use_bias: bool,
                             out_dtype: Type[torch.dtype] = torch.float16,
                             device: str = "cuda",
                             best_config:Optional[list] = None):
    # Test for a cutlass kernel with per-token activation quantization
    # and per-output channel weight quantization.
    a_fp16 = (torch.randn((m, k), device=device) * 5).to(dtype=out_dtype)
    b_fp16 = (torch.randn((k, n), device=device) * 5).to(dtype=out_dtype)
    
    a = to_int8(torch.randn((m, k), device=device) * 5)
    b = to_int8(torch.randn((n, k), device=device).t() * 5)
    
    m_a_scales = m if per_token_act_quant else 1
    n_b_scales = n if per_out_channel_weight_quant else 1

    scale_a = (torch.randn((m_a_scales,1), device=device,
                           dtype=torch.float32))
    scale_b = (torch.randn((n_b_scales,1), device=device,
                           dtype=torch.float32))
    if use_bias:
        bias = torch.rand((n, ), device=device, dtype=out_dtype) * 10
    else:
        bias = None

    #激活量化算子耗时统计
    for i in range(100):
        x_q, x_scale, x_zp = ops.scaled_int8_quant(a_fp16,
                                               None,
                                               None,
                                               symmetric=True)  
        
    elapsed_time0_gpu = triton.testing.do_bench(lambda:ops.scaled_int8_quant(a_fp16,
                                                                                None,
                                                                                None,
                                                                                symmetric=True),quantiles=None, return_mode="mean")*1000
    print("cost_time:{} us\n".format(elapsed_time0_gpu))

    for i in range(100):
        out=_cutlass_scaled_mm(a,b,scale_a,scale_b,out_dtype,bias)

    elapsed_time1_gpu = triton.testing.do_bench(lambda:_cutlass_scaled_mm(a,b,scale_a,scale_b,out_dtype,bias),quantiles=None, return_mode="mean")*1000
    print("cost_time:{} us\n".format(elapsed_time1_gpu))
    
    for i in range(100):
        out=dq_cutlass_scaled_mm(a_fp16,b,None,scale_b,out_dtype,bias)

    elapsed_time2_gpu = triton.testing.do_bench(lambda:dq_cutlass_scaled_mm(a_fp16,b,None,scale_b,out_dtype,bias),quantiles=None, return_mode="mean")*1000
    print("cost_time:{} us\n".format(elapsed_time2_gpu))

    #FP16 linear耗时
    for i in range(100):
        out=_fp16_gemm(a_fp16,b_fp16,out_dtype,bias)

    elapsed_time3_gpu = triton.testing.do_bench(lambda:_fp16_gemm(a_fp16,b_fp16,out_dtype,bias),quantiles=None, return_mode="mean")*1000
    print("cost_time:{} us\n".format(elapsed_time3_gpu))
    

    return elapsed_time0_gpu,elapsed_time1_gpu,elapsed_time2_gpu,elapsed_time3_gpu

    

def main(n_list, k_list ,excel_name="output.xlsx"):
    # m=1
    # n_list=[6144,4096,24576,4096]
    # k_list=[4096,4096,4096,12288]

    # # dpsk 671b
    # n_list = [256, 576, 1536, 1536, 2048, 2304, 7168, 7168, 7168,]
    # k_list = [7168, 7168, 1536, 7168, 512, 7168, 128, 1024, 1152,]
   
    triton_cost_gpu=[]
    act_cost_gpu=[]
    qtriton_cost_gpu=[]
    fp16_cost_gpu=[]

    bs=[]
    _m=[]
    _k=[]

    for i in range(0,len(k_list),1):      
    
        m1=[item if item < 17 else  1 << (item - 27) for item in range(1, 17)]
        m2=[item<<2  if item <17 else (item - 8)<<3  for item in range(5, 29)]
        m3=[2<<(item)  for item in range(7, 15)]
        
        m_list = m1+m2+m3
        for m in m_list:
            print("m:{} n:{} k:{} ".format(m,n_list[i],k_list[i]))
            
            #continue
            elapsed_time0_gpu,elapsed_time1_gpu,elapsed_time2_gpu,elapsed_time3_gpu=_int8_gemm_helper(m=m,n=n_list[i],k=k_list[i],per_token_act_quant=True,per_out_channel_weight_quant=True,use_bias=False)
            act_cost_gpu.append(elapsed_time0_gpu)
            triton_cost_gpu.append(elapsed_time1_gpu)
            qtriton_cost_gpu.append(elapsed_time2_gpu) 
            fp16_cost_gpu.append(elapsed_time3_gpu)  
            bs.append(m)
            _k.append(k_list[i])
            _m.append(n_list[i])
        
    # 创建一个包含这三个列表的 DataFrame
    df = pd.DataFrame({'n':bs,'m':_m,'k':_k,'激活量化GPU耗时': act_cost_gpu,'w8a8_cutslass GPU耗时': triton_cost_gpu,'激活+w8a8GPU耗时': qtriton_cost_gpu,'FP16 linear GPU耗时': fp16_cost_gpu})

    # 将 DataFrame 写入 Excel 文件
    df.to_excel(excel_name, index=False)

    print(f"表格已保存到 {excel_name}文件中。")

if __name__ == "__main__":
    # # # dpsk 671b
    n_list = [256, 576, 1536, 1536, 2048, 2304, 7168, 7168, 7168,]
    k_list = [7168, 7168, 1536, 7168, 512, 7168, 128, 1024, 1152,]
    main(n_list, k_list, excel_name='A100_dpsk_671b.xlsx')


    # # qwen3-8b-tp1
    n_list = [6144, 4096, 24576, 4096,]
    k_list = [4096, 4096, 4096, 12288,]
    main(n_list, k_list, excel_name='A100_qwen3_8b_tp1.xlsx')

    # # qwen3-32b-tp1
    n_list = [10240, 5120, 51200, 5120, ]
    k_list = [5120, 8192, 5120, 25600, ]

    main(n_list, k_list, excel_name='A100_qwen3_32b_tp1.xlsx')

    # # qwen3-32b-tp2
    n_list = [5120, 5120, 25600, 5120,]
    k_list = [5120, 4096, 5120, 12800, ]
    main(n_list, k_list, excel_name='A100_qwen3_32b_tp2.xlsx')

    # # qwen3-32b-tp4
    n_list = [2560, 5120, 12800, 5120,]
    k_list = [5120, 2048, 5120, 6400,]
    main(n_list, k_list, excel_name='A100_qwen3_32b_tp4.xlsx')
    

    # qwen3-30b-A3B-tp1
    n_list = [5120, 2048]
    k_list = [2048, 4096]
    main(n_list, k_list, excel_name='A100_qwen3_30b_A3B_tp1.xlsx')

    # qwen3-30b-A3B-tp2
    n_list = [2560, 2048]
    k_list = [2048, 2048]
    main(n_list, k_list, excel_name='A100_qwen3_30b_A3B_tp2.xlsx')

    # qwen3-30b-A3B-tp4
    n_list = [1280, 2048]
    k_list = [2048, 1024]
    main(n_list, k_list, excel_name='A100_qwen3_30b_A3B_tp4.xlsx')

    # # qwen3-235b-tp8
    n_list = [1280, 4096,]
    k_list = [4096, 1024,]
    main(n_list, k_list, excel_name='A100_qwen3_235b_tp8.xlsx')

    # # qwen3-235b-tp16
    n_list = [768, 4096,]
    k_list = [4096, 512,]
    main(n_list, k_list, excel_name='A100_qwen3_235b_tp16.xlsx')

    ## llama3.3-70b-tp2
    n_list = [28672, 8192, 5120, 8192,]
    k_list = [8192, 14336, 8192, 4096, ]
    main(n_list, k_list, excel_name='A100_llama3.3_70b_tp2.xlsx')

    ## llama3.3-70b-tp4
    n_list = [14336, 8192, 2560, 8192,]
    k_list = [8192, 7168, 8192, 2048, ]
    main(n_list, k_list, excel_name='A100_llama3.3_70b_tp4.xlsx')

    ## llama3.3-70b-tp8
    n_list = [7168, 8192, 1280, 8192,]
    k_list = [8192, 3584, 8192, 1024, ]
    main(n_list, k_list, excel_name='A100_llama3.3_70b_tp8.xlsx')
