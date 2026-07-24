import torch
import time
from typing import Optional, Type
import pandas as pd
import os
import json
import triton
from vllm import _custom_ops as ops
from common.w8a8_cache_json import W8a8GetCacheJSON

from lightop.gemm_ops import (
    gemm_w8a8_smooth,
    rocblas_scaled_mm_nn,
    triton_scaled_mm,
)
from lightop.quant import per_token_quant_int8


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

def _lightop_scaled_mm(a: torch.Tensor,
                      b: torch.Tensor,
                      scale_a: torch.Tensor,
                      scale_b: torch.Tensor,
                      out_dtype: torch.dtype,
                      bias: Optional[torch.Tensor] = None
                      ) -> torch.Tensor:
    return gemm_w8a8_smooth(a, b,scale_a,scale_b,bias,out_dtype)

def dq_lightop_scaled_mm(a: torch.Tensor,
                      b: torch.Tensor,
                      scale_a: torch.Tensor,
                      scale_b: torch.Tensor,
                      out_dtype: torch.dtype,
                      bias: Optional[torch.Tensor] = None,
                      ) -> torch.Tensor:
    
    x_q, x_scale = per_token_quant_int8(a)   
     
    return gemm_w8a8_smooth(x_q, b,x_scale,scale_b,bias,out_dtype)


def _triton_scaled_mm(a: torch.Tensor,
                      b: torch.Tensor,
                      scale_a: torch.Tensor,
                      scale_b: torch.Tensor,
                      out_dtype: torch.dtype,
                      bias: Optional[torch.Tensor] = None,
                      best_config:Optional[list] = None) -> torch.Tensor:
    return triton_scaled_mm(a, b,scale_a,scale_b,out_dtype,bias,best_config)

def dq_triton_scaled_mm(a: torch.Tensor,
                      b: torch.Tensor,
                      scale_a: torch.Tensor,
                      scale_b: torch.Tensor,
                      out_dtype: torch.dtype,
                      bias: Optional[torch.Tensor] = None,
                      best_config:Optional[list] = None) -> torch.Tensor:
    
    x_q, x_scale = per_token_quant_int8(a)   
     
    return triton_scaled_mm(x_q, b,x_scale,scale_b,out_dtype,bias,best_config)


def _rocblas_scaled_mm(a: torch.Tensor,   #input
                            b: torch.Tensor,  #weight
                            scale_a: torch.Tensor,
                            scale_b: torch.Tensor,
                            out_dtype: torch.dtype,
                            bias: Optional[torch.Tensor] = None) -> torch.Tensor:
    return rocblas_scaled_mm_nn(a,b,scale_a,scale_b,out_dtype,bias)

def dq_rocblas_scaled_mm(a: torch.Tensor,   #input
                            b: torch.Tensor,  #weight
                            scale_a: torch.Tensor,
                            scale_b: torch.Tensor,
                            out_dtype: torch.dtype,
                            bias: Optional[torch.Tensor] = None) -> torch.Tensor:
    x_q, x_scale = per_token_quant_int8(a)    
    return rocblas_scaled_mm_nn(x_q,b,x_scale,scale_b,out_dtype,bias)


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
    #b = to_int8(torch.randn((k,n), device=device) * 5)
    b = to_int8(torch.randn((n, k), device=device).t() * 5)
    
    m_a_scales = m if per_token_act_quant else 1
    n_b_scales = n if per_out_channel_weight_quant else 1

    scale_a = (torch.randn((m_a_scales, 1), device=device,
                           dtype=torch.float32))
    scale_b = (torch.randn((n_b_scales,1), device=device,
                           dtype=torch.float32))
    if use_bias:
        bias = torch.rand((n, ), device=device, dtype=out_dtype) * 10
    else:
        bias = None

    quantiles = [0.5, 0.2, 0.8]


    for i in range(100):
        status,out = _lightop_scaled_mm(a,b,scale_a,scale_b,out_dtype,bias)  
        
    elapsed_time7_gpu = triton.testing.do_bench(lambda:_lightop_scaled_mm(a,b,scale_a,scale_b,out_dtype,bias),quantiles=None, return_mode="mean")*1000
    # print("triton_time:{} us\n".format(elapsed_time7_gpu))

    #测试激活量化算子耗时
    for i in range(100):
        status,out = dq_lightop_scaled_mm(a_fp16,b,None,scale_b,out_dtype,bias)  
        
    elapsed_time8_gpu = triton.testing.do_bench(lambda:dq_lightop_scaled_mm(a_fp16,b,None,scale_b,out_dtype,bias),quantiles=None, return_mode="mean")*1000
    # print("triton_time:{} us\n".format(elapsed_time8_gpu))
      
    # #测试激活量化算子耗时
    # for i in range(100):
    #     x_q = per_token_quant_int8(a_fp16)  
        
    # elapsed_time0_gpu = triton.testing.do_bench(lambda:per_token_quant_int8(a_fp16),quantiles=None, return_mode="mean")*1000
    # # print("triton_time:{} us\n".format(elapsed_time0_gpu))
    
            
    # for i in range(100):
    #     triton_out=_triton_scaled_mm(a,b,scale_a,scale_b,out_dtype,bias,best_config)

    # elapsed_time1_gpu = triton.testing.do_bench(lambda:_triton_scaled_mm(a,b,scale_a,scale_b,out_dtype,bias,best_config),quantiles=None, return_mode="mean")*1000
    # # print("triton_time:{} us\n".format(elapsed_time1_gpu))
    # torch.cuda.synchronize()  
    # for i in range(100):
    #     triton_out=dq_triton_scaled_mm(a_fp16,b,None,scale_b,out_dtype,bias,best_config)

    # elapsed_time2_gpu = triton.testing.do_bench(lambda:dq_triton_scaled_mm(a_fp16,b,None,scale_b,out_dtype,bias,best_config),quantiles=None, return_mode="mean")*1000
    # # print("triton_time:{} us\n".format(elapsed_time2_gpu))
    # torch.cuda.synchronize()     

    # b=b.contiguous().reshape(k,-1)
    
    # for i in range(100):
    #     out = _rocblas_scaled_mm(a,b,scale_a,scale_b,out_dtype,bias)
 
    # elapsed_time3_gpu = triton.testing.do_bench(lambda:_rocblas_scaled_mm(a,b,scale_a,scale_b,out_dtype,bias),quantiles=None, return_mode="mean")*1000
    # # print("rocblas_scaled_mm:{} us\n".format(elapsed_time3_gpu))
    # torch.cuda.synchronize()  
    # for i in range(100):
    #     out = dq_rocblas_scaled_mm(a_fp16,b,None,scale_b,out_dtype,bias)
 
    # elapsed_time4_gpu = triton.testing.do_bench(lambda:dq_rocblas_scaled_mm(a_fp16,b,None,scale_b,out_dtype,bias),quantiles=None, return_mode="mean")*1000
    # torch.cuda.synchronize()  
    # # print("rocblas_scaled_mm:{} us\n".format(elapsed_time4_gpu))   
    
    elapsed_time0_gpu = 0
    elapsed_time1_gpu = 0
    elapsed_time2_gpu = 0
    elapsed_time3_gpu = 0
    elapsed_time4_gpu = 0
    elapsed_time5_gpu = 0
    elapsed_time6_gpu = 0
    torch.cuda.synchronize()   
    return elapsed_time0_gpu,elapsed_time1_gpu,elapsed_time2_gpu,elapsed_time3_gpu,elapsed_time4_gpu,elapsed_time5_gpu,elapsed_time6_gpu,elapsed_time7_gpu,elapsed_time8_gpu

    

def get_triton_cache(file_path,n,k):
    #会将所报错的json文件以字典的形式return出来
    #先读取指定的文件，该文件地址不存在则会读默认路径

    cache_json_file=file_path
    
    if os.path.exists(file_path):
    #try:
        with open(cache_json_file, 'r') as file:
            cachedata = json.load(file)
    else:
        cachedata = {}
        # 写入空数据到新的JSON文件
        print("*********no bestconfig*******")
                 
    #把所有的cache解析成key:config的形式：[M_N_K]:[config]
    configs_dict={}
    for key, value in cachedata.items():
        for sub_key, sub_value in value.items():
            configs_key= f"{sub_key}_{key}"
            configs_dict[configs_key]=sub_value
    return configs_dict

def getspec_config(configs_dict,M,N,K):
    if f"{M}_{N}_{K}" in configs_dict:
        return configs_dict[f"{M}_{N}_{K}"]
    else:
        return None  

def main(n_list, k_list ,excel_name="output.xlsx"):
    session=W8a8GetCacheJSON()

    # m_list = [8192]
    # n_list = [24576]
    # k_list = [4096]
    
   
    triton_json_dir=session.triton_json_dir
    print("triton_json_dir:",triton_json_dir)
    act_cost_gpu=[]
    triton_cost_gpu=[]
    qtriton_cost_gpu=[]
    lightop_cost_gpu=[]
    qlightop_cost_gpu=[]
    rocblas_cost_gpu=[]
    qrocblas_cost_gpu=[]
    fp16gemm_cost_gpu=[] 
    int8gemm_cost_gpu=[]
    bs=[]
    _m=[]
    _k=[]
    arch_name = torch.cuda.get_device_properties("cuda").gcnArchName.split(':')[0]
    arch_cu = torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count
    name =arch_name+'_'+str(arch_cu)+'cu'
    # name=torch.cuda.get_device_properties('cuda').name.replace(" ", "_")
    for i in range(0,len(k_list),1):      
        cache_json_file=f"{triton_json_dir}/W8A8_{n_list[i]}_{k_list[i]}_{name}.json"
        configs_dict=get_triton_cache(cache_json_file,n_list[i],k_list[i])
        
        m1=[item if item < 17 else  1 << (item - 27) for item in range(1, 17)]
        m2=[item<<2  if item <17 else (item - 8)<<3  for item in range(5, 29)]
        m3=[2<<(item)  for item in range(7, 15)]
        
        m_list = m1+m2+m3
        for m in m_list:
            print("m:{} n:{} k:{} ".format(m,n_list[i],k_list[i]))
            best_config=getspec_config(configs_dict,m,n_list[i],k_list[i])
            
            if best_config:
                print("found_config:",best_config)
            else:
                print("best config does not found\n")
                #continue
            elapsed_time0_gpu,elapsed_time1_gpu,elapsed_time2_gpu,elapsed_time3_gpu,elapsed_time4_gpu,elapsed_time5_gpu,elapsed_time6_gpu,elapsed_time7_gpu,elapsed_time8_gpu=_int8_gemm_helper(m=m,n=n_list[i],k=k_list[i],per_token_act_quant=True,per_out_channel_weight_quant=True,use_bias=False,best_config=best_config)

            act_cost_gpu.append(elapsed_time0_gpu)
            lightop_cost_gpu.append(elapsed_time7_gpu)
            qlightop_cost_gpu.append(elapsed_time8_gpu)
            triton_cost_gpu.append(elapsed_time1_gpu)
            qtriton_cost_gpu.append(elapsed_time2_gpu)
            rocblas_cost_gpu.append(elapsed_time3_gpu) 
            qrocblas_cost_gpu.append(elapsed_time4_gpu) 
            fp16gemm_cost_gpu.append(elapsed_time5_gpu)   
            int8gemm_cost_gpu.append(elapsed_time6_gpu)      
            bs.append(m)
            _k.append(k_list[i])
            _m.append(n_list[i])
        
    # 创建一个包含这三个列表的 DataFrame
    df = pd.DataFrame({'n':bs,'m':_m,'k':_k,'激活量化 GPU耗时': act_cost_gpu,'w8a8_triton GPU耗时': triton_cost_gpu,'激活量化+w8a8_triton GPU耗时': qtriton_cost_gpu, 'w8a8_rocblas GPU耗时': rocblas_cost_gpu, '激活量化+w8a8_rocblas GPU耗时': qrocblas_cost_gpu,'w8a8_lightop GPU耗时': lightop_cost_gpu, '激活量化+w8a8_lightop GPU耗时': qlightop_cost_gpu,'fp16 gemm GPU耗时':fp16gemm_cost_gpu,'int8 gemm GPU耗时':int8gemm_cost_gpu})

    # 将 DataFrame 写入 Excel 文件
    df.to_excel(excel_name, index=False)

    print(f"表格已保存到 {excel_name}文件中。")

if __name__ == "__main__":

     # # # dpsk 671b
    n_list = [256, 576, 1536, 1536, 2048, 2304, 7168, 7168, 7168,]
    k_list = [7168, 7168, 1536, 7168, 512, 7168, 128, 1024, 1152,]
    main(n_list, k_list, excel_name='dpsk_671b.xlsx')


    # # qwen3-8b-tp1
    n_list = [6144, 4096, 24576, 4096,]
    k_list = [4096, 4096, 4096, 12288,]
    main(n_list, k_list, excel_name='qwen3_8b_tp1.xlsx')

    # # qwen3-32b-tp1
    n_list = [10240, 5120, 51200, 5120, ]
    k_list = [5120, 8192, 5120, 25600, ]

    main(n_list, k_list, excel_name='qwen3_32b_tp1.xlsx')

    # # qwen3-32b-tp2
    n_list = [5120, 5120, 25600, 5120,]
    k_list = [5120, 4096, 5120, 12800, ]
    main(n_list, k_list, excel_name='qwen3_32b_tp2.xlsx')

    # # qwen3-32b-tp4
    n_list = [2560, 5120, 12800, 5120,]
    k_list = [5120, 2048, 5120, 6400,]
    main(n_list, k_list, excel_name='qwen3_32b_tp4.xlsx')
    

    # qwen3-30b-A3B-tp1
    n_list = [5120, 2048]
    k_list = [2048, 4096]
    main(n_list, k_list, excel_name='qwen3_30b_A3B_tp1.xlsx')

    # qwen3-30b-A3B-tp2
    n_list = [2560, 2048]
    k_list = [2048, 2048]
    main(n_list, k_list, excel_name='qwen3_30b_A3B_tp2.xlsx')

    # qwen3-30b-A3B-tp4
    n_list = [1280, 2048]
    k_list = [2048, 1024]
    main(n_list, k_list, excel_name='qwen3_30b_A3B_tp4.xlsx')

    # # qwen3-235b-tp8
    n_list = [1280, 4096,]
    k_list = [4096, 1024,]
    main(n_list, k_list, excel_name='qwen3_235b_tp8.xlsx')

    # # qwen3-235b-tp16
    n_list = [768, 4096,]
    k_list = [4096, 512,]
    main(n_list, k_list, excel_name='qwen3_235b_tp16.xlsx')

    ## llama3.3-70b-tp2
    n_list = [28672, 8192, 5120, 8192,]
    k_list = [8192, 14336, 8192, 4096, ]
    main(n_list, k_list, excel_name='llama3.3_70b_tp2.xlsx')

    ## llama3.3-70b-tp4
    n_list = [14336, 8192, 2560, 8192,]
    k_list = [8192, 7168, 8192, 2048, ]
    main(n_list, k_list, excel_name='llama3.3_70b_tp4.xlsx')

    ## llama3.3-70b-tp8
    n_list = [7168, 8192, 1280, 8192,]
    k_list = [8192, 3584, 8192, 1024, ]
    main(n_list, k_list, excel_name='llama3.3_70b_tp8.xlsx')
