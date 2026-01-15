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

gemm1_compare_failed_list = []
gemm2_compare_failed_list = []

def to_int8(tensor: torch.Tensor):
    return torch.round(tensor.clamp(min=-128, max=127)).to(dtype=torch.int8)

def batch_rocblas_scaled_mm(a: torch.Tensor,        # [n, b, p]
                            b: torch.Tensor,        # [n, l, p]
                            scale_a: torch.Tensor,  # [n, b, 1] 
                            scale_b: torch.Tensor,  # [n, l, 1]
                            out_dtype: torch.dtype = torch.bfloat16,
                            bias: Optional[torch.Tensor] = None) -> torch.Tensor: # [b, l]
    #scale_a.shape=[b,m,1]   fp32
    #scale_b.shape=[b,n,1]   fp32
    #a.shape=[b,m,k]         int8
    #b.shape=[b,n,k]         int8
    #output.shape=[b,m,n]    bf16/fp16
    #bias [b,n]              bf16/fp16

    batch_size = a.shape[0]
    output = torch.empty(batch_size, a.shape[1], b.shape[1], dtype=out_dtype)
    
    for i in range(batch_size):
        scales = scale_a[i] * scale_b[i].T   # [b, l]
        # 使用 torch 实现 int8 矩阵乘法，输出 int32
        # a[i]: [b, p] int8, b[i]: [l, p] int8
        # 计算 a[i] @ b[i].T，结果: [b, l] int32
        gemmoutput = torch.matmul(a[i].cpu().to(dtype=torch.int32), b[i].T.cpu().to(dtype=torch.int32))  # [b, l]
        output[i] = (scales.cpu() * gemmoutput.to(dtype=torch.float32)).to(out_dtype)
        if bias is not None:
            output[i] = output[i] + bias[i].cpu()
    return output.cuda()

def fused_bmm_trans_impl_int8(input: torch.Tensor,                              # b，n, p
                                w_uk: torch.Tensor,                             # n, l, p
                                input_scale: Optional[torch.Tensor] = None,     # n, b, 1
                                w_uk_scale: Optional[torch.Tensor] = None,      # n, l, 1
                                bias: Optional[torch.Tensor] = None,            # b, l
                                config: Optional[Dict[str, Any]] = None,
                                mode: Optional[List[int]]=None):
    
    # input = torch.ones_like(input)
    # w_uk的shape是[n,l,p]，第0,1维保持不变，第2维从0-p-1增长
    # w_uk = torch.arange(w_uk.shape[2], dtype=w_uk.dtype, device=w_uk.device).view(1, 1, -1).expand_as(w_uk)
    # w_uk = torch.ones_like(w_uk)*2
    # input_scale = torch.ones_like(input_scale)
    # w_uk_scale = torch.ones_like(w_uk_scale)

    inputc_clone = input.clone()
    torch_input = inputc_clone.transpose(0,1).contiguous()  # [b, n, p] -> [n, b, p]
    torch_uk = w_uk.clone()
    torch_input_scale = input_scale.clone()
    torch_w_uk_scale = w_uk_scale.clone()
    torch_bias = bias.clone() if bias is not None else None
    
    
    # fused_bmm_output 应该是 [n, b, l] 的形状
    fused_torch_output = batch_rocblas_scaled_mm(torch_input, torch_uk, torch_input_scale, torch_w_uk_scale, out_dtype=torch.bfloat16, bias=torch_bias)   # [n, b, l]
    torch_output = fused_torch_output.transpose(0,1).contiguous() # [b, n, l]
    cuda_output = torch.zeros_like(torch_output, dtype=torch.bfloat16)
    
    # 验证lightop kernel
    # torch.set_printoptions(profile="full") 
    # print("w_ukw_uk: ", w_uk[0,:15,:64])
    fused_bmm(input, w_uk, cuda_output, input_scale, w_uk_scale, bias, mode)
    torch.cuda.synchronize()
    
    if not (torch.allclose(torch_output, cuda_output, rtol=1e-2, atol=1e-2, equal_nan=True)):
        print(f"kernel 精度检查不合格!!!")
        
        torch.set_printoptions(profile="full")  
        ftriton = open('cuda_output.txt', 'w')   
        print(f'cuda_output: {cuda_output}', file=ftriton)
        ftriton.close
        ftorch = open('bench_output.txt', 'w')   
        print(f'bench_output: {torch_output}', file=ftorch)
        ftorch.close
    else:
        print(f"kernel 精度检查合格.")
    return True

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
    input = torch.randint(-128, 127, (batch, N, P), device=device, dtype=torch.int8)  # [b, n, p]
    # w_uk_t = torch.randint(-128, 127, (N, P ,L), device=device, dtype=torch.int8)    # [n, p, l]
    w_uk = torch.randint(-128, 127, (N, L, P), device=device, dtype=torch.int8)       # [n, l, p]
    input_scale = (torch.randn((N, batch ,1), device=device,                          # [n, b, 1]
                           dtype=torch.float32))
    w_uk_scale = (torch.randn((N, L, 1), device=device,                             # [n, l, 1]
                           dtype=torch.float32))   
    
    return input, w_uk, input_scale, w_uk_scale

def run_trans_bmm_trans_vaild():
    # batch_list=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,32,64,128,256,512,1024,2048,4096,8192,16384,32768]
    # batch_list=[1, 8, 16, 32, 33, 64, 65, 128] #decode
    # batch_list=[256,512,1024,2048,4096,8192,16384] #prefill
    # batch_list=[144,160,176,192,208,224,240,240,256,288,320,]
    batch_list=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,32]
    # batch_list = [16384]
    
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

    # mode_ranges = [i for i in range(len(blockM_configs))]
    # mode_ranges = [mode for mode in blockM_configs.keys()]
    mode_ranges = [0]
    # compare_fail = [42,43,46,52,54,60]
    # compare_fail = [64,66,67,69,70,71,77,78,79,80,81,82,83,84,85,94]
    # compare_fail = [6,11,13,42,43,46,52,54,60,64,66,67,69,70,71,77,78,79,80,81,82,83,84,85,94]
    #deepseek-471b-tp16

    # # decode corrtnese
    # mode_ranges = [i for i in range(len(blockM_configs_decode))]
    # exclude_modes = [13, 32, 33, 34, 35]
    # #mode_ranges = [3,4,6,8,10,12,14]
    # #exclude_modes = [13] # nk 4096 192 gemm2<16

    #mode_ranges = [3]

    
    for i in range(0,len(p_list),1):
        for batch_size in batch_list:
            for mode in mode_ranges:

                print("mode:{} batch:{} n:{} p:{} l:{} ".format(mode, batch_size, n_list[i],p_list[i],l_list[i]))
                input, w_uk_t, input_scale, w_uk_t_scale=_fused_bmm_helper(batch_size, n_list[i], p_list[i], l_list[i])
                # triton moe: output
                output=fused_bmm_trans_impl_int8(input, w_uk_t, input_scale, w_uk_t_scale, bias=None, mode=mode)

                     
    print("gemm1_compare_failed_list: ", gemm1_compare_failed_list)
    print("gemm2_compare_failed_list: ", gemm2_compare_failed_list)
            
def main():
    # run_tuning()
    run_trans_bmm_trans_vaild()

if __name__ == "__main__":
    main()
    # HIP_VISIBLE_DEVICES=1 python3 test_fused_trans_bmm_trans.py