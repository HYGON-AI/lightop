import torch
torch.set_printoptions(precision=4, profile="default", sci_mode=False)
import time
import pandas as pd
import functools
import json
import os
import math
from typing import Any, Callable, Dict, List, Optional, Tuple,Type,Any
from vllm import _custom_ops as ops
import triton
import itertools
from lightop import moe_groupgemm_w4a8
from lightop import get_moe_groupgemm_config_w4a8
import triton.language as tl
import torch.nn.functional as F
import ast  
import torch.multiprocessing as mp
from lmslim.layers.gemm.int8_utils import (
    per_token_group_quant_int8, per_token_quant_int8)
import argparse
from w4a8_moe_groupgemm_marlin_quant import w4a8_2_marlin_weight  # pyright: ignore[reportMissingImports]

from vllm.platforms import current_platform
device_name = current_platform.get_device_name().replace(" ", "_")
num_cus= torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count

def to_int8(tensor: torch.Tensor):
    return torch.round(tensor.clamp(min=-128, max=127)).to(dtype=torch.int8)

def unpack_int8_to_int4(tensor_int8: torch.Tensor) -> torch.Tensor:
    if tensor_int8.dtype != torch.int8:
        raise ValueError("Input tensor must be of type torch.int8")

    E,N, K_half = tensor_int8.shape
    tensor_int8 = tensor_int8.to(torch.int8)
    high4 = tensor_int8 & 0XF0
    low4 = (tensor_int8 << 4) & 0xF0
    unpacked = torch.empty((E, N, K_half * 2), dtype=torch.int8, device=tensor_int8.device)
    unpacked[:, :, 0::2] = high4.to(torch.int8)
    unpacked[:, :, 1::2] = low4.to(torch.int8)

    return unpacked

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

def fused_experts_impl_w4a8_ep(hidden_states,weight13, weight2, w13_scale, w2_scale, tokens_per_expert):
    
    # w13:[num_experts/ep, N, K // 2] [16, 4096, 3584]
    # w2: [num_experts/ep, K, N//2//2] [16, 7168, 1024]
    # w13_scale:[num_experts/ep, N, 1] [16, 4096, 1]
    # w2_scale: [num_experts/ep, K, 1] [16, 7168, 1]
            
    outputs = []
    start_idx = 0
    w13 = unpack_int8_to_int4(weight13)
    w2 = unpack_int8_to_int4(weight2)
    for i, num_tokens in enumerate(tokens_per_expert):
        end_idx = start_idx + num_tokens
        if num_tokens == 0:
            continue
        q_a1, qa1_scale = per_token_quant_int8(hidden_states[start_idx:end_idx])

        current_w13 = w13[i]
        current_w2 = w2[i]
        current_w13_s = w13_scale[i]
        current_w2_s = w2_scale[i]
        inter_out = native_w8a8_per_token_matmul(
            q_a1, current_w13 , qa1_scale, current_w13_s, output_dtype=hidden_states.dtype
        )
        d = inter_out.shape[-1] // 2
        act_out = F.silu(inter_out[..., :d]) * inter_out[..., d:]
        act_out_q, act_out_s = per_token_quant_int8(act_out)

        expert_out= native_w8a8_per_token_matmul(
            act_out_q, current_w2, act_out_s, current_w2_s, output_dtype=hidden_states.dtype
        )
        outputs.append(expert_out)
        start_idx = end_idx
    if len(outputs) > 0:
        expert_output = torch.cat(outputs, dim=0)
    else:
        assert hidden_states.numel() == 0, f"sorted_tokens: should be empty, but got {hidden_states.shape}"
        expert_output = hidden_states

    return expert_output
           
def fused_experts_impl_w4a8_ep_correctness(hidden_states,weight13, weight2, w13_scale, w2_scale, tokens_per_expert):
    
    # w13:[num_experts/ep, N, K // 2] [16, 4096, 3584]
    # w2: [num_experts/ep, K, N//2//2] [16, 7168, 1024]
    # w13_scale:[num_experts/ep, N, 1] [16, 4096, 1]
    # w2_scale: [num_experts/ep, K, 1] [16, 7168, 1]
            
    outputs = []
    start_idx = 0
    w13 = unpack_int8_to_int4(weight13)
    w2 = unpack_int8_to_int4(weight2)
    inter_out_gemm1 = []
    inter_out_gemm2 = []  # pyright: ignore[reportUnusedVariable]
    act_out_gemm2 = []
    q_a1_all, qa1_scale_all = per_token_quant_int8(hidden_states)
    # q_a1_all = torch.ones_like(q_a1_all)*5
    # qa1_scale_all = torch.ones_like(qa1_scale_all)
    # w13_scale = torch.ones_like(w13_scale)
    # import pdb
    # pdb.set_trace()
    for i, num_tokens in enumerate(tokens_per_expert):
        end_idx = start_idx + num_tokens
        if num_tokens == 0:
            continue
        # q_a1, qa1_scale = per_token_quant_int8(hidden_states[start_idx:end_idx])
        q_a1 = q_a1_all[start_idx:end_idx]
        qa1_scale = qa1_scale_all[start_idx:end_idx]

        current_w13 = w13[i]
        current_w2 = w2[i]
        current_w13_s = w13_scale[i]
        current_w2_s = w2_scale[i]
        inter_out = native_w8a8_per_token_matmul(
            q_a1, current_w13 , qa1_scale, current_w13_s, output_dtype=hidden_states.dtype
        )
        inter_out_gemm1.append(inter_out)
        d = inter_out.shape[-1] // 2
        act_out = F.silu(inter_out[..., :d]) * inter_out[..., d:]
        act_out_gemm2.append(act_out)
        act_out_q, act_out_s = per_token_quant_int8(act_out)
        # import pdb
        # pdb.set_trace()

        expert_out= native_w8a8_per_token_matmul(
            act_out_q, current_w2, act_out_s, current_w2_s, output_dtype=hidden_states.dtype
        )
        outputs.append(expert_out)
        start_idx = end_idx
    if len(outputs) > 0:
        expert_output = torch.cat(outputs, dim=0)
    else:
        assert hidden_states.numel() == 0, f"sorted_tokens: should be empty, but got {hidden_states.shape}"
        expert_output = hidden_states

    if len(inter_out_gemm1) > 0:
        inter_out_gemm1_out_ref = torch.cat(inter_out_gemm1, dim=0)
        act_out_cat = torch.cat(act_out_gemm2, dim=0)
        gemm1_out_cuda = torch.zeros_like(inter_out_gemm1_out_ref)
        gemm2_out_cuda = torch.zeros_like(expert_output)
        act_out_q, act_out_s = per_token_quant_int8(act_out_cat)

    E = len(tokens_per_expert)
    M = hidden_states.size(0)
    N1 = weight13.size(1)
    K1 = hidden_states.size(1)//2
    N2 = K1*2
    K2 = N1 // 4

    dtype = hidden_states.dtype
    config1, config2, status = get_moe_groupgemm_config_w4a8(E, M, N1, K1, N2, K2, device_name, num_cus, dtype, None, None)
        
    problem_size = tokens_per_expert
    problem_size_max = problem_size.max()
    expert_offsets = list(itertools.accumulate(tokens_per_expert, initial=0))[:-1]
    assert len(expert_offsets) == len(problem_size)
    # tensor_problem_size = torch.tensor(problem_size, dtype=torch.int32, device=hidden_states.device)
    tensor_problem_size = problem_size.to(device=hidden_states.device, dtype=torch.int32)
    tensor_expert_offsets = torch.tensor(expert_offsets, dtype=torch.int32, device=hidden_states.device)

    print("tensor_problem_size: ", tensor_problem_size)
    print("tensor_expert_offsets: ", tensor_expert_offsets)
    print("problem_size_max: ", problem_size_max)
    
    # # gemm1
    moe_groupgemm_w4a8(q_a1_all, 
                        weight13, 
                        gemm1_out_cuda,
                        qa1_scale_all, 
                        w13_scale,
                        tensor_problem_size, 
                        tensor_expert_offsets,
                        problem_size_max,
                        config1)
    torch.cuda.synchronize()
    if not (torch.allclose(inter_out_gemm1_out_ref, gemm1_out_cuda, rtol=1e-2, atol=1e-2)):
        print("error moe_groupgemm_w4a8 gemm1 精度检查不合格！！！")
    else:
        print("success moe_groupgemm_w4a8 gemm1 精度检查合格")
    
    # import pdb
    # pdb.set_trace()
    # gemm2
    moe_groupgemm_w4a8(act_out_q, 
                        weight2, 
                        gemm2_out_cuda,
                        act_out_s, 
                        w2_scale,
                        tensor_problem_size, 
                        tensor_expert_offsets,
                        problem_size_max,
                        config2)
    if not (torch.allclose(expert_output, gemm2_out_cuda, rtol=1e-2, atol=1e-2)):
        print("error moe_groupgemm_w4a8 gemm2 精度检查不合格！！！")
        # torch.set_printoptions(profile="full")  
        # fasm = open('asm_output.txt', 'w')   
        # print(f'asm_output: {out_cuda}', file=fasm)
        # fasm.close
        # ftorch = open('torch_output.txt', 'w')   
        # print(f'torch_output: {inter_out_gemm1_out_ref}', file=ftorch)
        # ftorch.close
    else:
        print("success moe_groupgemm_w4a8 gemm2 精度检查合格")
    
    return expert_output

def fused_experts_impl_w4a8_ep_dobench(hidden_states,weight13, weight2, w13_scale, w2_scale, tokens_per_expert):
    
    # w13:[num_experts/ep, N, K // 2] [16, 4096, 3584]
    # w2: [num_experts/ep, K, N//2//2] [16, 7168, 1024]
    # w13_scale:[num_experts/ep, N, 1] [16, 4096, 1]
    # w2_scale: [num_experts/ep, K, 1] [16, 7168, 1]
            
    outputs = []
    start_idx = 0
    
    # -------- 处理 gemm1 权重矩阵 ------------------
    w1_marlin_list = []
    for i in range(weight13.shape[0]):
        w1_marlin_in = w4a8_2_marlin_weight(weight13[i])
        w1_marlin_list.append(w1_marlin_in)
    weight13_marlin = torch.stack(w1_marlin_list, dim=0)
    
    # -------- 处理 gemm2 权重矩阵 ------------------
    w2_marlin_list = []
    for i in range(weight2.shape[0]):
        w2_marlin_in = w4a8_2_marlin_weight(weight2[i])
        w2_marlin_list.append(w2_marlin_in)
    weight2_marlin = torch.stack(w2_marlin_list, dim=0)
    
    w13 = unpack_int8_to_int4(weight13)
    w2 = unpack_int8_to_int4(weight2)
    inter_out_gemm1 = []
    act_out_gemm2 = []
    torch_gemm1_cost_list = []
    torch_gemm2_cost_list = []
    q_a1_all, qa1_scale_all = per_token_quant_int8(hidden_states)

    for i, num_tokens in enumerate(tokens_per_expert):
        end_idx = start_idx + num_tokens
        if num_tokens == 0:
            continue
        q_a1 = q_a1_all[start_idx:end_idx]
        qa1_scale = qa1_scale_all[start_idx:end_idx]

        current_w13 = w13[i]
        current_w2 = w2[i]
        current_w13_s = w13_scale[i]
        current_w2_s = w2_scale[i]
        inter_out = native_w8a8_per_token_matmul(
            q_a1, current_w13 , qa1_scale, current_w13_s, output_dtype=hidden_states.dtype
        )
        # warp
        for i in range(10):
            inter_out = native_w8a8_per_token_matmul(q_a1, current_w13 , qa1_scale, current_w13_s, output_dtype=hidden_states.dtype)
        cost1=triton.testing.do_bench(lambda: native_w8a8_per_token_matmul(q_a1, current_w13 , qa1_scale, current_w13_s, output_dtype=hidden_states.dtype),
                                             quantiles=None, return_mode="mean")*1000
        torch_gemm1_cost_list.append(cost1)
        
        inter_out_gemm1.append(inter_out)
        d = inter_out.shape[-1] // 2
        act_out = F.silu(inter_out[..., :d]) * inter_out[..., d:]
        act_out_gemm2.append(act_out)
        act_out_q, act_out_s = per_token_quant_int8(act_out)
        expert_out= native_w8a8_per_token_matmul(
            act_out_q, current_w2, act_out_s, current_w2_s, output_dtype=hidden_states.dtype
        )
        # warp
        for i in range(10):
            expert_out= native_w8a8_per_token_matmul(act_out_q, current_w2, act_out_s, current_w2_s, output_dtype=hidden_states.dtype)
        cost2=triton.testing.do_bench(lambda: native_w8a8_per_token_matmul(act_out_q, current_w2, act_out_s, current_w2_s, output_dtype=hidden_states.dtype),
                                             quantiles=None, return_mode="mean")*1000
        torch_gemm2_cost_list.append(cost2)
        
        outputs.append(expert_out)
        start_idx = end_idx
    if len(outputs) > 0:
        expert_output = torch.cat(outputs, dim=0)
    else:
        assert hidden_states.numel() == 0, f"sorted_tokens: should be empty, but got {hidden_states.shape}"
        expert_output = hidden_states

    if len(inter_out_gemm1) > 0:
        inter_out_gemm1_out_ref = torch.cat(inter_out_gemm1, dim=0)
        act_out_cat = torch.cat(act_out_gemm2, dim=0)
        gemm1_out_cuda = torch.zeros_like(inter_out_gemm1_out_ref)
        gemm2_out_cuda = torch.zeros_like(expert_output)
        act_out_q, act_out_s = per_token_quant_int8(act_out_cat)

    torch_gemm1_cost=torch.sum(torch.tensor(torch_gemm1_cost_list)).item()
    torch_gemm2_cost=torch.sum(torch.tensor(torch_gemm2_cost_list)).item()

    E = len(tokens_per_expert)
    M = hidden_states.size(0)
    N1 = weight13.size(1)
    K1 = hidden_states.size(1)//2
    N2 = K1*2
    K2 = N1 // 4

    dtype = hidden_states.dtype
    config1, config2, status = get_moe_groupgemm_config_w4a8(E, M, N1, K1, N2, K2, device_name, num_cus, dtype, None, None)
        
    problem_size = tokens_per_expert
    expert_offsets = list(itertools.accumulate(tokens_per_expert, initial=0))[:-1]
    assert len(expert_offsets) == len(problem_size)
    tensor_problem_size = torch.tensor(problem_size, dtype=torch.int32, device=hidden_states.device)
    tensor_expert_offsets = torch.tensor(expert_offsets, dtype=torch.int32, device=hidden_states.device)
    torch.cuda.synchronize()
    
    # gemm1
    # warp
    for i in range(10):
        moe_groupgemm_w4a8(q_a1_all, 
                            weight13_marlin, 
                            gemm1_out_cuda,
                            qa1_scale_all, 
                            w13_scale,
                            tensor_problem_size, 
                            tensor_expert_offsets,
                            config1)
        
    asm_gemm1_cost=triton.testing.do_bench(lambda: moe_groupgemm_w4a8(q_a1_all, 
                                                                        weight13_marlin, 
                                                                        gemm1_out_cuda,
                                                                        qa1_scale_all, 
                                                                        w13_scale,
                                                                        tensor_problem_size, 
                                                                        tensor_expert_offsets,
                                                                        config1),quantiles=None, return_mode="mean")*1000
    
    # gemm2
    # warmup
    for i in range(10):
        moe_groupgemm_w4a8(act_out_q, 
                            weight2_marlin, 
                            gemm2_out_cuda,
                            act_out_s, 
                            w2_scale,
                            tensor_problem_size, 
                            tensor_expert_offsets,
                            config2)
        
    asm_gemm2_cost=triton.testing.do_bench(lambda: moe_groupgemm_w4a8(act_out_q, 
                                                                        weight2_marlin, 
                                                                        gemm2_out_cuda,
                                                                        act_out_s, 
                                                                        w2_scale,
                                                                        tensor_problem_size, 
                                                                        tensor_expert_offsets,
                                                                        config2),quantiles=None, return_mode="mean")*1000

    return expert_output, torch_gemm1_cost, torch_gemm2_cost, asm_gemm1_cost, asm_gemm2_cost

def _fused_moe_helper(m: int,
                             n: int,
                             k: int,
                             n2:int,
                             k2:int,
                             E:int,
                             out_dtype: Type[torch.dtype] = torch.float16,
                             device: str = "cuda",
                             ):
    # Test for a cutlass kernel with per-token activation quantization
    # and per-output channel weight quantization.
    input = (torch.randn((m, n2), device=device) * 5).to(dtype=out_dtype)

    weight13 = to_int8(torch.randn((E,n ,k // 2), device=device) )
    weight2 = to_int8(torch.randn((E,n2 ,k2 // 2), device=device))

    weight13_scale = (torch.randn((E,n,1), device=device, dtype=torch.float32))
    weight2_scale = (torch.randn((E,n2,1), device=device, dtype=torch.float32))        
    
    cuts = torch.randint(1, m + 1, (E - 1,))
    cuts, _ = torch.sort(cuts)
    tokens_per_expert = torch.diff(torch.cat([torch.tensor([0]), cuts, torch.tensor([m])]))
    return input,weight13,weight2,weight13_scale,weight2_scale,tokens_per_expert




def main():
    # PyTorch CPU随机种子
    seed = 0
    torch.manual_seed(seed)
    
    # PyTorch GPU随机种子
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 多GPU情况
    # 创建解析器对象
    parser = argparse.ArgumentParser(description='这是一个示例脚本的描述')
    
    # 添加参数
    parser.add_argument('--E', type=int, default=16, help='E 参数')
    parser.add_argument('--n1-list', type=str, required=True, 
                       help='N1 参数列表，如 "[1536,768,384,384,192]"')
    parser.add_argument('--n2-list', type=str, required=True,
                       help='N2 参数列表，如 "[2048,2048,2048,4096,4096]"')
    parser.add_argument('--k-list', type=str, required=True,
                       help='K 参数列表，如 "[768,374,192,192,96]"')
    parser.add_argument('--topk', type=int, default=8, 
                       help='topk 参数')
    parser.add_argument('--out-dtype', type=torch.dtype, default=torch.bfloat16,
                       help='out-dtype')
    # parser.add_argument('--use-channel-wise',action='store_true', default=False,
    #                    help='use-channel-wise')

    # 解析参数
    args = parser.parse_args()
 
    def parse_list(list_str):
        try:
            return ast.literal_eval(list_str)
        except (ValueError, SyntaxError) as e:
            raise ValueError(f"无法解析列表参数: {list_str}。错误: {str(e)}")
    
    
    # m_list=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,32,64,128,256,512,1024,2048,4096]
    m_list=[2048]
    E=args.E 
    out_dtype=args.out_dtype
    
    n_list_1=parse_list(args.n1_list) #[256,512]
    k_list_1=parse_list(args.n2_list) #[7168,7168]
    
    n_list_2=parse_list(args.n2_list) #[7168,7168]
    k_list_2=parse_list(args.k_list) # [128,256]
    block_size=None
    
    topk=args.topk
    best_config = None

    torch_gemm1_cost_list=[]
    torch_gemm2_cost_list=[]
    asm_gemm1_cost_list=[]
    asm_gemm2_cost_list=[]
    _m=[]
    _n1=[]
    _n2=[]
    _k=[]  
    
    
    for i in range(0,len(k_list_1),1):

        for m in m_list:
            print("m:{} n1:{} n2:{} k:{} ".format(m,n_list_1[i],n_list_2[i],k_list_2[i]))
            input,weight13,weight2,weight13_scale,weight2_scale,tokens_per_expert=_fused_moe_helper(m=m,n=n_list_1[i],k=k_list_1[i],n2=n_list_2[i],k2=k_list_2[i],E=E,out_dtype=out_dtype)
            native_cost=triton.testing.do_bench(lambda:fused_experts_impl_w4a8_ep(
                                                            input,weight13, weight2, weight13_scale, weight2_scale, tokens_per_expert
                                                            ),quantiles=None, return_mode="mean")*1000
            
            expert_output, torch_gemm1_cost, torch_gemm2_cost, asm_gemm1_cost, asm_gemm2_cost=fused_experts_impl_w4a8_ep_dobench(input,weight13, weight2, weight13_scale, weight2_scale, tokens_per_expert)
            torch_gemm1_cost_list.append(torch_gemm1_cost) 
            torch_gemm2_cost_list.append(torch_gemm2_cost) 
            asm_gemm1_cost_list.append(asm_gemm1_cost) 
            asm_gemm2_cost_list.append(asm_gemm2_cost) 
            
            _m.append(m)
            _n1.append(n_list_1[i])
            _n2.append(n_list_2[i])
            _k.append(k_list_2[i])   

    # 创建一个包含这三个列表的 DataFrame
    df = pd.DataFrame({'m':_m,'n1':_n1,'n2':_n2,'k':_k,'torch gemm1 耗时(us)': torch_gemm1_cost_list,'torch gemm2 耗时(us)': torch_gemm2_cost_list,'asm gemm1 耗时(us)': asm_gemm1_cost_list,'asm gemm2 耗时(us)': asm_gemm2_cost_list})

    # 将 DataFrame 写入 Excel 文件
    excel_name="channel-wsie_output.xlsx"
        
    df.to_excel(excel_name, index=False)

    print("表格已保存到 {}文件中。".format(excel_name))           
            
        

if __name__ == "__main__":
    main()
    #dp2
    #HIP_VISIBLE_DEVICES=3 python moe_groupgemm_w4a8/w4a8_moe_benchmark_ep.py  --E 16 --n1-list '[4096]'  --n2-list '[7168]' --k-list '[2048]'
    #dp4
    #HIP_VISIBLE_DEVICES=3 python moe_groupgemm_w4a8/w4a8_moe_benchmark_ep.py  --E 8 --n1-list '[4096]'  --n2-list '[7168]' --k-list '[2048]'