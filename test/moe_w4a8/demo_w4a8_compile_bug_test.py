import torch


import torch.nn.functional as F
from typing import  Tuple
import time
import vllm
import triton

import numpy as np
# from vllm.model_executor.layers.quantization.utils.int8_utils import per_token_group_quant_int8
# from test_gzh_fusemoe_opt3_kernel_tunning2ntszinfertest_acconly_readdata import fuse_moe
from vllm import _custom_ops as ops 
import pdb
from vllm.platforms import current_platform
import triton.language as tl
from lightop.moe import moe_gemm_w4a8
from vllm.model_executor.layers.fused_moe.fused_moe import moe_kernel_prepare_input
@triton.jit
def _per_token_group_quant_int8(
    y_ptr,
    y_q_ptr,
    y_s_ptr,
    group_size,
    M,
    K,
    N,
    SIZE,
    eps,
    int8_min,
    int8_max,
    BLOCK:tl.constexpr,
    s_sum:tl.constexpr,
):
    gid = tl.program_id(0)
    y_ptr += gid * BLOCK
    y_q_ptr += gid * BLOCK
    y_s_ptr += gid * s_sum
    cols = tl.arange(0,BLOCK)
    s_cols = tl.arange(0,s_sum)

    mask = gid * BLOCK + cols < SIZE
    y = tl.load(y_ptr + cols,mask = mask ,other=0.0).to(tl.float32)
    y = tl.reshape(y,(s_sum,128))

    _absmax = tl.maximum(tl.max(tl.abs(y),axis=1),eps)
    y_s = (_absmax / int8_max).reshape(s_sum,1)

    y_q = tl.clamp(y/y_s,int8_min,int8_max).to(y_q_ptr.dtype.element_ty)
    y_q = tl.reshape(y_q,(s_sum *128))
    y_s = tl.reshape(y_s,s_sum)
    tl.store(y_q_ptr+cols,y_q,mask=mask)
    tl.store(y_s_ptr+s_cols,y_s.to(y_s_ptr.dtype.element_ty))


def per_token_group_quant_int8(
    x : torch.Tensor,
    group_size:int,
    best_config:dict=None,
    eps:float=1e-10,
    dtype:torch.dtype = torch.int8,
) -> Tuple[torch.Tensor,torch.Tensor]:
    
    M, K = x.shape
    assert(
        K % group_size ==0
    ),"groupsize fault"
    assert x.is_contiguous(),"x is not contiguous"
    iinfo = torch.iinfo(dtype)
    int8_max = iinfo.max
    int8_min = iinfo.min
    x_q = torch.empty_like(x,device=x.device,dtype=dtype)
    N= group_size
    grid = lambda META:(triton.cdiv(x.numel(),META['BLOCK']),)
    x_s = torch.empty(
        x.shape[:-1] + (x.shape[-1] // group_size,),
        device = x.device,
        dtype = torch.float32,
    )
    if M < 32:
        config = {"BLOCK": 512, "s_sum": 4}
    elif M < 512:
        config = {"BLOCK": 1024, "s_sum": 8}
    else:
        config = {"BLOCK": 2048, "s_sum": 16}
    
    _per_token_group_quant_int8[grid](
        x,
        x_q,
        x_s,
        group_size,
        M,
        K,
        N,
        x.numel(),
        eps,
        int8_min = int8_min,
        int8_max = int8_max,
        **config,
    )
    return x_q,x_s

def from_linear(weight, w_bit, device):
    # 获取weight的原始形状
    original_shape = weight.shape
    e, n = original_shape[0], original_shape[1]
    
    w_bit = 4
    weight = weight.view(-1, weight.shape[-1]).to(torch.int32)
    pack_num = 32 // w_bit

    qweights = torch.zeros(
        (weight.shape[0], weight.shape[1] // pack_num),
        dtype=torch.int32,
        device=device,
    )
    
    for col in range(weight.shape[1] // pack_num):
        if w_bit == 4:
            # order_map = [0, 2, 4, 6, 1, 3, 5, 7]
            order_map = [1, 0, 3, 2, 5, 4, 7, 6]
        else:
            raise NotImplementedError("只支持4-bit量化")
        for i in range(pack_num):
            qweight_col = weight[:, col * pack_num + order_map[i]]
            qweights[:, col] |= qweight_col << (i * w_bit)

    qweights.view(weight.shape[0], weight.shape[1] // pack_num)
    return qweights


def to_int8(tensor: torch.Tensor):
    return torch.round(tensor.clamp(min=-128,max=127)).to(dtype=torch.int8)




# This is used by the Deepseek V2/V3/R1 series models
@torch.compile(dynamic=True, backend=current_platform.simple_compile_backend)
def grouped_topk(
    hidden_states: torch.Tensor,
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
    num_expert_group: int = 0,
    topk_group: int = 0,
):
    assert hidden_states.shape[0] == gating_output.shape[0], "Number of tokens mismatch"

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

def moe_forward_w4a8(hidden_states, w1, w2, scale_zero, scale_zero2, zero, zero2, topk_weights, topk_ids, group_size=128, gemm1_mode=11, gemm2_mode=27,gating_output = None,            
                     combined_tensor = None,combined_tensor2 = None,w11_Triton = None,w21_Triton = None,):
    
    

    topk_weights1, topk_ids1 = grouped_topk(
                hidden_states=hidden_states,
                gating_output=gating_output,
                topk=8,
                renormalize=True,
                num_expert_group=8,
                topk_group=4,
            )


    sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
            topk_ids1, 16, 256,hidden_states.shape[0])
    
    # weight13 = to_int8(torch.randn((256,512 ,3584), device=device) * 5)
    # weight2 = to_int8(torch.randn((256,7168 ,128), device=device) * 5)

    weight13_scale = (torch.randn((256, 512 ,1), device='cuda',
                           dtype=torch.float32))
    weight2_scale = (torch.randn((256, 7168, 1), device='cuda',
                           dtype=torch.float32))  
        
    hidden_states1, A_scale = moe_kernel_prepare_input(
            A=hidden_states,
            B=w11_Triton,
            A_scale=None,
            B_scale=weight13_scale,
            use_fp8_w8a8=False,
            use_int8_w8a8=True,
            use_int8_w8a16=False,
            use_int4_w4a16=False,
            per_channel_quant=True,
            block_shape=None) 
 
    # hidden_states1, A_scale = per_token_group_quant_int8(hidden_states, 128)

    m, topk = topk_ids1.shape

    k = hidden_states.shape[1]
    n = w2.shape[2] * 8
    device = hidden_states.device
    hidden_states2 = torch.ones_like(hidden_states1)
   
    gemm1_out = torch.empty((m,topk, 2 * n), device=device, dtype=torch.float16)
    gemm2_out = torch.empty((m ,topk, k), device=device, dtype=torch.float16)
    config1 = {'BLOCK_SIZE_M': 16, 'MODE': 11, 'DELTA': 1}
    config2 = {'BLOCK_SIZE_M': 16, 'MODE': 27, 'DELTA': 1}  
    print("hidden_states1",hidden_states1.size())
    print("w11_Triton",w11_Triton.size())
    print("gemm1_out",gemm1_out.size())
    print("A_scale",A_scale.size())
    print("B_scale",weight13_scale.size())
    print("sorted_token_ids",sorted_token_ids.size())
    print("expert_ids",expert_ids.size())
    print("num_tokens_post_padded",num_tokens_post_padded)
    import pdb 
    pdb.set_trace()  
    moe_gemm_w4a8(hidden_states2, 
                    w11_Triton, 
                    gemm1_out,
                    A_scale, 
                    weight13_scale,
                    None,
                    sorted_token_ids, 
                    expert_ids,
                    num_tokens_post_padded, 
                    8,
                    config1)
    


    intermediate_cache2 = torch.empty(
            (m * topk_ids.shape[1], 256),
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )

    torch.ops._C.silu_and_mul(intermediate_cache2, gemm1_out.view(-1, 512))
    # intermediate_cache3 = torch.ones_like(intermediate_cache2)
    hidden_states2, A_scale2 = moe_kernel_prepare_input(
            A=intermediate_cache2,
            B=w21_Triton,
            A_scale=None,
            B_scale=weight2_scale,
            use_fp8_w8a8=False,
            use_int8_w8a8=True,
            use_int8_w8a16=False,
            use_int4_w4a16=False,
            per_channel_quant=True,
            block_shape=None)
      
    moe_gemm_w4a8(hidden_states2, 
                    w21_Triton, 
                    gemm2_out,
                    A_scale2, 
                    weight2_scale,
                    topk_weights1,
                    sorted_token_ids, 
                    expert_ids,
                    num_tokens_post_padded, 
                    1,
                    config2) 


    out_hidden_states2 = torch.empty_like(hidden_states)
    ops.moe_sum(
                gemm2_out.view(*gemm2_out.shape),
                out_hidden_states2[0:m],
            )
    
    return gemm1_out



def setup_test_data(m, n, k, e, topk, group_size=64, dtype=torch.float16):
    """
    创建测试所需的数据
    
    参数:
        m: 批次大小
        n: 隐藏层维度
        k: 输入特征维度
        e: 专家数量
        topk: 选择的top-k专家数量
        group_size: 量化组大小
        dtype: 数据类型
        
    返回:
        创建好的测试数据
    """
    # 设置随机种子以确保结果可重现
    seed = 0
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # 创建输入张量
    hidden_states = torch.randn((m, k), device='cuda', dtype=torch.float16)  # 输入特征
    
    hidden_states = hidden_states.contiguous()
    
    # 创建第一个专家权重
    w1_fp16 = torch.randint(0, 2, (e, 2 * n, k), device='cuda').to(dtype)
    w1_fp16 = w1_fp16.contiguous()
    # 将w1_fp16转换为量化后的w1
    w1 = from_linear(w1_fp16, 4, 'cuda')
    w1 = w1.view(e, 2 * n, k//8).contiguous()

  


    scale_zero = torch.randn((e,k//group_size,2 * n),device='cuda',dtype=torch.float16)
    zero = torch.randint(0,2,(e,k//group_size,2 * n),device='cuda',dtype=torch.uint8)
    zero_int16 = zero.to(torch.int32)
    w1_zp = (zero_int16 << 16) 
    cale_int16 = scale_zero.view(torch.int16)
    result_int32 = torch.zeros_like(cale_int16, dtype=torch.int32)
    result_int32 = result_int32 | cale_int16.to(dtype=torch.int32)
    result_int32 = result_int32 &0xFFFF

    # scale_zeroINT16 =  result_int32.to(torch.int16).view(torch.float16)
    #这里负数会出问题
    combined_tensor = (w1_zp | result_int32).contiguous()  

    w2_fp16 = torch.randint(0, 2, (e, k, n), device='cuda').to(dtype)
    w2_fp16 = w2_fp16.contiguous()
    # 将w2_fp16转换为量化后的w2
    w2 = from_linear(w2_fp16, 4, 'cuda')
    w2 = w2.view(e, k, n//8).contiguous()

    scale_zero2 = torch.randn((e, n//group_size, k), device='cuda', dtype=torch.float16).contiguous()
    zero2 = torch.randint(0,2,(e, n//group_size, k), device='cuda', dtype=torch.uint8).contiguous()
    # print("zero2",zero2)
    zero2_int16 = zero2.to(torch.int32)
    w2_zp = (zero2_int16 << 16) 
    cale_int162 = scale_zero2.view(torch.int16)
    result_int322 = torch.zeros_like(cale_int162, dtype=torch.int32)
    result_int322 = result_int322 | cale_int162.to(dtype=torch.int32)
    result_int322 = result_int322 &0xFFFF
    combined_tensor2 = (w2_zp | result_int322).contiguous()  
    # print("scale_zero2 size",scale_zero2.size())
    # 创建路由信息
    gating_output = torch.randn((m, e), device='cuda', dtype=dtype)      # 门控输出
    topk_weights = torch.softmax(gating_output, dim=-1)[:, :topk].to(dtype=torch.float32).contiguous()        # top-k权重
    topk_ids = torch.randint(0, e, (m, topk), device='cuda', dtype=torch.int32).contiguous()            # top-k专家索引

    binary_str = "00000001000000010000000100000001"

    binary_str2 = "00010001000100010001000100010001"
    binary_str3 = "00000001"
    # 将二进制字符串转换为整数
    value_int = int(binary_str2, 2)
    value_int2 = int(binary_str3, 2)
    w11 = torch.full((256,512,896), fill_value=value_int, dtype=torch.int32,device='cuda')
    w21 = torch.full((256,7168,32), fill_value=value_int, dtype=torch.int32,device='cuda')
    w11_Triton = torch.full((256,512,3584), fill_value=value_int2, dtype=torch.int8,device='cuda')
    w21_Triton = torch.full((256,7168,128), fill_value=value_int2, dtype=torch.int8,device='cuda')





    return {
        'hidden_states': hidden_states,
        'w1':w11,
        'w2': w21,
        'w1_fp16': w1_fp16,
        'w2_fp16': w2_fp16,
        'scale_zero': scale_zero,
        'scale_zero2': scale_zero2,
        'zero': zero,
        'zero2': zero2,
        'topk_weights': topk_weights,
        'topk_ids': topk_ids,
        'gating_output':gating_output,
        'combined_tensor':combined_tensor,
        'combined_tensor2':combined_tensor2,
        'w11_Triton':w11_Triton,
        'w21_Triton':w21_Triton,
    }



enable_moe_align_block_size_triton = 0
def moe_align_block_size(
    topk_ids: torch.Tensor, block_size: int, num_experts: int,num_tokens:int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

    if num_tokens < block_size:
        max_num_tokens_padded = min(topk_ids.numel()*block_size,topk_ids.numel() + num_experts * (block_size - 1))
    else:
        max_num_tokens_padded = topk_ids.numel() + num_experts * (block_size - 1)
    
    sorted_ids = torch.empty(
        (max_num_tokens_padded,), dtype=torch.int32, device=topk_ids.device
    )
    sorted_ids.fill_(topk_ids.numel())
    max_num_m_blocks = triton.cdiv(max_num_tokens_padded, block_size)
    expert_ids = torch.empty(
        (max_num_m_blocks,), dtype=torch.int32, device=topk_ids.device
    )
    num_tokens_post_pad = torch.empty((1), dtype=torch.int32, device=topk_ids.device)
    ops.sgl_moe_align_block_size(
            topk_ids,
            num_experts,
            block_size,
            sorted_ids,
            expert_ids,
            num_tokens_post_pad,
            # token_cnts_buffer,
            # cumsum_buffer,
        )
    return sorted_ids, expert_ids, num_tokens_post_pad

def run_w4a8_benchmark(data, group_size, gemm1_mode, gemm2_mode, iterations=100):

    hidden_states = data['hidden_states']
    w1 = data['w1']
    w2 = data['w2']
    scale_zero = data['scale_zero']
    scale_zero2 = data['scale_zero2']
    zero = data['zero']
    zero2 = data['zero2']

    topk_weights = data['topk_weights']
    topk_ids = data['topk_ids']
    gating_output = data['gating_output']
    combined_tensor = data['combined_tensor']
    combined_tensor2 =  data['combined_tensor2']
    w11_Triton =    data['w11_Triton']
    w21_Triton =    data['w21_Triton']

    print("执行W4A8版MOE前向传播...")
    w4a8_output = moe_forward_w4a8(
        hidden_states, 
        w1, 
        w2, 
        scale_zero, 
        scale_zero2,
        zero,
        zero2, 
        topk_weights, 
        topk_ids, 
        group_size=group_size,
        gemm1_mode=gemm1_mode, 
        gemm2_mode=gemm2_mode,
        gating_output = gating_output,
        combined_tensor = combined_tensor,
        combined_tensor2 = combined_tensor2,
        w11_Triton = w11_Triton,
        w21_Triton = w21_Triton,
    )
    print(f"W4A8版本输出形状: {w4a8_output.shape}")
    
    return 0

def main():
    """主函数，执行所有测试流程"""
    # 设置基本参数
    m, n, k, e, topk = 32, 64, 7168, 256, 8
    group_size = 128
    dtype = torch.float16
    gemm2_mode = 146 # 根据profiler得到的最佳mode
    gemm1_mode = 75
    iterations = 100
    
    data2 = setup_test_data(16, 256, 7168, 256, 8, 128, dtype)

    
    us = run_w4a8_benchmark(
                    data2, group_size, gemm1_mode, gemm2_mode, iterations
                )
    # w4a8_output, w4a8_avg_time = run_w4a8_benchmark(
    #     data2, group_size, gemm1_mode, gemm2_mode, iterations
    # )
    # bi = None
    # bj = None
    # best_time=float('inf')
    # for i in range(11,12,1):
    #     for j in range(27,28,1):
    #         gemm1_mode = i
    #         gemm2_mode = j
    # #         # print("i",i)
    # #         # print("j",j)
    #         try:
    #             us = run_w4a8_benchmark(
    #                 data2, group_size, gemm1_mode, gemm2_mode, iterations
    #             )
    #         except triton.runtime.autotuner.OutOfResources:
    #                     # Some configurations may be invalid and fail to compile.
    #                     # 捕获lds资源溢出错误
    #             continue
    #         if(us < best_time):
    #             bi = i
    #             bj = j
    #             best_time = us 
    # print("kernel best ms:",best_time)
    # print("best i :",bi)
    # print("best j :",bj)
    # run_w4a8_benchmark(
    #             data2, group_size, gemm1_mode, gemm2_mode, iterations
    #         )
    # print("run_w4a8_benchmark w4a8_output ",w4a8_output)
    # print("run_w4a8_benchmark w4a8_avg_time ",w4a8_avg_time)
    
    print("\n完成所有操作！")

if __name__ == "__main__":
    main() 