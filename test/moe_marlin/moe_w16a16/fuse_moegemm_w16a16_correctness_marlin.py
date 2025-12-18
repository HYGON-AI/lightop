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
# from lmslim.layers.fused_moe.fuse_moe_int8 import moe_kernel_prepare_input
from vllm import _custom_ops as ops
# from test_int8_kernel import torch_w8a8_per_column_moe
# import _custom_ops as ops
from lightop import moe_gemm_w8a8
from marlin_quant import w16a16_marlin_weight
from lmslim.layers.gemm.int8_utils import (
    per_token_group_quant_int8, per_token_quant_int8)

from fused_moe_valid_w16a16_marlin import fused_moe_valid_w16a16_marlin, blockM_all_configs

gemm1_compare_failed_list = []
gemm2_compare_failed_list = []

def grouped_topk(hidden_states: torch.Tensor,
                 gating_output: torch.Tensor,
                 topk: int,
                 renormalize: bool,
                 num_expert_group: int = 0,
                 topk_group: int = 0,
                 scoring_func: str = "softmax",
                 e_score_correction_bias: Optional[torch.Tensor] = None):

    assert hidden_states.shape[0] == gating_output.shape[0], (
        "Number of tokens mismatch")

    if scoring_func == "softmax":
        scores = torch.softmax(gating_output, dim=-1)
    elif scoring_func == "sigmoid":
        scores = gating_output.sigmoid()
    else:
        raise ValueError(f"Unsupported scoring function: {scoring_func}")

    if e_score_correction_bias is not None:
        # Store original scores before applying correction bias. We use biased
        # scores for expert selection but original scores for routing weights
        original_scores = scores
        scores = scores + e_score_correction_bias.unsqueeze(0)

    num_token = scores.shape[0]
    group_scores = scores.view(num_token, num_expert_group,
                               -1).max(dim=-1).values  # [n, n_group]
    group_idx = torch.topk(group_scores, k=topk_group, dim=-1,
                           sorted=False)[1]  # [n, top_k_group]
    group_mask = torch.zeros_like(group_scores)  # [n, n_group]
    group_mask.scatter_(1, group_idx, 1)  # [n, n_group]
    score_mask = group_mask.unsqueeze(-1).expand(
        num_token, num_expert_group,
        scores.shape[-1] // num_expert_group).reshape(num_token, -1)  # [n, e]
    tmp_scores = scores.masked_fill(~score_mask.bool(), 0.0)  # [n, e]

    if e_score_correction_bias is not None:
        topk_ids = torch.topk(tmp_scores, k=topk, dim=-1, sorted=False)[1]
        # Use original unbiased scores for the routing weights
        topk_weights = original_scores.gather(1, topk_ids)
    else:
        topk_weights, topk_ids = torch.topk(tmp_scores,
                                            k=topk,
                                            dim=-1,
                                            sorted=False)

    if renormalize:
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

    return topk_weights.to(torch.float32), topk_ids.to(torch.int32)

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

def benchmark_config(
    num_tokens: int,
    num_experts: int,
    shard_intermediate_size: int,
    hidden_size: int,
    topk: int,
    modes: Optional[List[int]]=None,
    dtype: Optional[torch.dtype]=None
) -> float:

    print("m = ", num_tokens, " 性能测试......")
    # print("num_experts: ", num_experts)
    # print("shard_intermediate_size: ", shard_intermediate_size)
    # print("hidden_size: ", hidden_size)
    # print("topk: ", topk)

    torch.manual_seed(0)

    # 随机100次 专家选择
    num_iters = 100


    #  =========================================== 准备 gemm1 相关数据 =============================================
    x = torch.randn((num_tokens, hidden_size),device='cuda',dtype=dtype) / 10 # [256*8, 7168] (bf16)
    w1 = torch.randn((num_experts, shard_intermediate_size, hidden_size), device=x.device, dtype=dtype)
    # -------- 处理 gemm1 权重矩阵 ------------------
    w1_marlin_list = []
    for i in range(w1.shape[0]):
        w1_marlin_in = w16a16_marlin_weight(w1[i])
        w1_marlin_list.append(w1_marlin_in)
    w1_marlin = torch.stack(w1_marlin_list, dim=0)
    w1_scale = (torch.randn((num_experts, shard_intermediate_size ,1), dtype=torch.float32, device=x.device))
    
    #  =========================================== 准备 gemm2 相关数据 =============================================
    w2 = torch.randn((num_experts, hidden_size, shard_intermediate_size // 2), device=x.device, dtype=dtype)
    # -------- 处理 gemm2 权重矩阵 ------------------
    w2_marlin_list = []
    for i in range(w2.shape[0]):
        w2_marlin_in = w16a16_marlin_weight(w2[i])
        w2_marlin_list.append(w2_marlin_in)
    w2_marlin = torch.stack(w2_marlin_list, dim=0)
    w2_scale = (torch.randn((num_experts, hidden_size, 1), dtype=torch.float32, device=x.device))

    # ============================================ 准备 gating_output 和 score ======================================
    gating_output = torch.randn(num_iters, num_tokens, num_experts, dtype=torch.float32)
    score = torch.randn((num_tokens, num_experts), device="cuda", dtype=torch.float32)

    topk_weights, topk_ids = grouped_topk(x, score, topk, True, 8, 4)
    # 先进行准确度测试
    # GEMM1_Peak,GEMM2_Peak,bandwith1,bandwith2 = fused_marlin_moe_valid(   
    fused_moe_valid_w16a16_marlin(                      
                                    x,
                                    w1,
                                    w2,
                                    w1_marlin,
                                    w2_marlin,
                                    score,
                                    topk_weights,
                                    topk_ids,
                                    modes=modes)

def run_moe_w16a16_perchannel_vaild():
    # m_list=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,32,64,128,256,512,1024,2048,4096,8192,16384,32768]
    # m_list=[1, 8, 16, 32, 33, 64, 65, 128] #decode
    m_list=[256,512,1024,2048,4096,8192,16384] #prefill
    # m_list=[144,160,176,192,208,224,240,240,256,288,320,]
    # m_list=[1, 2, 4, 8, 16, 32, 64, 128]
    # m_list = [512]
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
    
    # #qwen3-235b-A22B-tp8
    # E=128
    # topk=8
    # n_list_1=[384]
    # k_list_1=[4096]
    # n_list_2=[4096]
    # k_list_2=[192]
    
    #qwen3-480b-A35B-tp8
    E=160
    topk=8
    n_list_1=[640]
    k_list_1=[6144]
    n_list_2=[6144]
    k_list_2=[320]
    
    out_dtype=torch.bfloat16
    # out_dtype = torch.float16


    # configs=BlockSizeMConfigGen()
    best_config = None
    triton_json_dir="./cache/moe/"
    os.makedirs(triton_json_dir, exist_ok=True)
    
    device_name=torch.cuda.get_device_properties('cuda').name.replace(" ","_")

    # mode_ranges = [i for i in range(len(blockM_configs))]
    # mode_ranges = [mode for mode in blockM_all_configs.keys()]
    mode_ranges = [1000]
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

    
    for i in range(0,len(k_list_1),1):
        for m in m_list:
            for mode in mode_ranges:
                mode_list=[mode,mode]
                print("mode:{} m:{} n:{} k:{} ".format(mode_list, m,n_list_1[i],k_list_1[i]))
                num_tokens = m
                num_experts = E
                shard_intermediate_size = n_list_1[i]
                hidden_size = k_list_1[i]
                topk = topk
                result = benchmark_config(
                                num_tokens, 
                                num_experts,
                                shard_intermediate_size, 
                                hidden_size,
                                topk,
                                mode_list,
                                out_dtype)

                     
    # print("gemm1_compare_failed_list: ", gemm1_compare_failed_list)
    # print("gemm2_compare_failed_list: ", gemm2_compare_failed_list)
            
def main():
    # run_tuning()
    run_moe_w16a16_perchannel_vaild()

if __name__ == "__main__":
    main()
    # HIP_VISIBLE_DEVICES=1 python3 fuse_moegemm_w16a16_correctness_marlin.py