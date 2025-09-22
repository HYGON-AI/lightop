# SPDX-License-Identifier: Apache-2.0

import argparse
import time
from datetime import datetime
from itertools import product
from typing import Any, Dict, List, Tuple, TypedDict
import numpy as np
import ray
import torch
torch.set_printoptions(precision=4, profile="default", sci_mode=False)
from ray.experimental.tqdm_ray import tqdm
from transformers import AutoConfig
# from vllm.model_executor.layers.fused_moe.fused_moe import *
# from fused_moe import *
# from fused_moe_3 import *
from typing import Optional
import os
import pandas as pd
import triton
import triton.language as tl
from collections import defaultdict
from w4a8_moe_groupgemm_marlin_quant import w4a8_2_marlin_weight
# 引入适配后的 marlin fused_moe实现

from groupgemm_marlin_moe_valid import groupgemm_marlin_moe_valid
from groupgemm_marlin_moe import groupgemm_marlin_moe


from vllm.platforms import current_platform

FP8_DTYPE = torch.float8_e4m3fnuz if current_platform.is_rocm(
) else torch.float8_e4m3fn

def determine_expert_map(
        ep_size: int, ep_rank: int,
        global_num_experts: int) -> tuple[int, Optional[torch.Tensor]]:
    """
        Calculates how many experts should be assigned to each rank for EP and
        creates a mapping from global to local expert index. Experts are
        distributed evenly across ranks. Any remaining are assigned to the
        last rank.

        Args:
            ep_size (int): The size of the expert parallel group
            global_num_experts (int): The total number of experts in the model.

        Returns:
            tuple[int, Optional[torch.Tensor]]: A tuple containing:
                - local_num_experts (int): The number of experts assigned
                    to the current rank.
                - expert_map (Optional[torch.Tensor]): A tensor of shape
                    (global_num_experts,) mapping from global to local index.
                    Contains -1 for experts not assigned to the current rank.
                    Returns None if ep_size is 1.
        """
    assert ep_size > 0
    if ep_size == 1:
        return (global_num_experts, None)

    local_num_experts = global_num_experts // ep_size

    # Create a tensor of size num_experts filled with -1
    expert_map = torch.full((global_num_experts, ), -1, dtype=torch.int32)
    # Create a expert map for the local experts
    if ep_rank < (ep_size - 1):
        # Each non-last rank gets local_num_experts experts.
        expert_map[ep_rank * local_num_experts:
                        (ep_rank + 1) * local_num_experts] = \
            torch.arange(0, local_num_experts, dtype=torch.int32)
    else:
        # All remaining experts are assigned to the last rank.
        local_num_experts = (global_num_experts - ep_rank * local_num_experts)

        expert_map[-local_num_experts:] = \
            torch.arange(0, local_num_experts, dtype=torch.int32)
    expert_map = expert_map.cuda(0)
    return (local_num_experts, expert_map)

# 生成
def get_ep_topk_ids_weight(
    M,
    top_k,
    global_num_experts,
    ep_size=8,
    ep_rank=0,
    num_experts_select=1,
    device="cuda:0"
):
    assert global_num_experts % ep_size == 0, "global_num_experts must be divisible by ep_size"
    assert num_experts_select <= top_k, "num_experts_select cannot exceed top_k"
    assert ep_rank < ep_size, "ep_rank must be less than ep_size"
    
    # ================================ 生成 top_ids   ================================
    # 计算当前卡上的专家数
    local_num_experts = global_num_experts // ep_size
    # 生成返回的 topk_ids Tensor，填充255代表其他卡上的 专家索引
    topk_ids = torch.full((M, top_k), global_num_experts - 1, dtype=torch.int32, device=device)
    
    # 生成 [M, num_experts_select] Tensor，标记在当前卡专家上选择的专家
    local_experts_select = torch.randint(
        low=0, 
        high=local_num_experts, 
        size=(M, num_experts_select), 
        dtype=torch.int32,
        device=device
    )

    # 插入到 topk_ids 前 num_experts_select 个位置中
    topk_ids[:, :num_experts_select] = local_experts_select

    # 打乱每行里面的专家索引
    rows, cols = topk_ids.shape
    rand_indices = torch.argsort(torch.rand(rows, cols, device=device), dim=1)
    shuffled_topk_ids = topk_ids.gather(1, rand_indices)

    # ================================ 生成 top_weight ================================
    topk_weights = torch.randn((M, top_k), dtype=torch.float32, device=device)

    # 返回打乱后的结果
    return shuffled_topk_ids, topk_weights

def benchmark_config(
    num_tokens: int, # num_toens
    global_num_experts: int,  # 32
    shard_intermediate_size: int, # 4096
    hidden_size: int, # 7168
    topk: int,  # 8
    ep_size: int, # 8
    ep_rank: int, # 8
    num_experts_select: int, # 8
) -> float:
    print("===============================================================================")
    print("m = ", num_tokens, "num_experts_select = ", num_experts_select," 性能测试......")
    # print("num_experts: ", num_experts)
    # print("shard_intermediate_size: ", shard_intermediate_size)
    # print("hidden_size: ", hidden_size)
    # print("topk: ", topk)

    torch.manual_seed(0)

    # 随机100次 topk_ids 的生成
    num_iters = 100

    #  =========================================== 准备 gemm1 相关数据 =============================================
    x = torch.randn((num_tokens*ep_size, hidden_size),device='cuda',dtype=torch.bfloat16) / 10 # [256*8, 7168] (bf16)
    w1 = torch.randint(-127, 127, (num_experts, shard_intermediate_size, hidden_size//2), device=x.device, dtype=torch.int8)
    w1_scale = (torch.randn((num_experts, shard_intermediate_size ,1), dtype=torch.float32, device=x.device))
              
    # -------- 处理 gemm1 权重矩阵 ------------------
    w1_marlin_list = []
    for i in range(w1.shape[0]):
        w1_marlin_in = w4a8_2_marlin_weight(w1[i])
        w1_marlin_list.append(w1_marlin_in)
    w1_marlin = torch.stack(w1_marlin_list, dim=0)

    #  =========================================== 准备 gemm2 相关数据 =============================================
    w2 = torch.randint(-127, 127, (num_experts, hidden_size, shard_intermediate_size // 4), device=x.device, dtype=torch.int8)
    w2_scale = (torch.randn((num_experts, hidden_size, 1), dtype=torch.float32, device=x.device))
    # -------- 处理 gemm2 权重矩阵 ------------------
    w2_marlin_list = []
    for i in range(w2.shape[0]):
        w2_marlin_in = w4a8_2_marlin_weight(w2[i])
        w2_marlin_list.append(w2_marlin_in)
    w2_marlin = torch.stack(w2_marlin_list, dim=0)

    # ============================================ 准备 gating_output 和 score ======================================
    topk_ids, topk_weights = get_ep_topk_ids_weight(
                                                    num_tokens,
                                                    topk, 
                                                    global_num_experts,
                                                    ep_size=ep_size,
                                                    ep_rank=ep_rank,
                                                    num_experts_select=num_experts_select,
                                                    device=x.device,
                                                    )
    tokens_per_expert = [0]*32
    for element in topk_ids.view(-1):
        tokens_per_expert[element] = tokens_per_expert[element] + 1

    import pdb
    pdb.set_trace()
    print("topk_ids 0: ", topk_ids[0,:])
    print("real m: ", x.shape[0])
    # 做掩码
    local_num_experts, expert_map = determine_expert_map(
                                                        ep_size=ep_size,
                                                        ep_rank=ep_rank,
                                                        global_num_experts=global_num_experts
                                                        )
    w1 = w1[:32,:,:]
    w2 = w2[:32,:,:]
    w1_marlin = w1_marlin[:32,:,:]
    w2_marlin = w2_marlin[:32,:,:]
    w1_scale = w1_scale[:32,:,:]
    w2_scale = w2_scale[:32,:,:]
    print("expert_map: ", expert_map)
    # 正确性验证
    groupgemm_marlin_moe_valid(                      
                            x,
                            w1,
                            w2,
                            w1_marlin,
                            w2_marlin,
                            w1_scale,
                            w2_scale,
                            tokens_per_expert,
                            )

    # assert False
    # 预热
    for i in range(10):
        # 设置当前的随机种子
        torch.manual_seed(i)
        torch.cuda.manual_seed_all(i)
        # 根据当前的随机种子获取 topk_ids
        topk_ids, topk_weights = get_ep_topk_ids_weight(
                                                    num_tokens,
                                                    topk, 
                                                    global_num_experts,
                                                    ep_size=ep_size,
                                                    ep_rank=ep_rank,
                                                    num_experts_select=num_experts_select,
                                                    device=x.device,
                                                    )

        tokens_per_expert = [0]*32
        for element in topk_ids.view(-1):
            tokens_per_expert[element] = tokens_per_expert[element] + 1
        
        for j in range(10): # 一次随机专家运行10次
            groupgemm_marlin_moe(                      
                                 x,
                                 w1,
                                 w2,
                                 w1_marlin,
                                 w2_marlin,
                                 w1_scale,
                                 w2_scale,
                                 tokens_per_expert,
                                 )

    gemm1_time_us_total = 0 
    gemm2_time_us_total = 0
    for i in range(num_iters):
        # 设置当前的随机种子
        torch.manual_seed(i)
        torch.cuda.manual_seed_all(i)
        # 根据当前的随机种子获取 topk_ids
        topk_ids, topk_weights = get_ep_topk_ids_weight(
                                                    num_tokens,
                                                    topk, 
                                                    global_num_experts,
                                                    ep_size=ep_size,
                                                    ep_rank=ep_rank,
                                                    num_experts_select=num_experts_select,
                                                    device=x.device,
                                                    )
        for j in range(10): # 一次随机专家运行10次
            gemm1_time_us, gemm2_time_us = groupgemm_marlin_moe(                      
                                                                x,
                                                                w1,
                                                                w2,
                                                                w1_marlin,
                                                                w2_marlin,
                                                                w1_scale,
                                                                w2_scale,
                                                                tokens_per_expert,
                                                                )
            
            gemm1_time_us_total += gemm1_time_us
            gemm2_time_us_total += gemm2_time_us

    print("      GEMM1_Time_us: ", gemm1_time_us_total/num_iters/10, "GEMM2_Time_us: ", gemm2_time_us_total/num_iters/10)
    return gemm1_time_us_total/num_iters/10, gemm2_time_us_total/num_iters/10

if __name__ == "__main__":

    test_list = [
        # {"num_tokens": 32, "num_experts_select": 1},
        # {"num_tokens": 1024, "num_experts_select": 8},
        # {"num_tokens": 64, "num_experts_select": 1},
        # {"num_tokens": 64, "num_experts_select": 8},
        # {"num_tokens": 1,     "num_experts_select": 8},
        # {"num_tokens": 2,     "num_experts_select": 8},
        # {"num_tokens": 4,     "num_experts_select": 8},
        # {"num_tokens": 8,     "num_experts_select": 8},
        # {"num_tokens": 16,    "num_experts_select": 8},
        # {"num_tokens": 32,    "num_experts_select": 8},
        # {"num_tokens": 64,    "num_experts_select": 8},
        # {"num_tokens": 128,   "num_experts_select": 8},
        # {"num_tokens": 256,   "num_experts_select": 8},
        # {"num_tokens": 512,   "num_experts_select": 8},
        {"num_tokens": 1024,  "num_experts_select": 8},
        # {"num_tokens": 2048,  "num_experts_select": 8},
        # {"num_tokens": 4096,  "num_experts_select": 8},
        # {"num_tokens": 8192,  "num_experts_select": 8},
        # {"num_tokens": 16384, "num_experts_select": 8},
    ]

    ep_size = 8
    ep_rank = 0
    top_k = 8
    num_experts = 256
    K = 7168
    shard_intermediate_size = 4096

    prof_results = defaultdict(list)
    for test_example_dict in test_list:
        num_tokens = test_example_dict["num_tokens"]
        num_experts_select = test_example_dict["num_experts_select"]
        
        gemm1_time_us, gemm2_time_us = benchmark_config(
                                                        num_tokens, 
                                                        num_experts, 
                                                        shard_intermediate_size, 
                                                        K, 
                                                        top_k, 
                                                        ep_size=ep_size,
                                                        ep_rank=ep_rank,
                                                        num_experts_select=num_experts_select,
                                                        )
        prof_results[num_experts_select].append({
            "num_tokens": num_tokens,
            "gemm1_time_us": gemm1_time_us,
            "gemm2_time_us": gemm2_time_us
        })

    with pd.ExcelWriter("benchmark_results.xlsx", engine="openpyxl") as writer:
        # 为每个num_experts_select创建一张表
        for select_num, results in prof_results.items():
            # 转换为DataFrame
            df = pd.DataFrame(results)
            # 按num_tokens升序排序
            df = df.sort_values(by="num_tokens")
            # 写入工作表，表名格式为"select_{num}"
            df.to_excel(writer, sheet_name=f"select_{select_num}", index=False)

    print("测试结果已写入 benchmark_results.xlsx")