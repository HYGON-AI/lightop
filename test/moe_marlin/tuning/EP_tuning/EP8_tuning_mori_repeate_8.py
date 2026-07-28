import argparse
import time
from datetime import datetime
from itertools import product
from typing import Any, Dict, List, Tuple, TypedDict
import numpy as np
import ray
import torch
torch.set_printoptions(precision=4, profile="default", sci_mode=False)
import triton
from ray.experimental.tqdm_ray import tqdm
from transformers import AutoConfig
from typing import Optional
import os
import pandas as pd
import triton
import triton.language as tl
from collections import defaultdict
from marlin_quant import w4a8_2_marlin_weight
import random
# 引入适配后的 marlin fused_moe实现

from fused_marlin_moe_valid import fused_marlin_moe_valid

from fused_marlin_moe import fused_marlin_moe
import openpyxl
from openpyxl import Workbook  # 显式导入Workbook

import pandas as pd
import json

is_print = False

def grouped_topk(
                 gating_output: torch.Tensor,
                 topk: int,
                 renormalize: bool,
                 num_expert_group: int = 0,
                 topk_group: int = 0,
                 scoring_func: str = "softmax",
                 e_score_correction_bias: Optional[torch.Tensor] = None):
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

def determine_expert_map(
        ep_size: int, ep_rank: int,
        global_num_experts: int) -> tuple[int, Optional[torch.Tensor]]:
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
    device="cuda:0",
    seed=0
):
    assert global_num_experts % ep_size == 0, "global_num_experts must be divisible by ep_size"
    assert num_experts_select <= top_k, "num_experts_select cannot exceed top_k"
    assert ep_rank < ep_size, "ep_rank must be less than ep_size"

    set_random_seed(seed=seed)

    # ================================ 生成 top_ids   ================================
    # 计算当前卡上的专家数
    local_num_experts = global_num_experts // ep_size
    # 生成返回的 topk_ids Tensor，填充255代表其他卡上的 专家索引
    topk_ids = torch.full((M, top_k), global_num_experts - 1, dtype=torch.int32, device=device)
    
    # # 生成 [M, num_experts_select] Tensor，标记在当前卡专家上选择的专家
    # local_experts_select = torch.randint(
    #     low=0, 
    #     high=local_num_experts, 
    #     size=(M, num_experts_select), 
    #     dtype=torch.int32,
    #     device=device
    # )

    # 生成每一行专家索引 不重复的 local_experts_select 
    local_experts_select = torch.stack([
    torch.randperm(local_num_experts, device=device)[:num_experts_select] for _ in range(M) ])

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

def set_random_seed(seed=42):
    # Python随机种子
    random.seed(seed)
    # NumPy随机种子
    np.random.seed(seed)
    # PyTorch随机种子
    torch.manual_seed(seed)
    # CUDA随机种子
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # 多GPU时使用
        # 确保CUDA卷积操作的确定性
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# 根据 传入的 num_tokens 、 local_num_experts 决定参与 tuning 的所有 config
def get_marlin_asm_config(num_tokens, local_num_experts):
    test_config= []
    # ================== 汇编作为基础config ===================
    test_config.append(
                        {
                            "MODE": 1000,
                            "DELTA": 1,
                            "BLOCK_SIZE_M": 128,
                            "thread_n": 256,
                            "thread_k": 128,
                            "blocks_per_cu": 1 
                        }
                    )

    # 添加 128 block_m 的 config
    base_config = {
                    "MODE": 500,
                    "DELTA": 1,
                    "BLOCK_SIZE_M": 128,
                    }
    for threads_n, threads_k in [
                        (512, 64)
                        ]:        
        for blocks_per_cu in [1, 2, 3]:
            for is_reduce in [True, False]:
                config = base_config.copy()
                config["thread_n"] = threads_n         
                config["thread_k"] = threads_k         
                config["blocks_per_cu"] = blocks_per_cu
                config["is_reduce"] = is_reduce
                test_config.append(config)


    # # ================== 添加cuda 手写版本 config ===========
    # test_config.append(
    #                     {
    #                         "MODE": 402,
    #                         "DELTA": 1,
    #                         "BLOCK_SIZE_M": 16,
    #                         "thread_n": 256,
    #                         "thread_k": 128,
    #                         "blocks_per_cu": 1 
    #                     }
    #                 )
    # test_config.append(
    #                     {
    #                         "MODE": 403,
    #                         "DELTA": 1,
    #                         "BLOCK_SIZE_M": 16,
    #                         "thread_n": 256,
    #                         "thread_k": 128,
    #                         "blocks_per_cu": 1 
    #                     }
    #                 )

    # # ================== 添加 marlin config ================= 
    # for block_m in[16, 32, 48, 80]:
    #     base_config = {
    #                     "MODE": 500,
    #                     "DELTA": 1,
    #                     "BLOCK_SIZE_M": block_m,
    #                     }
    #     for threads_n, threads_k in [
    #                         (256, 128),
    #                         (512, 64)
    #                         ]:        
    #         for blocks_per_cu in [1, 2, 3]:
    #             for is_reduce in [True, False]:
    #                 config = base_config.copy()
    #                 config["thread_n"] = threads_n         
    #                 config["thread_k"] = threads_k         
    #                 config["blocks_per_cu"] = blocks_per_cu
    #                 config["is_reduce"] = is_reduce
    #                 test_config.append(config)

    # # 添加 128 block_m 的 config
    # base_config = {
    #                 "MODE": 500,
    #                 "DELTA": 1,
    #                 "BLOCK_SIZE_M": 128,
    #                 }
    # for threads_n, threads_k in [
    #                     (512, 64)
    #                     ]:        
    #     for blocks_per_cu in [1, 2, 3]:
    #         for is_reduce in [True, False]:
    #             config = base_config.copy()
    #             config["thread_n"] = threads_n         
    #             config["thread_k"] = threads_k         
    #             config["blocks_per_cu"] = blocks_per_cu
    #             config["is_reduce"] = is_reduce
    #             test_config.append(config)

    # num_experts_select = 1
    # # 当前根据任务量最大选block_m
    # for block_size_m_1 in [16, 32, 48, 80, 128]:
    #     if num_tokens * num_experts_select / local_num_experts / block_size_m_1 < 0.9:
    #         break

    # if block_size_m_1 != 128:
    #     base_config = {
    #                     "MODE": 500,
    #                     "DELTA": 1,
    #                     "BLOCK_SIZE_M": block_size_m_1,
    #                     }
    #     for threads_n, threads_k in [
    #                         (256, 128),
    #                         (512, 64)
    #                         ]:        
    #         for blocks_per_cu in [1, 2, 3]:
    #             # for is_reduce in [0, 1]:
    #                 config = base_config.copy()
    #                 config["thread_n"] = threads_n         
    #                 config["thread_k"] = threads_k         
    #                 config["blocks_per_cu"] = blocks_per_cu
    #                 # config["is_reduce"] = is_reduce
    #                 test_config.append(config)
    # else:
    #     base_config = {
    #                     "MODE": 500,
    #                     "DELTA": 1,
    #                     "BLOCK_SIZE_M": 128,
    #                     }
    #     for threads_n, threads_k in [
    #                         (512, 64)
    #                         ]:        
    #         for blocks_per_cu in [1, 2, 3]:
    #             # for is_reduce in [0, 1]:
    #                 config = base_config.copy()
    #                 config["thread_n"] = threads_n         
    #                 config["thread_k"] = threads_k         
    #                 config["blocks_per_cu"] = blocks_per_cu
    #                 # config["is_reduce"] = is_reduce
    #                 test_config.append(config)

    # num_experts_select = 8
    # # 当前根据任务量最大选block_m
    # for block_size_m_8 in [16, 32, 48, 80, 128]:
    #     if num_tokens * num_experts_select / local_num_experts / block_size_m_8 < 0.9:
    #         break
    # if block_size_m_8 != block_size_m_1:
    #     if block_size_m_8 != 128:
    #         base_config = {
    #                         "MODE": 500,
    #                         "DELTA": 1,
    #                         "BLOCK_SIZE_M": block_size_m_8,
    #                         }
    #         for threads_n, threads_k in [
    #                             (256, 128),
    #                             (512, 64)
    #                             ]:        
    #             for blocks_per_cu in [1, 2, 3]:
    #                 # for is_reduce in [0, 1]:
    #                     config = base_config.copy()
    #                     config["thread_n"] = threads_n         
    #                     config["thread_k"] = threads_k         
    #                     config["blocks_per_cu"] = blocks_per_cu
    #                     # config["is_reduce"] = is_reduce
    #                     test_config.append(config)
    #     else:
    #         base_config = {
    #                         "MODE": 500,
    #                         "DELTA": 1,
    #                         "BLOCK_SIZE_M": 128,
    #                         }
    #         for threads_n, threads_k in [
    #                             (512, 64)
    #                             ]:        
    #             for blocks_per_cu in [1, 2, 3]:
    #                 # for is_reduce in [0, 1]:
    #                     config = base_config.copy()
    #                     config["thread_n"] = threads_n         
    #                     config["thread_k"] = threads_k         
    #                     config["blocks_per_cu"] = blocks_per_cu
    #                     # config["is_reduce"] = is_reduce
    #                     test_config.append(config)

    return test_config

def count_rows_by_range(topk_ids: torch.Tensor, low_idx: int, high_idx: int) -> int:
    # 创建布尔 mask，表示哪些元素在 [low_idx, high_idx) 范围内
    in_range_mask = (topk_ids >= low_idx) & (topk_ids < high_idx)

    # 统计每一行是否有至少一个元素在范围内
    row_has_in_range = in_range_mask.any(dim=1)  # shape: [M]

    # 提取符合条件的行
    filtered_rows = topk_ids[row_has_in_range]
    num_rows = row_has_in_range.sum().item()

    return filtered_rows, num_rows

def get_max_tokens_rank(topk_ids: torch.Tensor, global_num_experts: int, ep_size: int) -> int:
    local_experts_num = global_num_experts // ep_size
    
    max_tokens_num = 0
    max_tokens_num_rank = 0

    # 遍历每个rank
    for ep_rank in range(0, ep_size):
        row_idx = ep_rank * local_experts_num
        high_idx = (ep_rank + 1) * local_experts_num
        # 统计每一行中符合条件的元素个数
        mask = (topk_ids >= row_idx) & (topk_ids < high_idx)
        occurrences = mask.sum(dim=1).sum().item()

        # print("ep_rank: ", ep_rank, ";occurrences: ", occurrences)

        if occurrences > max_tokens_num:
            max_tokens_num = occurrences
            max_tokens_num_rank = ep_rank

    return max_tokens_num_rank

def benchmark_config(
    batch_size: int, # num_toens
    global_num_experts: int,  # 256
    local_num_experts,
    shard_intermediate_size: int, # 4096
    hidden_size: int, # 7168
    topk: int,  # 8
    ep_size: int, # 8
    x,
    w1,
    w1_scale,
    w1_marlin,
    w2,
    w2_scale,
    w2_marlin
) -> float:
    print("=============================", "batch_size: ", batch_size , "tuning", "=======================================")
    # 获取 marlin 和 asm实现 的所有 config
    test_config_list = get_marlin_asm_config(batch_size, local_num_experts)
    print("test_config_list: ", test_config_list)
    num_iters = 100

    # 声明最优执行时间和对应的config
    best_total_time = 1000000
    best_gemm1_time = 1000000
    best_gemm2_time = 1000000

    best_gemm1_config = None
    best_gemm2_config = None  

    # 遍历所有的config
    for cur_config_gemm1 in test_config_list: # gemm1 config
        if is_print:
            print("gemm1 测试 config: ", cur_config_gemm1)
        block_m_1 = cur_config_gemm1["BLOCK_SIZE_M"]
        for cur_config_gemm2 in test_config_list:  # 确定 gemm2 config #n gemm2 
            block_m_2 = cur_config_gemm2["BLOCK_SIZE_M"]
            if(block_m_2 != block_m_1): # 如果 gemm2 config不等于 gemm1 config，则进入下一个config验证
                continue

            if is_print:
                print("     gemm2 测试 config: ", cur_config_gemm2)

            cur_total_time_us_1_gemm1 = 0.0
            cur_total_time_us_1_gemm2 = 0.0

            #  生成 100 次随机的 token 得分
            gating_output = torch.randn(100, batch_size, global_num_experts, dtype=torch.float32)
            #  每次使用的 token 得分
            score = torch.randn((batch_size, global_num_experts), device="cuda", dtype=torch.float32)
            # for i in range(0, 100): # duoc fused_moe
            for i in range(0, 1): # duoc fused_moe
                # 获取到本次随机生成的 token 得分
                score.copy_(gating_output[i]) # 
                # 获取当前 得分下的 topk_ids
                topk_weights, topk_ids = grouped_topk(score, topk, True, 8, 4)
                if is_print:
                    print("topk_ids: ", topk_ids.size(), topk_ids.dtype)
                    print(topk_ids)
                    print("topk_weights: ", topk_weights.size(), topk_weights.dtype)
                    
                # topk_ids 复制8份
                topk_ids_repeated = torch.cat([topk_ids] * 8, dim=0)
                topk_weights_repeated = torch.cat([topk_weights] * 8, dim=0)
                if is_print:
                    print("topk_ids_repeated: ", topk_ids_repeated.size(), topk_ids_repeated.dtype)
                    print("topk_weights_repeated: ", topk_weights_repeated.size(), topk_weights_repeated.dtype)
                
                # ====================================  读取模拟的topk_ids 进行性能评估 ========================== 
                # 遍历获取 当前 rank 卡上需要处理的 token，及其对应的 topk_ids
                ep_rank_max = get_max_tokens_rank(topk_ids_repeated, global_num_experts, ep_size)
                
                # # ==================================  读取 卡 上实际 的 topk_ids 进行 计算 ====================== 
                # ep_rank_max = 2
                # topk_ids_repeated = torch.from_numpy(np.load("./bs16_true/topk_ids/rank_2_round_93.npy")).to("cuda:0")


                rank_cur_topk_ids, rank_cur_topk_ids_num = count_rows_by_range(topk_ids_repeated, (ep_rank_max * local_num_experts), ((ep_rank_max + 1) * local_num_experts))
                # print("rank_cur_topk_ids_num: ", rank_cur_topk_ids_num)
                if is_print:
                    print("rank_cur_topk_ids_num: ", rank_cur_topk_ids_num)
                    print("rank_cur_topk_ids: ", rank_cur_topk_ids)
                
                # =====================================================  ===============================================
                #  x [128, 7168] topk_ids [128, 8] topk_weight: [128, 8] num_local_tokens [32]
                # 将有效索引填充到 总的 topk_ids缓冲区中，并新增  local_num_tokens 张量标志 前多少行有效
                local_num_tokens = torch.tensor(rank_cur_topk_ids_num, dtype=torch.int32).to("cuda:0")
                topk_ids_repeated[0:rank_cur_topk_ids_num] = rank_cur_topk_ids
                
                if is_print:
                    print("local_num_tokens: ", local_num_tokens)

                _, expert_map = determine_expert_map(ep_size=ep_size, ep_rank=ep_rank_max, global_num_experts=global_num_experts)
                # 验证正确性
                # is_true_1, is_true_2 = fused_marlin_moe_valid(
                #                                                 x, 
                #                                                 w1, 
                #                                                 w2, 
                #                                                 w1_marlin, 
                #                                                 w2_marlin, 
                #                                                 w1_scale, 
                #                                                 w2_scale, 
                #                                                 topk_weights_repeated, 
                #                                                 topk_ids_repeated, 
                #                                                 cur_config_gemm1, 
                #                                                 cur_config_gemm2, 
                #                                                 global_num_experts=global_num_experts, 
                #                                                 expert_map=expert_map,
                #                                                 ep_size=ep_size
                #                                                 )
                # if (not is_true_1) or (not is_true_2):
                #     print("num_experts_select = 8 精确度测试未通过： ", is_true_1, is_true_2)
                #     continue

                # 性能测试
                # 预热10次
                for i in range (0, 10):  
                    gemm1_time_us, gemm2_time_us = fused_marlin_moe(                      
                                                x,
                                                w1_marlin,
                                                w2_marlin,
                                                w1_scale,
                                                w2_scale,
                                                topk_weights_repeated,
                                                topk_ids_repeated,
                                                cur_config_gemm1,
                                                cur_config_gemm2,
                                                global_num_experts=global_num_experts,
                                                expert_map=expert_map,
                                                ep_size=ep_size,
                                                local_num_tokens=local_num_tokens # 
                                                )
                # 实际性能测试
                for i in range (0, 10):
                    gemm1_time_us, gemm2_time_us = fused_marlin_moe(                      
                                                                    x,
                                                                    w1_marlin,
                                                                    w2_marlin,
                                                                    w1_scale,
                                                                    w2_scale,
                                                                    topk_weights_repeated,
                                                                    topk_ids_repeated,
                                                                    cur_config_gemm1,
                                                                    cur_config_gemm2,
                                                                    global_num_experts=global_num_experts,
                                                                    expert_map=expert_map,
                                                                    ep_size=ep_size,
                                                                    local_num_tokens=local_num_tokens
                                                                    )
                    cur_total_time_us_1_gemm1 += gemm1_time_us
                    cur_total_time_us_1_gemm2 += gemm2_time_us
            
            # cur_total_time_us_1_gemm1 = cur_total_time_us_1_gemm1 / 100 / 10
            # cur_total_time_us_1_gemm2 = cur_total_time_us_1_gemm2 / 100 / 10
            cur_total_time_us_1_gemm1 = cur_total_time_us_1_gemm1 / 10
            cur_total_time_us_1_gemm2 = cur_total_time_us_1_gemm2 / 10
            if (cur_total_time_us_1_gemm1 + cur_total_time_us_1_gemm2) < best_total_time:
                best_total_time = cur_total_time_us_1_gemm1 + cur_total_time_us_1_gemm2
                best_gemm1_time = cur_total_time_us_1_gemm1
                best_gemm2_time = cur_total_time_us_1_gemm2
                best_gemm1_config = cur_config_gemm1
                best_gemm2_config = cur_config_gemm2
                
    print("best_gemm1_time : ", best_gemm1_time, "best_gemm1_config: ", best_gemm1_config)
    print("best_gemm2_time : ", best_gemm2_time, "best_gemm2_config: ", best_gemm2_config)

    best_config = {
        "batch_size": batch_size,
        "best_1_gemm1_time": best_gemm1_time,
        "best_1_gemm2_time": best_gemm2_time,
        "gemm1_config": best_gemm1_config,
        "gemm2_config": best_gemm2_config,
    }

    return best_config

# 主函数
if __name__ == "__main__":
    torch.set_printoptions(threshold=5000, edgeitems=10)  # 默认会显示整个 tensor 内容
    set_random_seed(seed=0)
    
    # ep_sizes = [8, 16, 32]  # 要遍历的 ep_size 值
    ep_sizes = 8  # 要遍历的 ep_size 值
    global_num_experts = 256
    local_num_experts = global_num_experts // ep_sizes
    hidden_size = 7168
    shard_intermediate_size = 4096
    top_k = 8

    local_num_experts = global_num_experts // ep_sizes

    # ======================== 预先生成 gemm1 的相关数据 ==================
    w1 = torch.randint(-127, 127, (local_num_experts, shard_intermediate_size, hidden_size // 2), device='cuda:0', dtype=torch.int8)
    w1_scale = torch.randn((local_num_experts, shard_intermediate_size, 1), dtype=torch.float32, device='cuda:0')

    # 处理 gemm1 权重矩阵
    w1_marlin_list = []
    for i in range(w1.shape[0]):
        w1_marlin_in = w4a8_2_marlin_weight(w1[i])
        w1_marlin_list.append(w1_marlin_in)
    w1_marlin = torch.stack(w1_marlin_list, dim=0)

    # ======================== 预先生成 gemm2 的相关数据 ==================
    w2 = torch.randint(-127, 127, (local_num_experts, hidden_size, shard_intermediate_size // 4), device='cuda:0', dtype=torch.int8)
    w2_scale = torch.randn((local_num_experts, hidden_size, 1), dtype=torch.float32, device='cuda:0')

    # 处理 gemm2 权重矩阵
    w2_marlin_list = []
    for i in range(w2.shape[0]):
        w2_marlin_in = w4a8_2_marlin_weight(w2[i])
        w2_marlin_list.append(w2_marlin_in)
    w2_marlin = torch.stack(w2_marlin_list, dim=0)

    total_config_info = []
    # for batch_size in [1, 2]:
    # for batch_size in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]:
    for batch_size in [16]:
        # 初始化一个 batch_size * ep_size 的激活矩阵
        x = torch.randn((batch_size * ep_sizes, hidden_size), device='cuda:0', dtype=torch.bfloat16) / 10  # [256*8, 7168] (bf16)

        # 执行实际的算子 获取最优config
        best_info = benchmark_config(
            batch_size,
            global_num_experts,
            local_num_experts,
            shard_intermediate_size,
            hidden_size,
            top_k,
            ep_size=ep_sizes,
            x=x,
            w1=w1,
            w1_scale=w1_scale,
            w1_marlin=w1_marlin,
            w2=w2,
            w2_scale=w2_scale,
            w2_marlin=w2_marlin
        )   
        total_config_info.append(best_info)

    print("total_config_info: ", total_config_info)

    # 把嵌套字典转换为字符串（可读性更强的 JSON 字符串）
    data_for_excel = []
    for entry in total_config_info:
        row = {
            'batch_size': entry['batch_size'],
            'best_1_gemm1_time': entry['best_1_gemm1_time'],
            'best_1_gemm2_time': entry['best_1_gemm2_time'],
            'gemm1_config': json.dumps(entry['gemm1_config'], indent=2),
            'gemm2_config': json.dumps(entry['gemm2_config'], indent=2)
        }
        data_for_excel.append(row)

    # 转为 DataFrame
    df = pd.DataFrame(data_for_excel)

    # 写入 Excel 文件
    output_file = 'config_info_bested.xlsx'
    df.to_excel(output_file, index=False)

    print(f"已成功写入 {output_file}")