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
import triton
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
from marlin_quant import w4a8_2_marlin_weight
# 引入适配后的 marlin fused_moe实现

from fused_marlin_moe_valid import fused_marlin_moe_valid
from fused_marlin_moe import fused_marlin_moe
import openpyxl
from openpyxl import Workbook  # 显式导入Workbook
from vllm.model_executor.layers.fused_moe.fused_moe import fused_topk

from vllm.platforms import current_platform
from vllm.utils import FlexibleArgumentParser
from vllm.scalar_type import scalar_types

FP8_DTYPE = torch.float8_e4m3fnuz if current_platform.is_rocm(
) else torch.float8_e4m3fn

is_print = False

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

def benchmark_config(
    num_tokens: int, # num_toens
    global_num_experts: int,  # 32
    local_num_experts,
    shard_intermediate_size: int, # 4096
    hidden_size: int, # 7168
    topk: int,  # 8
    ep_size: int, # 8
    ep_rank: int, # 8
    x,
    w1,
    w1_scale,
    w1_marlin,
    w2,
    w2_scale,
    w2_marlin
) -> float:
    print("===============================================================================")
    print("m = ", num_tokens, " config tuning......")

    num_iters = 100

    # ======================= 正确性验证  ====================
    #       num_experts_select = 1 正确性验证
    num_experts_select = 1
    topk_ids, topk_weights = get_ep_topk_ids_weight(
                                            num_tokens,
                                            topk, 
                                            global_num_experts,
                                            ep_size=ep_size,
                                            ep_rank=ep_rank,
                                            num_experts_select=num_experts_select,
                                            device=x.device,
                                            )
    local_num_experts, expert_map = determine_expert_map(
                                                        ep_size=ep_size,
                                                        ep_rank=ep_rank,
                                                        global_num_experts=global_num_experts
                                                        )
    is_true_1 = fused_marlin_moe_valid(                      
                            x,
                            w1,
                            w2,
                            w1_marlin,
                            w2_marlin,
                            w1_scale,
                            w2_scale,
                            topk_weights,
                            topk_ids,
                            global_num_experts=global_num_experts,
                            ep_size=ep_size,
                            expert_map=expert_map
                            )

    # ======================= num_experts_select = 8 正确性验证和性能测试 ============================
    num_experts_select = 8
    topk_ids, topk_weights = get_ep_topk_ids_weight(
                                            num_tokens,
                                            topk, 
                                            global_num_experts,
                                            ep_size=ep_size,
                                            ep_rank=ep_rank,
                                            num_experts_select=num_experts_select,
                                            device=x.device,
                                            )
    local_num_experts, expert_map = determine_expert_map(
                                                        ep_size=ep_size,
                                                        ep_rank=ep_rank,
                                                        global_num_experts=global_num_experts
                                                        )
    is_true_8 = fused_marlin_moe_valid(                      
                            x,
                            w1,
                            w2,
                            w1_marlin,
                            w2_marlin,
                            w1_scale,
                            w2_scale,
                            topk_weights,
                            topk_ids,
                            global_num_experts=global_num_experts,
                            ep_size=ep_size,
                            expert_map=expert_map
                            )

    if is_print:
        print("正确性测试结果: ", is_true_1, is_true_8)

    if (not is_true_1) and (not is_true_8): # 正确性验证不过时，跳过此 config
        print("正确性验证未通过！！！")



    # 预热 10 轮
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
        for j in range(10): # 一次随机专家运行10次
            fused_marlin_moe(                      
                            x,
                            w1_marlin,
                            w2_marlin,
                            w1_scale,
                            w2_scale,
                            topk_weights,
                            topk_ids,
                            global_num_experts=global_num_experts,
                            ep_size=ep_size,
                            expert_map=expert_map
                            )

    num_experts_select = 1
    gemm1_time_us_total_1 = 0 
    gemm2_time_us_total_1 = 0
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
            gemm1_time_us, gemm2_time_us = fused_marlin_moe(                      
                                                    x,
                                                    w1_marlin,
                                                    w2_marlin,
                                                    w1_scale,
                                                    w2_scale,
                                                    topk_weights,
                                                    topk_ids,
                                                    global_num_experts=global_num_experts,
                                                    ep_size=ep_size,
                                                    expert_map=expert_map
                                                    )
            
            gemm1_time_us_total_1 += gemm1_time_us
            gemm2_time_us_total_1 += gemm2_time_us

    cur_config_gemm1_time_1 = gemm1_time_us_total_1 / num_iters / 10
    cur_config_gemm2_time_1 = gemm2_time_us_total_1 / num_iters / 10

    #============================ num_experts_select = 8 的性能
    num_experts_select = 8

    gemm1_time_us_total_8 = 0 
    gemm2_time_us_total_8 = 0
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
            gemm1_time_us, gemm2_time_us = fused_marlin_moe(                      
                                                    x,
                                                    w1_marlin,
                                                    w2_marlin,
                                                    w1_scale,
                                                    w2_scale,
                                                    topk_weights,
                                                    topk_ids,
                                                    global_num_experts=global_num_experts,
                                                    ep_size=ep_size,
                                                    expert_map=expert_map
                                                    )
            
            gemm1_time_us_total_8 += gemm1_time_us
            gemm2_time_us_total_8 += gemm2_time_us

    cur_config_gemm1_time_8 = gemm1_time_us_total_8 / num_iters / 10
    cur_config_gemm2_time_8 = gemm2_time_us_total_8 / num_iters / 10

    if is_print:
        print("gemm1_1 time : ", cur_config_gemm1_time_1)
        print("gemm1_8 time : ", cur_config_gemm1_time_8)

        print("gemm2_1 time : ", cur_config_gemm2_time_1)
        print("gemm2_8 time : ", cur_config_gemm2_time_8)

    best_asm_1 = {
        "num_tokens": num_tokens,
        "num_experts_select": 1,
        "gemm1_time_1": cur_config_gemm1_time_1,
        "gemm2_time_1": cur_config_gemm2_time_1,
    }

    best_asm_8 = {
        "num_tokens": num_tokens,
        "num_experts_select": 8,
        "gemm1_time_8": cur_config_gemm1_time_8,
        "gemm2_time_8": cur_config_gemm2_time_8,
    }

    return best_asm_1, best_asm_8

# 将 权重生成部分移动到遍历外以 减少生成部分的开销
if __name__ == "__main__":

    ep_size = 16
    ep_rank = 0
    top_k = 8
    global_num_experts = 256
    hidden_size = 7168
    shard_intermediate_size = 4096

    local_num_experts = global_num_experts // ep_size

    # ======================== 预先 生成 gemm1 的 相关数据 ==================    
    w1 = torch.randint(-127, 127, (local_num_experts, shard_intermediate_size, hidden_size//2), device='cuda:0', dtype=torch.int8)
    w1_scale = torch.randn((local_num_experts, shard_intermediate_size ,1), dtype=torch.float32, device='cuda:0')

    # -------- 处理 gemm1 权重矩阵 ------------------
    w1_marlin_list = []
    for i in range(w1.shape[0]):
        w1_marlin_in = w4a8_2_marlin_weight(w1[i])
        w1_marlin_list.append(w1_marlin_in)
    w1_marlin = torch.stack(w1_marlin_list, dim=0)

    # ======================== 预先 生成 gemm2 的 相关数据 ==================    
    w2 = torch.randint(-127, 127, (local_num_experts, hidden_size, shard_intermediate_size // 4), device='cuda:0', dtype=torch.int8)
    w2_scale = (torch.randn((local_num_experts, hidden_size, 1), dtype=torch.float32, device='cuda:0'))
    # -------- 处理 gemm2 权重矩阵 ------------------
    w2_marlin_list = []
    for i in range(w2.shape[0]):
        w2_marlin_in = w4a8_2_marlin_weight(w2[i])
        w2_marlin_list.append(w2_marlin_in)
    w2_marlin = torch.stack(w2_marlin_list, dim=0)

    prof_results_1 = []
    prof_results_8 = []
    for num_tokens in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 6144, 8192, 16384]:
    # for num_tokens in [32]:
        x = torch.randn((num_tokens, hidden_size), device='cuda:0', dtype=torch.bfloat16) / 10 # [256*8, 7168] (bf16)
        
        if(is_print):
            print("x: ", x.size(), x.dtype)
            print('w1: ', w1.size(), w1.dtype)
            print("w1_marlin: ", w1_marlin.size(), w1_marlin.dtype)
            print('w1_scale: ', w1_scale.size(), w1_scale.dtype)

            print("w2: ", w2.size(), w2.dtype)
            print("w2_marlin: ", w2_marlin.size(), w2_marlin.dtype)
            print("w2_scale: ", w2_scale.size(), w2_scale.dtype)
    
        best_asm_1, best_asm_8 = benchmark_config(
                                                num_tokens, 
                                                global_num_experts, 
                                                local_num_experts,
                                                shard_intermediate_size, 
                                                hidden_size, 
                                                top_k, 
                                                ep_size=ep_size,
                                                ep_rank=ep_rank,
                                                x=x,
                                                w1=w1,
                                                w1_scale=w1_scale,
                                                w1_marlin=w1_marlin,
                                                w2=w2,
                                                w2_scale=w2_scale,
                                                w2_marlin=w2_marlin,
                                                )

        prof_results_1.append(best_asm_1)
        prof_results_8.append(best_asm_8)
    
    print("prof_results_1: ", prof_results_1)
    print("prof_results_8: ", prof_results_8)

    # 创建工作簿和工作表
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "性能测试结果"

    # 写入prof_results_1的表头和数据（从第1行开始）
    if prof_results_1:
        # 获取字典的键作为表头（第1行）
        headers_1 = list(prof_results_1[0].keys())
        for col, header in enumerate(headers_1, 1):  # 从第1列开始
            ws.cell(row=1, column=col, value=header)
        
        # 写入数据（从第2行开始）
        for row, data_dict in enumerate(prof_results_1, 2):
            for col, key in enumerate(headers_1, 1):
                ws.cell(row=row, column=col, value=data_dict[key])

    # 空一行分隔两个数据集
    next_row = len(prof_results_1) + 3  # 数据行+表头行+2行空行

    # 写入prof_results_8的表头和数据（从next_row行开始）
    if prof_results_8:
        # 获取字典的键作为表头
        headers_8 = list(prof_results_8[0].keys())
        for col, header in enumerate(headers_8, 1):
            ws.cell(row=next_row, column=col, value=header)
        
        # 写入数据
        for row, data_dict in enumerate(prof_results_8, next_row + 1):
            for col, key in enumerate(headers_8, 1):
                ws.cell(row=row, column=col, value=data_dict[key])

    # 调整列宽以适应内容
    for col in range(1, max(len(headers_1), len(headers_8)) + 1):
        ws.column_dimensions[openpyxl.utils.get_column_letter(col)].width = 20
    ep_size_str = str(ep_size)
    # 保存文件
    file_name = f"prof_ep{ep_size_str}.xlsx"
    wb.save(file_name)
    print(f"数据已成功写入 {file_name}")