# SPDX-License-Identifier: Apache-2.0
from tqdm import tqdm  # 导入 tqdm
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
import random
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

import pandas as pd
import json
from openpyxl import Workbook

is_print = False

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

def get_gemm1_config(num_tokens, topk, num_experts):
    test_config = []
    # ======================= 汇编config =========================
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
    # return test_config
    # # ======================= 手写config ==========================
    test_config.append(
                        {
                            "MODE": 402,
                            "DELTA": 1,
                            "BLOCK_SIZE_M": 16,
                            "thread_n": 256,
                            "thread_k": 128,
                            "blocks_per_cu": 1 
                        }
                    )
    test_config.append(
                        {
                            "MODE": 403,
                            "DELTA": 1,
                            "BLOCK_SIZE_M": 16,
                            "thread_n": 256,
                            "thread_k": 128,
                            "blocks_per_cu": 1 
                        }
                    )
    test_config.append(
                        {
                            "MODE": 413,
                            "DELTA": 1,
                            "BLOCK_SIZE_M": 16,
                            "thread_n": 256,
                            "thread_k": 128,
                            "blocks_per_cu": 1 
                        }
                    )
    test_config.append(
                        {
                            "MODE": 414,
                            "DELTA": 1,
                            "BLOCK_SIZE_M": 16,
                            "thread_n": 256,
                            "thread_k": 128,
                            "blocks_per_cu": 1 
                        }
                    )
    test_config.append(
                        {
                            "MODE": 415,
                            "DELTA": 1,
                            "BLOCK_SIZE_M": 32,
                            "thread_n": 256,
                            "thread_k": 128,
                            "blocks_per_cu": 1 
                        }
                    )
    test_config.append(
                        {
                            "MODE": 416,
                            "DELTA": 1,
                            "BLOCK_SIZE_M": 32,
                            "thread_n": 256,
                            "thread_k": 128,
                            "blocks_per_cu": 1 
                        }
                    )
    test_config.append(
                        {
                            "MODE": 417,
                            "DELTA": 1,
                            "BLOCK_SIZE_M": 32,
                            "thread_n": 256,
                            "thread_k": 128,
                            "blocks_per_cu": 1 
                        }
                    )
    test_config.append(
                        {
                            "MODE": 418,
                            "DELTA": 1,
                            "BLOCK_SIZE_M": 32,
                            "thread_n": 256,
                            "thread_k": 128,
                            "blocks_per_cu": 1 
                        }
                    )
    test_config.append(
                        {
                            "MODE": 419,
                            "DELTA": 1,
                            "BLOCK_SIZE_M": 64,
                            "thread_n": 256,
                            "thread_k": 128,
                            "blocks_per_cu": 1 
                        }
                    )
    test_config.append(
                        {
                            "MODE": 420,
                            "DELTA": 1,
                            "BLOCK_SIZE_M": 64,
                            "thread_n": 256,
                            "thread_k": 128,
                            "blocks_per_cu": 1 
                        }
                    )
    test_config.append(
                        {
                            "MODE": 421,
                            "DELTA": 1,
                            "BLOCK_SIZE_M": 128,
                            "thread_n": 256,
                            "thread_k": 128,
                            "blocks_per_cu": 1 
                        }
                    )
    test_config.append(
                        {
                            "MODE": 422,
                            "DELTA": 1,
                            "BLOCK_SIZE_M": 128,
                            "thread_n": 256,
                            "thread_k": 128,
                            "blocks_per_cu": 1 
                        }
                    )
    # ======================= marlin config ========================
    for block_size_m in [16, 32, 48, 80, 128]:
        if num_tokens * topk / num_experts / block_size_m < 0.9:
            break

    if block_size_m != 128:
        base_config = {
                        "MODE": 500,
                        "DELTA": 1,
                        "BLOCK_SIZE_M": block_size_m,
                        }
        for threads_n, threads_k in [
                            (256, 128),
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
    else:
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

    return test_config

def get_gemm2_config(num_tokens, topk, num_experts):
    test_config = []
    # ======================= 汇编config =========================
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
    # return test_config
    # # ======================= 手写config ==========================
    test_config.append(
                        {
                            "MODE": 401,
                            "DELTA": 1,
                            "BLOCK_SIZE_M": 16,
                            "thread_n": 256,
                            "thread_k": 128,
                            "blocks_per_cu": 1 
                        }
                    )
    test_config.append(
                        {
                            "MODE": 402,
                            "DELTA": 1,
                            "BLOCK_SIZE_M": 16,
                            "thread_n": 256,
                            "thread_k": 128,
                            "blocks_per_cu": 1 
                        }
                    )
    test_config.append(
                        {
                            "MODE": 415,
                            "DELTA": 1,
                            "BLOCK_SIZE_M": 32,
                            "thread_n": 256,
                            "thread_k": 128,
                            "blocks_per_cu": 1 
                        }
                    )
    test_config.append(
                        {
                            "MODE": 416,
                            "DELTA": 1,
                            "BLOCK_SIZE_M": 32,
                            "thread_n": 256,
                            "thread_k": 128,
                            "blocks_per_cu": 1 
                        }
                    )
    test_config.append(
                        {
                            "MODE": 417,
                            "DELTA": 1,
                            "BLOCK_SIZE_M": 64,
                            "thread_n": 256,
                            "thread_k": 128,
                            "blocks_per_cu": 1 
                        }
                    )
    test_config.append(
                        {
                            "MODE": 418,
                            "DELTA": 1,
                            "BLOCK_SIZE_M": 64,
                            "thread_n": 256,
                            "thread_k": 128,
                            "blocks_per_cu": 1 
                        }
                        )   

    # ======================= marlin config ========================
    # 当前根据任务量最大选block_m
    for block_size_m in [16, 32, 48, 80, 128]:
        if num_tokens * topk / num_experts / block_size_m < 0.9:
            break

    if block_size_m != 128:
        base_config = {
                        "MODE": 500,
                        "DELTA": 1,
                        "BLOCK_SIZE_M": block_size_m,
                        }
        for threads_n, threads_k in [
                            (256, 128),
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
    else:
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

    return test_config

def benchmark_config(
    num_tokens: int, # num_toens
    num_experts: int,  # 256
    shard_intermediate_size: int, # 4096
    hidden_size: int, # 7168
    topk: int,  # 8
    x,
    w1,
    w1_scale,
    w1_marlin,
    w2,
    w2_scale,
    w2_marlin
) -> float:
    print("=============================", "num_tokens: ", num_tokens , "tuning", "=======================================")


    if is_print:
        print("x: ", x.size())
        print("w1: ", w1.size())
        print("w1_scale: ", w1_scale.size())
        print("w1_marlin: ", w1_marlin.size())
        print("w2: ", w2.size())
        print("w2_scale: ", w2_scale.size())
        print("w2_marlin: ", w2_marlin.size())

    # ========================== 获取遍历的所有config ======================================
    gemm1_config = get_gemm1_config(num_tokens, topk, num_experts)
    gemm2_config = get_gemm2_config(num_tokens, topk, num_experts)

    num_iters = 100

    # 声明最优执行时间和对应的config
    best_time = 1000000

    best_gemm1_time = 0.0
    best_gemm2_time = 0.0

    best_gemm1_config = None
    best_gemm2_config = None  

    # 遍历所有的config
    for cur_config_gemm1 in tqdm(gemm1_config, desc="num_tokens: " + str(num_tokens)):
        if is_print:
            print("gemm1 测试 config: ", cur_config_gemm1)
        block_m_1 = cur_config_gemm1["BLOCK_SIZE_M"]
        for cur_config_gemm2 in gemm2_config:  # 确定 gemm2 config
            block_m_2 = cur_config_gemm2["BLOCK_SIZE_M"]
            if(block_m_2 != block_m_1): # 如果 gemm2 config不等于 gemm1 config，则进入下一个config验证
                continue

            if is_print:
                print("     gemm2 测试 config: ", cur_config_gemm2)

            # ============================================ 准备 gating_output 和 score ======================================
            gating_output = torch.randn(num_iters, num_tokens, num_experts, dtype=torch.float32)
            score = torch.randn((num_tokens, num_experts), device="cuda", dtype=torch.float32)

            def prepare(i: int):
                score.copy_(gating_output[i])

            topk_weights, topk_ids = grouped_topk(score, topk, True, 8, 4)

            # 先进行准确度测试
            gemm1_flag, gemm2_flag = fused_marlin_moe_valid(                      
                                                            x,
                                                            w1,
                                                            w2,
                                                            w1_marlin,
                                                            w2_marlin,
                                                            w1_scale,
                                                            w2_scale,
                                                            topk_weights,
                                                            topk_ids,
                                                            cur_config_gemm1,
                                                            cur_config_gemm2
                                                            )
            if (not gemm1_flag) or (not gemm2_flag):# 正确性验证通过后，进行性能测试
                print("正确性验证失败！！！！")
                print("cur_config_gemm1: ", cur_config_gemm1)
                print("cur_config_gemm2: ", cur_config_gemm2)
                continue 

            # ==============================   性能测试   ===============================
            # 预热 10 轮
            for i in range(10):
                # 使用新的专家得分
                prepare(i)
                # 获取新专家得分的 算子时间
                topk_weights, topk_ids = grouped_topk(score, topk, True, 8, 4)
                for j in range(10): # 一次随机专家运行10次
                    fused_marlin_moe(
                                    x,
                                    w1_marlin,
                                    w2_marlin,
                                    w1_scale,
                                    w2_scale,
                                    score,
                                    topk_weights,
                                    topk_ids,
                                    cur_config_gemm1,
                                    cur_config_gemm2
                                    )


            # 统计100轮随机后 计算 的时间  
            gemm1_time_us_total = 0 
            gemm2_time_us_total = 0
            for i in range(num_iters):
                # 使用新的专家得分
                prepare(i)
                # 获取新专家得分的 算子时间
                topk_weights, topk_ids = grouped_topk(score, topk, True, 8, 4)
                for j in range(10): # 一次随机专家运行10次
                    gemm1_time_us, gemm2_time_us =  fused_marlin_moe(
                                                                    x,
                                                                    w1_marlin,
                                                                    w2_marlin,
                                                                    w1_scale,
                                                                    w2_scale,
                                                                    score,
                                                                    topk_weights,
                                                                    topk_ids,
                                                                    cur_config_gemm1,
                                                                    cur_config_gemm2
                                                                    )
                    
                    gemm1_time_us_total += gemm1_time_us
                    gemm2_time_us_total += gemm2_time_us
                    
            cur_config_gemm1_time = gemm1_time_us_total / num_iters / 10
            cur_config_gemm2_time = gemm2_time_us_total / num_iters / 10

            if(is_print):
                print("cur_config_gemm1_time: ", cur_config_gemm1_time, "cur_config_gemm2_time: ", cur_config_gemm2_time) 

            if ((cur_config_gemm1_time + cur_config_gemm2_time) < best_time):
                if is_print:
                    print("============ 替换最优config!!! ==============")
                    print("best_gemm1_time: ", cur_config_gemm1_time)
                    print("best_gemm2_time: ", cur_config_gemm2_time)
                    print("cur_config_gemm1: ", cur_config_gemm1)
                    print("cur_config_gemm2: ", cur_config_gemm2)

                best_time = cur_config_gemm1_time + cur_config_gemm2_time

                best_gemm1_time = cur_config_gemm1_time
                best_gemm2_time = cur_config_gemm2_time

                best_gemm1_config = cur_config_gemm1
                best_gemm2_config = cur_config_gemm2

    print("best_gemm1_time : ", best_gemm1_time, "best_gemm2_time : ", best_gemm2_time)
    print("best_gemm1_config : ", best_gemm1_config, "best_gemm2_config : ", best_gemm2_config)

    best_info = {
        "num_tokens": num_tokens,
        "best_gemm1_time": best_gemm1_time,
        "best_gemm1_config": best_gemm1_config,
        "best_gemm2_time": best_gemm2_time,
        "best_gemm2_config": best_gemm2_config,
    }

    return best_info

# 主函数
if __name__ == "__main__":
    set_random_seed(seed=0)
    
    num_experts = 256
    hidden_size = 7168
    top_k = 8

    # for tp_sizes in [1, 2, 4, 8]:
    for tp_sizes in [1]:
        shard_intermediate_size = 4096 // tp_sizes
        # ======================== 预先生成 gemm1 的相关数据 ==================
        w1 = torch.randint(-127, 127, (num_experts, shard_intermediate_size, hidden_size // 2), device='cuda:0', dtype=torch.int8)
        w1_scale = torch.randn((num_experts, shard_intermediate_size, 1), dtype=torch.float32, device='cuda:0')

        # 处理 gemm1 权重矩阵
        w1_marlin_list = []
        for i in range(w1.shape[0]):
            w1_marlin_in = w4a8_2_marlin_weight(w1[i])
            w1_marlin_list.append(w1_marlin_in)
        w1_marlin = torch.stack(w1_marlin_list, dim=0)

        # ======================== 预先生成 gemm2 的相关数据 ==================
        w2 = torch.randint(-127, 127, (num_experts, hidden_size, shard_intermediate_size // 4), device='cuda:0', dtype=torch.int8)
        w2_scale = torch.randn((num_experts, hidden_size, 1), dtype=torch.float32, device='cuda:0')

        # 处理 gemm2 权重矩阵
        w2_marlin_list = []
        for i in range(w2.shape[0]):
            w2_marlin_in = w4a8_2_marlin_weight(w2[i])
            w2_marlin_list.append(w2_marlin_in)
        w2_marlin = torch.stack(w2_marlin_list, dim=0)

        # 用于保存性能数据
        prof_results = []

        for num_tokens in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 6144, 8192, 16384]:
        # for num_tokens in [1, 2, 4, 8, 16, 32]:
        # for num_tokens in [32]:
            # 激活矩第一维度： batch_size * ep_size     
            x = torch.randn((num_tokens, hidden_size), device='cuda:0', dtype=torch.bfloat16) / 10  # [256*8, 7168] (bf16)
            # 遍历所有config从中选取性能最好的config, 并返回 最优的config 和时间
            best_config_info = benchmark_config(
                num_tokens,
                num_experts,
                shard_intermediate_size,
                hidden_size,
                top_k,
                x=x,
                w1=w1,
                w1_scale=w1_scale,
                w1_marlin=w1_marlin,
                w2=w2,
                w2_scale=w2_scale,
                w2_marlin=w2_marlin
            )

            prof_results.append(best_config_info)

        # print(prof_results)

        # 创建Excel工作簿和工作表
        wb = Workbook()
        ws = wb.active
        ws.title = "GEMM数据"  # 工作表名称

        # 表头（与数据字典的键对应）
        headers = [
            'num_tokens',
            'best_gemm1_time',
            'best_gemm1_config',
            'best_gemm2_time',
            'best_gemm2_config'
        ]
        # 写入表头（第一行）
        for col, header in enumerate(headers, 1):  # 列从1开始（Excel列索引从1起）
            ws.cell(row=1, column=col, value=header)

        # 写入数据（从第二行开始）
        for row_idx, item in enumerate(prof_results, 2):  # row_idx为行号（从2开始）
            for col_idx, key in enumerate(headers, 1):  # col_idx为列号
                value = item[key]
                # 若为字典（即config字段），转为JSON字符串
                if isinstance(value, dict):
                    value = json.dumps(value, ensure_ascii=False, indent=2)  # indent=2让格式更易读
                # 写入单元格
                ws.cell(row=row_idx, column=col_idx, value=value)

        file_name = "moe_w4a8_i8_marlin_tp" + str(tp_sizes) + ".xlsx"
        # 保存Excel文件
        wb.save(file_name)
        print("数据已成功写入 ", file_name)

        gemm1_n = shard_intermediate_size 
        gemm1_k = hidden_size 
        gemm2_n = hidden_size
        gemm2_k = shard_intermediate_size // 2 

        # 3. 构建gemm1_config的JSON结构
        gemm1_top_key = f"{gemm1_n}_{gemm1_k}"  # 顶层键（如"7168_1024"）
        gemm1_data = {gemm1_top_key: {}}        # 初始化结构

        # 3.1 遍历数据，填充gemm1_config
        for item in prof_results:
            num_tokens = str(item['num_tokens'])  # 子键：num_tokens转为字符串
            gemm1_config = item['best_gemm1_config']  # 对应的值：config字典
            gemm1_data[gemm1_top_key][num_tokens] = gemm1_config

        # 4. 构建gemm2_config的JSON结构（逻辑同上）
        gemm2_top_key = f"{gemm2_n}_{gemm2_k}"
        gemm2_data = {gemm2_top_key: {}}

        for item in prof_results:
            num_tokens = str(item['num_tokens'])
            gemm2_config = item['best_gemm2_config']
            gemm2_data[gemm2_top_key][num_tokens] = gemm2_config

        # 5. 写入JSON文件（带缩进，便于阅读）
        json_gemm1_name = "MOE_BLOCKINT4_CUDA_MARLIN_" + str(gemm1_n) + "_" + str(gemm1_k) + "_gfx936_64.json"
        with open(json_gemm1_name, "w", encoding="utf-8") as f:
            json.dump(gemm1_data, f, indent=4, ensure_ascii=False)

        json_gemm2_name = "MOE_BLOCKINT4_CUDA_MARLIN_" + str(gemm2_n) + "_" + str(gemm2_k) + "_gfx936_64.json"
        with open(json_gemm2_name, "w", encoding="utf-8") as f:
            json.dump(gemm2_data, f, indent=4, ensure_ascii=False)

        print("JSON文件生成完成：gemm1_config.json 和 gemm2_config.json")