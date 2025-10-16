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

import openpyxl
from openpyxl import Workbook  # 显式导入Workbook
from vllm.model_executor.layers.fused_moe.fused_moe import fused_topk

from vllm.platforms import current_platform
from vllm.utils import FlexibleArgumentParser
from vllm.scalar_type import scalar_types

FP8_DTYPE = torch.float8_e4m3fnuz if current_platform.is_rocm(
) else torch.float8_e4m3fn

is_print = True

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

def benchmark_config(
    num_tokens: int, # num_toens
    global_num_experts: int,  # 32
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
    print("num_tokens = ", num_tokens, " test......")

    test_results = []

    score = torch.randn((num_tokens, global_num_experts), device="cuda", dtype=torch.float32)
    topk_weights, topk_ids = grouped_topk(x, score, topk, True, 8, 4)

    is_true = fused_marlin_moe_valid(                      
                            x,
                            w1,
                            w2,
                            w1_marlin,
                            w2_marlin,
                            w1_scale,
                            w2_scale,
                            topk_weights,
                            topk_ids,
                            global_num_experts
                            )

    if is_true:
        test_results.append({
                            "num_tokens": num_tokens,
                            "acc": True,
                            }) 
    else:
        test_results.append({
                            "num_tokens": num_tokens,
                            "acc": False,
                            }) 

    return test_results


if __name__ == "__main__":
    for tp_size in [1, 2, 4, 8]:
        test_results_total = []
        top_k = 8
        global_num_experts = 256
        hidden_size = 7168

        shard_intermediate_size = 4096 // tp_size

        # ======================== 预先 生成 gemm1 的 相关数据 ==================    
        w1 = torch.randint(-127, 127, (global_num_experts, shard_intermediate_size, hidden_size // 2), device='cuda:0', dtype=torch.int8)
        w1_scale = torch.randn((global_num_experts, shard_intermediate_size , 1), dtype=torch.float32, device='cuda:0')

        # -------- 处理 gemm1 权重矩阵 ------------------
        w1_marlin_list = []
        for i in range(w1.shape[0]):
            w1_marlin_in = w4a8_2_marlin_weight(w1[i])
            w1_marlin_list.append(w1_marlin_in)
        w1_marlin = torch.stack(w1_marlin_list, dim=0)

        # ======================== 预先 生成 gemm2 的 相关数据 ==================    
        w2 = torch.randint(-127, 127, (global_num_experts, hidden_size, shard_intermediate_size // 4), device='cuda:0', dtype=torch.int8)
        w2_scale = (torch.randn((global_num_experts, hidden_size, 1), dtype=torch.float32, device='cuda:0'))
        # -------- 处理 gemm2 权重矩阵 ------------------
        w2_marlin_list = []
        for i in range(w2.shape[0]):
            w2_marlin_in = w4a8_2_marlin_weight(w2[i])
            w2_marlin_list.append(w2_marlin_in)
        w2_marlin = torch.stack(w2_marlin_list, dim=0)

        
        # for num_tokens in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 6144, 8192, 16384]:
        # for num_tokens in [128, 256]:
        for num_tokens in [32]:
            x = torch.randn((num_tokens, hidden_size), device='cuda:0', dtype=torch.bfloat16) / 10 # [256*8, 7168] (bf16)
            test_results_cur = benchmark_config(
                                                    num_tokens, 
                                                    global_num_experts, 
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
            test_results_total.extend(test_results_cur)

        
        print("test_results_total: ", test_results_total)

        file_name = "test_w4a8_fused_moe_tp" + str(tp_size) + ".xlsx"
        # 列表转换为 pandas DataFrame（表格数据结构）
        df = pd.DataFrame(test_results_total)

        # 写入 Excel 文件（index=False 表示不写入行索引）
        df.to_excel(file_name, index=False, engine='openpyxl')

        print("数据已成功写入 ", file_name)