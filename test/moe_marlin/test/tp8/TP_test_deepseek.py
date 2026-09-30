# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
from marlin_quant import w4a8_2_marlin_weight
from fused_marlin_moe_valid import fused_marlin_moe_valid
from typing import Optional
import lightop

def grouped_topk(hidden_states: torch.Tensor,
                 gating_output: torch.Tensor,
                 topk: int,
                 renormalize: bool,
                 num_expert_group: int = 0,
                 topk_group: int = 0,
                 scoring_func: str = "softmax",
                 e_score_correction_bias: Optional[torch.Tensor] = None):
    """保留原有的grouped_topk函数"""
    assert hidden_states.shape[0] == gating_output.shape[0], (
        "Number of tokens mismatch")

    if scoring_func == "softmax":
        scores = torch.softmax(gating_output, dim=-1)
    elif scoring_func == "sigmoid":
        scores = gating_output.sigmoid()
    else:
        raise ValueError(f"Unsupported scoring function: {scoring_func}")

    if e_score_correction_bias is not None:
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
        topk_weights = original_scores.gather(1, topk_ids)
    else:
        topk_weights, topk_ids = torch.topk(tmp_scores,
                                            k=topk,
                                            dim=-1,
                                            sorted=False)

    if renormalize:
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

    return topk_weights.to(torch.float32), topk_ids.to(torch.int32)

# 定义测试参数组合
@pytest.mark.parametrize("tp_size", [8])
@pytest.mark.parametrize("num_tokens", [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384])
# @pytest.mark.parametrize("num_tokens", [1, 2, 4, 8])
def test_fused_marlin_moe(tp_size, num_tokens):
    """测试fused marlin moe的正确性"""
    # 测试参数设置
    top_k = 8
    global_num_experts = 256
    hidden_size = 7168
    shard_intermediate_size = 4096 // tp_size

    # 确保CUDA可用
    assert torch.cuda.is_available(), "CUDA is not available for testing"
    device = torch.device(f'cuda:0')

    # 生成gemm1的相关数据
    w1 = torch.randint(-127, 127, 
                      (global_num_experts, shard_intermediate_size, hidden_size // 2), 
                      device=device, dtype=torch.int8)
    w1_scale = torch.randn((global_num_experts, shard_intermediate_size , 1), 
                          dtype=torch.float32, device=device)

    # 处理gemm1权重矩阵 python实现
    w1_marlin_list = []
    for i in range(w1.shape[0]):
        w1_marlin_in = w4a8_2_marlin_weight(w1[i])
        w1_marlin_list.append(w1_marlin_in)
    w1_marlin = torch.stack(w1_marlin_list, dim=0)

    # 生成gemm2的相关数据
    w2 = torch.randint(-127, 127, 
                      (global_num_experts, hidden_size, shard_intermediate_size // 4), 
                      device=device, dtype=torch.int8)
    w2_scale = torch.randn((global_num_experts, hidden_size, 1), 
                          dtype=torch.float32, device=device)

    # 处理gemm2权重矩阵 python 实现
    w2_marlin_list = []
    for i in range(w2.shape[0]):
        w2_marlin_in = w4a8_2_marlin_weight(w2[i])
        w2_marlin_list.append(w2_marlin_in)
    w2_marlin = torch.stack(w2_marlin_list, dim=0)

    # 生成输入数据
    x = torch.randn((num_tokens, hidden_size), device=device, dtype=torch.bfloat16) / 10

    # 计算topk权重和索引
    score = torch.randn((num_tokens, global_num_experts), device=device, dtype=torch.float32)
    topk_weights, topk_ids = grouped_topk(x, score, topk=top_k, renormalize=True, 
                                         num_expert_group=8, topk_group=4)

    # 执行验证
    is_valid = fused_marlin_moe_valid(                      
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

    # 断言验证结果为True
    assert is_valid, f"Validation failed for tp_size={tp_size}, num_tokens={num_tokens}"