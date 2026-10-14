# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
from marlin_quant import w4a8_2_marlin_weight
from fused_marlin_moe_valid import fused_marlin_moe_valid
from typing import Optional
import lightop

def determine_expert_map(
        ep_size: int, ep_rank: int,
        global_num_experts: int) -> tuple[int, Optional[torch.Tensor]]:
    """计算专家映射关系（保留原逻辑）"""
    assert ep_size > 0
    if ep_size == 1:
        return (global_num_experts, None)

    local_num_experts = global_num_experts // ep_size

    # 创建全局到本地专家的映射，-1表示非当前rank的专家
    expert_map = torch.full((global_num_experts, ), -1, dtype=torch.int32)
    if ep_rank < (ep_size - 1):
        expert_map[ep_rank * local_num_experts : (ep_rank + 1) * local_num_experts] = \
            torch.arange(0, local_num_experts, dtype=torch.int32)
    else:
        local_num_experts = (global_num_experts - ep_rank * local_num_experts)
        expert_map[-local_num_experts:] = torch.arange(0, local_num_experts, dtype=torch.int32)
    
    expert_map = expert_map.cuda(0)
    return (local_num_experts, expert_map)


def get_ep_topk_ids_weight(
    M,
    top_k,
    global_num_experts,
    ep_size=8,
    ep_rank=0,
    num_experts_select=1,
    device="cuda:0"
):
    """生成专家并行场景下的topk_ids和权重（保留原逻辑）"""
    assert global_num_experts % ep_size == 0, "global_num_experts must be divisible by ep_size"
    assert num_experts_select <= top_k, "num_experts_select cannot exceed top_k"
    assert ep_rank < ep_size, "ep_rank must be less than ep_size"
    
    local_num_experts = global_num_experts // ep_size
    topk_ids = torch.full((M, top_k), global_num_experts - 1, dtype=torch.int32, device=device)

    # 生成当前rank上的本地专家索引
    local_experts_select = torch.stack([
        torch.randperm(local_num_experts, device=device)[:num_experts_select] 
        for _ in range(M)
    ])

    # 插入本地专家并打乱顺序
    topk_ids[:, :num_experts_select] = local_experts_select
    rows, cols = topk_ids.shape
    rand_indices = torch.argsort(torch.rand(rows, cols, device=device), dim=1)
    shuffled_topk_ids = topk_ids.gather(1, rand_indices)

    # 生成随机权重
    topk_weights = torch.randn((M, top_k), dtype=torch.float32, device=device)

    return shuffled_topk_ids, topk_weights


# 测试参数组合：ep_size、num_tokens、num_experts_select
# @pytest.mark.parametrize("num_experts_select", [1, 2, 3, 4, 5, 6, 7, 8])
@pytest.mark.parametrize("num_experts_select", [1, 8])
# @pytest.mark.parametrize("num_tokens", [16, 32])  # 可扩展更多值
@pytest.mark.parametrize("num_tokens", [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384])  # 可扩展更多值
# @pytest.mark.parametrize("ep_size", [8, 16, 32])
@pytest.mark.parametrize("ep_size", [8])
def test_fused_marlin_moe_ep(ep_size, num_tokens, num_experts_select):
    """测试专家并行场景下的fused marlin moe正确性"""
    # 基础参数配置
    ep_rank = 0
    top_k = 8
    global_num_experts = 256
    hidden_size = 7168
    shard_intermediate_size = 4096

    # 检查CUDA可用性
    assert torch.cuda.is_available(), "CUDA is required for this test"
    device = torch.device("cuda:0")

    # 计算本地专家数量和映射
    local_num_experts, expert_map = determine_expert_map(
        ep_size=ep_size,
        ep_rank=ep_rank,
        global_num_experts=global_num_experts
    )

    # 生成gemm1相关数据
    w1 = torch.randint(
        -127, 127,
        (local_num_experts, shard_intermediate_size, hidden_size // 2),
        device=device, dtype=torch.int8
    )
    w1_scale = torch.randn(
        (local_num_experts, shard_intermediate_size, 1),
        dtype=torch.float32, device=device
    )
    # 处理gemm1权重为marlin格式 python
    w1_marlin = torch.stack([
        w4a8_2_marlin_weight(w1[i]) for i in range(w1.shape[0])
    ], dim=0)


    # 生成gemm2相关数据
    w2 = torch.randint(
        -127, 127,
        (local_num_experts, hidden_size, shard_intermediate_size // 4),
        device=device, dtype=torch.int8
    )
    w2_scale = torch.randn(
        (local_num_experts, hidden_size, 1),
        dtype=torch.float32, device=device
    )
    # 处理gemm2权重为marlin格式  python格式
    w2_marlin = torch.stack([
        w4a8_2_marlin_weight(w2[i]) for i in range(w2.shape[0])
    ], dim=0)

    # 生成输入数据
    x = torch.randn(
        (num_tokens, hidden_size),
        device=device, dtype=torch.bfloat16
    ) / 10  # 缩小值范围，避免数值问题

    # 生成topk_ids和权重
    topk_ids, topk_weights = get_ep_topk_ids_weight(
        M=num_tokens,
        top_k=top_k,
        global_num_experts=global_num_experts,
        ep_size=ep_size,
        ep_rank=ep_rank,
        num_experts_select=num_experts_select,
        device=device
    )

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
        global_num_experts=global_num_experts,
        ep_size=ep_size,
        expert_map=expert_map
    )

    # 断言验证结果（失败时显示详细参数）
    assert is_valid, (
        f"Validation failed! "
        f"ep_size={ep_size}, num_tokens={num_tokens}, "
        f"num_experts_select={num_experts_select}"
    )