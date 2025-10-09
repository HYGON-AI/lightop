# -*- coding: UTF-8 -*-
"""
@Date    :2025/9/23 11:11
@Author  :ZhangShuai
@Desc    :测试miopen_moe_swiglu_dynamic_quant函数
"""
import torch
import time
import numpy as np

import lightop
from read_test_size import read_moe_size


def get_moe_swiglu_data(hidden_size, num_tokens, world_size, rank, num_shared_experts,
                        dp_size, sp_size, num_experts, topk, ep_size, use_half=False):
    """生成miopen_moe_swiglu_dynamic_quant测试数据"""
    # 确保参数为整数类型
    hidden_size = int(hidden_size)
    num_tokens = int(num_tokens)
    world_size = int(world_size)
    num_shared_experts = int(num_shared_experts)
    topk = int(topk)

    # 计算相关参数
    num_tokens_per_rank = num_tokens * world_size
    allocated_tokens = num_tokens_per_rank * topk
    real_scatter_tokens = num_shared_experts * num_tokens_per_rank + allocated_tokens

    # 每个rank的专家数
    total_expert_num = int(num_shared_experts + (num_experts // ep_size))

    # 生成输入数据
    scatter_tokens = torch.randn(topk, hidden_size * 2, dtype=torch.bfloat16).cuda()
    smooth = torch.randn(total_expert_num, hidden_size, dtype=torch.float32).cuda()

    # 生成专家token计数和起始位置
    # 假设每个专家处理的token数大致平均分配
    # tokens_per_expert = real_scatter_tokens // total_expert_num
    experts_tokens_count = torch.randint(0, total_expert_num, (total_expert_num,), dtype=torch.int32).cuda()

    # 处理余数，将剩余的token分配给最后一个专家
    exports_tokens_start = torch.randint(0, total_expert_num, (total_expert_num,), dtype=torch.int32).cuda()
    return scatter_tokens, smooth, experts_tokens_count, exports_tokens_start


def test_speed(hidden_size, num_tokens, world_size, num_shared_experts, dp_size, sp_size,
               num_experts, topk, ep_size, rank=0, use_half=False, forward_only=False):
    if not forward_only:
        raise ValueError(f'当前仅支持前向性能及精度测试，暂不支持其他类型测试任务')

    # 获取测试数据
    scatter_tokens, smooth, experts_tokens_count, exports_tokens_start = get_moe_swiglu_data(
        hidden_size, num_tokens, world_size, rank, num_shared_experts,
        dp_size, sp_size, num_experts, topk, ep_size, use_half
    )

    # 预热
    y, per_token_scale = lightop.miopen_moe_swiglu_dynamic_quant(
        scatter_tokens,
        smooth,
        experts_tokens_count,
        exports_tokens_start,
        1.0,
    )
    torch.cuda.synchronize()

    # 性能测试
    start_time = time.time()
    for i in range(10):
        y, per_token_scale = lightop.miopen_moe_swiglu_dynamic_quant(
            scatter_tokens,
            smooth,
            experts_tokens_count,
            exports_tokens_start,
            1.0,
        )
    torch.cuda.synchronize()
    end_time = time.time()

    # 计算平均时间
    avg_time = (end_time - start_time) * 1000 / 10
    print(f'miopen_moe_swiglu_dynamic_quant time: {avg_time:.3f} ms')
    # 输出信息
    print(f"Input scatter_tokens shape: {scatter_tokens.shape}")
    print(f"Input smooth shape: {smooth.shape}")
    print(f"Output y shape: {y.shape}")
    print(f"Output per_token_scale shape: {per_token_scale.shape}")
    print(f"Output dtype: {y.dtype}")
    print("-" * 50)


if __name__ == '__main__':
    print('=' * 30 + '测试miopen_moe_swiglu_dynamic_quant Size' + '=' * 30)
    forward_only = True
    use_half = True
    moe_size_path = r'../test_size/moe_swiglu_dynamic_quant.csv'  # 修改为对应的CSV文件路径
    # 读取CSV参数
    moe_params_list = read_moe_size(moe_size_path)
    for i, params in enumerate(moe_params_list):
        print(f'测试sizeid: {i + 1}')
        test_speed(use_half=use_half, forward_only=forward_only, **params)