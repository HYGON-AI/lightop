# -*- coding: UTF-8 -*-
"""
@Date    :2025/9/11 14:23
@Author  :ZhangShuai
@Desc    :
"""
import torch
import time
import numpy as np

import lightop
from read_test_size import read_moe_size


def get_moe_gather_data(hidden_size, num_tokens, world_size, rank, num_shared_experts,
                        dp_size, sp_size, num_experts, topk, ep_size, use_half=False):
    """生成miopen_moe_gather测试数据"""
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

    scatter_tokens = torch.randn(real_scatter_tokens, hidden_size, dtype=torch.bfloat16).cuda()
    # 生成token偏移量，范围在[0, num_tokens-1]之间
    token_offset = torch.randint(0, 2, (real_scatter_tokens,), dtype=torch.int32).cuda()

    return scatter_tokens, token_offset, real_scatter_tokens


def test_speed(hidden_size, num_tokens, world_size, num_shared_experts, dp_size, sp_size,
               num_experts, topk, ep_size, rank=0, use_half=False, forward_only=False):
    if not forward_only:
        raise ValueError(f'当前仅支持前向性能及精度测试，暂不支持其他类型测试任务')
    # 获取测试数据
    scatterDesc, tokenoffsetDesc, real_scatter_tokens = get_moe_gather_data(
        hidden_size, num_tokens, world_size, rank, num_shared_experts,
        dp_size, sp_size, num_experts, topk, ep_size, use_half
    )

    # 预热
    y = lightop.miopen_moe_gather(
        scatterDesc,
        tokenoffsetDesc,
        int(real_scatter_tokens),
        int(num_tokens),
        int(hidden_size),
        int(topk),
        int(num_shared_experts)
    )
    torch.cuda.synchronize()

    # 性能测试
    start_time = time.time()
    for i in range(10):
        y = lightop.miopen_moe_gather(
            scatterDesc,
            tokenoffsetDesc,
            int(real_scatter_tokens),
            int(num_tokens),
            int(hidden_size),
            int(topk),
            int(num_shared_experts)
        )
    torch.cuda.synchronize()
    end_time = time.time()

    # 计算平均时间
    avg_time = (end_time - start_time) * 1000 / 10
    print(f'miopen_moe_gather time: {avg_time:.3f} ms')

    # 输出信息
    print(f"Input shape: {scatterDesc.shape}")
    print(f"Output shape: {y.shape}")
    print(f"Output dtype: {y.dtype}")
    print("-" * 50)


if __name__ == '__main__':
    print('=' * 30 + '测试miopen_moe_gather Size' + '=' * 30)
    forward_only = True
    use_half = True
    moe_size_path = r'../test_size/moe.csv'
    moe_params_list = read_moe_size(moe_size_path)
    for i, params in enumerate(moe_params_list):
        print(f'测试sizeid: {i}')
        test_speed(use_half=use_half, forward_only=forward_only, **params)
