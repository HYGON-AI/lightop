# -*- coding: UTF-8 -*-
"""
@Date    :2025/9/12 10:56
@Author  :ZhangShuai
@Desc    :测试miopen_moe_scatter_dynamic_quant函数
"""

import torch
import time
import numpy as np
import gc

import lightop
from read_test_size import read_moe_size


def clear_gpu_memory():
    """彻底清理GPU内存"""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        gc.collect()
        torch.cuda.empty_cache()
        # 给CUDA一些时间完成清理
        time.sleep(0.1)


def get_moe_scatter_data(hidden_size, num_tokens, world_size, rank, num_shared_experts,
                         dp_size, sp_size, num_experts, topk, ep_size, use_half=False):
    """生成miopen_moe_scatter_dynamic_quant测试数据"""
    # 使用从CSV读取的参数
    num_tokens = int(num_tokens)
    hidden_size = int(hidden_size)
    topk = int(topk)
    num_shared_experts_per_rank = int(num_shared_experts)

    expert_start_idx = 0  # 可能需要根据rank计算
    expert_end_idx = 8
    num_experts_per_rank = expert_end_idx - expert_start_idx
    num_tokens_offset = 0  # 根据rank计算偏移
    num_tokens_per_rank = 0  # 每个rank的token数

    max_scatter_tokens = num_shared_experts_per_rank * num_tokens_per_rank + num_tokens * topk

    # 生成输入数据
    hidden = torch.randn(num_tokens, hidden_size, dtype=torch.bfloat16).cuda()
    selected_experts = torch.randint(0, 2, (num_tokens, topk), dtype=torch.int32).cuda()
    smooth = torch.randn(num_shared_experts_per_rank + num_experts_per_rank, hidden_size, dtype=torch.float32).cuda()
    moe_weights = torch.ones(num_tokens, topk, dtype=torch.float32, device="cuda")

    # 预分配输出张量
    out_scatter_tokens = torch.full((max_scatter_tokens, hidden_size), 0, dtype=torch.int8, device="cuda")
    out_scatter_per_token_scale = torch.full((max_scatter_tokens,), 0, dtype=torch.float32, device="cuda")
    out_scatter_tokens_offset = torch.full((max_scatter_tokens,), 0, dtype=torch.int32, device="cuda")
    out_experts_token_count = torch.full((num_shared_experts_per_rank + num_experts_per_rank,), 0, dtype=torch.int32,
                                         device="cuda"
                                         )
    out_experts_token_start = torch.full((num_shared_experts_per_rank + num_experts_per_rank,), 0, dtype=torch.int32,
                                         device="cuda")

    return (
        hidden, selected_experts, moe_weights, smooth,
        out_scatter_tokens, out_scatter_per_token_scale, out_scatter_tokens_offset,
        out_experts_token_count, out_experts_token_start,
        expert_start_idx, expert_end_idx, num_shared_experts_per_rank,
        num_tokens_offset, num_tokens_per_rank, num_tokens, max_scatter_tokens
    )


def test_speed_moe_scatter(hidden_size, num_tokens, world_size, num_shared_experts, dp_size, sp_size,
                           num_experts, topk, ep_size, rank=0, use_half=False, forward_only=False):
    if not forward_only:
        raise ValueError(f'当前仅支持前向性能及精度测试，暂不支持其他类型测试任务')
    # 测试前清理内存
    clear_gpu_memory()

    try:
        # 获取测试数据
        (hidden, selected_experts, moe_weights, smooth,
         out_scatter_tokens, out_scatter_per_token_scale, out_scatter_tokens_offset,
         out_experts_token_count, out_experts_token_start,
         expert_start_idx, expert_end_idx, num_shared_experts_val,
         num_tokens_offset, num_tokens_per_rank, num_tokens_val, max_scatter_tokens) = (get_moe_scatter_data
                                                                                        (hidden_size,
                                                                                         num_tokens,
                                                                                         world_size,
                                                                                         rank,
                                                                                         num_shared_experts,
                                                                                         dp_size,
                                                                                         sp_size,
                                                                                         num_experts,
                                                                                         topk,
                                                                                         ep_size,
                                                                                         use_half))

        # 预热
        result = lightop.miopen_moe_scatter_dynamic_quant(
            hidden,
            selected_experts,
            moe_weights,
            smooth,
            out_scatter_tokens,
            out_scatter_per_token_scale,
            out_scatter_tokens_offset,
            out_experts_token_count,
            out_experts_token_start,
            int(expert_start_idx),
            int(expert_end_idx),
            int(num_shared_experts_val),
            int(num_tokens_offset),
            int(num_tokens_per_rank),
            int(num_tokens_val),
        )
        # 解包结果
        scatter_tokens, scatter_per_token_scale, scatter_token_offset, experts_token_count, experts_token_start = result
        torch.cuda.synchronize()

        # 性能测试
        start_time = time.time()
        for i in range(2):  # 改为10次循环以获得更稳定的时间测量
            result = lightop.miopen_moe_scatter_dynamic_quant(
                hidden,
                selected_experts,
                moe_weights,
                smooth,
                out_scatter_tokens,
                out_scatter_per_token_scale,
                out_scatter_tokens_offset,
                out_experts_token_count,
                out_experts_token_start,
                int(expert_start_idx),
                int(expert_end_idx),
                int(num_shared_experts_val),
                int(num_tokens_offset),
                int(num_tokens_per_rank),
                int(num_tokens_val),
            )
            scatter_tokens, scatter_per_token_scale, scatter_token_offset, experts_token_count, experts_token_start = result

        torch.cuda.synchronize()
        end_time = time.time()
        # 计算平均时间
        avg_time = (end_time - start_time) * 1000 / 10
        print(f'miopen_moe_scatter_dynamic_quant time: {avg_time:.3f} ms')
        print(f"Input hidden shape: {hidden.shape}")
        print(f"Output scatter_tokens shape: {scatter_tokens.shape}")
        print(f"Output scatter_per_token_scale shape: {scatter_per_token_scale.shape}")
        print(f"Output scatter_token_offset shape: {scatter_token_offset.shape}")
        print(f"Output experts_token_count shape: {experts_token_count.shape}")
        print(f"Output experts_token_start shape: {experts_token_start.shape}")

    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print(f"内存不足，跳过该测试: hidden_size={hidden_size}, num_tokens={num_tokens}")
            return
        else:
            raise e
    finally:
        # 测试后清理内存
        clear_gpu_memory()


if __name__ == '__main__':
    print('=' * 30 + '测试miopen_moe_scatter_dynamic_quant Size' + '=' * 30)
    forward_only = True
    use_half = True
    moe_size_path = r'../test_size/moe.csv'
    moe_params_list = read_moe_size(moe_size_path)
    # 初始内存清理
    clear_gpu_memory()
    for i, params in enumerate(moe_params_list):
        print(f'测试sizeid: {i + 1}')
        test_speed_moe_scatter(use_half=use_half, forward_only=forward_only, **params)
        print("-" * 50)
    print('所有测试完成！')
