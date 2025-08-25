# -*- coding:utf-8 -*-
"""
author: niutao
date: 2025年08月11日
description：测试MHA算子。
"""

import lightop
import torch
import torch.nn.functional as F
from torch.profiler import profile, ProfilerActivity

from utils import verify_accuracy
from read_test_size import read_mha_size

# 执行性能测试时，注释掉下面三句命令；执行精度测试时，打开下面三句命令
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(True)


# 初始化数据
def get_data(batch_size, head_nums, sequence_length, dim, use_half=False):
    if use_half:
        input_q = torch.randn(batch_size, head_nums, sequence_length, dim).cuda().half()
        input_k = torch.randn(batch_size, head_nums, sequence_length, dim).cuda().half()
        input_v = torch.randn(batch_size, head_nums, sequence_length, dim).cuda().half()
    else:
        raise ValueError(f'暂不支持 MHA 单精度性能测试')

    Q1 = input_q.clone().detach()
    Q2 = input_q.clone().detach()
    K1 = input_k.clone().detach()
    K2 = input_k.clone().detach()
    V1 = input_v.clone().detach()
    V2 = input_v.clone().detach()

    return Q1, K1, V1, Q2, K2, V2


def test_speed(batch_size, head_nums, sequence_length, dim, use_half=False, forward_only=False):
    if not forward_only:
        raise ValueError(f'当前仅支持前向性能及精度测试，暂不支持其他类型测试任务')

    print(f'测试size, batch_size : {batch_size}, head_nums: {head_nums}, sequence_length: {sequence_length}, dim: {dim}')
    Q1, K1, V1, Q2, K2, V2 = get_data(batch_size, head_nums, sequence_length, dim, use_half=use_half)
    assert torch.equal(Q1, Q2)
    assert torch.equal(K1, K2)
    assert torch.equal(V1, V2)

    # 测试torch接口
    # warmup 10次
    for i in range(10):
        y1 = F.scaled_dot_product_attention(Q1, K1, V1, attn_mask=None, dropout_p=0.0, is_causal=False)

    # 测试torch API耗时
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    start_event.record()
    for i in range(10):
        y1 = F.scaled_dot_product_attention(Q1, K1, V1, None, 0.0, False)
    end_event.record()
    torch.cuda.synchronize()
    torch_api_time = start_event.elapsed_time(end_event) / 10
    print(f'Self torch api time: {round(torch_api_time, 3)}ms')

    # 测试torch prof耗时
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
        y1 = F.scaled_dot_product_attention(Q1, K1, V1, None, 0.0, False)
    print(prof.key_averages(group_by_input_shape=True).table(sort_by='self_cuda_time_total'))

    # 测试miopen接口，调用miopen api warmup
    for i in range(10):
        y2 = lightop.miopen_scaled_dot_product_attention(Q2, K2, V2)

    # 统计miopen API耗时
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    start_event.record()
    for i in range(10):
        y2 = lightop.miopen_scaled_dot_product_attention(Q2, K2, V2)
    end_event.record()
    torch.cuda.synchronize()
    miopen_api_time = start_event.elapsed_time(end_event) / 10
    print(f'Self miopen api time: {round(miopen_api_time, 3)}ms')

    # 测试miopen prof耗时
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
        y2 = lightop.miopen_scaled_dot_product_attention(Q2, K2, V2)
    print(prof.key_averages(group_by_input_shape=True).table(sort_by='self_cuda_time_total'))

    # 验证精度
    print(f'y1_shape: {y1.shape}, y2_shape: {y2.shape}')
    verify_accuracy(y1, y2, rtol=1e-1, atol=5e-2, use_half=use_half)


if __name__ == "__main__":
    print('=' * 30 + 'Test MHA Size' + '=' * 30)
    forward_only = True
    use_half = True

    mha_size_path = r'../test_size/mha.csv'
    mha_size_list = read_mha_size(mha_size_path)
    for idx, params in enumerate(mha_size_list):
        print(f'size id: {idx}')
        test_speed(**params, use_half=use_half, forward_only=forward_only)
        print('=' * 100)

