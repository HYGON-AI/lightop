# -*- coding:utf-8 -*-
"""
author: niutao
date: 2025年07月30日
description: 测试conv_bias融合算子，输出torch_api, torch_prof, miopen_api, miopen_prof耗时。仅支持前向，FP16性能及精度测试，默认数据布局为NHWC。
"""

import torch
import torch.nn.functional as F
from lightop import miopen_conv_bias
from torch.profiler import profile, ProfilerActivity

from read_test_size import read_convfusion_size
from utils import verify_accuracy


# 初始化数据
def get_data(batch_size, in_channels, out_channels, kernel_h, kernel_w, Height, Width, use_half=False):
    if use_half:
        input_data = torch.rand((batch_size, in_channels, Height, Width)).cuda().half()
        input_data = input_data.to(memory_format=torch.channels_last)
        weight = torch.rand((out_channels, in_channels, kernel_h, kernel_w)).cuda().half()
        weight = weight.to(memory_format=torch.channels_last)
        bias = torch.rand(out_channels).cuda().half()
        add = torch.rand((batch_size, out_channels, Height, Width)).cuda().half()
        add = add.to(memory_format=torch.channels_last)

    else:
        raise ValueError('暂不支持 conv_bias 单精度测试')

    x1 = input_data.clone().detach()
    x2 = weight.clone().detach()
    x3 = bias.clone().detach()
    x4 = add.clone().detach()

    return x1, x2, x3, x4


def test_speed(batch_size, channels, Height, Width, out_channels, kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w, dilation_h, dilation_w, use_half=False, forward_only=False):
    if not forward_only:
        raise ValueError(f'当前仅支持前向性能及精度测试，暂不支持其他类型测试任务')

    input_data, weight, bias, add = get_data(batch_size, channels, out_channels, kernel_h, kernel_w, Height, Width, use_half=use_half)

    stride = (stride_h, stride_w)
    padding = (pad_h, pad_w)
    dilation = (dilation_h, dilation_w)

    # warmup 10次
    for i in range(10):
        y1 = F.conv2d(input_data, weight, bias, stride, padding, dilation, 1)

    # 测试torch API耗时
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    start_event.record()
    for i in range(10):
        y1 = F.conv2d(input_data, weight, bias, stride, padding, dilation, 1)
    end_event.record()
    torch.cuda.synchronize()
    torch_api_time = start_event.elapsed_time(end_event) / 10
    print(f'Self torch api time: {round(torch_api_time, 3)}ms')

    # 测试torch prof耗时
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
        y1 = F.conv2d(input_data, weight, bias, stride, padding, dilation, 1)
    print(prof.key_averages(group_by_input_shape=True).table(sort_by='self_cuda_time_total'))

    # 调用miopen api warmup
    for i in range(10):
        y2 = miopen_conv_bias(input_data, weight, bias, padding, stride, dilation)

    # 统计miopen API耗时
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    start_event.record()
    for i in range(10):
        y2 = miopen_conv_bias(input_data, weight, bias, padding, stride, dilation)
    end_event.record()
    torch.cuda.synchronize()
    miopen_api_time = start_event.elapsed_time(end_event) / 10
    print(f'Self miopen api time: {round(miopen_api_time, 3)}ms')

    # 测试miopen prof耗时
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
        y2 = miopen_conv_bias(input_data, weight, bias, padding, stride, dilation)
    print(prof.key_averages(group_by_input_shape=True).table(sort_by='self_cuda_time_total'))

    # 验证精度
    verify_accuracy(y1, y2, rtol=1e-2, atol=1e-5, use_half=use_half)
    print('=' * 100 + '\n')


if __name__ == '__main__':
    print('=' * 30 + '测试Conv_Bias Size' + '=' * 30)
    forward_only = True
    use_half = True
    fusion_size_path = r'../test_size/conv_bias.csv'
    fusion_params_list = read_convfusion_size(fusion_size_path)
    for i, params in enumerate(fusion_params_list):
        print(f'测试sizeid: {i}')
        test_speed(use_half=use_half, forward_only=forward_only, **params)
