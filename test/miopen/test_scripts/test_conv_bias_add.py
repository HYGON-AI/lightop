# -*- coding:utf-8 -*-
"""
author: niutao
date: 2025年07月30日
description: 测试conv_bias_add融合算子，输出torch_api, torch_prof, miopen_api, miopen_prof耗时。仅支持前向，FP16性能及精度测试，默认数据布局为NHWC。
"""

from itertools import product

import torch
import torch.nn.functional as F
from lightop import miopen_conv_bias_add
from torch.profiler import profile, ProfilerActivity

from read_test_size import read_convfusion_size
from utils import verify_accuracy, get_data


def test_speed(batch_size, channels, Height, Width, out_channels, kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w, dilation_h, dilation_w,
               layout_type, data_type, forward_only=False):
    if not forward_only:
        raise ValueError(f'当前仅支持前向性能及精度测试，暂不支持其他类型测试任务')

    input_data, weight, bias, add = get_data(batch_size, channels, out_channels, kernel_h, kernel_w, Height, Width, layout_type, data_type)

    stride = (stride_h, stride_w)
    padding = (pad_h, pad_w)
    dilation = (dilation_h, dilation_w)

    # warmup 10次
    for i in range(10):
        out = F.conv2d(input_data, weight, bias, stride, padding, dilation, 1)
        y1 = out + add

    # 测试torch API耗时
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    start_event.record()
    for i in range(10):
        out = F.conv2d(input_data, weight, bias, stride, padding, dilation, 1)
        y1 = out + add
    end_event.record()
    torch.cuda.synchronize()
    torch_api_time = start_event.elapsed_time(end_event) / 10
    print(f'Self torch api time: {round(torch_api_time, 3)}ms')

    # 测试torch prof耗时
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
        out = F.conv2d(input_data, weight, bias, stride, padding, dilation, 1)
        y1 = out + add
    print(prof.key_averages(group_by_input_shape=True).table(sort_by='self_cuda_time_total'))

    # 调用miopen api warmup
    for i in range(10):
        y2 = miopen_conv_bias_add(input_data, weight, bias, add, padding, stride, dilation)

    # 统计miopen API耗时
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    start_event.record()
    for i in range(10):
        y2 = miopen_conv_bias_add(input_data, weight, bias, add, padding, stride, dilation)
    end_event.record()
    torch.cuda.synchronize()
    miopen_api_time = start_event.elapsed_time(end_event) / 10
    print(f'Self miopen api time: {round(miopen_api_time, 3)}ms')

    # 测试miopen prof耗时
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
        y2 = miopen_conv_bias_add(input_data, weight, bias, add, padding, stride, dilation)

    print(prof.key_averages(group_by_input_shape=True).table(sort_by='self_cuda_time_total'))

    # 验证精度
    verify_accuracy(y1, y2, rtol=1e-2, atol=1e-5, precision=data_type)
    print('=' * 100 + '\n')


if __name__ == '__main__':
    print('=' * 30 + '测试Conv_Bias_Add Size' + '=' * 30)
    forward_only = True
    precision_list = ['FP16']
    data_layout_list = ['NHWC']
    fusion_size_path = r'../test_size/conv_bias_add.csv'
    fusion_params_list = read_convfusion_size(fusion_size_path)
    for precision, data_layout in list(product(precision_list, data_layout_list)):
        for i, params in enumerate(fusion_params_list):
            print(f'sizeid: {i}')
            print(f'test_type: {precision}_{data_layout}')
            test_speed(**params, layout_type=data_layout, data_type=precision, forward_only=forward_only)
