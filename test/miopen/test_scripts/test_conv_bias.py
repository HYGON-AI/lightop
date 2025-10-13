# -*- coding:utf-8 -*-
"""
author: niutao
date: 2025年07月30日
description: 测试conv_bias融合算子，输出torch_api, torch_prof, miopen_api, miopen_prof耗时。仅支持前向，FP16性能及精度测试，默认数据布局为NHWC。
"""

from itertools import product

import torch
import torch.nn.functional as F
from lightop import miopen_conv_bias
from torch.profiler import profile, ProfilerActivity

from read_test_size import read_convfusion_size
from utils import verify_accuracy


dtype_map = {
    'FP16': torch.float16,
    'FP32': torch.float32
    }


def get_data(batchsize, in_channels, out_channels, kernel_h, kernel_w, Height, Width, data_layout, precision_type):
    """
    创建指定配置的张量数据

    Args:
        batchsize: 批量大小
        in_channels: 输入通道数
        out_channels: 输出通道数
        kernel_h: 卷积核高度
        kernel_w: 卷积核宽度
        Height: 输入数据高度
        Weight: 输入数据宽度
        data_layout: 'NHWC' 或 'NCHW'
        precision_type: 'FP16', 'BF16', 'FP32', 'TF32'
        device: 指定设备，默认为 CUDA（如果可用）

    returns:
        x1_tensor, x2_tensor, x3_tensor, x4_tensor
    """

    # 指定设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 创建浮点精度类型
    dtype = dtype_map.get(precision_type, torch.float16)
    input_data = torch.rand((batchsize, in_channels, Height, Width), device=device, dtype=dtype)
    weight = torch.rand((out_channels, in_channels, kernel_h, kernel_w), device=device, dtype=dtype)
    bias = torch.rand((out_channels), device=device, dtype=dtype)
    add = torch.rand((batchsize, out_channels, Height, Width), device=device, dtype=dtype)


    # 设置布局
    if data_layout == 'NCHW':
        input_data = input_data.to(memory_format=torch.contiguous_format)
        weight = weight.to(memory_format=torch.contiguous_format)
        add = add.to(memory_format=torch.contiguous_format)
    elif data_layout == 'NHWC':
        input_data = input_data.to(memory_format=torch.channels_last)
        weight = weight.to(memory_format=torch.channels_last)
        add = add.to(memory_format=torch.channels_last)

    x1 = input_data.clone().detach()
    x2 = weight.clone().detach()
    x3 = bias.clone().detach()
    x4 = add.clone().detach()
    return x1, x2, x3, x4


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
    verify_accuracy(y1, y2, rtol=1e-2, atol=1e-5, precision=data_type)
    print('=' * 100 + '\n')


if __name__ == '__main__':
    print('=' * 30 + '测试Conv_Bias Size' + '=' * 30)
    forward_only = True
    precision_list = ['FP16']
    data_layout_list = ['NHWC']
    fusion_size_path = r'../test_size/conv_bias.csv'
    fusion_params_list = read_convfusion_size(fusion_size_path)
    for precision, data_layout in list(product(precision_list, data_layout_list)):
        for i, params in enumerate(fusion_params_list):
            print(f'sizeid: {i}')
            print(f'test_type: {precision}_{data_layout}')
            test_speed(**params, layout_type=data_layout, data_type=precision, forward_only=forward_only)
