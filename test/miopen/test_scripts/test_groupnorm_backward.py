# -*- coding:utf-8 -*-
"""
author: niutao
date: 2025年09月25日
description: 测试groupnorm的miopen lightop bwd接口耗时，测试size布局支持NHWC及NCHW，精度支持fp32，mode仅支持1。
"""

import torch
from itertools import product

from lightop import miopen_groupnorm_backward, groupnorm_backward
from read_test_size import read_gn_size
from utils import verify_accuracy, find_topk_errors

# 创建全局变量dtype_map
dtype_map = {
    'FP16': torch.float16,
    'FP32': torch.float32,
}


def get_data(batchsize, seqs, cols, rows, num_groups, data_layout, precision_type):
    """
    创建指定配置的张量数据

    Args:
        batchsize: 批量大小
        seqs: 序列长度/通道数
        cols: 宽度
        rows: 高度
        num_groups: 分组的组数
        data_layout: 'NHWC' 或 'NCHW'
        precision_type: 'FP16', 'BF16', 'FP32', 'TF32'
        device: 指定设备，默认为 CUDA（如果可用）

    returns:
        input_tensor, x2_tensor, grad_tensor
    """

    # 指定设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 创建浮点精度类型
    dtype = dtype_map.get(precision_type, torch.float16)
    x = torch.randn((batchsize, seqs, cols, rows), device=device, dtype=dtype)
    dy = torch.randn((batchsize, seqs, cols, rows), device=device, dtype=dtype)
    weight = torch.randn((1, seqs, 1, 1,), device=device, dtype=dtype)
    mean = torch.randn((batchsize * num_groups, 1, 1, 1), device=device, dtype=dtype)
    rstd = torch.randn((batchsize * num_groups, 1, 1, 1), device=device, dtype=dtype)

    # 设置布局
    if data_layout == 'NCHW':
        x = x.to(memory_format=torch.contiguous_format)
        dy = dy.to(memory_format=torch.contiguous_format)
        x_ = x.clone()
        dy_ = dy.clone()
    elif data_layout == 'NHWC':
        x = x.to(memory_format=torch.channels_last)
        dy = dy.to(memory_format=torch.channels_last)
        x_ = x.to(memory_format=torch.contiguous_format)
        dy_ = dy.to(memory_format=torch.contiguous_format)

    return x, dy, x_, dy_, weight, mean, rstd


def test_speed(batch_size, seqs, cols, rows, num_groups, repeat_num, precision_type, data_layout, backward_only=False):
    if not backward_only:
        raise ValueError(f'当前仅支持反向性能及精度测试，暂不支持其他类型测试任务')

    print(f'size大小 n: {batch_size}, c: {seqs}, H: {cols}, W: {rows}, groups: {num_groups}')
    input_tensor, dy, input_tensor_, dy_, weight, mean, rstd = get_data(batch_size, seqs, cols, rows, num_groups, data_layout,
                                                    precision_type)

    # 测试lightop原库接口性能
    for i in range(10):
        groupnorm_backward(input_tensor_, dy_, weight, mean, rstd)

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    start_event.record()
    for i in range(repeat_num):
        lop_output = groupnorm_backward(input_tensor_, dy_, weight, mean, rstd)
    end_event.record()
    torch.cuda.synchronize()
    lop_time = start_event.elapsed_time(end_event)

    # 测试miopen_lightop接口性能
    for i in range(10):
        miopen_groupnorm_backward(input_tensor, dy, weight, mean, rstd, num_groups)

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    start_event.record()
    for i in range(repeat_num):
        mlop_output = miopen_groupnorm_backward(input_tensor, dy, weight, mean, rstd, num_groups)
    end_event.record()
    torch.cuda.synchronize()
    mlop_time = start_event.elapsed_time(end_event)

    print('lop_time:', format(lop_time / repeat_num, '.3f'), 'ms. mlop_time:',
          format(mlop_time / repeat_num, '.3f'), 'ms. speedup:', format(lop_time / mlop_time - 1, '.2f'))

    # 精度验证
    print(f'lop_shape: {lop_output[0].shape}, mlop_shape: {mlop_output[0].shape}')
    assert lop_output[0].shape == mlop_output[0].shape, 'groupnorm_backward API的输出dx shape与输入shape不一致，请检查'
    if data_layout == 'NCHW':
        verify_accuracy(lop_output[0], mlop_output[0], rtol=5e-3, atol=5e-3, precision=precision)
        verify_accuracy(lop_output[1], mlop_output[1], rtol=5e-3, atol=5e-3, precision=precision)
        verify_accuracy(lop_output[2], mlop_output[2], rtol=5e-3, atol=5e-3, precision=precision)
    else:
        lop_output_ = lop_output[0].permute(0, 2, 3, 1).contiguous(memory_format=torch.channels_last)
        mlop_output_ = mlop_output[0].permute(0, 2, 3, 1).contiguous(memory_format=torch.channels_last)
        verify_accuracy(lop_output_, mlop_output_, rtol=5e-3, atol=5e-3, precision=precision)
        verify_accuracy(lop_output[1], mlop_output[1], rtol=5e-3, atol=5e-3, precision=precision)
        verify_accuracy(lop_output[2], mlop_output[2], rtol=5e-3, atol=5e-3, precision=precision)


if __name__ == '__main__':
    print('=' * 30 + '测试GN backward size' + '=' * 30)
    # 设置基础参数
    precision_list = ['FP32']
    data_layout_list = ['NCHW', 'NHWC']
    repeat_num = 100
    gn_size_path = r'../test_size/groupnorm.csv'
    gn_size_list = read_gn_size(gn_size_path)
    for precision, data_layout in list(product(precision_list, data_layout_list)):
        for size_id, params in enumerate(gn_size_list):
            print(f'size_id: {size_id}')
            print(f'test_type: {precision}_{data_layout}')
            test_speed(**params, repeat_num=repeat_num, precision_type=precision, data_layout=data_layout,
                       backward_only=True)
            print('-' * 50)
