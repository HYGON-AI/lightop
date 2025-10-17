# -*- coding:utf-8 -*-
"""
author: niutao
date: 2025年08月02日
description：测试GNSilu算子，比较miopenGroupNorm(mode=1)+torch.silu()与miopenGroupNorm(mode=10)的性能比较。仅支持前向性能及精度验证，默认数据布局为NHWC。
"""

import torch
from itertools import product
from lightop import GNSilu, miopenGroupNorm

from read_test_size import read_gn_size
from utils import verify_accuracy


# 创建全局变量dtype_map, gn_silu仅支持fp16
dtype_map = {
    'FP16': torch.float16,
}


def get_data(batchsize, seqs, cols, rows, data_layout, precision_type):

    # 指定设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 创建浮点精度类型
    dtype = dtype_map.get(precision_type, torch.float16)
    x = torch.rand((batchsize, seqs, cols, rows), device=device, dtype=dtype)
    grad = torch.rand((batchsize, seqs, cols, rows), device=device, dtype=dtype)

    # 设置布局
    if data_layout == 'NCHW':
        x = x.to(memory_format=torch.contiguous_format)
        grad = grad.to(memory_format=torch.contiguous_format)
    elif data_layout == 'NHWC':
        x = x.to(memory_format=torch.channels_last)
        grad = grad.to(memory_format=torch.channels_last)

    x.requires_grad_(True)
    x1 = x.clone().requires_grad_()
    x2 = x.clone().requires_grad_()
    return x1, x2, grad


def get_model(seqs, num_groups, precision_type, mode):
    # 指定设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 指定精度类型
    if precision_type not in dtype_map:
        raise ValueError(f'LightOP gn_silu算子暂不支持{precision_type}精度测试, 请完善测试用例或修改精度类型')
    dtype = dtype_map[precision_type]

    # 设置随机种子
    torch.manual_seed(0)
    # 初始化模型中的weight及bias，LightOP库中的weight默认为全1，bias默认为全0
    if mode == 10:
        weight = torch.rand(seqs, device=device, dtype=dtype)
        bias = torch.rand(seqs, device=device, dtype=dtype)
    elif mode == 11:
        weight = torch.ones(seqs, device=device, dtype=dtype)
        bias = torch.zeros(seqs, device=device, dtype=dtype)

    # 创建模型实例
    miopen_gnsilu = miopenGroupNorm(num_groups, seqs, mode).to(device=device, dtype=dtype)
    lightop_gnsilu = GNSilu(num_groups, seqs, device=device, dtype=dtype)

    # 将初始化后的weight及bias数据传入model
    lightop_gnsilu.weight.data = weight.clone().detach()
    lightop_gnsilu.bias.data = bias.clone().detach()
    miopen_gnsilu.weight.data = weight.clone().detach()
    miopen_gnsilu.bias.data = bias.clone().detach()

    return lightop_gnsilu, miopen_gnsilu


def test_speed(batch_size, seqs, cols, rows, num_groups, repeat_num, precision_type, data_layout, mode,
               forward_only=False):
    if not forward_only:
        raise ValueError(f'当前仅支持前向性能及精度测试，暂不支持其他类型测试任务')

    x1, x2, grad = get_data(batch_size, seqs, cols, rows, data_layout, precision_type)
    lightop_model, miopen_model = get_model(seqs, num_groups, precision_type, mode)
    if forward_only:
        lightop_model = lightop_model.eval()
        miopen_model = miopen_model.eval()
    print(f'size大小 n: {batch_size}, c: {seqs}, H: {cols}, W: {rows}, groups: {num_groups}')

    # lightop model warm up
    for i in range(10):
        y1 = lightop_model(x1)
        if not forward_only:
            y1.backward(grad)

    # lightop model time count
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    start_event.record()
    for i in range(repeat_num):
        x1.grad = None
        lightop_model.weight.grad = None
        lightop_model.bias.grad = None
        y1 = lightop_model(x1)
        if not forward_only:
            y1.backward(grad)
    end_event.record()
    torch.cuda.synchronize()
    lop_time = start_event.elapsed_time(end_event)

    for i in range(10):
        y2 = miopen_model(x2)
        if not forward_only:
            y2.backward(grad)

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    start_event.record()
    for i in range(repeat_num):
        x2.grad = None
        miopen_model.weight.grad = None
        miopen_model.bias.grad = None
        y2 = miopen_model(x2)
        if not forward_only:
            y2.backward(grad)
    end_event.record()
    torch.cuda.synchronize()
    mlop_time = start_event.elapsed_time(end_event)
    print('lop_time:', format(lop_time / repeat_num, '.3f'), 'ms. mlop_time:',
          format(mlop_time / repeat_num, '.3f'), 'ms. speedup:', format(lop_time / mlop_time - 1, '.2f'))

    # 精度验证
    print(f'lightop_output_shape: {y1.shape}, lightop_miopen_output_shape: {y2[0].shape}')
    assert y1.shape == y2[0].shape, 'lightop接口与miopen lightop接口的输出tensor shape不一致'
    verify_accuracy(y1, y2[0], rtol=1e-2, atol=1e-2, precision=precision_type)


if __name__ == '__main__':
    print('=' * 30 + '测试GNSilu Size' + '=' * 30)
    # 设置基础参数
    forward_only = True
    mode_list = [10, 11]
    precision_list = ['FP16']
    data_layout_list = ['NHWC']
    repeat_num = 100
    gn_size_path = r'../test_size/groupnorm.csv'
    gn_size_list = read_gn_size(gn_size_path)
    for mode in mode_list:
        for precision, data_layout in list(product(precision_list, data_layout_list)):
            for size_id, params in enumerate(gn_size_list):
                print(f'size_id: {size_id}')
                print(f'test_type: mode{mode}_{precision}_{data_layout}')
                test_speed(**params, repeat_num=repeat_num, precision_type=precision, data_layout=data_layout,
                           mode=mode, forward_only=forward_only)
                print('-' * 50)