# -*- coding:utf-8 -*-
"""
author: niutao
date: 2025年08月01日
description: 测试groupnorm的lightop以及miopen耗时，仅支持forward性能及精度测试，且测试size布局默认为NHWC。
"""

import torch
from itertools import product

from lightop import GroupNorm, miopenGroupNorm
from read_test_size import read_gn_size
from utils import verify_accuracy


def get_data(batchsize, seqs, cols, rows, data_layout, precision_type):
    if precision_type == 'FP16':
        if data_layout == 'NHWC':
            x = torch.rand((batchsize, seqs, cols, rows)).to(memory_format=torch.channels_last).cuda().half()
            grad = torch.rand((batchsize, seqs, cols, rows)).to(memory_format=torch.channels_last).cuda().half()
        else:
            x = torch.rand((batchsize, seqs, cols, rows)).cuda().half()
            grad = torch.rand((batchsize, seqs, cols, rows)).cuda().half()
    elif precision_type == 'FP32':
        if data_layout == 'NHWC':
            x = torch.rand((batchsize, seqs, cols, rows)).to(memory_format=torch.channels_last).cuda()
            grad = torch.rand((batchsize, seqs, cols, rows)).to(memory_format=torch.channels_last).cuda()
        else:
            x = torch.rand((batchsize, seqs, cols, rows)).cuda()
            grad = torch.rand((batchsize, seqs, cols, rows)).cuda()
    else:
        raise ValueError(f'LightOP groupnorm算子暂不支持{precision_type}精度测试, 请完善测试用例或修改精度类型')

    x.requires_grad_(True)
    x1 = x.clone().detach().requires_grad_()
    x2 = x.clone().detach().requires_grad_()
    return x1, x2, grad


def get_model(seqs, num_groups, precision_type):
    if precision_type == 'FP16':
        miopen_groupnorm = miopenGroupNorm(num_groups, seqs, 1).cuda().half()
        lightop_groupnorm = GroupNorm(num_groups, seqs).cuda().half()
    elif precision_type == 'FP32':
        miopen_groupnorm = miopenGroupNorm(num_groups, seqs, 1).cuda()
        lightop_groupnorm = GroupNorm(num_groups, seqs).cuda()
    else:
        raise ValueError(f'LightOP groupnorm算子暂不支持{precision_type}精度测试, 请完善测试用例或修改精度类型')

    return lightop_groupnorm, miopen_groupnorm


def test_speed(batch_size, seqs, cols, rows, num_groups, repeat_num, precision_type, data_layout, forward_only=False):

    if not forward_only:
        raise ValueError(f'当前仅支持前向性能及精度测试，暂不支持其他类型测试任务')

    torch.manual_seed(0)
    x1, x2, grad = get_data(batch_size, seqs, cols, rows, data_layout, precision_type)
    lightop_model, miopen_model = get_model(seqs, num_groups, precision_type)
    if forward_only:
        lightop_model = lightop_model.eval()
        miopen_model = miopen_model.eval()
    torch.manual_seed(0)

    if precision_type == 'FP16':
        weight = torch.rand(seqs).cuda().half()
    else:
        weight = torch.rand(seqs).cuda()

    lightop_model.weight.data = weight.clone().detach()
    miopen_model.weight.data = weight.clone().detach()
    print(f'size = ({batch_size}, {seqs}, {cols}, {rows} {precision}, {data_layout})')

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
    lightop_time = start_event.elapsed_time(end_event)

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
    miopen_lightop_time = start_event.elapsed_time(end_event)
    print('lightop_time:', format(lightop_time / repeat_num, '.3f'), 'ms. miopen_lightop_time:',
          format(miopen_lightop_time / repeat_num, '.3f'), 'ms. speedup:', format(lightop_time / miopen_lightop_time - 1, '.2f'))

    # 精度验证
    print(f'lightop_output_shape: {y1.shape}, lightop_miopen_output_shape: {y2[0].shape}')
    assert y1.shape == y2[0].shape, 'lightop接口与miopen lightop接口的输出tensor shape不一致'
    verify_accuracy(y1, y2[0], rtol=5e-3, atol=5e-3, precision=precision)


if __name__ == '__main__':
    print('=' * 30 + '测试GN Size' + '=' * 30)
    # 设置基础参数
    forward_only = True
    precision_list = ['FP16', 'FP32']
    data_layout_list = ['NCHW', 'NHWC']
    repeat_num = 100
    gn_size_path = r'../test_size/groupnorm.csv'
    gn_size_list = read_gn_size(gn_size_path)
    for precision, data_layout in list(product(precision_list, data_layout_list)):
        for size_id, params in enumerate(gn_size_list):
            print(f'size_id: {size_id}')
            test_speed(**params, repeat_num=repeat_num, precision_type=precision, data_layout=data_layout, forward_only=forward_only)
