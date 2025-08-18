# -*- coding:utf-8 -*-
"""
author: niutao
date: 2025年08月01日
description: 测试groupnorm的lightop以及miopen耗时，仅支持forward性能及精度测试，且测试size布局默认为NHWC。
"""

import torch
from lightop import GroupNorm, miopenGroupNorm

from read_test_size import read_gn_size
from utils import verify_accuracy


def get_data(batchsize=128, seqs=50, cols=128, rows=128, use_half=False):
    if use_half:
        x = torch.rand((batchsize, seqs, cols, rows)).to(memory_format=torch.channels_last).cuda().half()
        grad = torch.rand((batchsize, seqs, cols, rows)).to(memory_format=torch.channels_last).cuda().half()
    else:
        x = torch.rand((batchsize, seqs, cols, rows)).to(memory_format=torch.channels_last).cuda()
        grad = torch.rand((batchsize, seqs, cols, rows)).to(memory_format=torch.channels_last).cuda()

    x.requires_grad_(True)
    x1 = x.clone().detach().requires_grad_()
    x2 = x.clone().detach().requires_grad_()
    return x1, x2, grad


def get_model(seqs=1024, num_groups=2, use_half=False):
    if use_half:
        miopen_groupnorm = miopenGroupNorm(num_groups, seqs, 1).cuda().half()
        lightop_groupnorm = GroupNorm(num_groups, seqs).cuda().half()
    else:
        miopen_groupnorm = miopenGroupNorm(num_groups, seqs, 1).cuda()
        lightop_groupnorm = GroupNorm(num_groups, seqs).cuda()

    return lightop_groupnorm, miopen_groupnorm


def test_speed(batch_size, seqs, cols, rows, num_groups, repeat_num, use_half=False, forward_only=False):

    if not forward_only:
        raise ValueError(f'当前仅支持前向性能及精度测试，暂不支持其他类型测试任务')

    torch.manual_seed(0)
    x1, x2, grad = get_data(batch_size, seqs, cols, rows, use_half=use_half)
    lightop_model, miopen_model = get_model(seqs=seqs, num_groups=num_groups, use_half=use_half)
    if forward_only:
        lightop_model = lightop_model.eval()
        miopen_model = miopen_model.eval()
    torch.manual_seed(0)
    weight = torch.rand(seqs).cuda()
    if use_half:
        weight = weight.half()
    lightop_model.weight.data = weight.clone().detach()
    miopen_model.weight.data = weight.clone().detach()
    print(f'size = ({batch_size}, {seqs}, {cols}, {rows})')

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
    miopen_time = start_event.elapsed_time(end_event)
    print('lightop time:', format(lightop_time / repeat_num, '.3f'), 'ms. miopen time:',
          format(miopen_time / repeat_num, '.3f'), 'ms speedup:', format(lightop_time / miopen_time - 1, '.2f'))

    # 精度验证
    verify_accuracy(y1, y2[0], rtol=1e-2, atol=1e-5, use_half=use_half)


if __name__ == '__main__':
    print('=' * 30 + '测试GN Size' + '=' * 30)
    # 设置基础参数
    forward_only = True
    use_half = True
    repeat_num = 100
    gn_size_path = r'../test_size/groupnorm.csv'
    gn_size_list = read_gn_size(gn_size_path)
    for params in gn_size_list:
        test_speed(**params, repeat_num=repeat_num, use_half=use_half, forward_only=forward_only)
