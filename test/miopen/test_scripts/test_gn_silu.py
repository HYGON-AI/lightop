# -*- coding:utf-8 -*-
"""
author: niutao
date: 2025年08月02日
description：测试GNSilu算子，比较miopenGroupNorm(mode=1)+torch.silu()与miopenGroupNorm(mode=10)的性能比较。仅支持前向性能及精度验证，默认数据布局为NHWC。
"""

import torch
import time
from lightop import GNSilu, miopenGroupNorm

from read_test_size import read_gn_size
from utils import verify_accuracy


class Torch_GNSilu(torch.nn.Module):

    def __init__(self, num_groups, seqs):
        super().__init__()
        self.seqs = seqs
        self.num_groups = num_groups
        self.silu = torch.nn.SiLU()
        # self.GroupNorm = torch.nn.GroupNorm(num_groups,seqs)
        self.GroupNorm = miopenGroupNorm(num_groups, seqs, 1)

    def forward(self, x):
        out = self.GroupNorm(x)[0]
        out = self.silu(out)
        return out


def get_data(batchsize=128, seqs=50, cols=128, rows=128, use_half=False):
    if use_half:
        x = torch.randn((batchsize, seqs, cols, rows)).to(memory_format=torch.channels_last).cuda().half()
        grad = torch.randn((batchsize, seqs, cols, rows)).to(memory_format=torch.channels_last).cuda().half()
    else:
        x = torch.randn((batchsize, seqs, cols, rows)).to(memory_format=torch.channels_last).cuda()
        grad = torch.randn((batchsize, seqs, cols, rows)).to(memory_format=torch.channels_last).cuda()

    x.requires_grad_(True)
    x1 = x.clone().detach().requires_grad_()
    x2 = x.clone().detach().requires_grad_()
    return x1, x2, grad


def get_model(seqs=1024, num_groups=2, use_half=False):
    if use_half:
        fuse_gn_silu = miopenGroupNorm(num_groups, seqs, 10).cuda().half()
        torch_gn_silu = Torch_GNSilu(num_groups, seqs).cuda().half()
    else:
        fuse_gn_silu = GNSilu(num_groups, seqs).cuda()
        torch_gn_silu = Torch_GNSilu(num_groups, seqs).cuda()

    return torch_gn_silu, fuse_gn_silu


def test_speed(batch_size, seqs, cols, rows, num_groups, repeat_num, use_half=False, forward_only=False):

    if not forward_only:
        raise ValueError(f'当前仅支持前向性能及精度测试，暂不支持其他类型测试任务')

    torch.manual_seed(0)
    x1, x2, grad = get_data(batch_size, seqs, cols, rows, use_half=use_half)
    f1, f2 = get_model(seqs=seqs, num_groups=num_groups, use_half=use_half)
    if forward_only:
        f1 = f1.eval()
        f2 = f2.eval()
    torch.manual_seed(1)
    weight = torch.rand(seqs).cuda()
    if use_half:
        weight = weight.half()
    f1.GroupNorm.weight.data = weight.clone().detach()
    f2.weight.data = weight.clone().detach()
    print(f'size = ({batch_size}, {seqs}, {cols}, {rows})')

    y1 = f1(x1)
    if not forward_only:
        y1.backward(grad)

    torch.cuda.synchronize()
    start_time = time.time()
    for i in range(repeat_num):
        x1.grad = None
        f1.GroupNorm.weight.grad = None
        f1.GroupNorm.bias.grad = None
        y1 = f1(x1)
        if not forward_only:
            y1.backward(grad)
    torch.cuda.synchronize()
    end_time = time.time()
    fc1_espl = end_time - start_time

    y2 = f2(x2)
    if not forward_only:
        y2.backward(grad)
    torch.cuda.synchronize()
    start_time = time.time()
    for i in range(repeat_num):
        x2.grad = None
        f2.weight.grad = None
        f2.bias.grad = None
        y2 = f2(x2)
        if not forward_only:
            y2.backward(grad)
    torch.cuda.synchronize()
    end_time = time.time()
    fc2_espl = end_time - start_time
    print('miopen_gn_torch_silu time:', format(fc1_espl * 1000 / repeat_num, '.3f'), 'ms. miopen time:',
          format(fc2_espl * 1000 / repeat_num, '.3f'), 'ms speedup:', format(fc1_espl / fc2_espl - 1, '.2f'))

    # 执行精度验证
    verify_accuracy(y1, y2[0], rtol=1e-2, atol=1e-5, use_half=use_half)


if __name__ == "__main__":
    print('=' * 30 + '测试GN_Silu Size' + '=' * 30)
    # 设置基础参数
    forward_only = True
    use_half = True
    repeat_num = 100
    gn_size_path = r'../test_size/groupnorm.csv'
    gn_size_list = read_gn_size(gn_size_path)
    for params in gn_size_list:
        test_speed(**params, repeat_num=repeat_num, use_half=use_half, forward_only=forward_only)