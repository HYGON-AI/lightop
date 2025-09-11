# -*- coding:utf-8 -*-
"""
author: zhangsw
date: 2025年09月10日
description: 测试layernorm的lightop以及miopen耗时，仅支持forward性能及精度测试，且测试size布局默认为NCHW。
"""

import torch
from lightop import LayerNorm, miopenLayerNorm

from read_test_size import read_ln5d_size
from utils import verify_accuracy


class Torch_Layernorm(torch.nn.Module):

    def __init__(self, cols):
        super(Torch_Layernorm, self).__init__()
        self.cols = cols
        self.LayerNorm = torch.nn.LayerNorm(cols)

    def forward(self, x):
        out = self.LayerNorm(x)
        return out

# 初始化数据
def get_data(batch_size=2, dima=1, dimb=1, seqs=32, cols=122880, use_half=False):
    if use_half:
        x = torch.rand((batch_size, dima, dimb, seqs, cols)).cuda().half()
        grad = torch.rand((batch_size, dima, dimb, seqs, cols)).cuda().half()
    else:
        x = torch.rand((batch_size, dima, dimb, seqs, cols)).cuda()
        grad = torch.rand((batch_size, dima, dimb, seqs, cols)).cuda()

    x.requires_grad_(True)  #计算X张量的梯度
    x1 = x.clone().detach().requires_grad_()
    x2 = x.clone().detach().requires_grad_()
    x3 = x.clone().detach().requires_grad_()
    return x1, x2, x3, grad


def get_model(cols=122880, use_half=False):
    if use_half:
        miopen_layernorm = miopenLayerNorm(cols).cuda().half()
        lightop_layernorm = LayerNorm(cols).cuda().half()
        torch_layernorm = Torch_Layernorm(cols).cuda().half()
    else:
        miopen_layernorm = miopenLayerNorm(cols).cuda()
        lightop_layernorm = LayerNorm(cols).cuda()
        torch_layernorm = Torch_Layernorm(cols).cuda()

    return lightop_layernorm, miopen_layernorm, torch_layernorm


def test_speed(batch_size, dima, dimb, seqs, cols, repeat_num, use_half=False, forward_only=False):

    if not forward_only:
        raise ValueError(f'当前仅支持前向性能及精度测试，暂不支持其他类型测试任务')

    torch.manual_seed(0)
    x1, x2, x3, grad = get_data(batch_size, dima, dimb, seqs, cols, use_half=use_half)     #获取.csv中五维数据后，初始化数据
    lightop_model, miopen_model, torch_model = get_model(cols=cols, use_half=use_half)   #获取模型

    if forward_only:
        lightop_model = lightop_model.eval()
        miopen_model = miopen_model.eval()
        torch_layernorm = torch_model.eval()
    torch.manual_seed(0)
    print(f'normalized_shape = ({batch_size}, {dima}, {dimb}, {seqs}, {cols})')

    # torch performance test:

    # torch model warm up
    for i in range(10):
        y1 = torch_layernorm(x1)
        if not forward_only:
            y1.backward(grad)

    # torch model time count
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()  # 同步 GPU 计算流,确保 GPU 计算完成
    start_event.record()
    # torch test:
    for i in range(repeat_num):
        x1.grad = None
        y1 = torch_layernorm(x1)
        if not forward_only:
            y1.backward(grad)

    end_event.record()
    torch.cuda.synchronize()
    torch_time = start_event.elapsed_time(end_event)  # 统计torch测试时间

    # lightop performance test:

    # lightop model warm up
    for i in range(10):
        y2 = lightop_model(x2)
        if not forward_only:
            y2.backward(grad)

    # lightop model time count
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()    #同步 GPU 计算流,确保 GPU 计算完成
    start_event.record()
    # lightop test:
    for i in range(repeat_num):
        x2.grad = None
        y2 = lightop_model(x2)
        if not forward_only:
            y2.backward(grad)

    end_event.record()
    torch.cuda.synchronize()
    lightop_time = start_event.elapsed_time(end_event)  #统计lightop测试时间

    # ms. miopen performance test:

    # ms. miopen model warm up
    for i in range(10):
        y3 = miopen_model(x3)
        if not forward_only:
            y3.backward(grad)

    # ms. miopen model time count
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()    #同步 GPU 计算流,确保 GPU 计算完成
    start_event.record()
    # lightop test:
    for i in range(repeat_num):
        x3.grad = None
        y3 = miopen_model(x3)
        if not forward_only:
            y3.backward(grad)

    end_event.record()
    torch.cuda.synchronize()
    miopen_time = start_event.elapsed_time(end_event)   #统计miopen测试时间

    print('torch time:', format(torch_time / repeat_num, '.3f'), 'lightop time:', format(lightop_time / repeat_num, '.3f'), 'ms. miopen time:',format(miopen_time / repeat_num, '.3f'), 'ms speedup:', format(lightop_time / miopen_time - 1, '.2f'))

    # 精度验证
    verify_accuracy(y2, y3[0], rtol=1e-2, atol=1e-5, use_half=use_half)


if __name__ == '__main__':
    print('=' * 30 + '测试LN Size' + '=' * 30)
    # 设置基础参数
    forward_only = True
    use_half = True
    repeat_num = 100
    ln_size_path = r'../test_size/layernorm5d.csv'
    ln_size_list = read_ln5d_size(ln_size_path)
    for params in ln_size_list:
        test_speed(**params, repeat_num=repeat_num, use_half=use_half, forward_only=forward_only)