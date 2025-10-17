# -*- coding:utf-8 -*-
"""
author: zhangsw
date: 2025年09月10日
description: 测试tensortrans的lightop以及miopen耗时，仅支持forward性能及精度测试，且测试size布局默认为NCHW。
"""

import lightop
import torch
from lightop import miopen_transform_tensor

from read_test_size import read_tt_size
from utils import verify_accuracy
from torch.profiler import profile, ProfilerActivity
import re

def allclose(a, b, rtol=1e-05, atol=1e-08, equal_nan=False):
    # 精度验证
    return torch.allclose(a, b, rtol=rtol, atol=atol, equal_nan=equal_nan)

def test_reshape(batch_size, seqs, cols, rows):
    # 计算张量总元素个数
    total_elements = batch_size * seqs * cols * rows
    # 创建连续数字的一维张量
    sequence = torch.arange(1, total_elements + 1,
                            dtype=torch.half,
                            device="cuda")
    # 重塑为目标形状
    input_ = sequence.reshape(batch_size, seqs, cols, rows)
    input = input_.contiguous(memory_format=torch.channels_last)  # NHWC 格式

    return input_

def test_assert(expected_shape,actual_shape):
    if expected_shape == actual_shape:
        print("转换成功")
    else:
        print("转换失败")

def test_speed(batch_size, seqs, cols, rows, transpose_mode, repeat_num, use_half=False, forward_only=False):

    if not forward_only:
        raise ValueError(f'当前仅支持前向性能及精度测试，暂不支持其他类型测试任务')

    print('=' * 60)
    print(f'in Tensor shape = ({batch_size}, {seqs}, {cols}, {rows})')

    # ms. miopen performance test:

    # 测试miopen接口，调用miopen api warm up
    for i in range(10):
        miopen_model = miopen_transform_tensor(test_reshape(batch_size, seqs, cols, rows), transpose_mode)

    # 统计miopen API耗时
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()    #同步 GPU 计算流,确保 GPU 计算完成
    start_event.record()
    # lightop test:
    for i in range(repeat_num):
        miopen_model = miopen_transform_tensor(test_reshape(batch_size, seqs, cols, rows), transpose_mode)

    end_event.record()
    torch.cuda.synchronize()
    miopen_api_time = start_event.elapsed_time(end_event) / 10  #统计miopen测试时间
    print('Self miopen api time: ', round(miopen_api_time, 3))

    # 测试miopen prof耗时
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
        out_miopen = miopen_transform_tensor(test_reshape(batch_size, seqs, cols, rows), transpose_mode)
    print(prof.key_averages(group_by_input_shape=True).table(sort_by="self_cuda_time_total"))

    # 输出张量形状
    print("out Tensor shape :", out_miopen.shape)

    # 断言
    if seqs % 32 != 0:
        new_dim2 = seqs // 32 + 1
    else:
        new_dim2 = seqs // 32
    new_dim5 = 32
    expected_shape = [int(batch_size),new_dim2,int(cols),int(rows),new_dim5]
    actual_shape = list(map(int,re.findall(r"\d+", str(out_miopen.shape))))
    test_assert(expected_shape,actual_shape)

    #print('torch time:', format(torch_time / repeat_num, '.3f'), 'lightop time:', format(lightop_time / repeat_num, '.3f'), 'ms. miopen time:',format(miopen_api_time / repeat_num, '.3f'), 'ms speedup:', format(lightop_time / miopen_api_time - 1, '.2f'))
    print('ms. miopen time:', format(miopen_api_time / repeat_num, '.3f'))

if __name__ == '__main__':
    print('=' * 30 + '测试LN Size' + '=' * 30)
    # 设置基础参数
    forward_only = True
    use_half = True
    repeat_num = 100
    ln_size_path = r'../test_size/tensortrans.csv'
    ln_size_list = read_tt_size(ln_size_path)
    for params in ln_size_list:
        test_speed(**params, repeat_num=repeat_num, use_half=use_half, forward_only=forward_only)