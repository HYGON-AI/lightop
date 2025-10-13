# -*- coding:utf-8 -*-
"""
author: niutao
date: 2025年08月05日
description: 支持测试脚本执行的公共函数，如精度验证
"""

import torch
import numpy as np


dtype_map = {
    'FP16': torch.float16,
    'FP32': torch.float32
    }


def verify_accuracy(a, b, rtol=1e-2, atol=1e-5, equal_nan=False, precision='FP16'):
    """验证FP16, FP32的精度测试结果。"""

    if torch.allclose(a, b, rtol=rtol, atol=atol, equal_nan=equal_nan):
        print(f'{precision} accuracy verify: Passed')
    else:
        print(f'{precision} accuracy verify: Failed')


def find_topk_errors(tensor1, tensor2, k=5):
    """找出前K个最大误差点"""
    # 展平张量
    tensor1_flat = tensor1.flatten()
    tensor2_flat = tensor2.flatten()

    # 计算相对误差
    rel_error_flat = (tensor1_flat - tensor2_flat).abs() / (tensor2_flat.abs() + 1e-8)

    # 获取前K个最大误差
    topk_errors, topk_indices_flat = torch.topk(rel_error_flat, k)
    print('最大误差列表:', topk_errors, topk_indices_flat)

    print(f"\n前{k}个最大相对误差点:")
    for i, (error, idx_flat) in enumerate(zip(topk_errors, topk_indices_flat)):
        # 将展平索引转换为4维索引
        idx_4d = np.unravel_index(idx_flat.cpu().numpy(), tensor1.shape)
        val1 = tensor1_flat[idx_flat].item()
        val2 = tensor2_flat[idx_flat].item()
        print(f"{i + 1}. 索引{idx_4d}: 相对误差={error.item():.6f}")
        print(f"lop计算结果: {val1:.6f}, mlop计算结果: {val2:.6f}, 绝对误差={abs(val1 - val2):.6f}")


def get_data(batchsize, in_channels, out_channels, kernel_h, kernel_w, Height, Width, data_layout, precision_type):
    """
    创建指定配置的张量数据，适用于convfusion算子

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
