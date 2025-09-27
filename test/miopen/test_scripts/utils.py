# -*- coding:utf-8 -*-
"""
author: niutao
date: 2025年08月05日
description: 支持测试脚本执行的公共函数，如精度验证
"""

import torch
import numpy as np


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
