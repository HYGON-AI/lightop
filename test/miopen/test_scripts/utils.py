# -*- coding:utf-8 -*-
"""
author: niutao
date: 2025年08月05日
description: 支持测试脚本执行的公共函数，如精度验证
"""

import torch


def verify_accuracy(a, b, use_half, equal_nan=False):
    """验证FP16, FP32的精度测试结果。"""
    if use_half:
        rtol, atol = 1e-02, 1e-05
    else:
        rtol, atol = 1e-05, 1e-08

    if use_half:
        if torch.allclose(a, b, rtol=rtol, atol=atol, equal_nan=equal_nan):
            print('FP16 accuracy verify: Passed')
        else:
            print('FP16 accuracy verify: Failed')
    else:
        if torch.allclose(a, b, rtol=rtol, atol=atol, equal_nan=equal_nan):
            print('FP32 accuracy verify: Passed')
        else:
            print('FP32 accuracy verify: Failed')