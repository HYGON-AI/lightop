# -*- coding:utf-8 -*-
"""
author: niutao
date: 2025年08月05日
description: 支持测试脚本执行的公共函数，如精度验证
"""

import torch


def verify_accuracy(a, b, rtol=1e-2, atol=1e-5, equal_nan=False, precision='FP16'):
    """验证FP16, FP32的精度测试结果。"""

    if torch.allclose(a, b, rtol=rtol, atol=atol, equal_nan=equal_nan):
        print(f'{precision} accuracy verify: Passed')
    else:
        print(f'{precision} accuracy verify: Failed')
