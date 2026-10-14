# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

import torch
from torch import Tensor
from . import op


def bilinear_upsample(x: Tensor, output_size) -> Tensor:
    """双线性上采样前向，等价于 nn.Upsample(mode='bilinear', align_corners=False)。"""
    return op.bilinear_upsample(x, list(output_size))


def bilinear_upsample_backward(grad_output: Tensor, input_size) -> Tensor:
    """双线性上采样反向，返回与输入等形状的梯度张量。
    
    Args:
        grad_output: 上游梯度，形状 [N, C, H_out, W_out]
        input_size:  原始输入形状，长度为 4 的序列 [N, C, H_in, W_in]
    """
    return op.bilinear_upsample_backward(grad_output, list(input_size))