# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

from torch.autograd import Function
import torch
from torch import Tensor
from . import op

def destindex_copy_quantize_int8_cuda(input, scale_zero, output):
    op.destindex_copy_quantize_int8_cuda(input, scale_zero, output)

def destindex_dequantize_int8_cuda(input, scale_zero, output):
    op.destindex_dequantize_int8_cuda(input, scale_zero, output)
