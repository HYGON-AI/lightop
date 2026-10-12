# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

from torch.autograd import Function
import torch
from torch import Tensor

#from lightop import op
from . import op


class FuseBiasGeluFunction(Function):
    r"""Fused bias and GELU operator.
    
    Shape:
        - Input: :math:`(*, D)`
        - Bias: :math:`(D)`
        - Output: :math:`(*, D)`
    """
    @staticmethod
    def forward(ctx, input, bias)-> Tensor:
        ctx.save_for_backward(input, bias)
        return op.bias_gelu_forward(input, bias)
    
    #@staticmethod
    def backward(ctx, grad_output):
        input, bias = ctx.saved_tensors
        return op.bias_gelu_backward(input, bias, grad_output)

    
class FuseBiasGelu(torch.nn.Module):

    def __init__(self):
        super(FuseBiasGelu, self).__init__()
        
    def forward(self, input,bias):
        return FuseBiasGeluFunction.apply(input, bias)