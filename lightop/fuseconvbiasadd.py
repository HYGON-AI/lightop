from torch.autograd import Function
import torch
from torch import Tensor
from torch.nn import functional as F, init
from typing import Optional, Union
from lightop import op



class FuseConvBiasAdd(torch.nn.Module):
    def __init__(self,
                 in_channels,
                out_channels,
                kernel_size,
                stride = 1,
                padding = 0,
                dilation = 1,
                groups = 1,
                bias = True,
                padding_mode = "zeros", 
                device=None,
                dtype=None,) -> None:
        
        super(FuseConvBiasAdd, self).__init__(in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            padding_mode,
            device=device,
            dtype=dtype,)
        
    def forward(self, input: Tensor, weight: Tensor, bias: Tensor, add: Tensor):
          return op.conv_bias_add(
            input, weight, bias, extra_input, ctx.stride, ctx.padding, ctx.dilation, ctx.groups
        )