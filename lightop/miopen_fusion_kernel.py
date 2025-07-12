from torch.autograd import Function
import torch
from torch import Tensor
from torch.nn import functional as F, init
from typing import Optional, Union
from lightop import op
import torch.ao.nn.quantized as nnq

# _reverse_repeat_padding = nnq.modules.conv._reverse_repeat_padding

     
class FuseConvAdd(torch.nn.Module):
    def __init__(self,
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                dilation,
                groups=1,
                scale=1,
                zero_point=0,
                bias=False,
                padding_mode="zeros",
                layout : str  = "NCHW", 
                device=None,
                dtype=None,):
        
        super().__init__(in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            scale=scale,
            zero_point=zero_point,
            bias=bias,
            padding_mode=padding_mode,
            layout = layout,
            device=device,
            dtype=dtype,)
    
    def forward(self, input: Tensor, add: Tensor, weight: Tensor) -> Tensor:
        # if len(input.shape) != 4:
        #     raise ValueError("Input shape must be `(N, C, H, W)`!")
        # if ctx.padding_mode != "zeros":
        #     _reversed_padding_repeated_twice = _reverse_repeat_padding(ctx.padding)
        #     input = F.pad(
        #         input, _reversed_padding_repeated_twice, mode=ctx.padding_mode
        #     )
        return op.conv_add(
            input, add, weight, self.layout, self.stride, self.padding, self.dilation, self.groups, self.scale, self.zero_point
        )
    
    
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
                layout : str  = "NCHW", 
                device=None,
                dtype=None,) -> None:
        
        super().__init__(in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            padding_mode,
            layout,
            device=device,
            dtype=dtype,)
        
        
    def forward(self, input: Tensor, weight: Tensor, bias: Tensor, add: Tensor) -> Tensor:
        # if ctx.padding_mode != "zeros":
        #     return F.conv2d(
        #         F.pad(
        #             input, ctx._reversed_padding_repeated_twice, mode=ctx.padding_mode
        #         ),
        #         weight,
        #         bias,
        #         ctx.stride,
        #         _pair(0),
        #         ctx.dilation,
        #         ctx.groups,
        #     )
        return op.conv_bias_add(
            input, weight, bias, add, self.layout, self.stride, self.padding, self.dilation, self.groups
        )