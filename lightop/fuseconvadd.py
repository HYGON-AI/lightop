from torch.autograd import Function
import torch
from torch import Tensor
import torch.nn.functional as F
import torch.ao.nn.quantized as nnq
from lightop import op

_reverse_repeat_padding = nnq.modules.conv._reverse_repeat_padding

class FuseConvAddFunction(Function):
    
    @staticmethod
    def forward(ctx, input: Tensor, extra_input: Tensor) -> Tensor:
        # if len(input.shape) != 4:
        #     raise ValueError("Input shape must be `(N, C, H, W)`!")
        # if ctx.padding_mode != "zeros":
        #     _reversed_padding_repeated_twice = _reverse_repeat_padding(ctx.padding)
        #     input = F.pad(
        #         input, _reversed_padding_repeated_twice, mode=ctx.padding_mode
        #     )
        return op.conv_add(
            input, extra_input, ctx._packed_params, ctx.scale, ctx.zero_point
        )
        
class FuseConvAdd(torch.nn.Module):
    def __init__(self,
                in_channels,
                out_channels,
                kernel_size,
                stride=1,
                padding=0,
                dilation=1,
                groups=1,
                bias=False,
                padding_mode="zeros",
                device=None,
                dtype=None,):
        
        super(FuseConvAdd, self).__init__(in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
            device=device,
            dtype=dtype,)
    
    def forward(self, input:Tensor, extra_input: Tensor):
        return FuseConvAddFunction.apply(input, extra_input)