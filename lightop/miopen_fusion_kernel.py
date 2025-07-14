from torch.autograd import Function
import torch
from torch import Tensor
# from torch.nn import functional as F
# from torch.nn import init
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
                alhpa = 1,
                beta = 0, 
                device=None,
                dtype=None,):
        
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.scale = scale
        self.zero_point = zero_point
        self.bias = bias
        self.padding_mode = padding_mode
        self.layout = layout
        self.device = device
        self.dtype = dtype
        self.alpha = alhpa
        self.beta = beta
    
    def forward(self, input: Tensor, add: Tensor, weight: Tensor) -> Tensor:
        # if len(input.shape) != 4:
        #     raise ValueError("Input shape must be `(N, C, H, W)`!")
        # if ctx.padding_mode != "zeros":
        #     _reversed_padding_repeated_twice = _reverse_repeat_padding(ctx.padding)
        #     input = F.pad(
        #         input, _reversed_padding_repeated_twice, mode=ctx.padding_mode
        #     )
        return op.conv_add(
            input, add, weight, self.padding, self.stride, self.dilation, self.layout, self.groups, self.alpha, self.beta, self.scale, self.zero_point
        )
    
    
class FuseConvBiasAdd(torch.nn.Module):
    def __init__(self,
                 in_channels,
                out_channels,
                kernel_size,
                stride,
                padding ,
                dilation ,
                groups = 1,
                bias = True,
                padding_mode = "zeros",
                layout : str  = "NCHW",
                alhpa = 1,
                beta = 0, 
                device=None,
                dtype=None,) -> None:
        
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        self.padding_mode = padding_mode
        self.layout = layout
        self.device = device
        self.dtype = dtype
        self.alpha = alhpa
        self.beta = beta
        
        
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
            input, weight, bias, add, self.padding, self.stride, self.dilation, self.layout, self.alpha, self.beta
        )