import torch
from torch import Tensor
from lightop import op
from typing import Union
from torch.nn.common_types import _size_2_t


def miopen_conv_bias_add(input, weight, bias, add, padding, stride, dilation):
    """
    参数:
        input (Tensor): 输入张量，形状为 [N, C_in, H, W] (NCHW) 或 [N, H, W, C_in] (NHWC)
        weight (Tensor): 卷积核权重，形状为 [C_out, C_in, kH, kW]
        bias (Tensor): 偏置项，形状为 [C_out]
        add (Tensor): 需要加和的张量，形状应与卷积输出相同
        padding (int|tuple): 填充大小。(padH, padW) 元组
        stride (int|tuple): 卷积步长。 (strideH, strideW) 元组
        dilation (int|tuple): 卷积膨胀率。(dilationH, dilationW) 元组
    返回:
        Tensor: 计算结果张量
    """
    return op.miopen_conv_bias_add(input, weight, bias, add, padding, stride, dilation)


def miopen_conv_bias(x, weight, bias, padding, stride, dilation):
    """_summary_

    Args:
        x (Tensor): 输入张量
        add (Tensor): 需要加和的张量
        weight (Tensor): 卷积核权重
        pads (int|tuple): 填充大小。(padH, padW) 元组
        strides (int|tuple): 卷积步长。 (strideH, strideW) 元组
        dilations (int|tuple): 卷积膨胀率。(dilationH, dilationW) 元组
        groups (int): 是否为组卷积。1为普通卷积, >1 为组卷积
        alpha (float): _description_
        beta (float): _description_
        scale (double): _description_
        zero_point (double): _description_

    Returns:
        Tensor: 结果张量
    """
    return op.miopen_conv_bias(x, weight, bias, padding, stride, dilation)


def miopen_conv_bias_relu(input, weight, bias, padding, stride, dilation):
    """_summary_

    Args:
        input (Tensor): 输入张量
        weight (Tensor): 卷积核权重
        bias (Tensor): 偏置项
        ppadding (int|tuple): 填充大小。(padH, padW) 元组
        stride (int|tuple): 卷积步长。 (strideH, strideW) 元组
        dilation (int|tuple): 卷积膨胀率。(dilationH, dilationW) 元组

    Returns:
        Tensor: yDesc 计算结果张量
    """
    return op.miopen_conv_bias_relu(input, weight, bias, padding, stride, dilation)


class miopenConvBias(torch.nn.Conv2d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_2_t,
        stride: _size_2_t = 1,
        padding: Union[str, _size_2_t] = 0,
        dilation: _size_2_t = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        device=None,
        dtype=None,
    ):
        super().__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode, device, dtype
        )

    def forward(self, input):
        return op.miopen_conv_bias(
            input, self.weight, self.bias, list(self.padding), list(self.stride), list(self.dilation)
        )


class miopenConvBiasAdd(torch.nn.Conv2d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_2_t,
        stride: _size_2_t = 1,
        padding: Union[str, _size_2_t] = 0,
        dilation: _size_2_t = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        device=None,
        dtype=None,
    ):
        super().__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode, device, dtype
        )

    def forward(self, input: Tensor, extra_input: Tensor):
        return op.miopen_conv_bias_add(
            input, self.weight, self.bias, extra_input, list(self.padding), list(self.stride), list(self.dilation)
        )
