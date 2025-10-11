import torch
from torch import Tensor
from lightop import op
from typing import Union
from torch.nn.common_types import _size_2_t


def miopen_conv_bias_add(input, weight, bias, add, padding, stride, dilation):
    """
    参数:
        input (Tensor): 输入张量, 形状为 [N, C_in, H, W] (NCHW) 或 [N, H, W, C_in] (NHWC)
        weight (Tensor): 卷积核权重, 形状为 [C_out, C_in, kH, kW]
        bias (Tensor): 偏置项, 形状为 [C_out]
        add (Tensor): 需要加和的张量, 形状应与卷积输出相同
        padding (int|tuple): 填充大小, (padH, padW) 元组
        stride (int|tuple): 卷积步长, (strideH, strideW) 元组
        dilation (int|tuple): 卷积膨胀率, (dilationH, dilationW) 元组
    返回:
        Tensor: 计算结果张量 out
    """
    return op.miopen_conv_bias_add(input, weight, bias, add, padding, stride, dilation)


def miopen_conv_bias(x, weight, bias, padding, stride, dilation):
    """_summary_

    Args:
        x (Tensor): 输入张量, 形状为 [N, C_in, H, W] (NCHW) 或 [N, H, W, C_in] (NHWC)
        add (Tensor): 需要加和的张量, 形状应与卷积输出相同
        weight (Tensor): 卷积核权重, 形状为 [C_out, C_in, kH, kW]
        pads (int|tuple): 填充大小, (padH, padW) 元组
        strides (int|tuple): 卷积步长,  (strideH, strideW) 元组
        dilations (int|tuple): 卷积膨胀率, (dilationH, dilationW) 元组

    Returns:
        Tensor: 计算结果张量 out
    """
    return op.miopen_conv_bias(x, weight, bias, padding, stride, dilation)


def miopen_conv_bias_relu(input, weight, bias, padding, stride, dilation):
    """_summary_

    Args:
        input (Tensor): 输入张量, 形状为 [N, C_in, H, W] (NCHW) 或 [N, H, W, C_in] (NHWC)
        weight (Tensor): 卷积核权重, 形状为 [C_out, C_in, kH, kW]
        bias (Tensor): 偏置项, 形状为 [C_out]
        ppadding (int|tuple): 填充大小, (padH, padW) 元组
        stride (int|tuple): 卷积步长, (strideH, strideW) 元组
        dilation (int|tuple): 卷积膨胀率, (dilationH, dilationW) 元组

    Returns:
        Tensor: 计算结果张量 out
    """
    return op.miopen_conv_bias_relu(input, weight, bias, padding, stride, dilation)


def miopen_conv_bias_add_relu(input, weight, bias, add, padding, stride, dilation):
    """_summary_

    Args:
        input (Tensor): 输入张量, 形状为 [N, C_in, H, W] (NCHW) 或 [N, H, W, C_in] (NHWC)
        weight (Tensor): 卷积核权重, 形状为 [C_out, C_in, kH, kW]
        bias (Tensor): 偏置项, 形状为 [C_out]
        add (Tensor): 需要加和的张量, 形状应与卷积输出相同
        padding (int|tuple): 填充大小, (padH, padW) 元组
        stride (int|tuple): 卷积步长, (strideH, strideW) 元组
        dilation (int|tuple): 卷积膨胀率, (dilationH, dilationW) 元组

    Returns:
        Tensor: 计算结果张量 out
    """
    return op.miopen_conv_bias_add_relu(input, weight, bias, add, padding, stride, dilation)


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
        """_summary_

        Args:
            in_channels (int): 输入张量的通道数
            out_channels (int): 卷积核的通道数
            kernel_size (_size_2_t): 卷积核尺寸, 传入 x, 自动扩展为[x, x]
            stride (_size_2_t, optional): 卷积步长, 默认为 1, 自动扩展为[1, 1]
            padding (Union[str, _size_2_t], optional): 填充大小, 默认为 0.
            dilation (_size_2_t, optional): 卷积膨胀率. 默认为 1, 自动扩展为[1, 1].
            groups (int, optional): 分组数, 默认为 1, 即不分组.
            bias (bool, optional): 是否具有偏置项, 默认为 True.
            padding_mode (str, optional): 填充模式, 默认为 "zeros".
            device (_type_, optional): 设备类型, 默认为 None.
            dtype (_type_, optional): 数据类型, 默认为 None.
        
        Returns:
            Tensor: 计算结果张量 out
        """
        super().__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode, device, dtype
        )

    def forward(self, input):
        return op.miopen_conv_bias(
            input, self.weight, self.bias, list(self.padding), list(self.stride), list(self.dilation)
        )


class miopenConvBiasRelu(torch.nn.Conv2d):
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
        """_summary_

        Args:
            in_channels (int): 输入张量的通道数
            out_channels (int): 卷积核的通道数
            kernel_size (_size_2_t): 卷积核尺寸, 传入 x, 自动扩展为[x, x]
            stride (_size_2_t, optional): 卷积步长, 默认为 1, 自动扩展为[1, 1]
            padding (Union[str, _size_2_t], optional): 填充大小, 默认为 0.
            dilation (_size_2_t, optional): 卷积膨胀率. 默认为 1, 自动扩展为[1, 1].
            groups (int, optional): 分组数, 默认为 1, 即不分组.
            bias (bool, optional): 是否具有偏置项, 默认为 True.
            padding_mode (str, optional): 填充模式, 默认为 "zeros".
            device (_type_, optional): 设备类型, 默认为 None.
            dtype (_type_, optional): 数据类型, 默认为 None.
            
        Returns:
            Tensor: 计算结果张量 out
        """
        super().__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode, device, dtype
        )

    def forward(self, input):
        return op.miopen_conv_bias_relu(
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
        """_summary_

        Args:
            in_channels (int): 输入张量的通道数
            out_channels (int): 卷积核的通道数
            kernel_size (_size_2_t): 卷积核尺寸, 传入 x, 自动扩展为[x, x]
            stride (_size_2_t, optional): 卷积步长, 默认为 1, 自动扩展为[1, 1]
            padding (Union[str, _size_2_t], optional): 填充大小, 默认为 0.
            dilation (_size_2_t, optional): 卷积膨胀率. 默认为 1, 自动扩展为[1, 1].
            groups (int, optional): 分组数, 默认为 1, 即不分组.
            bias (bool, optional): 是否具有偏置项, 默认为 True.
            padding_mode (str, optional): 填充模式, 默认为 "zeros".
            device (_type_, optional): 设备类型, 默认为 None.
            dtype (_type_, optional): 数据类型, 默认为 None.
            
        Returns:
            Tensor: 计算结果张量 out
        """
        super().__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode, device, dtype
        )

    def forward(self, input: Tensor, extra_input: Tensor):
        return op.miopen_conv_bias_add(
            input, self.weight, self.bias, extra_input, list(self.padding), list(self.stride), list(self.dilation)
        )


class miopenConvBiasAddRelu(torch.nn.Conv2d):
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
        """_summary_

        Args:
            in_channels (int): 输入张量的通道数
            out_channels (int): 卷积核的通道数
            kernel_size (_size_2_t): 卷积核尺寸, 传入 x, 自动扩展为[x, x]
            stride (_size_2_t, optional): 卷积步长, 默认为 1, 自动扩展为[1, 1]
            padding (Union[str, _size_2_t], optional): 填充大小, 默认为 0.
            dilation (_size_2_t, optional): 卷积膨胀率. 默认为 1, 自动扩展为[1, 1].
            groups (int, optional): 分组数, 默认为 1, 即不分组.
            bias (bool, optional): 是否具有偏置项, 默认为 True.
            padding_mode (str, optional): 填充模式, 默认为 "zeros".
            device (_type_, optional): 设备类型, 默认为 None.
            dtype (_type_, optional): 数据类型, 默认为 None.
            
        Returns:
            Tensor: 计算结果张量 out
        """
        super().__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode, device, dtype
        )

    def forward(self, input: Tensor, extra_input: Tensor):
        return op.miopen_conv_bias_add_relu(
            input, self.weight, self.bias, extra_input, list(self.padding), list(self.stride), list(self.dilation)
        )