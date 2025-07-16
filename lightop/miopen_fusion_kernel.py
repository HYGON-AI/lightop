
import torch
from torch import Tensor
from lightop import op


def conv_bias_add(input, weight, bias, add, padding, stride, dilation, layout):
    """
    参数:
        input (Tensor): 输入张量，形状为 [N, C_in, H, W] (NCHW) 或 [N, H, W, C_in] (NHWC)
        weight (Tensor): 卷积核权重，形状为 [C_out, C_in, kH, kW]
        bias (Tensor): 偏置项，形状为 [C_out]
        add (Tensor): 需要加和的张量，形状应与卷积输出相同
        padding (int|tuple): 填充大小。(padH, padW) 元组
        stride (int|tuple): 卷积步长。 (strideH, strideW) 元组
        dilation (int|tuple): 卷积膨胀率。(dilationH, dilationW) 元组
        layout (str): 数据布局格式。支持的格式: "NCHW" | "NHWC"
    返回:
        Tensor: 计算结果张量
    """
    return op.conv_bias_add(input, weight, bias, add, padding, stride, dilation, layout)


def conv_add(x, weight, add, padding, stride, dilation, layout, groups = 1, alpha = 1 , beta = 0, scale = 1, zero_point = 0):
    """_summary_

    Args:
        x (Tensor): 输入张量
        add (Tensor): 需要加和的张量
        weight (Tensor): 卷积核权重
        pads (int|tuple): 填充大小。(padH, padW) 元组
        strides (int|tuple): 卷积步长。 (strideH, strideW) 元组
        dilations (int|tuple): 卷积膨胀率。(dilationH, dilationW) 元组
        layout (str): 数据布局格式。支持的格式: "NCHW" | "NHWC"
        groups (int): 是否为组卷积。1为普通卷积, >1 为组卷积
        alpha (float): _description_
        beta (float): _description_
        scale (double): _description_
        zero_point (double): _description_

    Returns:
        Tensor: 结果张量
    """
    return op.conv_add(x, weight, add, padding, stride, dilation, layout, groups, alpha, beta, scale, zero_point)


def add_rms_norm_dynamic_quant(x, x2, weight, smoothScale, epsilon):
    """
        (in) hidden_states
            [num_tokens, hidden_size]
            bfloat16
        (in) residual
            optional
            [num_tokens, hidden_size]
            bfloat16
        (in) weight
            [hidden_size]
            bfloat16 or float32
        (in) smooth_scale
            [hidden_size]
            bfloat16 or float32
        (out) after_res
            [num_tokens, hidden_size]
            bfloat16
        (out) y
            [num_tokens, hidden_size]
            int8
        (out) per_token_scale
            [num_tokens]
            float32
        (attr) epsilon
            float32
    """
    
    return op.add_rms_norm_dynamic_quant(x, x2, weight, smoothScale, epsilon)

def head_rms_norm(x, weight, epsilon, head_offset, head_num):
    return op.head_rms_norm(x, weight, epsilon, head_offset, head_num)
