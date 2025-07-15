
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
