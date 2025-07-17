
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

def conv_bias_relu(input, weight, bias, padding, stride, dilation, layout):
    """_summary_

    Args:
        input (Tensor): 输入张量
        weight (Tensor): 卷积核权重
        bias (Tensor): 偏置项
        ppadding (int|tuple): 填充大小。(padH, padW) 元组
        stride (int|tuple): 卷积步长。 (strideH, strideW) 元组
        dilation (int|tuple): 卷积膨胀率。(dilationH, dilationW) 元组
        layout (str): 数据布局格式。支持的格式: "NCHW" | "NHWC"

    Returns:
        Tensor: yDesc 计算结果张量
    """
    return op.conv_bias_relu(input, weight, bias, padding, stride, dilation, layout)

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

def rms_norm_dynamic_quant_without_residual(x, weight, smoothScale, epsilon):
    """_summary_

    Args:
        x (Tensor): 输入张量，形状为 [N, C_in, H, W] (NCHW) 或  [N, C_in, D, H, W] (NCDHW)
        weight (Tensor): 权值张量，形状为 [W]
        smoothScale (Tensor): 形状为 [W]
        epsilon (_tfloatype_): 用于稳定逆方差计算的值

    Returns:
        vector<Tensor>: 返回 addDesc 形状为  [N, C_in, H, W] (NCHW) 或 [N, C_in, D, H, W] (NCDHW)
                            yDesc 形状为 [N, C_in, H, W] (NCHW) 或 [N, C_in, D, H, W] (NCDHW)
                            perTokenScaleDesc/rstdDesc 形状为 [N, C_in, H] (NCHW) 或 [N, C_in, D, H] (NCDHW)
                            
    """
    return op.rms_norm_dynamic_quant_without_residual(x, weight, smoothScale, epsilon)

def head_rms_norm(x, weight, epsilon, head_offset, head_num):
    """_summary_

    Args:
        x (Tensor): 输入张量，形状为 [N, C_in, H, W] (NCHW) 或  [N, C_in, D, H, W] (NCDHW)
        weight (Tensor): 权重张量，形状为 [W]
        epsilon (float): 用于稳定逆方差计算的值
        head_offset (float): 用于稳定逆方差计算的值
        head_num (float): 用于稳定逆方差计算的值

    Returns:
        Tensor: 返回 yDesc, 形状为 [N, C_in, H, W] (NCHW) 或  [N, C_in, D, H, W] (NCDHW)
    """
    return op.head_rms_norm(x, weight, epsilon, head_offset, head_num)

def groupnorm(x, weight, bias, num_groups, epsilon, mode, layout):
    """_summary_

    Args:
        x (Tensor): 输入张量，形状为 [N, C_in, H, W] (NCHW) 或 [N, H, W, C_in] (NHWC) 或 [N, C_in, D, H, W] (NCDHW) 或 [N, D, H, W, C_in] (NDHWC)
        weight (Tensor): 滤波器权重，形状为 [1, C_in, 1, 1] (4D) 或 [1, C_in, 1, 1, 1] (5D)
        bias (Tensor): 偏置项，形状为 [1, C_in, 1, 1] (4D) 或 [1, C_in, 1, 1, 1] (5D)
        num_groups (int): 将通道分成的组数
        epsilon (float): 用于稳定逆方差计算的值
        mode (int): 组归一化模式，支持(0, 1, 10, 11)

    Returns:
        vector<Tensor>: 返回 yDesc, 形状为  [N, C_in, H, W] (NCHW) 或 [N, H, W, C_in] (NHWC) 或 [N, C_in, D, H, W] (NCDHW) 或 [N, D, H, W, C_in] (NDHWC)
                            meanDesc, 形状为 [N*num_groups, 1, 1, 1] (4D) 或 [N*num_groups, 1, 1, 1, 1] (5D)
                            rstdDesc, 形状为 [N*num_groups, 1, 1, 1] (4D) 或 [N*num_groups, 1, 1, 1, 1] (5D)
    """
    return op.groupnorm(x, weight, bias, num_groups, epsilon, mode, layout)

def add_groupnorm(x1, x2, weight, bias, num_groups, epsilon, mode, layout):
    """_summary_

    Args:
        x1 (Tensor): 输入张量，形状为 [N, C_in, H, W] (NCHW) 或 [N, H, W, C_in] (NHWC) 或 [N, C_in, D, H, W] (NCDHW) 或 [N, D, H, W, C_in] (NDHWC)
        x2 (Tensor): 输入张量，形状为 [N, C_in, H, W] (NCHW) 或 [N, H, W, C_in] (NHWC) 或 [N, C_in, D, H, W] (NCDHW) 或 [N, D, H, W, C_in] (NDHWC)
        weight (Tensor): 卷积核权重，形状为 [1, C_in, 1, 1] (4D) 或 [1, C_in, 1, 1, 1] (5D)
        bias (Tensor): 偏置项，形状为 [1, C_in, 1, 1] (4D) 或 [1, C_in, 1, 1, 1] (5D)
        num_groups (int): 将通道分成的组数
        epsilon (float): 用于稳定逆方差计算的值
        mode (int): 组归一化模式，支持(2, 3, 12, 13)

    Returns:
        vector<Tensor>: 返回 yDesc, 形状为  [N, C_in, H, W] (NCHW) 或 [N, H, W, C_in] (NHWC) 或 [N, C_in, D, H, W] (NCDHW) 或 [N, D, H, W, C_in] (NDHWC)
                            meanDesc, 形状为 [N*num_groups, 1, 1, 1] (4D) 或 [N*num_groups, 1, 1, 1, 1] (5D)
                            rstdDesc, 形状为 [N*num_groups, 1, 1, 1] (4D) 或 [N*num_groups, 1, 1, 1, 1] (5D)
    """
    return op.add_groupnorm(x1, x2, weight, bias, num_groups, epsilon, mode, layout)

def layernorm(x, weight, bias, epsilon, normalized_dim, mode):
    """_summary_

    Args:
        x (Tensor): 输入张量，形状为 [N, C_in, H, W] (NCHW) 或  [N, C_in, D, H, W] (NCDHW)
        weight (Tensor): 卷积核权重，形状为 [1] (normalized_dim = 4 | normalized_dim = 5), 或 [{x.size().begin() + normalized_dim, x.size().end()}]
        bias (Tesnor): 偏置项，形状为 [1] (normalized_dim = 4 | normalized_dim = 5), 或 [{x.size().begin() + normalized_dim, x.size().end()}]
        epsilon (float): 用于稳定逆方差计算的值
        normalized_dim (int): 归一化维度
        mode (int): 归一化模式，支持(0, 1)

    Returns:
        vector<Tensor>: 返回 yDesc, 形状为  [N, C_in, H, W] (NCHW) 或 [N, C_in, D, H, W] (NCDHW)
                            meanDesc, 形状为 [1] (normalized_dim = 0) 或 [{x.size().begin(), x.size().end() - (x.size()-normalized_dim)}] (normalized_dim != 0)
                            rstdDesc, 形状为 [1] (normalized_dim = 0) 或 [{x.size().begin(), x.size().end() - (x.size()-normalized_dim)}] (normalized_dim != 0)
    """
    return op.layernorm(x, weight, bias, epsilon, normalized_dim, mode)

def add_layernorm(x1, x2, weight, bias, epsilon, normalized_dim, mode):
    """_summary_

    Args:
        x1 (Tesnor): 输入张量，[N, C_in, H, W] (NCHW) 或  [N, C_in, D, H, W] (NCDHW)
        x2 (Tensor): 输入张量，[N, C_in, H, W] (NCHW) 或  [N, C_in, D, H, W] (NCDHW)
        weight (Tensor): 卷积核权重，形状为 [1] (normalized_dim = 4 | normalized_dim = 5), 或 [{x.size().begin() + normalized_dim, x.size().end()}]
        bias (Tesnor): 偏置项，形状为 [1] (normalized_dim = 4 | normalized_dim = 5), 或 [{x.size().begin() + normalized_dim, x.size().end()}]
        epsilon (float): 用于稳定逆方差计算的值
        normalized_dim (int): 归一化维度
        mode (int): 归一化模式，支持(0, 1)

    Returns:
        vector<Tensor>: 返回 yDesc, 形状为  [N, C_in, H, W] (NCHW) 或 [N, C_in, D, H, W] (NCDHW)
                            meanDesc, 形状为 [1] (normalized_dim = 0) 或 [{x.size().begin(), x.size().end() - (x.size()-normalized_dim)}] (normalized_dim != 0)
                            rstdDesc, 形状为 [1] (normalized_dim = 0) 或 [{x.size().begin(), x.size().end() - (x.size()-normalized_dim)}] (normalized_dim != 0)
    """
    return op.add_layernorm(x1, x2, weight, bias, epsilon, normalized_dim, mode)

def skip_layernorm(x1, x2, weight, bias, epsilon, normalized_dim, mode):
    """_summary_

    Args:
        x1 (Tesnor): 输入张量，[N, C_in, H, W] (NCHW) 或  [N, C_in, D, H, W] (NCDHW)
        x2 (Tensor): 输入张量，[N, C_in, H, W] (NCHW) 或  [N, C_in, D, H, W] (NCDHW)
        weight (Tensor): 卷积核权重，形状为 [1] (normalized_dim = 4 | normalized_dim = 5), 或 [{x.size().begin() + normalized_dim, x.size().end()}]
        bias (Tesnor): 偏置项，形状为 [1] (normalized_dim = 4 | normalized_dim = 5), 或 [{x.size().begin() + normalized_dim, x.size().end()}]
        epsilon (float): 用于稳定逆方差计算的值
        normalized_dim (int): 归一化维度
        mode (int): 归一化模式，支持(0, 1)

    Returns:
        vector<Tensor>: 返回 yDesc, 形状为  [N, C_in, H, W] (NCHW) 或 [N, C_in, D, H, W] (NCDHW)
                            skipDesc, 形状为 [N, C_in, H, W] (NCHW) 或 [N, C_in, D, H, W] (NCDHW)
                            meanDesc, 形状为 [1] (normalized_dim = 0) 或 [{x.size().begin(), x.size().end() - (x.size()-normalized_dim)}] (normalized_dim != 0)
                            rstdDesc, 形状为 [1] (normalized_dim = 0) 或 [{x.size().begin(), x.size().end() - (x.size()-normalized_dim)}] (normalized_dim != 0)
    """
    return op.skip_layernorm(x1, x2, weight, bias, epsilon, normalized_dim, mode)
