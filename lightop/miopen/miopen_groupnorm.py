import torch
from torch.autograd import Function
from lightop import op


def miopen_groupnorm(x, weight, bias, num_groups, epsilon: float = 1e-5, mode: int = 0):
    """_summary_

    Args:
        x (Tensor): 输入张量，形状为 [N, C_in, H, W] (NCHW) 或 [N, H, W, C_in] (NHWC) 或 [N, C_in, D, H, W] (NCDHW) 或 [N, D, H, W, C_in] (NDHWC)
        weight (Tensor): 滤波器权重，形状为 [1, C_in, 1, 1] (4D) 或 [1, C_in, 1, 1, 1] (5D)
        bias (Tensor): 偏置项，形状为 [1, C_in, 1, 1] (4D) 或 [1, C_in, 1, 1, 1] (5D)
        num_groups (int): 将通道分成的组数
        epsilon (float): 用于稳定逆方差计算的值
        mode (int): 组归一化模式，支持(
                                    0: MIOPEN_ELEMENTWISE_AFFINE [weights初始化为值为1的张量, bias初始化为值为0的张量] |
                                    1: MIOPEN_WEIGHT_BIAS [具有可学习参数的weights和bias] |
                                    10: MIOPEN_WEIGHT_BIAS_FUSION_SILU [适用于norm+silu场景的, 具有可学习参数的weights和bias] |
                                    11: MIOPEN_FUSION_SILU [适用于norm+silu场景的, weights初始化为值为1的张量, bias初始化为值为0的张量])

    Returns:
        vector<Tensor>:
            yDesc, 形状为  [N, C_in, H, W] (NCHW) 或 [N, H, W, C_in] (NHWC) 或 [N, C_in, D, H, W] (NCDHW) 或 [N, D, H, W, C_in] (NDHWC)
            meanDesc, 形状为 [N*num_groups, 1, 1, 1] (4D) 或 [N*num_groups, 1, 1, 1, 1] (5D)
            rstdDesc, 形状为 [N*num_groups, 1, 1, 1] (4D) 或 [N*num_groups, 1, 1, 1, 1] (5D)
    """
    result = op.miopen_groupnorm(x, weight, bias, num_groups, epsilon, mode)
    return result[0]


def miopen_add_groupnorm(x1, x2, weight, bias, num_groups, epsilon: float = 1e-5, mode: int = 0):
    """_summary_

    Args:
        x1 (Tensor): 输入张量，形状为 [N, C_in, H, W] (NCHW) 或 [N, H, W, C_in] (NHWC) 或 [N, C_in, D, H, W] (NCDHW) 或 [N, D, H, W, C_in] (NDHWC)
        x2 (Tensor): 输入张量，形状为 [N, C_in, H, W] (NCHW) 或 [N, H, W, C_in] (NHWC) 或 [N, C_in, D, H, W] (NCDHW) 或 [N, D, H, W, C_in] (NDHWC)
        weight (Tensor): 卷积核权重，形状为 [1, C_in, 1, 1] (4D) 或 [1, C_in, 1, 1, 1] (5D)
        bias (Tensor): 偏置项，形状为 [1, C_in, 1, 1] (4D) 或 [1, C_in, 1, 1, 1] (5D)
        num_groups (int): 将通道分成的组数
        epsilon (float): 用于稳定逆方差计算的值
        mode (int): 组归一化模式，支持(
                                    2: MIOPEN_ELEMENTWISE_AFFINE_FUSED_ADD [适用于add+norm场景的, weights初始化为值为1的张量, bias初始化为值为0的张量]|
                                    3: MIOPEN_WEIGHT_BIAS_FUSED_ADD [适用于add+norm场景的, 具有可学习参数的weights和bias]|
                                    12: MIOPEN_WEIGHT_BIAS_FUSION_SILU_ADD [适用于add+norm+silu场景的, 具有可学习参数的weights和bias] |
                                    13: MIOPEN_FUSION_SILU_ADD [适用于add+norm+silu场景的, weights初始化为值为1的张量, bias初始化为值为0的张量])

    Returns:
        vector<Tensor>: 
            yDesc, 形状为  [N, C_in, H, W] (NCHW) 或 [N, H, W, C_in] (NHWC) 或 [N, C_in, D, H, W] (NCDHW) 或 [N, D, H, W, C_in] (NDHWC)
            meanDesc, 形状为 [N*num_groups, 1, 1, 1] (4D) 或 [N*num_groups, 1, 1, 1, 1] (5D)
            rstdDesc, 形状为 [N*num_groups, 1, 1, 1] (4D) 或 [N*num_groups, 1, 1, 1, 1] (5D)
    """
    return op.miopen_add_groupnorm(x1, x2, weight, bias, num_groups, epsilon, mode)


def miopen_groupnorm_backward(x, dy, weight, mean, rstd, num_groups, epsilon: float = 1e-5, mode: int =1):
    """_summary_

    Args:
        x (Tensor): 输入张量
        dy (Tensor): 
        weight (Tensor): 权值张量
        mean (Tensor): 均值张量
        rstd (Tensor): 方差张量
        num_groups (int): 组数
        epsilon (float, optional): 用于稳定逆方差计算的值. 默认为 1e-5.
        mode (int, optional): 组归一化模式. 默认为 1,  MIOPEN_WEIGHT_BIAS [具有可学习参数的weights和bias]

    Returns:
        vector<Tensor>:
            dx: 形状与x相同
            dscale: 形状 [x[0]*x[1], 1, ,1 ,1]
            dbias: 形状 [x[0]*x[1], 1, ,1 ,1]
    """
    return op.miopen_groupnorm_backward(x, dy, weight, mean, rstd, num_groups, epsilon, mode)


class miopenGroupNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, num_groups, eps, mode):
        ctx.num_groups = num_groups
        ctx.eps = eps
        ctx.mode = mode
        result = op.miopen_groupnorm(x, weight, bias, num_groups, eps, mode)
        ctx.save_for_backward(x, weight, result[1], result[2])
        return result[0]
    
    @staticmethod
    def backward(ctx, grad_output):
        x, weight, mean, rstd = ctx.saved_tensors
        grad_output = grad_output.contiguous(memory_format=torch.channels_last) if x.is_contiguous() == False else grad_output
        num_groups = ctx.num_groups
        eps = ctx.eps
        mode = ctx.mode
        result = op.miopen_groupnorm_backward(x, grad_output, weight, mean, rstd, num_groups, eps, mode)
        return result[0], result[1].squeeze(), result[2].squeeze(), None, None, None


class miopenGroupNorm(torch.nn.Module):
    def __init__(self, num_groups: int, num_channels: int, mode: int, eps: float = 1e-5, device=None, dtype=None):
        """_summary_

        Args:
            num_groups (int): 分组数
            num_channels (int): 输入张量通道数
            mode (int): 归一化模式, 支持(
                                    0: MIOPEN_ELEMENTWISE_AFFINE [weights初始化为值为1的张量, bias初始化为值为0的张量] |
                                    1: MIOPEN_WEIGHT_BIAS [具有可学习参数的weights和bias] |
                                    10: MIOPEN_WEIGHT_BIAS_FUSION_SILU [适用于norm+silu场景的, 具有可学习参数的weights和bias] |
                                    11: MIOPEN_FUSION_SILU [适用于norm+silu场景的, weights初始化为值为1的张量, bias初始化为值为0的张量])
            eps (float, optional): 用于稳定逆方差计算的值,  默认为 1e-5.
            device (_type_, optional): 设备类型, 默认为 None.
            dtype (_type_, optional): 数据类型, 默认为 None.
            
        Returns:
        vector<Tensor>: 
            yDesc, 形状为  [N, C_in, H, W] (NCHW) 或 [N, H, W, C_in] (NHWC) 或 [N, C_in, D, H, W] (NCDHW) 或 [N, D, H, W, C_in] (NDHWC)
            meanDesc, 形状为 [N*num_groups, 1, 1, 1] (4D) 或 [N*num_groups, 1, 1, 1, 1] (5D)
            rstdDesc, 形状为 [N*num_groups, 1, 1, 1] (4D) 或 [N*num_groups, 1, 1, 1, 1] (5D)
        """
        super(miopenGroupNorm, self).__init__()
        self.eps = eps
        self.num_groups = num_groups
        self.num_channels = num_channels
        self.mode = mode
        factory_kwargs = {"device": device, "dtype": dtype}
        self.weight = torch.nn.Parameter(torch.empty(num_channels, **factory_kwargs))
        self.bias = torch.nn.Parameter(torch.empty(num_channels, **factory_kwargs))
        torch.nn.init.ones_(self.weight)
        torch.nn.init.zeros_(self.bias)

    def forward(self, x):
        return miopenGroupNormFunction.apply(x, self.weight, self.bias, self.num_groups, self.eps, self.mode)

    def extra_repr(self):
        return f"num_groups={self.num_groups},num_channels={self.num_channels},eps={round(self.eps,5):0.5f},mode={self.mode}"

