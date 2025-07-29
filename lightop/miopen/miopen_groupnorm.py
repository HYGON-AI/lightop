import torch
from lightop import op


def miopen_groupnorm(x, weight, bias, num_groups, epsilon, mode):
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
    return op.miopen_groupnorm(x, weight, bias, num_groups, epsilon, mode)


def miopen_add_groupnorm(x1, x2, weight, bias, num_groups, epsilon, mode):
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
    return op.miopen_add_groupnorm(x1, x2, weight, bias, num_groups, epsilon, mode)


class miopenGroupNorm(torch.nn.Module):
    # mode = 0 , MIOPEN_ELEMENTWISE_AFFINE
    # mode = 1 , MIOPEN_WEIGHT_BIAS
    # mode = 10 , MIOPEN_WEIGHT_BIAS_FUSION_SILU
    # mode = 11 ,  MIOPEN_FUSION_SILU
    def __init__(self, num_groups: int, num_channels: int, mode: int, eps: float = 1e-5, device=None, dtype=None):
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
        return op.miopen_groupnorm(x, self.weight, self.bias, self.num_groups, self.eps, self.mode)

    def extra_repr(self):
        return f"num_groups={self.num_groups},num_channels={self.num_channels},eps={round(self.eps,5):0.5f},mode={self.mode}"
