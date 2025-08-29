import numbers
import torch
from lightop import op
from .miopen_utils import _shape_t


def miopen_layernorm(x, weight, bias, epsilon: float = 1e-5, normalized_dim: int = 0, mode: int = 0):
    """_summary_

    Args:
        x (Tensor): 输入张量，形状为 [N, C_in, H, W] (NCHW) 或  [N, C_in, D, H, W] (NCDHW)
        weight (Tensor): 卷积核权重，形状为 [1] (normalized_dim = 4 | normalized_dim = 5), 或 [{x.size().begin() + normalized_dim, x.size().end()}]
        bias (Tesnor): 偏置项，形状为 [1] (normalized_dim = 4 | normalized_dim = 5), 或 [{x.size().begin() + normalized_dim, x.size().end()}]
        epsilon (float): 用于稳定逆方差计算的值
        normalized_dim (int): 归一化维度
        mode (int): 归一化模式，支持(
                                    0: MIOPEN_ELEMENTWISE_AFFINE [weights初始化为值为1的张量, bias初始化为值为0的张量] |
                                    1: MIOPEN_WEIGHT_BIAS [具有可学习参数的weights和bias] )

    Returns:
        vector<Tensor>: 返回 yDesc, 形状为  [N, C_in, H, W] (NCHW) 或 [N, C_in, D, H, W] (NCDHW)
                            meanDesc, 形状为 [1] (normalized_dim = 0) 或 [{x.size().begin(), x.size().end() - (x.size()-normalized_dim)}] (normalized_dim != 0)
                            rstdDesc, 形状为 [1] (normalized_dim = 0) 或 [{x.size().begin(), x.size().end() - (x.size()-normalized_dim)}] (normalized_dim != 0)
    """
    return op.miopen_layernorm(x, weight, bias, epsilon, normalized_dim, mode)


def miopen_add_layernorm(x1, x2, weight, bias, epsilon: float = 1e-5, normalized_dim: int = 0, mode: int = 0):
    """_summary_

    Args:
        x1 (Tesnor): 输入张量，[N, C_in, H, W] (NCHW) 或  [N, C_in, D, H, W] (NCDHW)
        x2 (Tensor): 输入张量，[N, C_in, H, W] (NCHW) 或  [N, C_in, D, H, W] (NCDHW)
        weight (Tensor): 卷积核权重，形状为 [1] (normalized_dim = 4 | normalized_dim = 5), 或 [{x.size().begin() + normalized_dim, x.size().end()}]
        bias (Tesnor): 偏置项，形状为 [1] (normalized_dim = 4 | normalized_dim = 5), 或 [{x.size().begin() + normalized_dim, x.size().end()}]
        epsilon (float): 用于稳定逆方差计算的值
        normalized_dim (int): 归一化维度
        mode (int): 归一化模式，支持(
                                    0: MIOPEN_ELEMENTWISE_AFFINE [weights初始化为值为1的张量, bias初始化为值为0的张量] |
                                    1: MIOPEN_WEIGHT_BIAS [具有可学习参数的weights和bias] )

    Returns:
        vector<Tensor>: 返回 yDesc, 形状为  [N, C_in, H, W] (NCHW) 或 [N, C_in, D, H, W] (NCDHW)
                            meanDesc, 形状为 [1] (normalized_dim = 0) 或 [{x.size().begin(), x.size().end() - (x.size()-normalized_dim)}] (normalized_dim != 0)
                            rstdDesc, 形状为 [1] (normalized_dim = 0) 或 [{x.size().begin(), x.size().end() - (x.size()-normalized_dim)}] (normalized_dim != 0)
    """
    return op.miopen_add_layernorm(x1, x2, weight, bias, epsilon, normalized_dim, mode)


def miopen_skip_layernorm(x1, x2, weight, bias, epsilon: float = 1e-5, normalized_dim: int = 0, mode: int = 0):
    """_summary_

    Args:
        x1 (Tesnor): 输入张量，[N, C_in, H, W] (NCHW) 或  [N, C_in, D, H, W] (NCDHW)
        x2 (Tensor): 输入张量，[N, C_in, H, W] (NCHW) 或  [N, C_in, D, H, W] (NCDHW)
        weight (Tensor): 卷积核权重，形状为 [1] (normalized_dim = 4 | normalized_dim = 5), 或 [{x.size().begin() + normalized_dim, x.size().end()}]
        bias (Tesnor): 偏置项，形状为 [1] (normalized_dim = 4 | normalized_dim = 5), 或 [{x.size().begin() + normalized_dim, x.size().end()}]
        epsilon (float): 用于稳定逆方差计算的值
        normalized_dim (int): 归一化维度
        mode (int): 归一化模式，支持(
                                    0: MIOPEN_ELEMENTWISE_AFFINE [weights初始化为值为1的张量, bias初始化为值为0的张量] |
                                    1: MIOPEN_WEIGHT_BIAS [具有可学习参数的weights和bias] )

    Returns:
        vector<Tensor>: 返回 yDesc, 形状为  [N, C_in, H, W] (NCHW) 或 [N, C_in, D, H, W] (NCDHW)
                            skipDesc, 形状为 [N, C_in, H, W] (NCHW) 或 [N, C_in, D, H, W] (NCDHW)
                            meanDesc, 形状为 [1] (normalized_dim = 0) 或 [{x.size().begin(), x.size().end() - (x.size()-normalized_dim)}] (normalized_dim != 0)
                            rstdDesc, 形状为 [1] (normalized_dim = 0) 或 [{x.size().begin(), x.size().end() - (x.size()-normalized_dim)}] (normalized_dim != 0)
    """
    return op.miopen_skip_layernorm(x1, x2, weight, bias, epsilon, normalized_dim, mode)


class miopenLayerNorm(torch.nn.Module):
    def __init__(
        self, normalized_shape: _shape_t, mode: int = 0, eps: float = 1e-5, bias: bool = True, device=None, dtype=None
    ):
        """_summary_

        Args:
            normalized_shape (_shape_t): 归一化维度, 可输入单个int类型值, 也可输入list类型
            mode (int, optional): 归一化模式，支持(
                                    0: MIOPEN_ELEMENTWISE_AFFINE [weights初始化为值为1的张量, bias初始化为值为0的张量] |
                                    1: MIOPEN_WEIGHT_BIAS [具有可学习参数的weights和bias] )
            eps (float, optional): 用于稳定逆方差计算的值. 默认为 1e-5.
            bias (bool, optional): 是否具有偏置项. 默认为 True.
            device (_type_, optional): 设备类型. 默认为 None.
            dtype (_type_, optional): 数据类型. 默认为 None.
        """
        super(miopenLayerNorm, self).__init__()
        factory_kwargs = {"device": device, "dtype": dtype}

        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)

        self.normalized_shape = tuple(normalized_shape)
        self.mode = mode
        self.eps = eps
        self.bias = bias

        self.weight = torch.nn.Parameter(torch.empty(self.normalized_shape, **factory_kwargs))
        self.bias = torch.nn.Parameter(torch.empty(self.normalized_shape, **factory_kwargs))

        torch.nn.init.ones_(self.weight)
        torch.nn.init.zeros_(self.bias)

    def forward(self, x):
        rank = x.dim()
        norm_shape_rank = len(self.normalized_shape)

        normalized_dim = rank - norm_shape_rank

        if normalized_dim >= rank or normalized_dim < 0:
            raise ValueError("normalized_dim out of range")

        return op.miopen_layernorm(x, self.weight, self.bias, self.eps, normalized_dim, self.mode)

    def extra_repr(self) -> str:
        return "normalized_shape = {normalized_shape}, eps = {eps}, " "elementwise_affine= {mode}".format(
            **self.__dict__
        )
