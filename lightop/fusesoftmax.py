import torch
from torch import Tensor
from torch.nn import init
from lightop import op
import warnings
from typing import Optional


def _get_softmax_dim(name: str, ndim: int, stacklevel: int) -> int:
    warnings.warn(
        "Implicit dimension choice for {} has been deprecated. "
        "Change the call to include dim=X as an argument.".format(name),
        stacklevel=stacklevel,
    )
    if ndim == 0 or ndim == 1 or ndim == 3:
        ret = 0
    else:
        ret = 1
    return ret

class FuseSoftmax(torch.nn.Module):
    dim: Optional[int]
    def __init__(self, half_to_float = False, dim = None) -> None:
        super(FuseSoftmax, self).__init__()
        self.half_to_float = half_to_float
        self.dim = dim
    

    def forward(self, input, **kargs):
        if self.dim is None:
            self.dim = _get_softmax_dim("softmax", input.dim(), 5)
        return op.softmax_forward_autograd(input, self.dim, self.half_to_float)

class FuseSoftmax2d(torch.nn.Module):
    r"""Applies SoftMax over features to each spatial location.

    When given an image of ``Channels x Height x Width``, it will
    apply `Softmax` to each location :math:`(Channels, h_i, w_j)`

    Shape:
        - Input: :math:`(N, C, H, W)` or :math:`(C, H, W)`.
        - Output: :math:`(N, C, H, W)` or :math:`(C, H, W)` (same shape as input)

    Returns:
        a Tensor of the same dimension and shape as the input with
        values in the range [0, 1]

    Examples::

        >>> m = nn.Softmax2d()
        >>> # you softmax over the 2nd dimension
        >>> input = torch.randn(2, 3, 12, 13)
        >>> output = m(input)
    """
    def __init__(self, half_to_float = False) -> None:
        super(FuseSoftmax2d, self).__init__()
        self.half_to_float = half_to_float

    def forward(self, input, **kargs):
        assert input.dim() == 4 or input.dim() == 3, 'FuseSoftmax2d requires a 3D or 4D tensor as input'
        return op.softmax_forward_autograd(input, -3, self.half_to_float)

class FuseLogSoftmax(torch.nn.Module):
    dim: Optional[int]
    def __init__(self, half_to_float = False, dim = None) -> None:
        super(FuseLogSoftmax, self).__init__()
        self.half_to_float = half_to_float
        self.dim = dim
    

    def forward(self, input, **kargs):
        if self.dim is None:
            self.dim = _get_softmax_dim("softmax", input.dim(), 5)
        return op.logsoftmax_forward_autograd(input, self.dim, self.half_to_float)
