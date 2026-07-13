import torch
import warnings
from typing import Optional
from typing import Any, List, Optional, Tuple, TYPE_CHECKING, Union

from torch import Tensor
#from lightop import op
from . import op


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
    r"""
    Applies the Softmax function to an n-dimensional input Tensor rescaling them so that the elements of the n-dimensional output Tensor lie in the range [0,1] and sum to 1.

    Softmax is defined as

        Softmax(xi) = exp(xi) / sum(exp(xi))

    Shape:
        Input: (∗) where * means, any number of additional dimensions
        Output: (∗) , same shape as the input

    Parameters
        dim (int) – A dimension along which Softmin will be computed (so every slice along dim will sum to 1).
        half_to_float (bool) – When the input is of type half, set whether the output is of type float.

    Returns
        a Tensor of the same dimension and shape as the input, with values in the range [0, 1]

    Examples:

        >>> m = nn.Softmax(dim=1)
        >>> input = torch.randn(2, 3)
        >>> output = m(input)
    """
    dim: Optional[int]

    def __init__(self, dim: int = -1, half_to_float: bool = False) -> Tensor:
        super(FuseSoftmax, self).__init__()
        self.dim = dim
        self.half_to_float = half_to_float

    def forward(self, input, **kargs):
        if self.dim is None:
            self.dim = _get_softmax_dim("softmax", input.dim(), 5)

        return op.softmax_forward_autograd(input, self.dim, self.half_to_float, self.training)

    def extra_repr(self):
        return 'dim={}, half_to_float={}'.format(self.dim, self.half_to_float)


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

    def __init__(self, half_to_float: bool = False) -> Tensor:
        super(FuseSoftmax2d, self).__init__()
        self.half_to_float = half_to_float
        self.dim = -3

    def forward(self, input, **kargs):
        assert input.dim() == 4 or input.dim() == 3, 'FuseSoftmax2d requires a 3D or 4D tensor as input'

        return op.softmax_forward_autograd(input, self.dim, self.half_to_float, self.training)

    def extra_repr(self):
        return 'dim={}, half_to_float={}'.format(self.dim, self.half_to_float)


class FuseLogSoftmax(torch.nn.Module):
    r"""
    Applies the log(Softmax(x)) function to an n-dimensional input Tensor.

    The LogSoftmax formulation can be simplified as:

    LogSoftmax(xi) = LogSoftmax(xi)=log(Softmax(xi))

    Shape:
        Input: (∗) where * means, any number of additional dimensions
        Output: (∗) , same shape as the input

    Parameters
        dim (int) – A dimension along which LogSoftmax will be computed.
        half_to_float (bool) – When the input is of type half, set whether the output is of type float.

    Returns
        a Tensor of the same dimension and shape as the input with values in the range [-inf, 0)

    Examples:

        >>> m = nn.LogSoftmax()
        >>> input = torch.randn(2, 3)
        >>> output = m(input)
    """
    dim: Optional[int]

    def __init__(self, dim: int = -1, half_to_float: bool = False) -> Tensor:
        super(FuseLogSoftmax, self).__init__()
        self.dim = dim
        self.half_to_float = half_to_float

    def forward(self, input, **kargs):
        if self.dim is None:
            self.dim = _get_softmax_dim("softmax", input.dim(), 5)

        return op.logsoftmax_forward_autograd(input, self.dim, self.half_to_float, self.training)

    def extra_repr(self):
        return 'dim={}, half_to_float={}'.format(self.dim, self.half_to_float)
