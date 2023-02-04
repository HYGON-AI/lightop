import torch
from torch import Tensor
from torch.autograd import Function
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
        assert input.dim() == 2 or input.dim() == 1, 'Softmax1d requires a 1D or 2D tensor as input'
        if self.dim is None:
            self.dim = _get_softmax_dim("softmax", input.dim(), 5)
        return op.softmax_forward_autograd(input, self.dim, self.half_to_float)

class FuseLogSoftmax(torch.nn.Module):
    dim: Optional[int]
    def __init__(self, half_to_float = False, dim = None) -> None:
        super(FuseLogSoftmax, self).__init__()
        self.half_to_float = half_to_float
        self.dim = dim
    

    def forward(self, input, **kargs):
        assert input.dim() == 2 or input.dim() == 1, 'Softmax1d requires a 1D or 2D tensor as input'
        if self.dim is None:
            self.dim = _get_softmax_dim("softmax", input.dim(), 5)
        return op.logsoftmax_forward_autograd(input, self.dim, self.half_to_float)
