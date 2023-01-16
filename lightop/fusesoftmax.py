# from asyncio import streams
# import ctypes
# import warnings
# from typing import Optional, Tuple

import torch
from torch import Tensor
# from . import functional as F
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

    
class _fuseSoftmax_function(Function):

    @staticmethod
    def forward(ctx, input, half_to_float, dim):
        forward_data= op.softmax_forward(input, dim, half_to_float)
        # ctx.save_for_backward(input, forward_data, torch.Tensor([dim]))
        ctx.save_for_backward(input, forward_data)
        ctx.dim = dim
        return forward_data


    @staticmethod
    def backward(ctx, gradOutput):
        save_input, save_output = ctx.saved_tensors
        grad_input_return = op.softmax_backward(
            gradOutput, 
            save_output,
            ctx.dim,
            save_input)
        return grad_input_return, None, None
    
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
        return _fuseSoftmax_function.apply(input, self.half_to_float, self.dim)

class _fuselogSoftmax_function(Function):

    @staticmethod
    def forward(ctx, input, half_to_float, dim):
        forward_data= op.log_softmax_forward(input, dim, half_to_float)
        # ctx.save_for_backward(input, forward_data, torch.Tensor([dim]))
        ctx.save_for_backward(input, forward_data)
        ctx.dim = dim
        return forward_data


    @staticmethod
    def backward(ctx, gradOutput):
        save_input, save_output = ctx.saved_tensors
        grad_input_return = op.log_softmax_backward(
            gradOutput, 
            save_output,
            ctx.dim,
            save_input)
        return grad_input_return, None, None


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
        return _fuselogSoftmax_function.apply(input, self.half_to_float, self.dim)
