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



# class _FuseBase(torch.nn.Module):

#     # def __init__(self, input, output, cols, rows, half_to_float, dim) -> None:
#     __constants__ = ['dim']
#     dim: Optional[int]
#     # def __init__(self, half_to_float:Optional[bool] = False, dim = None) -> None:
#     def __init__(self, half_to_float = False, dim = None) -> None:
#         self.half_to_float = half_to_float
#         self.dim = dim

#     def __setstate__(self, state):
#         self.__dict__.update(state)
#         if not hasattr(self, 'dim'):
#             self.dim = None
            
#     # def forward(self, input: Tensor)-> Tensor:

#     #     # if self.half_to_float is None:
#     #     #     self.half_to_float=False
#     #     if self.dim is None:
#     #         self.dim = _get_softmax_dim("softmax", input.dim(), 5)
#     #     return _fuseSoftmax_function.apply(input, self.half_to_float, self.dim)
    
    
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
        if self.dim is None:
            self.dim = _get_softmax_dim("softmax", input.dim(), 5)
        assert input.dim() == 2 or input.dim() == 1, 'Softmax1d requires a 1D or 2D tensor as input'
        return _fuseSoftmax_function.apply(input, self.half_to_float, self.dim)

class FuseSoftmax2d(FuseSoftmax):
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
    # def __init__(self, half_to_float = False, dim = None) -> None:
    #     super(FuseSoftmax1d, self).__init__()

        
    # def forward(self, input):
    #     if self.dim is None:
    #         self.dim = _get_softmax_dim("softmax", input.dim(), 5)
    #     assert input.dim() == 4 or input.dim() == 3, 'Softmax2d requires a 3D or 4D tensor as input'
    #     return _fuseSoftmax_function.apply(input, self.half_to_float, self.dim)
    # def __init__(self, half_to_float = False, dim = None) -> None:
    #     super(FuseSoftmax1d, self).__init__()

        
    def forward(self, input):
        self.dim = _get_softmax_dim("softmax", input.dim(), 5)
        assert input.dim() == 4 or input.dim() == 3, 'Softmax2d requires a 3D or 4D tensor as input'
        return _fuseSoftmax_function.apply(input, self.half_to_float, self.dim)