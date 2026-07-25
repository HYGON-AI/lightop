from torch.autograd import Function
import torch
from torch import Tensor

#from lightop import op
from . import op


class ReluDropoutFunction(Function):
    @staticmethod
    def forward(ctx, x, rate, use_relu, train) -> Tensor:
        ret, mask = op.relu_dropout_forward(x, rate, use_relu, train)
        ctx.save_for_backward(mask)
        return ret

    @staticmethod
    def backward(ctx, grad):
        mask, = ctx.saved_tensors
        # print("mask type is ", mask.type())
        ret = op.relu_dropout_backward(grad, mask)
        return ret, None, None, None


class ReluDropout(torch.nn.Module):
    rate: float

    def __init__(self, use_relu: bool = True, droprate: float = 0.5):
        super(ReluDropout, self).__init__()
        self.use_relu = use_relu
        self.rate = 1 - droprate

    def forward(self, input_x):
        return ReluDropoutFunction.apply(input_x, self.rate, self.use_relu, self.training)

    def extra_repr(self):
        return f'use_relu={self.use_relu},rate={round(self.rate,3):0.3f}'