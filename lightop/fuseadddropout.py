from torch.autograd import Function
import torch
from lightop import op


class AddDropoutFunction(Function):
    @staticmethod
    def forward(ctx, x, y, rate, train):
        ret,mask = op.add_dropout_forward(x, y, rate, train)
        ctx.save_for_backward(mask, torch.Tensor([rate]))
        return ret

    @staticmethod
    def backward(ctx, gradOutput):
        mask, rate = ctx.saved_tensors
        gradX, gradY = op.add_dropout_backward(gradOutput, mask, rate.item())
        return gradX, gradY, None, None


class AddDropout(torch.nn.Module):
    rate: float

    def __init__(self, rate=0.5):
        self.rate = rate
        super(AddDropout, self).__init__()

    def forward(self, inputA, inputB):
        return AddDropoutFunction.apply(inputA, inputB, self.rate, self.training)
