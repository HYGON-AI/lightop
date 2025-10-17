from torch.autograd import Function
import torch
from torch import Tensor
#from lightop import op
from . import op

class FuseBiasDropoutAddFunction(Function):

    @staticmethod
    def forward(ctx, input, bias,residual,droprate,training)-> Tensor:
        result, mask = op.bias_dropout_add_forward(input, bias,residual,droprate,training)
        ctx.save_for_backward(mask)
        ctx.droprate = droprate
        return result
    
    #@staticmethod
    def backward(ctx, grad_output):
        mask, = ctx.saved_tensors
        droprate = ctx.droprate
        grad_input,grad_input_bias=op.bias_dropout_add_backward(grad_output,mask,droprate)
        return grad_input,grad_input_bias,grad_output,None, None


class FuseBiasDropoutAdd(torch.nn.Module):

    rate: float
    def __init__(self,droprate= 0.5):
        self.droprate = droprate
        super(FuseBiasDropoutAdd, self).__init__()
        
    def forward(self, input,bias,residual):
        return FuseBiasDropoutAddFunction.apply(input, bias,residual,self.droprate,self.training)
    
    def extra_repr(self):
        return f'droprate={round(self.droprate,3):0.3f}'