import torch
#from lightop import op
from . import op

    
class FusedBiasSwiGLUFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, bias):
        input_ = input.contiguous()
        bias_ = bias.contiguous()
        output = op.FusedBiasSwiGLU_forward(input_, bias_)
        ctx.save_for_backward(input_, bias_)

        return output.view(*tuple(input.shape[:-1]), input.shape[-1] // 2)

    @staticmethod
    def backward(ctx, grad_output):
        input_, bias_ = ctx.saved_tensors
        input_grad, bias_grad = op.FusedBiasSwiGLU_backward(grad_output, input_, bias_)

        return input_grad.view(input_.shape), bias_grad, None, None
    
class FuseBiasSwiGLU(torch.nn.Module):

    def __init__(self):
        super(FuseBiasSwiGLU, self).__init__()
        
    def forward(self, input, bias):
        return FusedBiasSwiGLUFunction.apply(input, bias)

 
