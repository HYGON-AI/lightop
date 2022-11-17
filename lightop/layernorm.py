from torch.autograd import Function
import torch
from lightop import op

class LayerNormFunction(Function):
    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ret,mean,rstd = op.layernorm_forward(x, weight,bias,eps)
        ctx.save_for_backward(x, mean,rstd,weight)
        return ret

    @staticmethod
    def backward(ctx, gradOutput):
        X, mean, rstd, gamma = ctx.saved_tensors
        dx,dgamma,dbeta = op.layernorm_backward(gradOutput,X,mean,rstd,gamma)
        return  dx, dgamma, dbeta, None

class LayerNorm(torch.nn.Module):

    def __init__(self, normalized_shape:int,eps: float = 1e-5,device=None, dtype=None ):
        super(LayerNorm, self).__init__()
        self.eps = eps
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.weight = torch.nn.Parameter(torch.empty(normalized_shape, **factory_kwargs))
        self.bias = torch.nn.Parameter(torch.empty(normalized_shape, **factory_kwargs))
        torch.nn.init.ones_(self.weight)
        torch.nn.init.zeros_(self.bias)

    def forward(self, input, ):
        return LayerNormFunction.apply(input, self.weight,self.bias,self.eps)

