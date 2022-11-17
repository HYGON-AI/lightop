from torch.autograd import Function
import torch
from lightop import op

class LnAddDropoutFunction(Function):
    @staticmethod
    def forward(ctx, x, y, weight, bias, eps, rate, train):
        ret,ret2,mean,rstd = op.ln_add_dropout_forward(x, y, weight,bias,eps,rate, train)
        #rett,meant,rstdt =torch.native_layer_norm(ret2,tuple((768,)),weight,bias,eps)
        ctx.save_for_backward(ret2, mean,rstd,weight)
        return ret

    @staticmethod
    def backward(ctx, gradOutput):
        X, mean, rstd, gamma = ctx.saved_tensors
        dx1,dx2,dgamma,dbeta = op.ln_add_dropout_backward(gradOutput,X,mean,rstd,gamma)
        return  dx1, dx2, dgamma, dbeta, None, None, None

class LnAddDropout(torch.nn.Module):
    rate: float

    def __init__(self, normalized_shape:int,droprate: float = 0.5,eps: float = 1e-5,device=None, dtype=None ):
        super(LnAddDropout, self).__init__()
        self.rate = droprate
        self.eps = eps
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.weight = torch.nn.Parameter(torch.empty(normalized_shape, **factory_kwargs))
        self.bias = torch.nn.Parameter(torch.empty(normalized_shape, **factory_kwargs))
        torch.nn.init.ones_(self.weight)
        torch.nn.init.zeros_(self.bias)

    def forward(self, inputA, inputB):
        return LnAddDropoutFunction.apply(inputA, inputB, self.weight,self.bias,self.eps,self.rate, self.training)

