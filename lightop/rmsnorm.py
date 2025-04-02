import torch
from lightop import op
import numbers
import torch.nn as nn


def rmsnorm_forward(x,weight,eps:float = 1e-5,training:bool=True):
    return op.rmsnorm_forward(x,weight,eps,training)

def rmsnorm_backward(grad,x,rstd,gamma):
    return op.rmsnorm_backward(grad,x,rstd,gamma)

def rmsnorm_forward_torch(hidden_states,weight,eps):
    input_dtype = hidden_states.dtype
    variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    hidden_states = hidden_states * torch.rsqrt(variance + eps)
    if weight is not None:
        if weight.dtype in [torch.float16, torch.bfloat16]:
            hidden_states = hidden_states * weight.to(torch.float32)
            hidden_states = hidden_states.to(weight.dtype)
        else:
            hidden_states = hidden_states * weight
    else:
        hidden_states = hidden_states.to(input_dtype)
    return hidden_states

class RMSNorm(torch.nn.Module):

    def __init__(self, normalized_shape:int,eps: float = 1e-5,device=None, dtype=None ):
        super(RMSNorm, self).__init__()
        self.eps = eps
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.weight = torch.nn.Parameter(torch.empty(normalized_shape, **factory_kwargs))
        torch.nn.init.ones_(self.weight)
        self.rmsnorm_compile=torch.compile(rmsnorm_forward_torch)

    def forward(self, x ):
        if x.dtype == torch.bfloat16 and self.weight.numel()==128 and self.training==False:
            return self.rmsnorm_compile(x, self.weight,self.eps)
        return op.rmsnorm_forward_autograd(x, self.weight,self.eps, self.training)

    def extra_repr(self):
        return f'eps={round(self.eps,5):0.5f}'

class RMSNormTorch(torch.nn.Module):
    def __init__(self, dim, eps: float=1e-5, elementwise_affine: bool = True):
        super().__init__()

        self.eps = eps

        if isinstance(dim, numbers.Integral):
            dim = (dim,)

        self.dim = torch.Size(dim)

        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(dim))
        else:
            self.weight = None

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)
        # print(torch.rsqrt(variance + self.eps))
        if self.weight is not None:
            if self.weight.dtype in [torch.float16, torch.bfloat16]:
                hidden_states = hidden_states * self.weight.to(torch.float32)
                hidden_states = hidden_states.to(self.weight.dtype)
            else:
                hidden_states = hidden_states * self.weight
        else:
            hidden_states = hidden_states.to(input_dtype)

        return hidden_states

class RMSNormAdd(torch.nn.Module):
    rate: float

    def __init__(self, normalized_shape:int,eps: float = 1e-5,device=None, dtype=None ,inplace = False):
        super(RMSNormAdd, self).__init__()
        self.eps = eps
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.weight = torch.nn.Parameter(torch.empty(normalized_shape, **factory_kwargs))
        torch.nn.init.ones_(self.weight)
        self.inplace = inplace

    def forward(self, X, residual):
        return op.rn_add_forward_autograd(X, residual, self.weight,self.eps,self.training,self.inplace)

    def extra_repr(self):
        return f'rate={round(self.rate,3):0.3f},eps={round(self.eps,5):0.5f}'


class RMSNormDropout(torch.nn.Module):
    rate: float

    def __init__(self, normalized_shape:int,droprate: float = 0.5,eps: float = 1e-5,device=None, dtype=None ):
        super(RMSNormDropout, self).__init__()
        self.rate = droprate
        self.eps = eps
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.weight = torch.nn.Parameter(torch.empty(normalized_shape, **factory_kwargs))
        torch.nn.init.ones_(self.weight)

    def forward(self, inputA):
        return op.rn_dropout_forward_autograd(inputA, self.weight,self.eps,self.rate, self.training)

    def extra_repr(self):
        return f'rate={round(self.rate,3):0.3f},eps={round(self.eps,5):0.5f}'


class RMSNormAddDropout(torch.nn.Module):
    rate: float

    def __init__(self, normalized_shape:int,droprate: float = 0.5,eps: float = 1e-5,device=None, dtype=None ):
        super(RMSNormAddDropout, self).__init__()
        self.rate = droprate
        self.eps = eps
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.weight = torch.nn.Parameter(torch.empty(normalized_shape, **factory_kwargs))
        torch.nn.init.ones_(self.weight)

    def forward(self, X, residual):
        return op.rn_add_dropout_forward_autograd(X, residual, self.weight,self.eps,self.rate, self.training)

    def extra_repr(self):
        return f'rate={round(self.rate,3):0.3f},eps={round(self.eps,5):0.5f}'


class RMSNormAddDropoutBias(torch.nn.Module):
    rate: float

    def __init__(self, normalized_shape:int,droprate: float = 0.5,eps: float = 1e-5,device=None, dtype=None ):
        super(RMSNormAddDropoutBias, self).__init__()
        self.rate = droprate
        self.eps = eps
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.weight = torch.nn.Parameter(torch.empty(normalized_shape, **factory_kwargs))
        torch.nn.init.ones_(self.weight)

    def forward(self, X, residual, bias):
        return op.rn_add_dropout_bias_forward_autograd(X, residual, bias, self.weight, self.eps, self.rate, self.training)

    def extra_repr(self):
        return f'rate={round(self.rate,3):0.3f},eps={round(self.eps,5):0.5f}'
