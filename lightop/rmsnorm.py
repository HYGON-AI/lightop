import torch
from lightop import op

class RMSNorm(torch.nn.Module):

    def __init__(self, normalized_shape:int,eps: float = 1e-5,device=None, dtype=None ):
        super(RMSNorm, self).__init__()
        self.eps = eps
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.weight = torch.nn.Parameter(torch.empty(normalized_shape, **factory_kwargs))
        torch.nn.init.ones_(self.weight)

    def forward(self, x ):
        return op.rmsnorm_forward_autograd(x, self.weight,self.eps, self.training)

    def extra_repr(self):
        return f'eps={round(self.eps,5):0.5f}'

class RMSNormTorch(torch.nn.Module):
    def __init__(self, normalized_shape:int,eps: float = 1e-5,device=None, dtype=None):

        super(RMSNormTorch, self).__init__()
        self.eps = eps
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.weight = torch.nn.Parameter(torch.empty(normalized_shape, **factory_kwargs))
        torch.nn.init.ones_(self.weight)


    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)*self.weight

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
