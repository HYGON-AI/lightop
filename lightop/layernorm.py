import torch
from lightop import op

class LayerNorm(torch.nn.Module):

    def __init__(self, normalized_shape:int,eps: float = 1e-5,elementwise_affine: bool = True,
                 bias:bool = True,device=None, dtype=None):
        super(LayerNorm, self).__init__()
        self.eps = eps
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = torch.nn.Parameter(torch.empty(normalized_shape, **factory_kwargs))
            # torch.nn.init.ones_(self.weight)#lj
            if bias:
                self.bias = torch.nn.Parameter(torch.empty(normalized_shape, **factory_kwargs))
                # torch.nn.init.zeros_(self.bias)#lj
            else:
                self.register_parameter('bias',None)
        else:
            self.register_parameter('weight',None)
            self.register_parameter('bias',None)
        
        self.reset_parameters()
    
    def reset_parameters(self) -> None:
        if self.elementwise_affine:
            torch.nn.init.ones_(self.weight)#lj
            if self.bias is not None:
               torch.nn.init.zeros_(self.bias)#lj 
            
    def forward(self, x):
        return op.layernorm_forward_autograd(x,self.weight,self.bias,
                                             self.eps,self.training)

    def extra_repr(self):
        return f'eps={round(self.eps,5):0.5f}'

class AddLn(torch.nn.Module):
    def __init__(self,normalized_shape:int,eps: float = 1e-5,device=None, dtype=None ):
        super(AddLn,self).__init__()
        self.eps = eps
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.weight = torch.nn.Parameter(torch.empty(normalized_shape, **factory_kwargs))
        self.bias = torch.nn.Parameter(torch.empty(normalized_shape, **factory_kwargs))
        torch.nn.init.ones_(self.weight)
        torch.nn.init.zeros_(self.bias)
    def forward(self, X, residual):
        return op.add_ln_forward_autograd(X, residual, self.weight,self.bias,self.eps,self.training)
    def extra_repr(self):
        return f'eps={round(self.eps,5):0.5f}'

class LnDropout(torch.nn.Module):
    rate: float

    def __init__(self, normalized_shape:int,droprate: float = 0.5,eps: float = 1e-5,device=None, dtype=None ):
        super(LnDropout, self).__init__()
        self.rate = droprate
        self.eps = eps
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.weight = torch.nn.Parameter(torch.empty(normalized_shape, **factory_kwargs))
        self.bias = torch.nn.Parameter(torch.empty(normalized_shape, **factory_kwargs))
        torch.nn.init.ones_(self.weight)
        torch.nn.init.zeros_(self.bias)

    def forward(self, X):
        return op.ln_dropout_forward_autograd(X, self.weight,self.bias,self.eps,self.rate, self.training)

    def extra_repr(self):
        return f'rate={round(self.rate,3):0.3f},eps={round(self.eps,5):0.5f}'


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

    def forward(self, X, residual):
        return op.ln_add_dropout_forward_autograd(X, residual, self.weight,self.bias,self.eps,self.rate, self.training)

    def extra_repr(self):
        return f'rate={round(self.rate,3):0.3f},eps={round(self.eps,5):0.5f}'


class LnAddDropoutBias(torch.nn.Module):
    rate: float

    def __init__(self, normalized_shape:int,droprate: float = 0.5,eps: float = 1e-5,device=None, dtype=None ):
        super(LnAddDropoutBias, self).__init__()
        self.rate = droprate
        self.eps = eps
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.weight = torch.nn.Parameter(torch.empty(normalized_shape, **factory_kwargs))
        self.bias = torch.nn.Parameter(torch.empty(normalized_shape, **factory_kwargs))
        torch.nn.init.ones_(self.weight)
        torch.nn.init.zeros_(self.bias)

    def forward(self, X, residual,bias):
        return op.ln_add_dropout_bias_forward_autograd(X, residual, bias, self.weight,self.bias,self.eps,self.rate, self.training)

    def extra_repr(self):
        return f'rate={round(self.rate,3):0.3f},eps={round(self.eps,5):0.5f}'



class LnDropPath(torch.nn.Module):

    def __init__(self, cols:int, drop_prob: float = 0., eps: float = 1e-5, scale_by_keep: bool = True, device=None, dtype=None):
        super(LnDropPath, self).__init__()
        self.keep_prob = 1.0 - drop_prob
        self.scale_by_keep = scale_by_keep
        self.eps = eps
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.weight = torch.nn.Parameter(torch.empty(cols, **factory_kwargs))
        self.bias = torch.nn.Parameter(torch.empty(cols, **factory_kwargs))
        torch.nn.init.ones_(self.weight)
        torch.nn.init.zeros_(self.bias)

    def forward(self, x):
        return op.layernorm_droppath_forward_autograd(x, self.weight,self.bias,self.eps, self.keep_prob, self.scale_by_keep, self.training)

    def extra_repr(self):
        return f'keep_prob={round(self.keep_prob,3):0.3f},eps={round(self.eps,5):0.5f},scale_by_keep={self.scale_by_keep}'
