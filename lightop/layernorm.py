import torch
from lightop import op

class LayerNorm(torch.nn.Module):

    def __init__(self, normalized_shape:int,eps: float = 1e-5,device=None, dtype=None ):
        super(LayerNorm, self).__init__()
        self.eps = eps
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.weight = torch.nn.Parameter(torch.empty(normalized_shape, **factory_kwargs))
        self.bias = torch.nn.Parameter(torch.empty(normalized_shape, **factory_kwargs))
        torch.nn.init.ones_(self.weight)
        torch.nn.init.zeros_(self.bias)

    def forward(self, x ):
        return op.layernorm_forward_autograd(x, self.weight,self.bias,self.eps)

    def extra_repr(self):
        return f'eps={round(self.eps,5):0.5f}'