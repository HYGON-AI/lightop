import torch
from lightop import op

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

