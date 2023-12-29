import torch
from lightop import op

class GroupNorm(torch.nn.Module):

    def __init__(self,num_groups:int, num_channels:int,eps: float = 1e-5,device=None, dtype=None ):
        super(GroupNorm, self).__init__()
        self.eps = eps
        self.num_groups = num_groups
        self.num_channels = num_channels
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.weight = torch.nn.Parameter(torch.empty(num_channels, **factory_kwargs))
        self.bias = torch.nn.Parameter(torch.empty(num_channels, **factory_kwargs))
        torch.nn.init.ones_(self.weight)
        torch.nn.init.zeros_(self.bias)

    def forward(self, x):
        return op.groupnorm_forward_autograd(x, self.weight,self.bias,self.num_groups,self.num_channels,self.eps,self.training)

    def extra_repr(self):
        return f'num_groups={self.num_groups},num_channels={self.num_channels},eps={round(self.eps,5):0.5f}'


class CeluGN(torch.nn.Module):

    def __init__(self,num_groups:int, num_channels:int,eps: float = 1e-5,alpha: float = 1.,device=None, dtype=None ):
        super(CeluGN, self).__init__()
        self.eps = eps
        self.num_groups = num_groups
        self.num_channels = num_channels
        self.alpha = alpha
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.weight = torch.nn.Parameter(torch.empty(num_channels, **factory_kwargs))
        self.bias = torch.nn.Parameter(torch.empty(num_channels, **factory_kwargs))
        torch.nn.init.ones_(self.weight)
        torch.nn.init.zeros_(self.bias)

    def forward(self, x):
        return op.celu_gn_forward_autograd(x, self.weight,self.bias,self.num_groups,self.num_channels,self.eps,self.alpha,self.training)

    def extra_repr(self):
        return f'num_groups={self.num_groups},num_channels={self.num_channels},eps={round(self.eps,5):0.5f},alpha={round(self.alpha,1):0.1f}'