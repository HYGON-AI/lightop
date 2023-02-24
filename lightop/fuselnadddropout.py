import torch
from lightop import op

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
        return op.ln_add_dropout_forward_autograd(inputA, inputB, self.weight,self.bias,self.eps,self.rate, self.training)

