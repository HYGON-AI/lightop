import torch
from lightop import op

def mean(x, dim=-1):
    assert dim < x.dim() and dim >=-1
    x = x.contiguous()
    if dim == -1 :
        dim = x.dim()-1
    if x.size()[-1]%8 == 0:
      return op.mean_opt(x,dim)
    else :
      return torch.mean(x,dim)

def sum(x, dim=-1):
    assert dim < x.dim() and dim >=-1
    x = x.contiguous()
    if dim == -1 :
        dim = x.dim()-1
    if x.size()[-1]%8 == 0:
      return op.sum_opt(x,dim)
    else :
      return torch.sum(x,dim)