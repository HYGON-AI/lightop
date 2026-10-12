# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

import torch
#from lightop import op
from . import op

def mean(x, dim=-1):
    assert dim < x.dim() and dim >=-1
    x = x.contiguous()
    if dim == -1 :
        dim = x.dim()-1
    if x.size()[-1]%8 == 0:
      return op.mean_opt(x,dim)
    else :
      return torch.mean(x,dim)

def sum(x, dim=-1,keepdim=False):
    assert dim < x.dim() and dim >=-1
    x = x.contiguous()
    if dim == -1 :
        dim = x.dim()-1
    if x.size()[-1]%8 == 0:
      ret = op.sum_opt(x,dim)
      if(keepdim==True and x.dim()>ret.dim()):
         ret=ret.unsqueeze(dim)
      return ret
    else :
      return torch.sum(x,dim)

def topk(x,k, dim=-1,largest=True,sorted=True):
    assert dim < x.dim() and dim >=-1
    x = x.contiguous()
    if dim == -1 :
        dim = x.dim()-1
    if k <= 100 and x.size()[dim]<=32768 and largest and x.is_cuda:
      return op.topk_opt(x,k,dim,sorted)
    else :
      return torch.topk(x, k, dim=dim, largest=largest, sorted=sorted)