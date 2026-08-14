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
    # Specialized unsorted power-of-two topk from custom_ops (k in {2,4,8,16}).
    if (
        k in (2, 4, 8, 16)
        and largest
        and (not sorted)
        and x.is_cuda
        and dim == x.dim() - 1
        and x.dtype in (torch.float16, torch.bfloat16, torch.float32)
    ):
        value = torch.empty(*x.shape[:-1], k, device=x.device, dtype=x.dtype)
        indice = torch.empty(*x.shape[:-1], k, device=x.device, dtype=torch.int64)
        op.topk_power2_opt(value, indice, x, k)
        return value, indice
    if k <= 100 and x.size()[dim]<=32768 and largest and x.is_cuda:
      return op.topk_opt(x,k,dim,sorted)
    else :
      return torch.topk(x, k, dim=dim, largest=largest, sorted=sorted)