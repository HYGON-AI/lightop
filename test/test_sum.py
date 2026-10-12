# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

import torch 
from lightop.tensor import sum as lightop_sum
def allclose(a, b, rtol=1e-02, atol=1e-02, equal_nan=False):
    return torch.allclose(a, b, rtol=rtol, atol=atol, equal_nan=equal_nan)

batchsize=1024
cols=8192
dim=-1
x = torch.rand((batchsize, cols)).cuda().bfloat16()
x = x - 0.5
y1 = lightop_sum(x,dim=dim,keepdim=False)
# print(y1)
y2 = torch.sum(x,dim=dim,keepdim=False)
# print(y2)
print(allclose(y1,y2))