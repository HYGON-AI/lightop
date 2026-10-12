# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

import torch
#from lightop import op
from . import op

class DropPath(torch.nn.Module):

    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True ):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return op.droppath_forward_autograd(x, self.drop_prob,self.training,self.scale_by_keep)

    def extra_repr(self):
        return f'drop_prob={round(self.drop_prob,3):0.3f},scale_by_keep={self.scale_by_keep}'
