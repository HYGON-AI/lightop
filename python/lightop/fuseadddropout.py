# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

import torch
#from lightop import op
from . import op


class AddDropout(torch.nn.Module):
    rate: float

    def __init__(self, rate=0.5):
        self.rate = rate
        super(AddDropout, self).__init__()

    def forward(self, inputA, inputB):
        return op.add_dropout_forward_autograd(inputA, inputB, self.rate, self.training)

    def extra_repr(self):
        return f'rate={round(self.rate,3):0.3f}'