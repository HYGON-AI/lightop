# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

import torch
import enum
import torch.nn as nn
#from lightop import op
from . import op

def ds_cat(
            input_A: torch.Tensor,
            input_B: torch.Tensor,
            output: torch.Tensor,
            mode: int) -> torch.Tensor:
    op.ds_concat(input_A, input_B, output, mode)
    return output
