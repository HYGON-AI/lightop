# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

import torch
import enum
#from lightop import op
from . import op
from typing import Tuple

def fill_zeros(
            shape: tuple[int, ...],
            dtype: torch.dtype,
            device: torch.device | str = "cuda"
            ) -> torch.Tensor:
    out = torch.empty(shape, device=device, dtype=dtype)
    op.fill_zeros(out, shape[0], shape[1])
    return out
