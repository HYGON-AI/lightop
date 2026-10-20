# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

import torch
from typing import Optional, Tuple
from . import op  


def fuse_rms_mrope_cuda(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    mrope_section: list[int],
    head_size: int,
    rotary_dim: int,
    mrope_interleaved: bool,
    weight_q: torch.Tensor,       # [hidden_size] 
    weight_k: torch.Tensor,       # [hidden_size]
    residual_q: Optional[torch.Tensor],
    residual_k: Optional[torch.Tensor],
    epsilon: float = 1e-5,
) -> tuple[torch.Tensor, torch.Tensor]:
    
    op.fuse_rms_mrope_cuda(
        q,
        k,
        cos,
        sin,
        mrope_section,
        head_size,
        rotary_dim,
        mrope_interleaved,
        weight_q,
        weight_k,
        residual_q,
        residual_k,
        epsilon
    )
    
    return q, k
