import torch
import enum
import torch.nn as nn
from lightop import op
from typing import Optional, Tuple

def fuse_silu_mul_quant(input: torch.Tensor, 
                        num_local_tokens_tensor: Optional[torch.Tensor] = None,
                        topk:int=1) -> tuple[torch.Tensor, torch.Tensor]:
    leading, d2 = input.shape          # d2 == 2*d
    d = d2 // 2

    output = torch.empty(leading, d, dtype=torch.int8, device=input.device)
    scales = torch.empty((leading, 1),
                        device=input.device,
                        dtype=torch.float32)

    op.fuse_silu_mul_quant(input, output, scales, num_local_tokens_tensor, topk)
    return output, scales