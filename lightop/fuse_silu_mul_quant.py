import torch
import enum
import torch.nn as nn
from lightop import op

def fuse_silu_mul_quant(input: torch.Tensor
                        ) -> tuple[torch.Tensor, torch.Tensor]:
    leading, d2 = input.shape          # d2 == 2*d
    d = d2 // 2
    # output = torch.empty(leading, d, dtype=input.dtype, device=input.device)
    output = torch.empty(leading, d, dtype=torch.int8, device=input.device)
    scales = torch.empty((input.numel() // input.shape[-1], 1),
                        device=input.device,
                        dtype=torch.float32)
    # scales = torch.empty(leading, d, dtype=input.dtype, device=input.device)
    op.fuse_silu_mul_quant(input, output, scales)
    return output, scales