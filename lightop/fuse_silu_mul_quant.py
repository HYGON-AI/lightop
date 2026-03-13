import torch
import enum
import torch.nn as nn
##from lightop import op
from . import op
from typing import Optional, Tuple

def fuse_silu_and_mul(input: torch.Tensor,output: torch.Tensor) -> None:
    op.fuse_silu_and_mul(input, output)

def fuse_silu_mul_quant(input: torch.Tensor, 
                        num_local_tokens_tensor: Optional[torch.Tensor] = None,
                        topk:int=1,
                        expect_m:int=-1,
                        output: Optional[torch.Tensor] = None,
                        scales: Optional[torch.Tensor] = None,
                        expert_ids: Optional[torch.Tensor] = None) -> tuple[torch.Tensor, torch.Tensor]:
    leading, d2 = input.shape          # d2 == 2*d
    d = d2 // 2

    if output is None:
        output = torch.empty(leading, d, dtype=torch.int8, device=input.device)
    if scales is None:
        scales = torch.empty((leading, 1),
                            device=input.device,
                            dtype=torch.float32)

    op.fuse_silu_mul_quant(input, output, scales, num_local_tokens_tensor, topk, expect_m, expert_ids)
    return output, scales

def fuse_silu_mul_quant_ep(input: torch.Tensor, 
                        tokens_per_expert: Optional[torch.Tensor] = None,
                        num_local_tokens_tensor: Optional[torch.Tensor] = None,
                        topk:int=1,
                        expect_m:int=-1) -> tuple[torch.Tensor, torch.Tensor]:
    if input.dim() != 3:
        raise ValueError(f"Input tensor must be 3-dimensional [E, T, H], but got shape {input.shape}")

    E, T, H = input.shape
    d = H // 2

    output = torch.empty(E, T, d, dtype=torch.int8, device=input.device)
    scales = torch.empty((E, T, 1),
                        device=input.device,
                        dtype=torch.float32)

    op.fuse_silu_mul_quant_ep(input, output, scales, tokens_per_expert, num_local_tokens_tensor, topk, expect_m)
    return output, scales

def fuse_silu_mul_fp8_quant(input: torch.Tensor, 
                        fp8type:int=0, #floate4m3 = 0, floate5m2 = 1 
                        num_local_tokens_tensor: Optional[torch.Tensor] = None,
                        topk:int=1,
                        expect_m:int=-1,
                        output: Optional[torch.Tensor] = None,
                        scales: Optional[torch.Tensor] = None,
                        expert_ids: Optional[torch.Tensor] = None) -> tuple[torch.Tensor, torch.Tensor]:
    leading, d2 = input.shape          # d2 == 2*d
    d = d2 // 2

    if output is None:
        if fp8type==0:
            output = torch.empty(leading, d, dtype=torch.float8_e4m3fn, device=input.device)
        else:
            output = torch.empty(leading, d, dtype=torch.float8_e5m2, device=input.device)
    if scales is None:
        scales = torch.empty((leading, 1),
                            device=input.device,
                            dtype=torch.float32)

    op.fuse_silu_mul_fp8_quant(input, output.view(torch.uint8), scales, fp8type, num_local_tokens_tensor, topk, expect_m, expert_ids)
    return output, scales

def fuse_silu_mul_fp8_quant_ep(input: torch.Tensor,
                        fp8type:int=0, #floate4m3 = 0, floate5m2 = 1 
                        tokens_per_expert: Optional[torch.Tensor] = None,
                        num_local_tokens_tensor: Optional[torch.Tensor] = None,
                        topk:int=1,
                        expect_m:int=-1) -> tuple[torch.Tensor, torch.Tensor]:
    if input.dim() != 3:
        raise ValueError(f"Input tensor must be 3-dimensional [E, T, H], but got shape {input.shape}")

    E, T, H = input.shape
    d = H // 2

    if fp8type==0:
        output = torch.empty(E, T, d, dtype=torch.float8_e4m3fn, device=input.device)
    else:
        output = torch.empty(E, T, d, dtype=torch.float8_e5m2, device=input.device)
    scales = torch.empty((E, T, 1),
                        device=input.device,
                        dtype=torch.float32)

    op.fuse_silu_mul_fp8_quant_ep(input, output.view(torch.uint8), scales, fp8type, tokens_per_expert, num_local_tokens_tensor, topk, expect_m)
    return output, scales

#通用接口
def fuse_silu_mul_per_token_quant(input: torch.Tensor, 
                        dtype: torch.dtype = torch.int8,
                        num_local_tokens_tensor: Optional[torch.Tensor] = None,
                        topk: int = 1,
                        expect_m: int = -1,
                        output: Optional[torch.Tensor] = None,
                        scales: Optional[torch.Tensor] = None,
                        expert_ids: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:

    leading, d2 = input.shape          
    d = d2 // 2

    actual_dtype = output.dtype if output is not None else dtype

    if output is None:
        output = torch.empty(leading, d, dtype=actual_dtype, device=input.device)
        
    if scales is None:
        scales = torch.empty((leading, 1),
                             device=input.device,
                             dtype=torch.float32)

    if actual_dtype == torch.int8:
        op.fuse_silu_mul_quant(input, output, scales, num_local_tokens_tensor, topk, expect_m, expert_ids)
    elif actual_dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
        # fp8type: floate4m3 = 0, floate5m2 = 1 
        fp8type = 0 if actual_dtype == torch.float8_e4m3fn else 1
        op.fuse_silu_mul_fp8_quant(input, output.view(torch.uint8), scales, fp8type, num_local_tokens_tensor, topk, expect_m, expert_ids)
    else:
        raise ValueError(f"Unsupported dtype: {actual_dtype}. Supported dtypes are torch.int8, torch.float8_e4m3fn, torch.float8_e5m2")

    return output, scales