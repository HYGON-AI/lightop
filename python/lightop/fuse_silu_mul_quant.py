import torch
import enum
import torch.nn as nn
##from lightop import op
from . import op
from typing import Optional, Tuple


def _normalize_fp8_limit(limit: Optional[float]) -> float:
    if limit is None:
        return -1.0

    normalized_limit = float(limit)
    if normalized_limit != -1.0 and normalized_limit <= 0:
        raise ValueError(
            f"limit must be -1.0 (disabled) or positive, but got {limit}"
        )
    return normalized_limit


def fuse_silu_and_mul_ep(input: torch.Tensor,
                         output: torch.Tensor,
                         mask_m: torch.Tensor,
                         expect_m: int = -1) -> None:
    """Run masked EP SiLU-and-mul.

    mask_m contains one valid-token count per expert.
    expect_m controls the expected number of launched blocks per expert.
    """
    op.fuse_silu_and_mul_ep(input, output, mask_m, expect_m)

def fuse_silu_and_mul(input: torch.Tensor,
                      output: torch.Tensor,
                      mask_m: Optional[torch.Tensor] = None,
                      expect_m: int = -1) -> None:
    if mask_m is not None:
        fuse_silu_and_mul_ep(input, output, mask_m, expect_m)
        return
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

def fuse_silu_mul_clamp_quant(
    input: torch.Tensor,
    limit: float,
    output: Optional[torch.Tensor] = None,
    scales: Optional[torch.Tensor] = None,
    mask_m: Optional[torch.Tensor] = None,
    expect_m: int = -1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused GLM SwiGLU clamp, activation, and per-token INT8 quant.

    When ``mask_m`` is provided or ``expect_m != -1``, ``input`` uses the EP
    layout ``[experts, tokens, 2 * hidden]`` and ``mask_m[e]`` gives the valid
    token count for expert ``e``. ``expect_m`` controls the number of launched
    blocks per expert.
    """
    if mask_m is not None or expect_m != -1:
        if mask_m is None:
            raise ValueError("mask_m is required when expect_m != -1")
        return fuse_silu_mul_clamp_quant_ep(
            input,
            limit,
            mask_m,
            expect_m,
            output,
            scales,
        )

    if input.dim() != 2 or input.shape[-1] % 2 != 0:
        raise ValueError(
            "input must have shape [tokens, 2 * hidden]"
        )
    if not input.is_contiguous():
        raise ValueError("input must be contiguous")
    if limit <= 0:
        raise ValueError("limit must be positive")

    doubled_hidden = input.shape[-1]
    hidden = doubled_hidden // 2
    leading_shape = input.shape[:-1]
    if output is None:
        output = torch.empty(
            (*leading_shape, hidden), dtype=torch.int8, device=input.device
        )
    if scales is None:
        scales = torch.empty(
            (*leading_shape, 1), dtype=torch.float32, device=input.device
        )
    op.fuse_silu_mul_clamp_quant(
        input,
        output,
        scales,
        float(limit),
    )
    return output, scales


def fuse_silu_mul_clamp_quant_ep(
    input: torch.Tensor,
    limit: float,
    mask_m: torch.Tensor,
    expect_m: int = -1,
    output: Optional[torch.Tensor] = None,
    scales: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run masked EP GLM SwiGLU clamp and per-token INT8 quantization."""
    if input.dim() != 3 or input.shape[-1] % 2 != 0:
        raise ValueError(
            "input must have shape [experts, tokens, 2 * hidden]"
        )
    if not input.is_contiguous():
        raise ValueError("input must be contiguous")
    if limit <= 0:
        raise ValueError("limit must be positive")

    experts, tokens, doubled_hidden = input.shape
    hidden = doubled_hidden // 2
    if output is None:
        output = torch.empty(
            (experts, tokens, hidden),
            dtype=torch.int8,
            device=input.device,
        )
    if scales is None:
        scales = torch.empty(
            (experts, tokens, 1),
            dtype=torch.float32,
            device=input.device,
        )
    op.fuse_silu_mul_clamp_quant_ep(
        input,
        output,
        scales,
        float(limit),
        mask_m,
        expect_m,
    )
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
                        expert_ids: Optional[torch.Tensor] = None,
                        limit: Optional[float] = -1.0) -> tuple[torch.Tensor, torch.Tensor]:
    """Run SwiGLU followed by per-token FP8 quantization.

    A positive ``limit`` applies the DeepSeek V4 clamp in the input dtype
    before activation: ``gate <= limit`` and ``-limit <= up <= limit``.
    The default ``-1.0`` disables the clamp. ``None`` remains accepted for
    backward compatibility and also disables the clamp.
    """
    normalized_limit = _normalize_fp8_limit(limit)

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

    op.fuse_silu_mul_fp8_quant(
        input,
        output.view(torch.uint8),
        scales,
        fp8type,
        num_local_tokens_tensor,
        topk,
        expect_m,
        expert_ids,
        normalized_limit,
    )
    return output, scales

def fuse_silu_mul_fp8_quant_ep(input: torch.Tensor,
                        fp8type:int=0, #floate4m3 = 0, floate5m2 = 1 
                        tokens_per_expert: Optional[torch.Tensor] = None,
                        num_local_tokens_tensor: Optional[torch.Tensor] = None,
                        topk:int=1,
                        expect_m:int=-1,
                        limit: Optional[float] = None) -> tuple[torch.Tensor, torch.Tensor]:
    """Run EP SwiGLU followed by per-token FP8 quantization.

    When ``limit`` is provided, apply the DeepSeek V4 clamp in the input
    dtype before activation: ``gate <= limit`` and ``-limit <= up <= limit``.
    Leaving it as ``None`` preserves the original unclamped behavior.
    """
    if input.dim() != 3:
        raise ValueError(f"Input tensor must be 3-dimensional [E, T, H], but got shape {input.shape}")
    if limit is not None and limit <= 0:
        raise ValueError(f"limit must be positive when provided, but got {limit}")

    E, T, H = input.shape
    d = H // 2

    if fp8type==0:
        output = torch.empty(E, T, d, dtype=torch.float8_e4m3fn, device=input.device)
    else:
        output = torch.empty(E, T, d, dtype=torch.float8_e5m2, device=input.device)
    scales = torch.empty((E, T, 1),
                        device=input.device,
                        dtype=torch.float32)

    op.fuse_silu_mul_fp8_quant_ep(
        input,
        output.view(torch.uint8),
        scales,
        fp8type,
        tokens_per_expert,
        num_local_tokens_tensor,
        topk,
        expect_m,
        -1.0 if limit is None else limit,
    )
    return output, scales

#通用接口
def fuse_silu_mul_per_token_quant(input: torch.Tensor, 
                        dtype: torch.dtype = torch.int8,
                        num_local_tokens_tensor: Optional[torch.Tensor] = None,
                        topk: int = 1,
                        expect_m: int = -1,
                        output: Optional[torch.Tensor] = None,
                        scales: Optional[torch.Tensor] = None,
                        expert_ids: Optional[torch.Tensor] = None,
                        limit: Optional[float] = -1.0) -> Tuple[torch.Tensor, torch.Tensor]:

    leading, d2 = input.shape          
    d = d2 // 2

    actual_dtype = output.dtype if output is not None else dtype
    normalized_limit = _normalize_fp8_limit(limit)

    if output is None:
        output = torch.empty(leading, d, dtype=actual_dtype, device=input.device)
        
    if scales is None:
        scales = torch.empty((leading, 1),
                             device=input.device,
                             dtype=torch.float32)

    if actual_dtype == torch.int8:
        if normalized_limit != -1.0:
            raise ValueError("limit is only supported for FP8 output")
        op.fuse_silu_mul_quant(input, output, scales, num_local_tokens_tensor, topk, expect_m, expert_ids)
    elif actual_dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
        # fp8type: floate4m3 = 0, floate5m2 = 1 
        fp8type = 0 if actual_dtype == torch.float8_e4m3fn else 1
        op.fuse_silu_mul_fp8_quant(
            input,
            output.view(torch.uint8),
            scales,
            fp8type,
            num_local_tokens_tensor,
            topk,
            expect_m,
            expert_ids,
            normalized_limit,
        )
    else:
        raise ValueError(f"Unsupported dtype: {actual_dtype}. Supported dtypes are torch.int8, torch.float8_e4m3fn, torch.float8_e5m2")

    return output, scales

def moe_swiglu_dynamic_quant(
    scatter_tokens: torch.Tensor, 
    smooth: torch.Tensor, 
    experts_tokens_count: torch.Tensor,
    experts_tokens_start: torch.Tensor,
    beta: float = 1.0  
):
    
    leading, d2 = scatter_tokens.shape 
    d = d2 // 2
    
    output = torch.empty((leading, d), dtype=torch.int8, device=scatter_tokens.device)
    scales = torch.empty((leading, 1), dtype=torch.float32, device=scatter_tokens.device)
    
    op.moe_swiglu_dynamic_quant(
        scatter_tokens, 
        smooth, 
        experts_tokens_count, 
        experts_tokens_start, 
        output, 
        scales,
        beta  
    )
    
    return output, scales

def relu2(input: torch.Tensor) -> torch.Tensor:
    output = torch.empty_like(input)
    
    op.relu2_opt(output, input)
    return output

