# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

import torch
# from lightop import op
from . import op
import numbers
import torch.nn as nn
from typing import List, Optional, Tuple

def rms_norm(
        x: torch.Tensor, weight: torch.Tensor,
        variance_epsilon: float,
        out: Optional[torch.Tensor] = None,
        limit: float = -1.0) -> torch.Tensor:
    if out is None:
        out = x
    op.rms_norm_opt(
        out,
        x,
        weight,
        variance_epsilon,
        limit,
    )
    return out


def fused_add_rms_norm(
        x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor,
        variance_epsilon: float,
        limit: float = -1.0) -> tuple[torch.Tensor, torch.Tensor]:
    """Normalize an FP32 residual sum and update x/residual in place.

    For BF16/FP16 inputs, round only the final output and stored residual.
    The optional clamp applies to the normalized output, not the residual.
    """
    op.fused_add_rms_norm_opt(
        x,
        residual,
        weight,
        variance_epsilon,
        limit,
    )
    return x, residual

def gemma_fused_add_rmsnorm(
        x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor,
        variance_epsilon: float) -> tuple[torch.Tensor, torch.Tensor]:
    """FP32-sum in-place RMSNorm with the Gemma (1 + weight) convention."""
    op.gemma_fused_add_rmsnorm(
        x,
        residual,
        weight,
        variance_epsilon,
    )
    return x, residual
def gemma_rmsnorm(
        x: torch.Tensor, weight: torch.Tensor,
        variance_epsilon: float, out: torch.Tensor = None) -> torch.Tensor:
    if out is None:
        out = x  
    op.gemma_rmsnorm(
        out,
        x,
        weight,
        variance_epsilon,
    )
    return out
def rmsnorm_forward(x,weight,output=None,eps:float = 1e-5,training:bool=True):
    return op.rmsnorm_forward(x,weight,output,eps,training)

def rmsnorm_backward(grad,x,rstd,gamma):
    return op.rmsnorm_backward(grad,x,rstd,gamma)

def rmsnorm_forward_torch(hidden_states,weight,eps):
    input_dtype = hidden_states.dtype
    variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    hidden_states = hidden_states * torch.rsqrt(variance + eps)
    if weight is not None:
        if weight.dtype in [torch.float16, torch.bfloat16]:
            hidden_states = hidden_states * weight.to(torch.float32)
            hidden_states = hidden_states.to(weight.dtype)
        else:
            hidden_states = hidden_states * weight
    else:
        hidden_states = hidden_states.to(input_dtype)
    return hidden_states

class RMSNorm(torch.nn.Module):

    def __init__(self, normalized_shape:int,eps: float = 1e-5,device=None, dtype=None ):
        super(RMSNorm, self).__init__()
        self.eps = eps
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.weight = torch.nn.Parameter(torch.empty(normalized_shape, **factory_kwargs))
        torch.nn.init.ones_(self.weight)
        self.rmsnorm_compile=torch.compile(rmsnorm_forward_torch)

    def forward(self, x ):
        if x.dtype == torch.bfloat16 and self.weight.numel()==128 and self.training==False:
            return self.rmsnorm_compile(x, self.weight,self.eps)
        return op.rmsnorm_forward_autograd(x, self.weight,self.eps, self.training)

    def extra_repr(self):
        return f'eps={round(self.eps,5):0.5f}'

class RMSNormTorch(torch.nn.Module):
    def __init__(self, dim, eps: float=1e-5, elementwise_affine: bool = True):
        super().__init__()

        self.eps = eps

        if isinstance(dim, numbers.Integral):
            dim = (dim,)

        self.dim = torch.Size(dim)

        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(dim))
        else:
            self.weight = None

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)
        # print(torch.rsqrt(variance + self.eps))
        if self.weight is not None:
            if self.weight.dtype in [torch.float16, torch.bfloat16]:
                hidden_states = hidden_states * self.weight.to(torch.float32)
                hidden_states = hidden_states.to(self.weight.dtype)
            else:
                hidden_states = hidden_states * self.weight
        else:
            hidden_states = hidden_states.to(input_dtype)

        return hidden_states

class RMSNormAdd(torch.nn.Module):
    rate: float

    def __init__(self, normalized_shape:int,eps: float = 1e-5,device=None, dtype=None ,inplace = False):
        super(RMSNormAdd, self).__init__()
        self.eps = eps
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.weight = torch.nn.Parameter(torch.empty(normalized_shape, **factory_kwargs))
        torch.nn.init.ones_(self.weight)
        self.inplace = inplace

    def forward(self, X, residual):
        return op.rn_add_forward_autograd(X, residual, self.weight,self.eps,self.training,self.inplace)

    def extra_repr(self):
        return f'rate={round(self.rate,3):0.3f},eps={round(self.eps,5):0.5f}'


class RMSNormDropout(torch.nn.Module):
    rate: float

    def __init__(self, normalized_shape:int,droprate: float = 0.5,eps: float = 1e-5,device=None, dtype=None ):
        super(RMSNormDropout, self).__init__()
        self.rate = droprate
        self.eps = eps
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.weight = torch.nn.Parameter(torch.empty(normalized_shape, **factory_kwargs))
        torch.nn.init.ones_(self.weight)

    def forward(self, inputA):
        return op.rn_dropout_forward_autograd(inputA, self.weight,self.eps,self.rate, self.training)

    def extra_repr(self):
        return f'rate={round(self.rate,3):0.3f},eps={round(self.eps,5):0.5f}'


class RMSNormAddDropout(torch.nn.Module):
    rate: float

    def __init__(self, normalized_shape:int,droprate: float = 0.5,eps: float = 1e-5,device=None, dtype=None ):
        super(RMSNormAddDropout, self).__init__()
        self.rate = droprate
        self.eps = eps
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.weight = torch.nn.Parameter(torch.empty(normalized_shape, **factory_kwargs))
        torch.nn.init.ones_(self.weight)

    def forward(self, X, residual):
        return op.rn_add_dropout_forward_autograd(X, residual, self.weight,self.eps,self.rate, self.training)

    def extra_repr(self):
        return f'rate={round(self.rate,3):0.3f},eps={round(self.eps,5):0.5f}'


class RMSNormAddDropoutBias(torch.nn.Module):
    rate: float

    def __init__(self, normalized_shape:int,droprate: float = 0.5,eps: float = 1e-5,device=None, dtype=None ):
        super(RMSNormAddDropoutBias, self).__init__()
        self.rate = droprate
        self.eps = eps
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.weight = torch.nn.Parameter(torch.empty(normalized_shape, **factory_kwargs))
        torch.nn.init.ones_(self.weight)

    def forward(self, X, residual, bias):
        return op.rn_add_dropout_bias_forward_autograd(X, residual, bias, self.weight, self.eps, self.rate, self.training)

    def extra_repr(self):
        return f'rate={round(self.rate,3):0.3f},eps={round(self.eps,5):0.5f}'

def rms_norm_dynamic_per_token_quant(
    input: torch.Tensor,
    weight: torch.Tensor,
    epsilon: float,
    quant_dtype: torch.dtype,
    residual: Optional[torch.Tensor] = None,
    update_input: bool = True,
    limit: float = -1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    output = torch.empty(input.shape,device=input.device,dtype=quant_dtype)
    scales = torch.empty((input.numel() // input.shape[-1], 1),
                        device=input.device,
                        dtype=torch.float32)

    op.rms_norm_dynamic_per_token_quant(output, input, weight,
                                                    scales, epsilon,
                                                    residual, update_input, limit)
    return output, scales


SGLANG_NATIVE_RMS_QUANT_VERSION = 1


def rms_norm_dynamic_per_token_quant_sglang(
    input: torch.Tensor,
    weight: torch.Tensor,
    epsilon: float,
    quant_dtype: torch.dtype = torch.int8,
    residual: Optional[torch.Tensor] = None,
    update_input: bool = True,
    limit: float = -1.0,
    smooth_scale: Optional[torch.Tensor] = None,
    norm_output: Optional[torch.Tensor] = None,
    clamp_norm_output: bool = True,
    residual_output: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """RMSNorm + per-token INT8 quant with SGLang-compatible numerics.

    This is intentionally a separate API from the historical LightOp kernel.
    In particular, quantization observes the BF16/FP16-rounded RMSNorm output,
    and uses ``max(absmax, 1e-10) / 127`` as the returned scale.  When
    ``smooth_scale`` is provided the fused order is RMSNorm, optional clamp,
    smooth multiplication, activation-dtype rounding, and INT8 quantization.
    The FP32 residual sum feeds RMSNorm without an intermediate BF16 cast.
    Pass a distinct ``residual_output`` to preserve the input residual, as
    Torch native does. Omission retains the historical in-place residual ABI.
    """
    if quant_dtype != torch.int8:
        raise ValueError(f"Only torch.int8 is supported, got {quant_dtype}")

    output = torch.empty_like(input, dtype=torch.int8)
    scales = torch.empty(
        (input.numel() // input.shape[-1], 1),
        device=input.device,
        dtype=torch.float32,
    )
    op.rms_norm_dynamic_per_token_quant_sglang(
        output,
        input,
        weight,
        scales,
        epsilon,
        residual,
        update_input,
        limit,
        smooth_scale,
        norm_output,
        clamp_norm_output,
        residual_output,
    )
    return output, scales


def mla_qkv_a_rms_norm_dynamic_per_token_quant_sglang(
    packed_input: torch.Tensor,
    q_weight: torch.Tensor,
    kv_weight: torch.Tensor,
    q_epsilon: float,
    kv_epsilon: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Normalize packed MLA q/kv segments and dynamically quantize q.

    packed_input has layout [q_lora, kv_lora, q_rope]. The q and kv
    segments are normalized in place with their own weights and epsilons. The
    q_rope tail is untouched. Quantization observes the activation-dtype
    rounded q RMSNorm output and returns SGLang-compatible INT8 values/scales.
    """
    if packed_input.dim() != 2:
        raise ValueError("packed_input must be a 2D tensor")

    num_tokens = packed_input.shape[0]
    q_output = torch.empty(
        (num_tokens, q_weight.numel()),
        device=packed_input.device,
        dtype=torch.int8,
    )
    q_scales = torch.empty(
        (num_tokens, 1),
        device=packed_input.device,
        dtype=torch.float32,
    )
    op.mla_qkv_a_rms_norm_dynamic_per_token_quant_sglang(
        q_output,
        packed_input,
        q_weight,
        kv_weight,
        q_scales,
        q_epsilon,
        kv_epsilon,
    )
    return q_output, q_scales


def rms_norm_per_token_fp8_quant(
    input: torch.Tensor,
    weight: torch.Tensor,
    epsilon: float = 1e-5,
    fp8type: int = 0,
    residual: Optional[torch.Tensor] = None,
    update_input: bool = False
) -> tuple[torch.Tensor, torch.Tensor]:
    if fp8type == 0:
        output = torch.empty(input.shape,device=input.device,dtype=torch.float8_e4m3fn)
    else:
        output = torch.empty(input.shape,device=input.device,dtype=torch.float8_e5m2)
    scales = torch.empty((input.numel() // input.shape[-1], 1),
                        device=input.device,
                        dtype=torch.float32)

    op.rms_norm_fp8_quant(output.view(torch.uint8), input, weight, scales, epsilon,
                          residual, update_input, fp8type)
    return output, scales
def gemma_rms_norm_per_token_fp8_quant(
    input: torch.Tensor,
    weight: torch.Tensor,
    epsilon: float = 1e-5,
    fp8type: int = 0,
    residual: Optional[torch.Tensor] = None,
    update_input: bool = False
) -> tuple[torch.Tensor, torch.Tensor]:
    if fp8type == 0:
        output = torch.empty(input.shape, device=input.device, dtype=torch.float8_e4m3fn)
    else:
        output = torch.empty(input.shape, device=input.device, dtype=torch.float8_e5m2)
        
    scales = torch.empty((input.numel() // input.shape[-1], 1),
                        device=input.device,
                        dtype=torch.float32)

   
    op.gemma_rms_norm_fp8_quant(output.view(torch.uint8), input, weight, scales, epsilon,
                                residual, update_input, fp8type)
    return output, scales
def rms_norm_per_token_quant(
    input: torch.Tensor,
    weight: torch.Tensor,
    epsilon: float = 1e-5,
    dtype: torch.dtype = torch.int8,
    residual: Optional[torch.Tensor] = None,
    update_input: bool = False,
    limit: float = -1.0
) -> Tuple[torch.Tensor, torch.Tensor]:
  
    output = torch.empty(input.shape, device=input.device, dtype=dtype)
    scales = torch.empty((input.numel() // input.shape[-1], 1),
                         device=input.device,
                         dtype=torch.float32)

    if dtype == torch.int8:
        op.rms_norm_dynamic_per_token_quant(output, input, weight,
                                            scales, epsilon,
                                            residual, update_input, limit)
    elif dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
        if limit != -1.0:
            raise ValueError("limit is only supported when dtype is torch.int8")
        fp8type = 0 if dtype == torch.float8_e4m3fn else 1
        op.rms_norm_fp8_quant(output.view(torch.uint8), input, weight, 
                              scales, epsilon,
                              residual, update_input, fp8type)
    else:
        raise ValueError(f"Unsupported dtype: {dtype}. Supported dtypes are torch.int8, torch.float8_e4m3fn, torch.float8_e5m2")

    return output, scales

def fused_rms_norm_contiguous(
    out: torch.Tensor,
    input: torch.Tensor,
    weight: torch.Tensor,
    epsilon: float
) -> torch.Tensor:
    op.fused_rms_norm_contiguous(out, input, weight, epsilon)
    return out


# fused_rms_norm_rope_contiguous(positions,q_pe,k_pe,rot_dim,cos_sin_cache,is_neox_style,offsets)
def fused_rms_norm_rope_contiguous(
    positions: torch.Tensor,
    q_pe: torch.Tensor,
    k_pe: torch.Tensor,
    kv_c: torch.Tensor,
    kv_c_normed: torch.Tensor,
    weight: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    kv_cache: torch.Tensor,
    kv_cache_dtype_str: str,
    kv_cache_scale: float,
    is_neox_style: bool = False,
    epsilon: float = 1e-5,
    limit: float = -1.0,
    flashmla_fp8_layout: bool = False,
    kv_quant_epsilon: float = 1e-6,
    kv_smooth_scale: Optional[torch.Tensor] = None,
) -> None:
    op.fused_rms_norm_rope_contiguous(
        positions,
        q_pe,
        k_pe,
        kv_c,
        kv_c_normed,
        weight,
        cos_sin_cache,
        slot_mapping,
        kv_cache,
        kv_cache_dtype_str,
        kv_cache_scale,
        is_neox_style,
        epsilon,
        limit,
        flashmla_fp8_layout,
        kv_quant_epsilon,
        kv_smooth_scale,
    )
    return


def fused_rms_norm_rope_contiguous_q_cat(
    positions: torch.Tensor,
    q_pe: torch.Tensor,
    k_pe: torch.Tensor,
    kv_c: torch.Tensor,
    kv_c_normed: torch.Tensor,
    weight: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    kv_cache: torch.Tensor,
    q_nope: torch.Tensor,
    q_cat: torch.Tensor,
    kv_cache_dtype_str: str,
    kv_cache_scale: float,
    is_neox_style: bool = False,
    epsilon: float = 1e-5,
    limit: float = -1.0,
    flashmla_fp8_layout: bool = False,
    kv_quant_epsilon: float = 1e-6,
    kv_smooth_scale: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Run the existing RMS/RoPE/cache-store kernel and materialize Q[576].

    The K/RMS-specialized blocks are unchanged.  Q-specialized blocks rotate
    ``q_pe`` in place and write ``cat([q_nope, q_pe], -1)`` to the caller-owned
    contiguous ``q_cat`` buffer.
    """
    op.fused_rms_norm_rope_contiguous_q_cat(
        positions,
        q_pe,
        k_pe,
        kv_c,
        kv_c_normed,
        weight,
        cos_sin_cache,
        slot_mapping,
        kv_cache,
        q_nope,
        q_cat,
        kv_cache_dtype_str,
        kv_cache_scale,
        is_neox_style,
        epsilon,
        limit,
        flashmla_fp8_layout,
        kv_quant_epsilon,
        kv_smooth_scale,
    )
    return q_cat


def fuse_rmsnorm_rope_quant_qkv(
    positions: torch.Tensor,
    q_c: torch.Tensor,
    q_pe: torch.Tensor,
    q_quant: torch.Tensor,
    q_scale: torch.Tensor,
    k_pe: torch.Tensor,
    kv_c: torch.Tensor,
    kv_c_normed: torch.Tensor,
    weight: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    kv_cache: torch.Tensor,
    kv_cache_dtype_str: str,
    kv_cache_scale: float,
    is_neox_style: bool = False,
    epsilon: float = 1e-5,
    limit: float = -1.0,
)->None:
    op.fuse_rmsnorm_rope_quant_qkv(positions, q_c, q_pe, q_quant.view(torch.uint8), q_scale, k_pe, kv_c, kv_c_normed, weight, cos_sin_cache, slot_mapping, kv_cache.view(torch.uint8), kv_cache_dtype_str, kv_cache_scale, is_neox_style, epsilon, limit)
    return

def fuse_fill_rms_x2_concat(
    hidden_states_fuse,
    positions, 
    inputs_embeds, 
    previous_hidden_states, 
    weight_inputs_embeds, 
    weight_previous_hidden_states, 
    epsilon)->None:
    assert(inputs_embeds.shape == previous_hidden_states.shape, "inputs_embeds and previous_hidden_states must have the same shape")
    assert(hidden_states_fuse.shape[-1] == inputs_embeds.shape[-1] * 2, "hidden_states_fuse and inputs_embeds must have the same shape")
    assert(inputs_embeds.shape[-1]<=16384, "inputs_embeds size must be less than 16384")
    op.fuse_fill_rms_x2_concat(hidden_states_fuse,positions, inputs_embeds, previous_hidden_states, weight_inputs_embeds, weight_previous_hidden_states, epsilon)
    return
