import torch
#from lightop import op
from . import op
import numbers
import torch.nn as nn
from typing import List, Optional, Tuple

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
    update_input: bool = True
) -> tuple[torch.Tensor, torch.Tensor]:
    output = torch.empty(input.shape,device=input.device,dtype=quant_dtype)
    scales = torch.empty((input.numel() // input.shape[-1], 1),
                        device=input.device,
                        dtype=torch.float32)

    op.rms_norm_dynamic_per_token_quant(output, input, weight,
                                                    scales, epsilon,
                                                    residual, update_input)
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
) -> None:
    op.fused_rms_norm_rope_contiguous(positions, q_pe, k_pe, kv_c, kv_c_normed, weight, cos_sin_cache, slot_mapping, kv_cache, kv_cache_dtype_str, kv_cache_scale, is_neox_style, epsilon)
    return


def fuse_rmsnorm_rope_quant_gfx938(
    positions: torch.Tensor,
    q_c: torch.Tensor,
    q_pe: torch.Tensor,
    q_quant: torch.Tensor,
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
)->None:
    op.fuse_rmsnorm_rope_quant_gfx938(positions, q_c, q_pe, q_quant.view(torch.uint8), k_pe, kv_c, kv_c_normed, weight, cos_sin_cache, slot_mapping, kv_cache.view(torch.uint8), kv_cache_dtype_str, kv_cache_scale, is_neox_style, epsilon)
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