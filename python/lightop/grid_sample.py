import torch
# from lightop import op
from . import op
import numbers
import torch.nn as nn
from typing import List, Optional, Tuple

def grid_sample_forward(x,weight,output=None,eps:float = 1e-5,training:bool=True):
    return op.rmsnorm_forward(x,weight,output,eps,training)

def grid_sample_backward(grad,x,rstd,gamma):
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

class GridSample(torch.nn.Module):

    def __init__(self):
        super(GridSample, self).__init__()

    def forward(self, x, grid, interpolation_mode, padding_mode, align_corners):
        return op.grid_sample_forward_autograd(x, grid, interpolation_mode, padding_mode, align_corners)