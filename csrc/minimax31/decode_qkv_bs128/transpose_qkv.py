"""Compatibility entry points for the BS128 replay harness."""
import torch

from lightop.attention import minimax31_project_qkv
from lightop.attention import minimax_qk_norm_rope_fp8_cache as norm_cache
from lightop.minimax31_decode_lt import _plan


def plan_for(device):
    with torch.cuda.device(device):
        stream = torch.cuda.current_stream(device).cuda_stream
        _, plan, index = _plan(None, device, stream, 128, 9856, 6144, 26690)
    return plan, index


def project(w, ws, q, scales):
    if q.ndim != 2 or q.size(0) != 128:
        raise ValueError("expected 128 QKV rows")
    return minimax31_project_qkv(q, scales, w, ws)


def projection(kernel, layer, q, scales):
    if not kernel.supports_prequantized_qkv(layer, q.device):
        raise ValueError("unexpected QKV weight layout")
    w,ws,_,_,_=kernel._get_layer_params(layer)
    return project(w.t(),ws,q,scales)
