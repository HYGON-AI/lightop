"""Compatibility entry points for expanded VERIFY replay."""
import os

import torch

from lightop.attention import minimax31_project_qkv
from lightop.attention import minimax_qk_norm_rope_fp8_cache as norm_cache
from lightop.minimax31_decode_lt import _QKV_SOLUTIONS, _plan


TOKEN_COUNTS = tuple(_QKV_SOLUTIONS)


def plan_for(device, rows, solution):
    if rows not in TOKEN_COUNTS:
        raise ValueError(f"unsupported VERIFY QKV rows {rows}")
    with torch.cuda.device(device):
        stream = torch.cuda.current_stream(device).cuda_stream
        _, plan, index = _plan(None, device, stream, rows, 9856, 6144, solution)
    return plan, index


def project(w, ws, q, scales):
    selected = os.environ.get("M31_CROSS_QKV_SOLUTION")
    return minimax31_project_qkv(
        q, scales, w, ws, solution=int(selected) if selected else None,
    )


def projection(kernel, layer, q, scales):
    if not kernel.supports_prequantized_qkv(layer, q.device):
        raise ValueError("Unexpected Verify8 QKV weight layout")
    w, ws, _, _, _ = kernel._get_layer_params(layer)
    return project(w.t(), ws, q, scales)
