"""Execute Verify8 QKV with a measured solution for its actual token count."""
import atexit
from functools import lru_cache
import os

import torch
from m31_qkv_followup import qk_cache

from deploy_lt_row_major import library

TOKEN_COUNTS = (64, 80, 96, 128, 144)


@lru_cache(maxsize=None)
def plan_for(device, rows, solution):
    if rows not in TOKEN_COUNTS:
        raise ValueError(f"Unsupported Verify8 QKV rows {rows}")
    lib = library()
    scales_a = torch.ones(rows, device=device)
    scales_b = torch.ones(9856, device=device)
    plan = lib.create_plan(rows, 9856, 6144, scales_a.data_ptr(), scales_b.data_ptr(), 0)
    if not plan:
        raise RuntimeError(f"Cannot initialize hipBLASLt QKV for rows {rows}")
    for index in range(lib.plan_count(plan)):
        if lib.solution_id(plan, index) == solution and lib.solution_workspace(plan, index) == 0:
            atexit.register(lib.destroy_plan, plan)
            return plan, index
    lib.destroy_plan(plan)
    raise RuntimeError(f"hipBLASLt solution {solution} is unavailable for {rows} rows")


def project(w, ws, q, scales):
    rows = q.size(0)
    if (rows not in TOKEN_COUNTS or w.shape != (9856, 6144) or q.shape != (rows, 6144)
            or ws.shape != (9856, 1) or scales.shape != (rows, 1)
            or w.dtype != torch.int8 or q.dtype != torch.int8
            or ws.dtype != torch.float32 or scales.dtype != torch.float32
            or not all(t.is_contiguous() and t.device == q.device for t in (w, ws, q, scales))):
        raise ValueError("Unsupported transposed Verify8 QKV tensors")
    solution = int(os.environ["M31_CROSS_QKV_SOLUTION"])
    stream = torch.cuda.current_stream(q.device).cuda_stream
    plan, index = plan_for(q.device, rows, solution)
    out = torch.empty((9856, rows), device=q.device, dtype=torch.bfloat16)
    status = library().run_plan(plan, index, q.data_ptr(), w.data_ptr(), scales.data_ptr(),
                                ws.data_ptr(), out.data_ptr(), None, stream)
    if status:
        raise RuntimeError(f"hipBLASLt transposed QKV failed with status {status}")
    return out.t()


def projection(kernel, layer, q, scales):
    if not kernel.supports_prequantized_qkv(layer, q.device):
        raise ValueError("Unexpected Verify8 QKV weight layout")
    w, ws, _, _, _ = kernel._get_layer_params(layer)
    return project(w.t(), ws, q, scales)


def norm_cache(qkv, weights, positions, cos_sin, eps, locations, mapping, *cache):
    return qk_cache(qkv, list(weights), positions, cos_sin, eps, locations, mapping, *cache, 16)
