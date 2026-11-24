"""Reuse the HIPBLASLt INT32 projection in its native output layout."""
import atexit
from functools import lru_cache

import torch
from m31_qkv_followup import qk_cache

from deploy_lt_row_major import library


@lru_cache(maxsize=None)
def plan_for(device):
    lib = library()
    scales_a = torch.ones(128, device=device)
    scales_b = torch.ones(9856, device=device)
    plan = lib.create_plan(128, 9856, 6144, scales_a.data_ptr(), scales_b.data_ptr(), 0)
    if not plan:
        raise RuntimeError("cannot initialize transposed QKV hipBLASLt plan")
    for index in range(lib.plan_count(plan)):
        if lib.solution_id(plan, index) == 26690 and lib.solution_workspace(plan, index) == 0:
            atexit.register(lib.destroy_plan, plan)
            return plan,index
    lib.destroy_plan(plan)
    raise RuntimeError("transposed solution 26690 is unavailable")


def project(w, ws, q, scales):
    if (w.shape != (9856,6144) or q.shape != (128,6144) or
            ws.shape != (9856,1) or scales.shape != (128,1) or
            w.dtype != torch.int8 or q.dtype != torch.int8 or
            ws.dtype != torch.float32 or scales.dtype != torch.float32 or
            not all(t.is_contiguous() and t.device == q.device for t in (w,ws,q,scales))):
        raise ValueError("unsupported transposed QKV geometry")
    stream = torch.cuda.current_stream(q.device).cuda_stream
    plan,index = plan_for(q.device)
    out = torch.empty((9856,128),device=q.device,dtype=torch.bfloat16)
    status = library().run_plan(plan,index,q.data_ptr(),w.data_ptr(),scales.data_ptr(),
                                ws.data_ptr(),out.data_ptr(),None,stream)
    if status:
        raise RuntimeError(f"transposed QKV hipBLASLt status {status}")
    return out.t()


def projection(kernel, layer, q, scales):
    if not kernel.supports_prequantized_qkv(layer, q.device):
        raise ValueError("unexpected QKV weight layout")
    w,ws,_,_,_=kernel._get_layer_params(layer)
    return project(w.t(),ws,q,scales)


def norm_cache(qkv, weights, positions, cos_sin, eps, locations, mapping, *cache):
    return qk_cache(qkv, list(weights), positions, cos_sin, eps, locations,mapping,*cache,16)
