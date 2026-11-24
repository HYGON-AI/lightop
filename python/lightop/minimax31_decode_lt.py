"""hipBLASLt plans for the measured MiniMax VERIFY INT8 projections."""
import atexit
import ctypes
import os
from functools import lru_cache

import torch


@lru_cache(maxsize=None)
def _library(path):
    lib = ctypes.CDLL(path)
    handle, integer, size = ctypes.c_void_p, ctypes.c_int, ctypes.c_size_t
    for name, args, result in (
        ("create_plan", [integer, integer, integer, handle, handle, size], handle),
        ("destroy_plan", [handle], None),
        ("plan_count", [handle], integer),
        ("solution_id", [handle, integer], integer),
        ("solution_workspace", [handle, integer], size),
        ("run_plan", [handle, integer] + [handle] * 7, integer),
    ):
        fn = getattr(lib, name)
        fn.argtypes, fn.restype = args, result
    return lib


@lru_cache(maxsize=None)
def _plan(path, device, stream, left_rows, right_rows, k, solution):
    lib = _library(path)
    left_scale = torch.ones(left_rows, device=device)
    right_scale = torch.ones(right_rows, device=device)
    plan = lib.create_plan(
        left_rows, right_rows, k, left_scale.data_ptr(), right_scale.data_ptr(), 0,
    )
    if not plan:
        raise RuntimeError(f"Cannot construct Verify8 Lt plan for {left_rows}/{right_rows}/{k}")
    for index in range(lib.plan_count(plan)):
        if lib.solution_id(plan, index) == solution and lib.solution_workspace(plan, index) == 0:
            atexit.register(lib.destroy_plan, plan)
            return lib, plan, index
    lib.destroy_plan(plan)
    raise RuntimeError(f"Verify8 Lt solution {solution} is unavailable for {left_rows}/{right_rows}/{k}")



def warmup(library_path, device, left_rows, right_rows, k, solution):
    """Prepare an Lt plan on the caller's stream before pinning host KV."""
    with torch.cuda.device(device):
        stream = torch.cuda.current_stream(device).cuda_stream
        _plan(os.fspath(library_path), device, stream, left_rows, right_rows, k, solution)

def project(library_path, left, left_scale, right, right_scale, solution):
    """Execute the selected zero-workspace Lt solution on the caller's stream."""
    path = os.fspath(library_path)
    device = left.device
    left_rows, k = left.shape
    right_rows = right.size(0)
    with torch.cuda.device(device):
        stream = torch.cuda.current_stream(device).cuda_stream
        lib, plan, index = _plan(path, device, stream, left_rows, right_rows, k, solution)
        output = torch.empty((right_rows, left_rows), device=device, dtype=torch.bfloat16)
        status = lib.run_plan(
            plan, index, left.data_ptr(), right.data_ptr(), left_scale.data_ptr(),
            right_scale.data_ptr(), output.data_ptr(), None, stream,
        )
    if status:
        raise RuntimeError(f"Verify8 Lt failed for {left_rows}/{right_rows}/{k} with status {status}")
    return output
