"""hipBLASLt plans for the measured MiniMax VERIFY INT8 projections."""
import atexit
import ctypes
import os
from functools import lru_cache
from threading import Lock

import torch


_external_submission = Lock()
_QKV_SOLUTIONS = {64: 26685, 80: 26690, 96: 26690, 128: 26690, 144: 26741}


@lru_cache(maxsize=None)
def _library(path):
    if path is None:
        from . import op

        path, prefix = op.__file__, "lightop_minimax31_"
    else:
        prefix = ""
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
        fn = getattr(lib, prefix + name)
        fn.argtypes, fn.restype = args, result
        setattr(lib, name, fn)
    return lib


@lru_cache(maxsize=None)
def _plan(path, device, stream, left_rows, right_rows, k, solution):
    lib = _library(path)
    left_scale = torch.ones(left_rows, device=device, dtype=torch.float32)
    right_scale = torch.ones(right_rows, device=device, dtype=torch.float32)
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



def _path(library_path):
    return None if library_path is None else os.fspath(library_path)


def warmup(library_path, device, left_rows, right_rows, k, solution):
    """Prepare an Lt plan on the caller's stream before pinning host KV."""
    with torch.cuda.device(device):
        stream = torch.cuda.current_stream(device).cuda_stream
        _plan(_path(library_path), device, stream, left_rows, right_rows, k, solution)

def _validate_operands(left, left_scale, right, right_scale):
    if (left.ndim != 2 or right.ndim != 2 or left.size(1) != right.size(1)
            or not all(0 < n <= 2**31 - 1 for n in (*left.shape, right.size(0)))):
        raise ValueError("Lt projection requires nonempty [M,K] and [N,K] with the same K")
    for operand in (left, right):
        if (not operand.is_cuda or operand.device != left.device
                or operand.dtype != torch.int8 or not operand.is_contiguous()):
            raise ValueError("Lt projection requires contiguous INT8 inputs on one HIP device")
    for scale, rows in ((left_scale, left.size(0)), (right_scale, right.size(0))):
        if (scale.device != left.device or scale.dtype != torch.float32
                or not scale.is_contiguous() or scale.shape not in ((rows,), (rows, 1))):
            raise ValueError("Lt projection requires one contiguous FP32 scale per input row")


def project(library_path, left, left_scale, right, right_scale, solution):
    """Return contiguous BF16 [N,M]; use None for the installed LightOp library."""
    _validate_operands(left, left_scale, right, right_scale)
    path, device = _path(library_path), left.device
    left_rows, k = left.shape
    right_rows = right.size(0)
    with torch.cuda.device(device):
        stream = torch.cuda.current_stream(device).cuda_stream
        lib, plan, index = _plan(path, device, stream, left_rows, right_rows, k, solution)
        output = torch.empty((right_rows, left_rows), device=device, dtype=torch.bfloat16)
        args = (
            plan, index, left.data_ptr(), right.data_ptr(), left_scale.data_ptr(),
            right_scale.data_ptr(), output.data_ptr(), None, stream,
        )
        if path is None:
            status = lib.run_plan(*args)
        else:
            # An explicitly supplied pre-existing library may predate the native lock.
            with _external_submission:
                status = lib.run_plan(*args)
    if status:
        raise RuntimeError(f"Verify8 Lt failed for {left_rows}/{right_rows}/{k} with status {status}")
    return output


def minimax31_project_qkv(left, left_scale, right, right_scale, *, solution=None):
    """Project INT8 MiniMax QKV to BF16 [rows,9856] in dense transposed layout."""
    if (left.ndim != 2 or left.size(0) not in _QKV_SOLUTIONS or left.size(1) != 6144
            or right.shape != (9856, 6144)):
        raise ValueError("unsupported transposed MiniMax QKV geometry")
    if solution is None:
        solution = _QKV_SOLUTIONS[left.size(0)]
    return project(None, left, left_scale, right, right_scale, solution).t()
