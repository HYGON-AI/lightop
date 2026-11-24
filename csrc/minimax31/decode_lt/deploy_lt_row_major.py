"""Select zero-workspace hipBLASLt solutions for VERIFY O and shared gate/up."""
from functools import lru_cache
from pathlib import Path
import atexit
import ctypes

import torch


_handle = ctypes.c_void_p
_int = ctypes.c_int
_size = ctypes.c_size_t
SOLUTIONS = {8192: 26708, 6144: 26733}


@lru_cache(maxsize=1)
def library():
    lib = ctypes.CDLL(str(Path(__file__).with_name("lt_plans.so")))
    for name, parameters, result in (
        ("create_plan", [_int, _int, _int, _handle, _handle, _size], _handle),
        ("destroy_plan", [_handle], None),
        ("plan_count", [_handle], _int),
        ("solution_id", [_handle, _int], _int),
        ("solution_workspace", [_handle, _int], _size),
        ("run_plan", [_handle, _int] + [_handle] * 7, _int),
    ):
        fn = getattr(lib, name)
        fn.argtypes = parameters
        fn.restype = result
    return lib


@lru_cache(maxsize=None)
def find_plan(device, stream_id, k):
    # The scale addresses are only needed at creation for the support query.
    # run_plan sets the actual scale pointers before each launch.
    lib = library()
    m, n = 128, 6144
    scale_m = torch.ones((m,), device=device)
    scale_n = torch.ones((n,), device=device)
    plan = lib.create_plan(n, m, k, scale_n.data_ptr(), scale_m.data_ptr(), 0)
    if not plan:
        raise RuntimeError(f"cannot construct hipBLASLt plan for K={k}")
    for index in range(lib.plan_count(plan)):
        if lib.solution_id(plan, index) == SOLUTIONS[k] and lib.solution_workspace(plan, index) == 0:
            atexit.register(lib.destroy_plan, plan)
            return plan, index
    lib.destroy_plan(plan)
    raise RuntimeError(f"gfx938 hipBLASLt solution {SOLUTIONS[k]} is unavailable")


def run(w, q, ws, qs):
    k = q.shape[1]
    stream = torch.cuda.current_stream(q.device).cuda_stream
    plan, index = find_plan(q.device, stream, k)
    output = torch.empty((128,6144), device=q.device, dtype=torch.bfloat16)
    status = library().run_plan(plan, index, w.data_ptr(), q.data_ptr(), ws.data_ptr(),
                                qs.data_ptr(), output.data_ptr(), None, stream)
    if status != 0:
        raise RuntimeError(f"hipBLASLt row-major K={k} returned {status}")
    return output


def install(selected, device, shared_stream=None):
    from hcu_vllm_das.parallel.minimax31_verify_int8 import VerifyTransposeInt8Kernel

    streams = {torch.cuda.current_stream(device).cuda_stream}
    if shared_stream is not None:
        streams.add(shared_stream.cuda_stream)
    for stream in streams:
        for shape in selected:
            find_plan(device, stream, shape)

    original = VerifyTransposeInt8Kernel.apply_prequantized

    def apply_prequantized(self, layer, q, scales, dtype, *, row_major=False):
        w, ws, _, _, _ = self._get_layer_params(layer)
        valid = (q.ndim == 2 and q.shape[1] in selected and q.shape[0] == 128
                 and q.dtype == torch.int8 and q.is_contiguous() and dtype == torch.bfloat16
                 and w.shape == (q.shape[1], 6144) and w.stride() == (1, q.shape[1])
                 and w.dtype == torch.int8 and ws.shape == (6144, 1)
                 and scales.shape == (128, 1) and ws.dtype == scales.dtype == torch.float32
                 and all(t.device == q.device for t in (w, ws, scales))
                 and ws.is_contiguous() and scales.is_contiguous())
        if row_major or not valid:
            return original(self, layer, q, scales, dtype, row_major=row_major)
        return run(w.t(), q, ws, scales)

    VerifyTransposeInt8Kernel.apply_prequantized = apply_prequantized
    print("M31_LT_ROW_MAJOR", {k: SOLUTIONS[k] for k in selected}, flush=True)
