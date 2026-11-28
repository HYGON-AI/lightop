"""Select zero-workspace hipBLASLt solutions for VERIFY O and shared gate/up."""
import torch

from lightop.minimax31_decode_lt import _library, _plan, project


SOLUTIONS = {8192: 26708, 6144: 26733}


def library():
    return _library(None)


def find_plan(device, stream_id, k):
    with torch.cuda.device(device):
        _, plan, index = _plan(None, device, stream_id, 6144, 128, k, SOLUTIONS[k])
    return plan, index


def run(w, q, ws, qs):
    return project(None, w, ws, q, qs, SOLUTIONS[q.shape[1]])


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
