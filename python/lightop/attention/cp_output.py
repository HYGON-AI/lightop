# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""In-place rank-major to token-major CP output permutation.

The caller must exclusively own the gathered buffer. Only shape metadata is
examined on the CPU; tensor values never leave the device. Cycles partition the
rows, so GPU blocks need no inter-block synchronization or full-size workspace.
"""

from collections import OrderedDict
from threading import Lock

import torch

try:
    from .. import _cp_output
except ImportError:
    from .. import op as _cp_output

_PLANS = OrderedDict()
_PLAN_LOCK = Lock()
_MAX_PLANS = 32
# Captured graphs keep raw device pointers. Pin their small immutable plans for
# the process lifetime so later LRU eviction cannot invalidate graph replay.
_CAPTURED_PLANS = {}


def _cycle_leaders(rows: int, cp_size: int) -> list[int]:
    if cp_size < 1 or rows < 0 or rows % cp_size:
        raise ValueError("rows must be nonnegative and divisible by positive CP")
    local_rows = rows // cp_size
    visited = bytearray(rows)
    leaders = []
    for start in range(rows):
        if visited[start]:
            continue
        current = start
        length = 0
        while not visited[current]:
            visited[current] = 1
            length += 1
            current = (current % cp_size) * local_rows + current // cp_size
        if length > 1:
            leaders.append(start)
    return leaders


def interleave_cp_output_(gathered: torch.Tensor, cp_size: int) -> torch.Tensor:
    """Return the same buffer in token order, preserving all element bits.

    ``gathered`` is contiguous with shape ``[CP * local_rows, ...]``. All trailing
    dimensions and element storage widths are supported; CP need not be a power
    of two. Warm the shape once before CUDA Graph capture. The operation is for
    inference and cannot be applied to a buffer still read by communication.
    """
    if not gathered.is_cuda or not gathered.is_contiguous() or gathered.ndim < 1:
        raise ValueError("a contiguous GPU tensor with a row dimension is required")
    if gathered.requires_grad or cp_size < 1 or gathered.shape[0] % cp_size:
        raise ValueError("inference tensor and rows divisible by positive CP required")
    if gathered.numel() == 0 or cp_size == 1 or gathered.shape[0] == cp_size:
        return gathered

    device = gathered.device
    stream = torch.cuda.current_stream(device)
    with torch.cuda.device(device):
        capturing = torch.cuda.is_current_stream_capturing()
    key = (device.index, gathered.shape[0], cp_size)
    with _PLAN_LOCK:
        plan = _CAPTURED_PLANS.get(key) or _PLANS.get(key)
        if plan is None:
            if capturing:
                raise RuntimeError("warm the CP output shape before graph capture")
            # Shape-only CPU work supplies an allocation extent. Plan values
            # are generated on the GPU; no CPU tensor or plan is uploaded.
            cycles = len(_cycle_leaders(gathered.shape[0], cp_size))
            indices = torch.empty(
                (cycles,),
                dtype=torch.int64,
                device=device,
            )
            _cp_output.make_interleave_plan_(indices, gathered.shape[0], cp_size)
            ready = torch.cuda.Event()
            ready.record(stream)
            plan = (indices, ready)
        _PLANS[key] = plan
        _PLANS.move_to_end(key)
        if capturing:
            _CAPTURED_PLANS[key] = plan
        if len(_PLANS) > _MAX_PLANS:
            _PLANS.popitem(last=False)
    indices, ready = plan
    stream.wait_event(ready)
    indices.record_stream(stream)
    _cp_output.interleave_rows_(gathered, cp_size, indices)
    return gathered


__all__ = ["interleave_cp_output_"]
