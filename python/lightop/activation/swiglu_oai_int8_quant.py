# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""BF16-rounded OAI SwiGLU with symmetric per-row INT8 quantization."""
import threading
import torch
from lightop import op as _op

# Existing runtime packages may carry this separately compiled native extension.
# A normal LightOp build exposes the same functions through lightop.op.
if hasattr(_op, "swiglu_oai_build_int8_quant_table"):
    class _Native:
        contiguous = staticmethod(_op.swiglu_oai_int8_quant_contiguous)
        masked = staticmethod(_op.swiglu_oai_int8_quant_masked)
        build_gate_table = staticmethod(_op.swiglu_oai_build_gate_table)
        build_quant_table = staticmethod(_op.swiglu_oai_build_int8_quant_table)
        contiguous_tables = staticmethod(_op.swiglu_oai_int8_quant_contiguous_tables)
        contiguous_block_tables = staticmethod(_op.swiglu_oai_int8_quant_contiguous_block_tables)
        contiguous_direct = staticmethod(_op.swiglu_oai_int8_quant_contiguous_direct)
        masked_flat_tables = staticmethod(_op.swiglu_oai_int8_quant_masked_flat_tables)
    _native = _Native()
else:
    from lightop import lightop_oai_int8_candidate as _native

_tables = {}
_table_lock = threading.Lock()

def _get_tables(input, alpha, limit):
    key = (input.device.index, float(alpha), float(limit))
    cached = _tables.get(key)
    if cached is not None:
        return cached
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("Warm up OAI INT8 quantization before graph capture")
    with _table_lock:
        if key not in _tables:
            gate = torch.empty(65536, device=input.device, dtype=torch.float32)
            quant = torch.empty(128*9*128, device=input.device, dtype=torch.uint8)
            _native.build_gate_table(gate, alpha, limit)
            _native.build_quant_table(quant)
            # One initialization sync makes the immutable cache safe on any stream.
            torch.cuda.current_stream(input.device).synchronize()
            _tables[key] = (gate, quant)
        return _tables[key]

def _buffers(input, output, scales):
    if input.dtype != torch.bfloat16 or not input.is_cuda:
        raise ValueError("Expected BF16 HIP input")
    if input.ndim not in (2, 3) or input.shape[-1] <= 0 or input.shape[-1] % 2:
        raise ValueError("Expected [rows,2H] or [E,M,2H]")
    if output is None:
        output = torch.empty((*input.shape[:-1], input.shape[-1]//2),
                             dtype=torch.int8, device=input.device)
    if scales is None:
        scales = torch.empty((*input.shape[:-1], 1),
                             dtype=torch.float32, device=input.device)
    return output, scales

def _limit(limit):
    if limit is not None and limit <= 0:
        raise ValueError("Clamp limit must be positive")
    return 0.0 if limit is None else float(limit)

def swiglu_oai_int8_quant_contiguous(
    input, alpha=1.702, beta=1.0, limit=7.0, output=None, scales=None,
):
    """FP32 OAI -> BF16 -> absmax/127; RNE INT8 in [-127,127].

    Zero rows use scale 1e-10/127. Persistent output buffers support graph replay.
    """
    output, scales = _buffers(input, output, scales)
    limit = _limit(limit)
    alpha, beta = float(alpha), float(beta)
    if input.ndim != 2:
        raise ValueError("Contiguous entry requires [rows,2H]")
    if input.numel() == 0:
        _native.contiguous(input, output, scales, alpha, beta, limit)
    elif input.shape[-1] == 6144 and input.data_ptr()%16 == 0 and output.data_ptr()%8 == 0:
        gate, quant = _get_tables(input, alpha, limit)
        _native.contiguous_block_tables(input, output, scales, gate, quant,
                                       alpha, beta, limit, 256, min(input.shape[0],65536))
    else:
        _native.contiguous(input, output, scales, alpha, beta, limit)
    return output, scales

def swiglu_oai_int8_quant_masked(
    input, tokens_per_expert, alpha=1.702, beta=1.0, limit=7.0,
    output=None, scales=None,
):
    """Mask [E,M,2H]; overwrite padding with q=0, scale=1."""
    output, scales = _buffers(input, output, scales)
    _native.masked(input, output, scales, tokens_per_expert,
                   float(alpha), float(beta), _limit(limit))
    return output, scales

def swiglu_oai_int8_quant_masked_flat(
    input, tokens_per_expert, alpha=1.702, beta=1.0, limit=7.0,
    output=None, scales=None,
):
    """Quantize ONLY live rows of [16,M,2H]; padding remains unspecified.

    Requires H%8==0, H<=8192 and 16/8-byte input/output alignment. The consumer
    must mask padding. Counts stay on device and can change during graph replay.
    Other expert counts/layouts retain zero/unit padding through the fallback.
    """
    output, scales = _buffers(input, output, scales)
    alpha, beta, limit = float(alpha), float(beta), _limit(limit)
    if input.ndim != 3:
        raise ValueError("Masked entry requires [E,M,2H]")
    if (input.shape[0] == 16 and input.shape[-1]//2%8 == 0
        and input.data_ptr()%16 == 0 and output.data_ptr()%8 == 0 and input.numel()):
        gate, quant = _get_tables(input, alpha, limit)
        _native.masked_flat_tables(input, output, scales, tokens_per_expert,
                                  gate, quant, alpha, beta, limit, 256, 256)
    else:
        _native.masked(input, output, scales, tokens_per_expert, alpha, beta, limit)
    return output, scales
