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
    try:
        from lightop import lightop_oai_int8_candidate as _native
    except ImportError:
        _native = None

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
    if _native is None:
        raise RuntimeError(
            "OAI INT8 quantization is unavailable; rebuild LightOp with the "
            "OAI INT8 native operators before calling this API"
        )
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
    input, alpha=1.702, beta=1.0, limit=7.0, output=None, scales=None, *,
    vllm_compatible=False, block_threads=None,
):
    """FP32 OAI -> BF16 -> row INT8 and FP32 scale, with optional output buffers.

    The default floors absmax at 1e-10. With vllm_compatible=True, aligned
    contiguous BF16 [rows,6144] uses the unfloored row maximum and rounds
    after multiplying by float32 (127 / absmax). Zero rows return zero scale.
    Both modes support persistent output buffers and graph replay. Warm up
    each alpha and limit combination before capturing a graph.
    """
    output, scales = _buffers(input, output, scales)
    limit = _limit(limit)
    alpha, beta = float(alpha), float(beta)
    if input.ndim != 2:
        raise ValueError("Contiguous entry requires [rows,2H]")
    if vllm_compatible:
        native_vllm = getattr(_op, "swiglu_oai_int8_quant_vllm_contiguous", None)
        if native_vllm is None:
            raise RuntimeError("Rebuild LightOp for vLLM-compatible OAI INT8 quantization")
        gate = None
        if input.shape[1] in (6144, 12288) and input.numel() != 0:
            gate, _ = _get_tables(input, alpha, limit)
        if block_threads is None:
            native_vllm(input, output, scales, alpha, beta, limit, gate)
        else:
            native_vllm(input, output, scales, alpha, beta, limit, gate, block_threads)
    elif input.numel() == 0:
        _native.contiguous(input, output, scales, alpha, beta, limit)
    elif input.shape[-1] == 6144 and input.data_ptr()%16 == 0 and output.data_ptr()%8 == 0:
        gate, quant = _get_tables(input, alpha, limit)
        _native.contiguous_block_tables(input, output, scales, gate, quant,
                                       alpha, beta, limit, 256, min(input.shape[0],65536))
    else:
        _native.contiguous(input, output, scales, alpha, beta, limit)
    return output, scales

def swiglu_oai_int8_quant_transposed(
    input, alpha=1.702, beta=1.0, limit=7.0, output=None, scales=None, *, config=0,
):
    """Consume column-major shared gate/up directly, returning vLLM INT8 and scale."""
    output, scales = _buffers(input, output, scales)
    if input.ndim != 2 or input.shape[1] not in (6144, 12288):
        raise ValueError("Expected column-major MiniMax shared gate/up")
    limit = _limit(limit)
    gate, _ = _get_tables(input, float(alpha), limit)
    _op.swiglu_oai_int8_quant_vllm_transposed(
        input, output, scales, float(alpha), float(beta), limit, gate, config,
    )
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
    output=None, scales=None, *, block_threads=256, grid_blocks=256,
):
    """Quantize ONLY live rows of [16,M,2H]; padding remains unspecified.

    Requires H%8==0, H<=8192 and 16/8-byte input/output alignment. The consumer
    must mask padding. Counts stay on device and can change during graph replay.
    Other expert counts/layouts retain zero/unit padding through the fallback.
    block_threads and grid_blocks tune the live-only path on wave64 devices.
    """
    output, scales = _buffers(input, output, scales)
    alpha, beta, limit = float(alpha), float(beta), _limit(limit)
    if input.ndim != 3:
        raise ValueError("Masked entry requires [E,M,2H]")
    if (input.shape[0] == 16 and input.shape[-1]//2%8 == 0
        and input.data_ptr()%16 == 0 and output.data_ptr()%8 == 0 and input.numel()):
        gate, quant = _get_tables(input, alpha, limit)
        _native.masked_flat_tables(input, output, scales, tokens_per_expert,
                                  gate, quant, alpha, beta, limit, block_threads, grid_blocks)
    else:
        _native.masked(input, output, scales, tokens_per_expert, alpha, beta, limit)
    return output, scales
