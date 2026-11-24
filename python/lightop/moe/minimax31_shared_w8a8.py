# SPDX-License-Identifier: Apache-2.0
"""LightOP providers for the gfx938 MiniMax shared-expert W8A8 path."""

import torch


def minimax31_shared_gate_up_w8a8(quant, scales, weight, weight_scale):
    """Produce row-major BF16 gate/up from the existing KN weight view."""
    from lightop.gemmopt import gemm_w8a8_smooth
    from lightop._lmslim_native.quantize.quant_ops import hipblaslt_w8a8_channelwise_gemm

    if (
        quant.ndim != 2 or quant.shape[0] not in (1, 2, 4, 8, 16, 32, 64, 128)
        or quant.shape[1] != 6144 or quant.dtype != torch.int8 or not quant.is_cuda
        or not quant.is_contiguous() or scales.shape != (quant.shape[0], 1)
        or scales.dtype != torch.float32 or not scales.is_contiguous()
        or weight.ndim != 2 or weight.shape[0] != 6144
        or weight.shape[1] not in (6144, 12288)
        or weight.dtype != torch.int8 or weight.stride() != (1, 6144)
        or weight_scale.dtype != torch.float32 or not weight_scale.is_contiguous()
        or weight_scale.shape != (weight.shape[1], 1)
        or any(t.device != quant.device for t in (scales, weight, weight_scale))
    ):
        return False, None
    arch = getattr(torch.cuda.get_device_properties(quant.device), "gcnArchName", "")
    if not arch.startswith("gfx938"):
        return False, None

    m, n = quant.shape[0], weight.shape[1]
    if (m, n) == (128, 6144):
        ok, output = gemm_w8a8_smooth(
            quant, weight, scales, weight_scale, None, torch.bfloat16,
        )
    else:
        ok, output = hipblaslt_w8a8_channelwise_gemm(
            quant, weight.t(), scales, weight_scale, m, n, 6144, "NT", torch.bfloat16,
        )
    if not ok:
        return False, None
    if (output.shape not in ((m, n), (1, m, n))
            or not output.is_contiguous() or output.dtype != torch.bfloat16
            or output.device != quant.device):
        raise RuntimeError("Unexpected MiniMax shared gate/up output")
    return True, output.view(m, n)
