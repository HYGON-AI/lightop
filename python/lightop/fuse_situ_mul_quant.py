from typing import Optional, Tuple

import torch

from . import op


def fuse_situ_mul_quant_contiguous(
    input: torch.Tensor,
    situ_beta: float,
    situ_linear_beta: float,
    output: Optional[torch.Tensor] = None,
    scales: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Fused SiTU (SoftCap-GLU) + per-token INT8 quant for contiguous layout.

    Matches ``fuse_situ_mul_quant_contiguous`` from
    sglang-das/python/sglang/srt/layers/moe/ep_moe/layer.py.
    """
    if input.dim() != 2 or input.shape[-1] % 2 != 0:
        raise ValueError("input must have shape [tokens, 2 * hidden]")
    if not input.is_contiguous():
        raise ValueError("input must be contiguous")
    if situ_beta <= 0 or situ_linear_beta <= 0:
        raise ValueError("SiTU beta and linear_beta must be positive")

    tokens, doubled_hidden = input.shape
    hidden = doubled_hidden // 2

    if output is None:
        output = torch.empty((tokens, hidden), dtype=torch.int8, device=input.device)
    if scales is None:
        scales = torch.empty((tokens, 1), dtype=torch.float32, device=input.device)

    op.fuse_situ_mul_quant_contiguous(
        input, output, scales, float(situ_beta), float(situ_linear_beta)
    )
    return output, scales


def fuse_situ_mul_quant_ep(
    input: torch.Tensor,
    masked_m: torch.Tensor,
    situ_beta: float,
    situ_linear_beta: float,
    expect_m: int = -1,
    output: Optional[torch.Tensor] = None,
    scales: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Fused SiTU (SoftCap-GLU) + per-token INT8 quant for the EP layout.

    Matches ``fuse_situ_mul_quant_ep`` from
    sglang-das/python/sglang/srt/layers/moe/ep_moe/layer.py.

    Args:
        input: Contiguous ``[experts, tokens, 2 * hidden]``, fp32/fp16/bf16.
        masked_m: ``[experts]`` int32; ``masked_m[e]`` is the valid-token
            count for expert ``e``.
        situ_beta, situ_linear_beta: Positive SoftCap-GLU betas.
        expect_m: Scheduling hint, not a bound on valid tokens. On ROCm,
            BF16 inputs with E=28 and hidden=3072 use packed scheduling for
            hints 1..4 and a tuned per-expert reduction for hints 5..19.
            LIGHTOP_SITU_EP_PACKED=0 restores legacy scheduling;
            LIGHTOP_SITU_EP_TUNED=0 keeps only the earlier packed path.
            Both switches default on and are read once on the first call.
            Other cases retain the original grid-stride scheduling: a
            positive hint clamps the grid to min(expect_m, tokens per
            expert); otherwise the grid covers the full token capacity.
            Every path processes the actual masked_m counts, including
            counts greater than the hint.

    Returns:
        (output, scales) where output is ``[experts, tokens, hidden]`` int8
        and scales is ``[experts, tokens, 1]`` fp32. Rows with
        ``t >= masked_m[e]`` are never written by the kernel; they retain
        the buffer's initial contents. The downstream masked GEMM
        (``m_grouped_w4a8_gemm_nt_masked_hipc``) reads only valid rows, so
        the padding output is a genuine don't-care. Padding scales are
        prefilled with 1.0 so that any consumer that scans the whole tensor
        sees a finite value.
    """
    if input.dim() != 3 or input.shape[-1] % 2 != 0:
        raise ValueError("input must have shape [experts, tokens, 2 * hidden]")
    if not input.is_contiguous():
        raise ValueError("input must be contiguous")
    if masked_m.dim() != 1 or masked_m.shape[0] != input.shape[0]:
        raise ValueError(
            f"masked_m must have shape ({input.shape[0]},), got {tuple(masked_m.shape)}"
        )
    if masked_m.dtype != torch.int32:
        raise ValueError("masked_m must be int32")
    if situ_beta <= 0 or situ_linear_beta <= 0:
        raise ValueError("SiTU beta and linear_beta must be positive")

    experts, tokens, doubled_hidden = input.shape
    hidden = doubled_hidden // 2

    if output is None:
        output = torch.empty(
            (experts, tokens, hidden), dtype=torch.int8, device=input.device
        )
    if scales is None:
        # Prefill 1.0 so padding rows have a safe finite scale. This is
        # required because the kernel now leaves padding rows untouched.
        scales = torch.full(
            (experts, tokens, 1),
            1.0,
            dtype=torch.float32,
            device=input.device,
        )

    op.fuse_situ_mul_quant_ep(
        input,
        output,
        scales,
        masked_m,
        float(situ_beta),
        float(situ_linear_beta),
        int(expect_m),
    )
    return output, scales
