# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Correctness checks for the optional 656-byte FlashMLA KV-cache layout."""

import itertools

import torch

from lightop import op
from lightop.rmsnorm import fused_rms_norm_rope_contiguous


DTYPE = torch.bfloat16
Q_C_DIM = 128
KV_C_DIM = 512
ROT_DIM = 64
FLASHMLA_ENTRY_BYTES = 656
RMS_EPSILON = 1e-6
KV_QUANT_EPSILON = 1e-6


def rope_reference(
    value: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    is_neox_style: bool,
) -> torch.Tensor:
    selected = cos_sin_cache[positions]
    if is_neox_style:
        cos = selected[..., : ROT_DIM // 2].repeat(1, 2)
        sin = selected[..., ROT_DIM // 2 :].repeat(1, 2)
    else:
        cos = selected[..., : ROT_DIM // 2].repeat_interleave(2, dim=-1)
        sin = selected[..., ROT_DIM // 2 :].repeat_interleave(2, dim=-1)
    while cos.dim() < value.dim():
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)
    if is_neox_style:
        first, second = value.chunk(2, dim=-1)
        rotated = torch.cat((-second, first), dim=-1)
    else:
        pairs = value.reshape(*value.shape[:-1], ROT_DIM // 2, 2)
        rotated = torch.stack((-pairs[..., 1], pairs[..., 0]), dim=-1).flatten(-2)
    return value * cos + rotated * sin


def rms_reference(
    value: torch.Tensor,
    weight: torch.Tensor,
    limit: float,
    smooth_scale: torch.Tensor | None,
) -> torch.Tensor:
    output = value.float() * torch.rsqrt(
        value.float().square().mean(dim=-1, keepdim=True) + RMS_EPSILON
    )
    output = output * weight.float()
    if smooth_scale is None:
        if limit >= 0.0:
            output = torch.clamp(output, -limit, limit)
        return output.to(value.dtype)

    # tokenflow apply_input_smooth rounds RMSNorm to BF16, clamps BF16,
    # casts the FP32 smooth scale to BF16 first, then performs BF16 multiply.
    output = output.to(value.dtype)
    if limit >= 0.0:
        output = torch.clamp(output, -limit, limit)
    return output * smooth_scale.to(value.dtype)


def make_inputs(tokens: int, heads: int, three_dimensional_cache: bool) -> dict:
    torch.manual_seed(4100 + tokens + heads + int(three_dimensional_cache))
    q_storage = torch.randn(
        tokens, heads, Q_C_DIM + ROT_DIM, device="cuda", dtype=DTYPE
    ).contiguous()
    k_storage = torch.randn(
        tokens, KV_C_DIM + ROT_DIM, device="cuda", dtype=DTYPE
    ).contiguous()
    positions = torch.arange(tokens, device="cuda", dtype=torch.int64) % 256
    inv_freq = 1.0 / (
        10000.0
        ** (
            torch.arange(0, ROT_DIM, 2, device="cuda", dtype=torch.float32)
            / ROT_DIM
        )
    )
    freqs = torch.outer(torch.arange(256, device="cuda", dtype=torch.float32), inv_freq)
    cos_sin_cache = torch.cat((freqs.cos(), freqs.sin()), dim=-1).to(DTYPE)
    weight = torch.linspace(0.5, 2.5, KV_C_DIM, device="cuda", dtype=DTYPE)
    smooth_scale = torch.logspace(
        -1.0, 3.5, KV_C_DIM, device="cuda", dtype=torch.float32
    )

    slots = tokens + 7
    slot_mapping = torch.randperm(slots, device="cuda", dtype=torch.int64)[:tokens]
    if tokens >= 4:
        slot_mapping[0] = -1
        slot_mapping[1] = slots
    cache_shape = (
        (slots, 1, FLASHMLA_ENTRY_BYTES)
        if three_dimensional_cache
        else (slots, FLASHMLA_ENTRY_BYTES)
    )
    kv_cache = torch.full(cache_shape, 0xA5, device="cuda", dtype=torch.uint8)
    return {
        "positions": positions,
        "q_pe": q_storage[..., Q_C_DIM:],
        "q_before": q_storage[..., Q_C_DIM:].clone(),
        "k_pe": k_storage[..., KV_C_DIM:],
        "k_before": k_storage[..., KV_C_DIM:].clone(),
        "kv_c": k_storage[..., :KV_C_DIM],
        "kv_c_before": k_storage[..., :KV_C_DIM].clone(),
        "kv_c_normed": torch.full(
            (tokens, KV_C_DIM), 7.0, device="cuda", dtype=DTYPE
        ),
        "weight": weight,
        "smooth_scale": smooth_scale,
        "cos_sin_cache": cos_sin_cache,
        "slot_mapping": slot_mapping,
        "kv_cache": kv_cache,
        "slots": slots,
    }


def run_case(
    tokens: int,
    heads: int,
    three_dimensional_cache: bool,
    limit: float,
    is_neox_style: bool,
    apply_smooth: bool,
) -> None:
    data = make_inputs(tokens, heads, three_dimensional_cache)
    expected_cache = data["kv_cache"].clone()
    q_expected = rope_reference(
        data["q_before"],
        data["positions"],
        data["cos_sin_cache"],
        is_neox_style,
    )
    k_expected = data["k_before"].clone()
    rms_expected = torch.full_like(data["kv_c_normed"], 7.0)
    valid = (data["slot_mapping"] >= 0) & (data["slot_mapping"] < data["slots"])
    k_expected[valid] = rope_reference(
        data["k_before"][valid],
        data["positions"][valid],
        data["cos_sin_cache"],
        is_neox_style,
    )
    smooth_scale = data["smooth_scale"] if apply_smooth else None
    rms_expected[valid] = rms_reference(
        data["kv_c_before"][valid], data["weight"], limit, smooth_scale
    )

    fused_rms_norm_rope_contiguous(
        data["positions"],
        data["q_pe"],
        data["k_pe"],
        data["kv_c"],
        data["kv_c_normed"],
        data["weight"],
        data["cos_sin_cache"],
        data["slot_mapping"],
        data["kv_cache"],
        "fp8_e4m3",
        1.0,
        is_neox_style,
        RMS_EPSILON,
        limit,
        True,
        KV_QUANT_EPSILON,
        smooth_scale,
    )

    if valid.any():
        op.fused_quantize_and_store_mla_kv_cache(
            data["kv_c_normed"][valid],
            data["k_pe"][valid],
            expected_cache,
            data["slot_mapping"][valid],
            "fp8_e4m3",
            KV_QUANT_EPSILON,
        )

    torch.testing.assert_close(data["q_pe"], q_expected, rtol=0, atol=2e-2)
    torch.testing.assert_close(data["k_pe"], k_expected, rtol=0, atol=2e-2)
    torch.testing.assert_close(
        data["kv_c_normed"], rms_expected, rtol=1e-2, atol=2e-2
    )
    torch.testing.assert_close(data["kv_cache"], expected_cache, rtol=0, atol=0)


def test_flashmla_layout() -> None:
    token_head_pairs = (
        (1, 1),
        (4, 8),
        (5, 7),
        (16, 8),
        (33, 8),
        (128, 128),
        (1024, 8),
    )
    for (
        (tokens, heads),
        three_dimensional_cache,
        limit,
        is_neox_style,
        apply_smooth,
    ) in itertools.product(
        token_head_pairs,
        (False, True),
        (-1.0, 8.0, 0.0),
        (False, True),
        (False, True),
    ):
        run_case(
            tokens,
            heads,
            three_dimensional_cache,
            limit,
            is_neox_style,
            apply_smooth,
        )


def test_flashmla_layout_validation() -> None:
    data = make_inputs(4, 8, False)
    try:
        op.fused_rms_norm_rope_contiguous(
            data["positions"],
            data["q_pe"],
            data["k_pe"],
            data["kv_c"],
            data["kv_c_normed"],
            data["weight"],
            data["cos_sin_cache"],
            data["slot_mapping"],
            data["kv_cache"],
            "bf16",
            1.0,
            False,
            RMS_EPSILON,
            -1.0,
            True,
            KV_QUANT_EPSILON,
        )
    except RuntimeError:
        pass
    else:
        raise AssertionError("FlashMLA layout accepted a non-FP8 cache dtype")

    try:
        fused_rms_norm_rope_contiguous(
            data["positions"],
            data["q_pe"],
            data["k_pe"],
            data["kv_c"],
            data["kv_c_normed"],
            data["weight"],
            data["cos_sin_cache"],
            data["slot_mapping"],
            data["kv_cache"],
            "fp8_e4m3",
            1.0,
            True,
            RMS_EPSILON,
            8.0,
            True,
            KV_QUANT_EPSILON,
            data["smooth_scale"].to(torch.bfloat16),
        )
    except RuntimeError:
        pass
    else:
        raise AssertionError("FlashMLA layout accepted a non-FP32 smooth scale")

    misaligned_cache = torch.empty(
        (data["slots"], FLASHMLA_ENTRY_BYTES + 1),
        device="cuda",
        dtype=torch.uint8,
    )
    try:
        fused_rms_norm_rope_contiguous(
            data["positions"],
            data["q_pe"],
            data["k_pe"],
            data["kv_c"],
            data["kv_c_normed"],
            data["weight"],
            data["cos_sin_cache"],
            data["slot_mapping"],
            misaligned_cache,
            "fp8_e4m3",
            1.0,
            True,
            RMS_EPSILON,
            8.0,
            True,
            KV_QUANT_EPSILON,
        )
    except RuntimeError:
        pass
    else:
        raise AssertionError("FlashMLA layout accepted a misaligned token stride")

    cache_storage = torch.empty(
        data["slots"] * FLASHMLA_ENTRY_BYTES + 1,
        device="cuda",
        dtype=torch.uint8,
    )
    misaligned_base_cache = cache_storage[1:].view(
        data["slots"], FLASHMLA_ENTRY_BYTES
    )
    assert misaligned_base_cache.is_contiguous()
    try:
        fused_rms_norm_rope_contiguous(
            data["positions"],
            data["q_pe"],
            data["k_pe"],
            data["kv_c"],
            data["kv_c_normed"],
            data["weight"],
            data["cos_sin_cache"],
            data["slot_mapping"],
            misaligned_base_cache,
            "fp8_e4m3",
            1.0,
            True,
            RMS_EPSILON,
            8.0,
            True,
            KV_QUANT_EPSILON,
        )
    except RuntimeError:
        pass
    else:
        raise AssertionError("FlashMLA layout accepted a misaligned cache base")


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA/ROCm device is required")
    test_flashmla_layout()
    test_flashmla_layout_validation()
    print("FlashMLA 656-byte fused RMSNorm + RoPE cache checks passed.")


if __name__ == "__main__":
    main()
