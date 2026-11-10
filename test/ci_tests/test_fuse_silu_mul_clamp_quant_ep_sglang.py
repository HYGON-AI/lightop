# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Bitwise EP GELU/clamp/smooth/INT8 regression."""

import pytest
import torch
import torch.nn.functional as F

from lightop import fuse_silu_mul_clamp_quant_ep


lmslim_int8_utils = pytest.importorskip("lmslim.layers.gemm.int8_utils")
per_token_quant_int8 = lmslim_int8_utils.per_token_quant_int8


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a GPU")
@pytest.mark.parametrize("expect_m", [-1, 0, 1, 7, 17])
def test_ep_gelu_clamp_bf16_smooth_quant_matches_lmslim_bitwise(
    expect_m: int,
) -> None:
    """Valid masked rows must match the real unfused operator chain."""

    torch.manual_seed(20260809)
    num_experts, tokens_per_expert, hidden_size = 6, 17, 1024
    limit = 8.0
    gate_up = (
        torch.randn(
            (num_experts, tokens_per_expert, 2 * hidden_size),
            device="cuda",
            dtype=torch.bfloat16,
        )
        * 4
    )
    smooth_scale = torch.empty(
        (hidden_size,), device="cuda", dtype=torch.float32
    ).uniform_(0.05, 2.0)
    # Includes an empty expert and several valid counts greater than expect_m.
    mask_m = torch.tensor(
        [0, 1, 3, 9, 17, 5], device="cuda", dtype=torch.int32
    )

    actual_q, actual_scales = fuse_silu_mul_clamp_quant_ep(
        gate_up,
        limit,
        mask_m,
        expect_m=expect_m,
        use_gelu=True,
        smooth_scale=smooth_scale,
    )

    assert actual_q.shape == (
        num_experts,
        tokens_per_expert,
        hidden_size,
    )
    assert actual_scales.shape == (num_experts, tokens_per_expert, 1)

    smooth_scale_bf16 = smooth_scale.to(torch.bfloat16)
    for expert, valid_m in enumerate(mask_m.tolist()):
        if valid_m == 0:
            continue
        valid_gate_up = gate_up[expert, :valid_m]
        gate, up = valid_gate_up.chunk(2, dim=-1)
        # This is the real SGLang order and its materialization
        # points: exact GELU(BF16), Mul(BF16), clamp(BF16), then
        # InputSmooth with the stored FP32 scale cast to BF16.  lmslim is
        # the correctness oracle for scale flooring and INT8 rounding.
        activated = F.gelu(gate, approximate="none")
        multiplied = activated * up
        clamped = multiplied.clamp(-limit, limit)
        smoothed = clamped * smooth_scale_bf16
        expected_q, expected_scales = per_token_quant_int8(smoothed)

        torch.testing.assert_close(
            actual_q[expert, :valid_m], expected_q, rtol=0, atol=0
        )
        torch.testing.assert_close(
            actual_scales[expert, :valid_m],
            expected_scales,
            rtol=0,
            atol=0,
        )
