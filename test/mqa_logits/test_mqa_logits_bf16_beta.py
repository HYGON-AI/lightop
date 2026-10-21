# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.

"""BF16 beta regressions for nonfinite values and padded SmallKV heads."""

import pytest
import torch
import lightop


@pytest.fixture(autouse=True)
def bf16_beta(monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("requires a HIP GPU")
    arch = getattr(torch.cuda.get_device_properties(0), "gcnArchName", "")
    if not arch.startswith(("gfx936", "gfx938")):
        pytest.skip("BF16 beta requires gfx936 or gfx938")
    monkeypatch.setenv("LIGHTOP_MQA_LOGITS_BETA", "1")


SHAPES = [(1, 16, h) for h in [1, 2, 4, 8, 16, 32, 64, 128]] + [
    (32, 64, 8),       # Paired SmallKV.
    (17, 1025, 8),     # BM16 with M/N tails.
    (65, 1025, 8),     # BM32 with M/N tails.
    (64, 65537, 8),    # Long KV window with an N tail.
    (64, 65537, 1),    # BM64 and fewer than 16 heads.
    (16, 1024, 64),    # SplitHead.
    (1024, 1024, 64),  # BM32/BN256/GH1 with the batch NaN detector.
    (1025, 1025, 16),  # The same fast ReLU with M/N tails.
]


@pytest.mark.parametrize("m,n,h", SHAPES)
@pytest.mark.parametrize("clean", [False, True])
@pytest.mark.parametrize("case", [
    "q_nan", "q_negative_nan", "k_nan", "k_negative_nan",
    "q_inf", "q_negative_inf", "k_inf", "k_negative_inf",
    "weights_nan", "masked_q_nan", "masked_k_nan",
    "q_snan", "q_negative_snan", "k_snan", "k_negative_snan",
    "zero_times_inf", "inf_cancel",
])
def test_bf16_beta_nonfinite(m, n, h, clean, case):
    q = torch.ones((m, h, 128), device="cuda", dtype=torch.bfloat16)
    k = torch.ones((n, 128), device="cuda", dtype=torch.bfloat16)
    weights = torch.ones((m, h), device="cuda", dtype=torch.float32)
    ks = torch.zeros(m, device="cuda", dtype=torch.int32)
    ke = torch.full_like(ks, n)
    expected = torch.full((m, n), float(h * 128), device="cuda")
    if case in ("q_nan", "q_negative_nan", "masked_q_nan",
                "q_snan", "q_negative_snan"):
        bits = (-127 if "negative" in case else 0x7F81) if "snan" in case else (
            -64 if "negative" in case else 0x7FC0)
        q.view(torch.int16)[0, 0, 0] = bits
        expected[0] = float("nan")
        if case == "masked_q_nan":
            ke[0] = 0
    elif case in ("k_nan", "k_negative_nan", "masked_k_nan",
                  "k_snan", "k_negative_snan"):
        bits = (-127 if "negative" in case else 0x7F81) if "snan" in case else (
            -64 if "negative" in case else 0x7FC0)
        k.view(torch.int16)[0, 0] = bits
        expected[:, 0] = float("nan")
        if case == "masked_k_nan":
            ks.fill_(1)
    elif case in ("q_inf", "q_negative_inf"):
        q[0, 0, 0] = float("-inf") if case == "q_negative_inf" else float("inf")
        expected[0] = float((h - 1) * 128) if case == "q_negative_inf" else float("inf")
    elif case in ("k_inf", "k_negative_inf"):
        k[0, 0] = float("-inf") if case == "k_negative_inf" else float("inf")
        expected[:, 0] = 0. if case == "k_negative_inf" else float("inf")
    elif case == "zero_times_inf":
        q[0, 0, 0] = 0.
        k[0, 0] = float("inf")
        expected[0] = float(h * 128 - 1)
        expected[:, 0] = float("inf")
        expected[0, 0] = float("nan")
    elif case == "inf_cancel":
        q[0, 0, 0] = float("inf")
        q[0, 0, 1] = float("-inf")
        expected[0] = float("nan")
    else:
        weights[0, 0] = float("nan")
        expected[0] = float("nan")
    out = torch.full((m, n), 137., device="cuda")
    actual = lightop.mqa_logits(q, k, weights, ks, ke, clean_logit=clean, D_out=out)
    columns = torch.arange(n, device="cuda")[None, :]
    valid = (columns >= ks[:, None]) & (columns < ke[:, None])
    assert actual.data_ptr() == out.data_ptr()
    torch.testing.assert_close(actual[valid], expected[valid],
                               rtol=0, atol=0, equal_nan=True)
    if clean:
        assert bool(torch.isneginf(actual[~valid]).all())


@pytest.mark.parametrize("clean", [False, True])
@pytest.mark.parametrize("case", [
    "late_q_nan", "late_k_nan", "opposite_inf", "finite_detector_overflow",
    "underflow", "zero_weights",
])
def test_bf16_beta_relu_guard(clean, case):
    # Auto selects BM32/BN256/GH1; the final tile has both M and N tails.
    m, n, h = 1025, 1025, 16
    q = torch.ones((m, h, 128), device="cuda", dtype=torch.bfloat16)
    k = torch.ones((n, 128), device="cuda", dtype=torch.bfloat16)
    weights = torch.ones((m, h), device="cuda", dtype=torch.float32)
    ks = torch.zeros(m, device="cuda", dtype=torch.int32)
    ke = torch.full_like(ks, n)
    expected = torch.full((m, n), float(h * 128), device="cuda")
    if case == "late_q_nan":
        # Cover the second M16 fragment and the last head, not only dot[0][0].
        q.view(torch.int16)[m - 2, h - 1, 127] = -64
        expected[m - 2] = float("nan")
    elif case == "late_k_nan":
        k.view(torch.int16)[n - 2, 127] = -64
        expected[:, n - 2] = float("nan")
    elif case == "opposite_inf":
        k[0, 0] = float("inf")
        k[16, 0] = float("-inf")
        expected[:, 0] = float("inf")
        expected[:, 16] = 0.
    elif case == "finite_detector_overflow":
        # Every dot is finite (2**127), although their detection sum overflows.
        # Scaling each dot before accumulation keeps the result exactly 128*H.
        q.fill_(2.**60)
        k.fill_(2.**60)
        weights.fill_(2.**-120)
    elif case == "underflow":
        q.fill_(-2.**-100)
        k.fill_(2.**-100)
        weights.fill_(-1.)
        expected.zero_()
    else:
        weights.zero_()
        weights[:, ::2] = -0.
        expected.zero_()
    out = torch.full((m, n), 137., device="cuda")
    actual = lightop.mqa_logits(q, k, weights, ks, ke, clean_logit=clean, D_out=out)
    assert actual.data_ptr() == out.data_ptr()
    # The fast clamp may normalize -0 to +0; numerical equality is required.
    torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True)
