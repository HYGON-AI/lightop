# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import torch.nn.functional as F

from lightop.activation import step5_swiglustep_fp8_quant


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires HCU")


@pytest.mark.parametrize("id_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("limit", [1.0, 7.0])
def test_step5_rowwise_quantization(id_dtype, limit):
    generator = torch.Generator(device="cuda").manual_seed(732)
    x = torch.randn(17, 3072, device="cuda", dtype=torch.bfloat16, generator=generator) * 4
    ids = torch.arange(len(x), device=x.device, dtype=id_dtype)
    ids[1::4] = -1
    ids[3::7] = -2
    valid = ids >= 0
    x[~valid] = torch.nan
    x[0].zero_()
    q = torch.empty((len(x), 1536), device=x.device, dtype=torch.float8_e4m3fn)
    scales = torch.empty((len(x), 1), device=x.device, dtype=torch.float32)
    out, actual_scales = step5_swiglustep_fp8_quant(
        x, ids, output=q, scales=scales, limit=limit
    )
    assert out is q
    assert actual_scales is scales
    assert torch.count_nonzero(out.view(torch.uint8)[~valid]).item() == 0
    torch.testing.assert_close(
        scales[~valid], torch.full_like(scales[~valid], 1e-10 / 448), rtol=0, atol=0
    )
    gate, up = x[valid].float().chunk(2, dim=-1)
    y = (F.silu(gate).clamp_max(limit) * up.clamp(-limit, limit)).bfloat16().float()
    expected_scales = y.abs().amax(dim=-1, keepdim=True).clamp_min(1e-10) / 448
    torch.testing.assert_close(scales[valid], expected_scales, rtol=0.008, atol=0)
    # Allow FP8 rounding and one BF16 rounding step around SiLU ties.
    normalized = y / scales[valid]
    magnitude = normalized.abs()
    spacing = torch.exp2(torch.floor(torch.log2(magnitude.clamp_min(2**-6))) - 3)
    assert torch.all((out[valid].float() - normalized).abs() <= 0.5 * spacing + 0.008 * magnitude)


def test_step5_empty_rows():
    x = torch.empty((0, 3072), device="cuda", dtype=torch.bfloat16)
    ids = torch.empty(0, device=x.device, dtype=torch.int32)
    q, scales = step5_swiglustep_fp8_quant(x, ids)
    assert q.shape == (0, 1536) and q.dtype == torch.float8_e4m3fn
    assert scales.shape == (0, 1) and scales.dtype == torch.float32
