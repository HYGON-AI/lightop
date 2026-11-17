# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

import runpy
from types import SimpleNamespace

import pytest
import torch


@pytest.mark.parametrize("cus", [64, 72, 128])
def test_gfx946_dispatch(monkeypatch, cus):
    from lightop import envs
    from lightop.attention.mqa import _persistent_supported
    from lightop.config_smooth_gemm import gemm_w8a8_smooth_config

    props = SimpleNamespace(gcnArchName="gfx946:sramecc+", multi_processor_count=cus)
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda device: props)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setenv("LIGHTOP_GPU_CUS", str(cus))
    monkeypatch.setenv("LIGHTOP_GPU_TARGET", "gfx946")
    assert runpy.run_path(envs.__file__)["LMSLIM_GPU_NAME"] == f"gfx946_{cus}cu"
    assert gemm_w8a8_smooth_config.get_w8a8_smooth_quant_config_dict()

    q = torch.empty(1, 1, 32, 128, dtype=torch.int8)
    kv = torch.empty(1, 64, 1, 132, dtype=torch.int8)
    weights = torch.ones(1, 32)
    lens = torch.ones(1, dtype=torch.int32)
    table = torch.zeros(1, 1, dtype=torch.int32)
    args = (weights, lens, table, None, 64, True, False)
    assert _persistent_supported(q, kv, *args)
    assert not _persistent_supported(q, kv.to(torch.uint8), *args)


@pytest.mark.parametrize("batch", [1, 2])
def test_gfx946_scaled_gemm_fallbacks(monkeypatch, batch):
    from lightop.gemm_ops import (
        BlockSize,
        hipblaslt_w8a8_blockwise_gemm,
        hipblaslt_w8a8_channelwise_gemm,
        hipblaslt_w8a8_gemm,
    )

    props = SimpleNamespace(gcnArchName="gfx946")
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda device: props)
    m, n, k = 3, 7, 128
    rng = torch.Generator().manual_seed(946)
    a = torch.randint(-4, 5, (batch, m, k), generator=rng, dtype=torch.int8)
    b = torch.randint(-4, 5, (batch, n, k), generator=rng, dtype=torch.int8)
    sa = torch.rand(batch, m, 1, generator=rng)
    sb = torch.rand(batch, n, 1, generator=rng)
    bias = torch.rand(batch, n, generator=rng).half()
    ref = (a.float() @ b.float().transpose(1, 2)) * sa * sb.transpose(1, 2)
    ref = (ref + bias[:, None, :]).half()
    args = (a, b, sa, sb, m, n, k, "NT", torch.float16, bias)
    ok, result = hipblaslt_w8a8_channelwise_gemm(*args)
    assert ok
    torch.testing.assert_close(result, ref)
    ok, result = hipblaslt_w8a8_gemm(*args)
    assert ok
    torch.testing.assert_close(result, ref.squeeze(0))

    scale_b = torch.rand(batch, 1, 1, generator=rng)
    ok, result = hipblaslt_w8a8_blockwise_gemm(
        a, b.transpose(1, 2).contiguous(), sa, scale_b, m, n, k, "NN",
        torch.float16, BlockSize.block_128x128, bias,
    )
    assert ok
    ref = (a.float() @ b.float().transpose(1, 2)) * sa * scale_b
    torch.testing.assert_close(result, (ref + bias[:, None, :]).half())


@pytest.mark.parametrize("dtype", [torch.int8, torch.float8_e4m3fn])
@pytest.mark.parametrize("transpose", [False, True])
def test_fused_bmm_accumulation(dtype, transpose):
    if not torch.cuda.is_available() or not torch.version.hip:
        pytest.skip("requires a HIP device")
    arch = torch.cuda.get_device_properties(0).gcnArchName.split(":")[0]
    if arch not in ("gfx938", "gfx946"):
        pytest.skip("requires INT8 and E4M3 MMAC")
    from lightop import fused_bmm

    rng = torch.Generator().manual_seed(946)
    x = torch.randint(-3, 4, (17, 2, 128), generator=rng).to(dtype)
    w = torch.randint(-3, 4, (2, 64, 128), generator=rng).to(dtype)
    ref = torch.bmm(x.float().transpose(0, 1), w.float().transpose(1, 2))
    if transpose:
        x = x.transpose(0, 1).contiguous()
    else:
        ref = ref.transpose(0, 1)
    sx = torch.ones(*x.shape[:2], 1, device="cuda")
    sw = torch.ones(2, 64, 1, device="cuda")
    result = fused_bmm(x.cuda(), w.cuda(), sx, sw, dtype=torch.bfloat16,
                       transpose_bm=transpose, transpose_bm_in=transpose)
    torch.cuda.synchronize()
    torch.testing.assert_close(result.cpu(), ref.bfloat16(), atol=0, rtol=0)
