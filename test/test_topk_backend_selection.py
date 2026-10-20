# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Check cached backend selection in separate processes using real HIP kernels."""
import os
import subprocess
import sys

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.version.hip is None,
    reason="native adaptive backend dispatch requires wave64 HIP",
)

PROBE = r'''
import os, torch
from lightop.attention import fast_topk_transform_fused, fast_topk_transform_ragged_fused
from lightop.fuse_topk_transform import topk_transform_512, topk_transform_1024

mode = os.environ.get("SGL_USE_LIGHTOP_TOPK_BACKAND", "")
torch.manual_seed(42)
n = 8192
scores = torch.randn(2, n, device="cuda")
lengths = torch.full((2,), n, dtype=torch.int32, device="cuda")
pages = torch.arange(n, dtype=torch.int32, device="cuda")[None, :].repeat(2, 1)
cu = torch.arange(3, dtype=torch.int32, device="cuda")
offsets = torch.zeros(2, dtype=torch.int32, device="cuda")

def call(k):
    if k == 2048:
        return fast_topk_transform_fused(scores, lengths, pages, cu, k)
    out = torch.empty((2, k), dtype=torch.int32, device="cuda")
    (topk_transform_512 if k == 512 else topk_transform_1024)(scores, lengths, pages, out, 1)
    return out

def rejects(fn, text):
    try:
        fn()
    except RuntimeError as error:
        assert text in str(error), error
    else:
        raise AssertionError("expected rejection: " + text)

if mode == "3":
    rejects(lambda: call(512), "must be 0, 1, or 2")
    raise SystemExit(0)

ks = [512] if mode == "2" else [512, 1024, 2048]
for k in ks:
    expected = torch.topk(scores, k, dim=1).indices.sort(dim=1).values.int()
    assert torch.equal(call(k).sort(dim=1).values, expected), (mode, k)
torch.cuda.synchronize()
with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                       torch.profiler.ProfilerActivity.CUDA]) as prof:
    for k in ks:
        call(k)
    torch.cuda.synchronize()
names = [e.name for e in prof.events()]
if mode in ("", "0"):
    for k in ks:
        assert any("at::native::adaptive_topk_kernel<" + str(k) + "," in name for name in names), (k, names)
else:
    assert not any("adaptive_topk_kernel<" in name for name in names)
    expected_name = "radix_select_8bit" if mode == "1" else "vgpr_select_512"
    assert any(expected_name in name for name in names), names
if mode == "2":
    rejects(lambda: call(1024), "requires topK=512")
    rejects(lambda: call(2048), "only supported")
    rejects(lambda: fast_topk_transform_ragged_fused(scores, lengths, offsets, 2048), "only supported")
'''


@pytest.mark.parametrize("mode", [None, "", "0", "1", "2", "3"],
                         ids=["unset", "empty", "adaptive", "legacy1", "legacy2", "invalid"])
def test_backend_dispatch(mode):
    env = dict(os.environ)
    if mode is None:
        env.pop("SGL_USE_LIGHTOP_TOPK_BACKAND", None)
    else:
        env["SGL_USE_LIGHTOP_TOPK_BACKAND"] = mode
    # The removed switch must neither disable backend 0 nor override backend 1/2.
    env["LIGHTOP_ADAPTIVE_TOPK"] = "0" if mode in (None, "", "0") else "1"
    result = subprocess.run([sys.executable, "-c", PROBE], env=env,
                            capture_output=True, text=True, timeout=180)
    assert result.returncode == 0, result.stdout + result.stderr
