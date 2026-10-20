# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Target correctness and metadata tests for mHC repeat+CP specialization."""

import pytest
import torch


@pytest.fixture(scope="module")
def api():
    if not torch.version.hip or not torch.cuda.is_available():
        pytest.skip("Requires reviewed HIP/gfx936")
    if torch.cuda.get_device_properties(0).gcnArchName.split(":", 1)[0] != "gfx936":
        pytest.skip("Requires gfx936")
    from lightop.attention import (
        mhc_repeat_cp_sglang,
        supports_mhc_repeat_cp_sglang,
    )

    return supports_mhc_repeat_cp_sglang, mhc_repeat_cp_sglang


def bytes_equal(first, second):
    return torch.equal(first.contiguous().view(torch.uint8),
                       second.contiguous().view(torch.uint8))


@pytest.mark.parametrize("rank", [0, 3, 7])
def test_target_ranks_match_slice_repeat_bitwise(api, rank):
    supports, kernel = api
    generator = torch.Generator(device="cuda").manual_seed(20260905 + rank)
    source = torch.randn((32768, 4096), device="cuda", dtype=torch.float32,
                         generator=generator).to(torch.bfloat16)
    source[rank, 0:6] = torch.tensor(
        [0.0, -0.0, float("inf"), -float("inf"), float("nan"), 1.0],
        device="cuda", dtype=torch.bfloat16)
    before = source.clone()
    expected = source[rank::8].unsqueeze(1).repeat(1, 4, 1)
    assert supports(source, 4, 8, rank) is True
    actual = kernel(source, 4, 8, rank)
    torch.cuda.synchronize()
    assert actual.shape == (4096, 4, 4096) and actual.is_contiguous()
    assert bytes_equal(actual, expected)
    assert bytes_equal(source, before)


def test_metadata_rejections(api):
    supports, kernel = api
    source = torch.empty((32768, 4096), device="cuda", dtype=torch.bfloat16)
    cases = [
        (source[:32767], 4, 8, 0),
        (source[:, :4095], 4, 8, 0),
        (source.float(), 4, 8, 0),
        (source.detach().requires_grad_(True), 4, 8, 0),
        (source, 3, 8, 0),
        (source, 4, 4, 0),
        (source, 4, 8, -1),
        (source, 4, 8, 8),
    ]
    for args in cases:
        assert supports(*args) is False
    with pytest.raises(RuntimeError, match="inference-only"):
        kernel(source.detach().requires_grad_(True), 4, 8, 0)
