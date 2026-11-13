# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Correctness coverage for the gfx938 H64 native-FP8 sparse MQA path."""

import pytest
import torch

from lightop.gemmopt import page_mqa_logits_sparse_mask_grouped


def require_gfx938():
    if not torch.cuda.is_available() or torch.version.hip is None:
        pytest.skip("requires an HCU with the installed LightOp extension")
    props = torch.cuda.get_device_properties(0)
    arch = props.gcnArchName.split(":")[0]
    if arch != "gfx938" or props.multi_processor_count not in (64, 72):
        pytest.skip(f"requires gfx938/64CU/72CU, got {arch}/{props.multi_processor_count}CU")


def pack_fp8_kv(kv: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    blocks, page_size, _, head_dim = kv.shape
    storage = torch.empty(
        blocks,
        page_size * (head_dim + 4),
        dtype=torch.uint8,
        device=kv.device,
    )
    storage[:, : page_size * head_dim].copy_(
        kv.view(torch.uint8).reshape(blocks, page_size * head_dim)
    )
    storage[:, page_size * head_dim :].copy_(
        scales.contiguous().view(torch.uint8).reshape(blocks, page_size * 4)
    )
    return storage.view(blocks, page_size, 1, head_dim + 4)


def unpack_nonzero_mask(mask: torch.Tensor, width: int) -> torch.Tensor:
    bits = 1 << torch.arange(16, dtype=torch.int32, device=mask.device)
    return ((mask.to(torch.int32).unsqueeze(-1) & bits) != 0).flatten(1)[:, :width]


def fp32_reference(q, kv, scales, weights, context_lens, block_table, width):
    result = torch.zeros(q.shape[0], width, dtype=torch.float32, device=q.device)
    for row in range(q.shape[0]):
        length = int(context_lens[row])
        pages = block_table[row, : (length + 63) // 64].long()
        keys = kv[pages].reshape(-1, 128)[:length].float()
        key_scales = scales[pages].reshape(-1, 1)[:length]
        keys = keys * key_scales
        dots = torch.matmul(q[row, 0].float(), keys.transpose(0, 1))
        result[row, :length] = (
            torch.relu(dots) * weights[row].unsqueeze(1)
        ).sum(dim=0)
    return result


@torch.no_grad()
def test_sparse_mqa_h64_native_fp8_matches_fp32():
    require_gfx938()
    torch.manual_seed(20261102)
    batch, heads, width = 2, 64, 192
    q = torch.randint(-2, 3, (batch, 1, heads, 128), device="cuda")
    q = q.float().to(torch.float8_e4m3fn).contiguous()
    kv = torch.randint(-2, 3, (4, 64, 1, 128), device="cuda")
    kv = kv.float().to(torch.float8_e4m3fn)
    kv[:, ::5].zero_()
    scales = torch.full((4, 64, 1), 0.5, dtype=torch.float32, device="cuda")
    packed_kv = pack_fp8_kv(kv, scales)
    weights = torch.linspace(0.25, 1.0, heads, device="cuda").repeat(batch, 1)
    context_lens = torch.tensor([129, 150], dtype=torch.int32, device="cuda")
    block_table = torch.tensor(
        [[2, 0, 3], [1, 3, 0]], dtype=torch.int32, device="cuda"
    )

    values, mask = page_mqa_logits_sparse_mask_grouped(
        q,
        packed_kv,
        weights,
        context_lens,
        block_table,
        None,
        width,
        group_size=1,
    )
    reference = fp32_reference(
        q, kv, scales, weights, context_lens, block_table, width
    )
    nonzero = unpack_nonzero_mask(mask, width)
    actual = torch.where(nonzero, values, torch.zeros_like(values))
    expected_nonzero = reference > 0
    torch.cuda.synchronize()

    assert torch.equal(nonzero, expected_nonzero)
    torch.testing.assert_close(actual, reference, atol=2.0e-3, rtol=2.0e-4)


@torch.no_grad()
def test_sparse_mqa_h64_fp8_rejects_grouped_rows():
    require_gfx938()
    q = torch.zeros(
        3, 1, 64, 128, dtype=torch.float8_e4m3fn, device="cuda"
    )
    kv = torch.zeros(3, 64, 1, 132, dtype=torch.uint8, device="cuda")
    weights = torch.ones(3, 64, dtype=torch.float32, device="cuda")
    context_lens = torch.full((3,), 64, dtype=torch.int32, device="cuda")
    block_table = torch.arange(3, dtype=torch.int32, device="cuda").view(3, 1)

    with pytest.raises(ValueError, match="64 heads requires group_size=1"):
        page_mqa_logits_sparse_mask_grouped(
            q,
            kv,
            weights,
            context_lens,
            block_table,
            None,
            64,
            group_size=3,
        )
