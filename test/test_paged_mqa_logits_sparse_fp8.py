"""gfx938 native-FP8 grouped sparse Page-MQA correctness checks."""

import pytest
import torch

from lightop.attention import page_mqa_logits_sparse_mask_grouped


def _is_gfx938_64cu() -> bool:
    if not torch.cuda.is_available():
        return False
    props = torch.cuda.get_device_properties(0)
    arch = getattr(props, "gcnArchName", props.name).split(":")[0]
    return arch == "gfx938" and props.multi_processor_count == 64


def _pack_fp8_kv(kv: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Build the [page, 64, 1, 128+4] storage consumed by Page MQA."""
    assert kv.dtype == torch.float32
    scales = kv.abs().amax(dim=-1).clamp_min(1.0e-4) / 448.0
    kv_fp8 = (kv / scales.unsqueeze(-1)).to(torch.float8_e4m3fn)
    # The logical [64, 1, 132] view deliberately preserves the historical
    # ABI, while its page bytes are planar: 64*128 FP8 K bytes followed by 64
    # FP32 scales.  Do not fill it as token-interleaved [128 + 4] records.
    pages, page_size, _, head_dim = kv.shape
    packed_bytes = torch.empty(
        (pages, page_size * (head_dim + 4)),
        device=kv.device,
        dtype=torch.uint8,
    )
    packed_bytes[:, :page_size * head_dim].copy_(
        kv_fp8.reshape(pages, page_size * head_dim).view(torch.uint8)
    )
    packed_bytes[:, page_size * head_dim:].view(torch.float32).copy_(
        scales.reshape(pages, page_size)
    )
    packed = packed_bytes.view(pages, page_size, 1, head_dim + 4)
    return packed, kv_fp8.to(torch.float32) * scales.unsqueeze(-1)


def _unpack_mask(mask: torch.Tensor, width: int) -> torch.Tensor:
    bit_offsets = torch.arange(16, device=mask.device, dtype=torch.int32)
    words = mask.to(torch.int32).unsqueeze(-1) & 0xFFFF
    return ((words >> bit_offsets) & 1).to(torch.bool).flatten(1)[:, :width]


def _reference(
        q: torch.Tensor,
        kv: torch.Tensor,
        weights: torch.Tensor,
        context_lens: torch.Tensor,
        block_table: torch.Tensor,
        max_context_len: int) -> torch.Tensor:
    result = torch.zeros(
        (q.shape[0], max_context_len), device=q.device, dtype=torch.float32)
    q_fp32 = q.to(torch.float32).squeeze(1)
    for row, context_len in enumerate(context_lens.cpu().tolist()):
        page_ids = block_table[row, : (context_len + 63) // 64].long()
        keys = kv[page_ids].reshape(-1, 128)[:context_len]
        dots = q_fp32[row] @ keys.transpose(0, 1)
        result[row, :context_len] = (
            torch.relu(dots) * weights[row].unsqueeze(1)
        ).sum(dim=0)
    return result


@pytest.mark.parametrize("group_size", [1, 3, 4, 5])
def test_grouped_sparse_page_mqa_native_fp8(group_size: int) -> None:
    if not _is_gfx938_64cu():
        pytest.skip("native grouped FP8 Page MQA is gfx938/64CU-only")

    torch.manual_seed(20260817 + group_size)
    requests = 2
    rows = requests * group_size
    max_context_len = 128
    q = (torch.randn(
        (rows, 1, 32, 128), device="cuda", dtype=torch.float32
    ) * 0.125).to(torch.float8_e4m3fn)
    weights = torch.randn((rows, 32), device="cuda", dtype=torch.float32)

    # Give each request a distinct physical-page pair.  The first page is
    # exactly zero so the sparse producer must clear all its mask bits while
    # retaining the nonzero second-page scores.
    kv = torch.randn((requests * 2, 64, 1, 128), device="cuda") * 0.125
    kv[::2].zero_()
    packed_kv, decoded_kv = _pack_fp8_kv(kv)
    block_table = torch.empty(
        (rows, max_context_len // 64), device="cuda", dtype=torch.int32
    )
    for request in range(requests):
        block_table[request * group_size:(request + 1) * group_size] = (
            torch.tensor(
                [request * 2, request * 2 + 1],
                device="cuda",
                dtype=torch.int32,
            )
        )
    context_lens = torch.tensor(
        [64 + (row * 11) % 64 for row in range(rows)],
        device="cuda",
        dtype=torch.int32,
    )

    values, sparse_mask = page_mqa_logits_sparse_mask_grouped(
        q,
        packed_kv,
        weights,
        context_lens,
        block_table,
        schedule_meta=None,
        max_context_len=max_context_len,
        group_size=group_size,
    )
    expected = _reference(
        q, decoded_kv, weights, context_lens, block_table, max_context_len
    )
    actual_mask = _unpack_mask(sparse_mask, max_context_len)
    expected_mask = expected != 0
    positions = torch.arange(max_context_len, device="cuda").unsqueeze(0)
    expected_mask &= positions < context_lens.unsqueeze(1)

    torch.cuda.synchronize()
    assert torch.equal(actual_mask, expected_mask)
    assert torch.allclose(
        values[actual_mask],
        expected[actual_mask],
        rtol=2.0e-3,
        atol=2.0e-3,
    )


def test_group4_persistent_pipeline_crosses_256_token_tile() -> None:
    if not _is_gfx938_64cu():
        pytest.skip("native grouped FP8 Page MQA is gfx938/64CU-only")

    torch.manual_seed(20260821)
    requests, group_size = 2, 4
    rows = requests * group_size
    max_context_len = 512
    pages_per_request = max_context_len // 64
    q = (torch.randn(
        (rows, 1, 32, 128), device="cuda", dtype=torch.float32
    ) * 0.125).to(torch.float8_e4m3fn)
    weights = torch.randn((rows, 32), device="cuda", dtype=torch.float32)
    kv = torch.randn(
        (requests * pages_per_request, 64, 1, 128),
        device="cuda",
        dtype=torch.float32,
    ) * 0.125
    packed_kv, decoded_kv = _pack_fp8_kv(kv)

    request_pages = torch.arange(
        requests * pages_per_request, device="cuda", dtype=torch.int32
    ).view(requests, pages_per_request)
    block_table = request_pages.repeat_interleave(
        group_size, dim=0
    ).contiguous()
    context_lens = torch.tensor(
        [257, 319, 383, 447, 271, 335, 399, 511],
        device="cuda",
        dtype=torch.int32,
    )

    values, sparse_mask = page_mqa_logits_sparse_mask_grouped(
        q,
        packed_kv,
        weights,
        context_lens,
        block_table,
        schedule_meta=None,
        max_context_len=max_context_len,
        persistent_ctas=1,
        group_size=group_size,
    )
    expected = _reference(
        q, decoded_kv, weights, context_lens, block_table, max_context_len
    )
    actual_mask = _unpack_mask(sparse_mask, max_context_len)
    positions = torch.arange(max_context_len, device="cuda").unsqueeze(0)
    expected_mask = (expected != 0) & (
        positions < context_lens.unsqueeze(1)
    )

    torch.cuda.synchronize()
    assert torch.equal(actual_mask, expected_mask)
    assert torch.allclose(
        values[actual_mask],
        expected[actual_mask],
        rtol=2.0e-3,
        atol=2.0e-3,
    )


def test_grouped_sparse_page_mqa_fp8_nan_scale_sets_mask_bits() -> None:
    if not _is_gfx938_64cu():
        pytest.skip("native grouped FP8 Page MQA is gfx938/64CU-only")

    q = torch.zeros(
        (1, 1, 32, 128), device="cuda", dtype=torch.float8_e4m3fn
    )
    weights = torch.ones((1, 32), device="cuda", dtype=torch.float32)
    packed_kv, _ = _pack_fp8_kv(
        torch.zeros((1, 64, 1, 128), device="cuda", dtype=torch.float32)
    )
    # The cache ABI is planar within each page, so scales begin at byte 8192.
    packed_kv.reshape(1, -1)[:, 64 * 128:].view(torch.float32).fill_(
        float("nan")
    )
    context_lens = torch.tensor([16], device="cuda", dtype=torch.int32)
    block_table = torch.zeros((1, 1), device="cuda", dtype=torch.int32)

    values, sparse_mask = page_mqa_logits_sparse_mask_grouped(
        q,
        packed_kv,
        weights,
        context_lens,
        block_table,
        schedule_meta=None,
        max_context_len=64,
        group_size=1,
    )
    actual_mask = _unpack_mask(sparse_mask, 64)
    torch.cuda.synchronize()

    assert torch.equal(
        actual_mask,
        torch.tensor(
            [[True] * 16 + [False] * 48], device="cuda", dtype=torch.bool
        ),
    )
    assert torch.isnan(values[0, :16]).all()


def test_grouped_sparse_page_mqa_fp8_negative_zero_scale_sets_mask_bits() -> None:
    if not _is_gfx938_64cu():
        pytest.skip("native grouped FP8 Page MQA is gfx938/64CU-only")

    q = torch.zeros(
        (1, 1, 32, 128), device="cuda", dtype=torch.float8_e4m3fn
    )
    weights = torch.ones((1, 32), device="cuda", dtype=torch.float32)
    packed_kv, _ = _pack_fp8_kv(
        torch.zeros((1, 64, 1, 128), device="cuda", dtype=torch.float32)
    )
    # A raw all-zero KV tile is only skippable for a non-negative scale:
    # +0.0 * -0.0 is -0.0, which must remain visible to sparse TopK.
    packed_kv.reshape(1, -1)[:, 64 * 128:].view(torch.float32).fill_(-0.0)
    context_lens = torch.tensor([16], device="cuda", dtype=torch.int32)
    block_table = torch.zeros((1, 1), device="cuda", dtype=torch.int32)

    values, sparse_mask = page_mqa_logits_sparse_mask_grouped(
        q,
        packed_kv,
        weights,
        context_lens,
        block_table,
        schedule_meta=None,
        max_context_len=64,
        group_size=1,
    )
    actual_mask = _unpack_mask(sparse_mask, 64)
    torch.cuda.synchronize()

    assert torch.equal(
        actual_mask,
        torch.tensor(
            [[True] * 16 + [False] * 48], device="cuda", dtype=torch.bool
        ),
    )
    assert torch.signbit(values[0, :16]).all()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
