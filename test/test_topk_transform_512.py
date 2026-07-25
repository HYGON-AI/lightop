import time

import torch

from lightop.attention import topk_transform_512

MAX_CONTEXT_LEN = 512 * 1024
TOPK = 512
PAGE_SIZE = 64


def topk_transform_512_reference(
    scores: torch.Tensor,
    seq_lens: torch.Tensor,
    page_tables: torch.Tensor,
    page_size: int,
):
    batch_size = scores.size(0)
    positions = torch.arange(scores.size(1), device=scores.device)
    valid_scores = positions.unsqueeze(0) < seq_lens.unsqueeze(1)
    masked_scores = scores.masked_fill(~valid_scores, float("-inf"))
    selected = torch.topk(masked_scores, TOPK, dim=1, sorted=False).indices

    sequential = positions[:TOPK].unsqueeze(0).expand(batch_size, -1)
    short_rows = seq_lens.unsqueeze(1) <= TOPK
    raw_indices = torch.where(short_rows, sequential, selected)
    valid_indices = raw_indices < seq_lens.unsqueeze(1)
    safe_raw_indices = raw_indices.masked_fill(~valid_indices, 0)

    page_slots = torch.div(
        safe_raw_indices, page_size, rounding_mode="floor"
    )
    page_offsets = safe_raw_indices.remainder(page_size).to(torch.int32)
    page_indices = (
        torch.gather(page_tables, 1, page_slots) * page_size + page_offsets
    )
    page_indices.masked_fill_(~valid_indices, -1)
    raw_indices = raw_indices.to(torch.int32)
    raw_indices.masked_fill_(~valid_indices, -1)

    return page_indices, raw_indices


def make_inputs(seq_lens_host, row_padding=17):
    max_seq_len = max(max(seq_lens_host), TOPK)
    batch_size = len(seq_lens_host)
    score_storage = torch.randn(
        batch_size,
        max_seq_len + row_padding,
        dtype=torch.float32,
        device="cuda",
    )
    scores = score_storage[:, :max_seq_len]
    seq_lens = torch.tensor(seq_lens_host, dtype=torch.int32, device="cuda")
    num_pages = (max_seq_len + PAGE_SIZE - 1) // PAGE_SIZE
    page_tables = torch.stack(
        [
            torch.randperm(num_pages, device="cuda", dtype=torch.int64).to(
                torch.int32
            )
            for _ in range(batch_size)
        ]
    )
    return scores, seq_lens, page_tables


def check_case(seq_lens_host, return_raw_indices):
    scores, seq_lens, page_tables = make_inputs(seq_lens_host)
    out_page_indices = torch.empty(
        len(seq_lens_host), TOPK, dtype=torch.int32, device="cuda"
    )
    out_raw_indices = (
        torch.empty_like(out_page_indices) if return_raw_indices else None
    )

    topk_transform_512(
        scores,
        seq_lens,
        page_tables,
        out_page_indices,
        PAGE_SIZE,
        out_raw_indices,
    )
    ref_page_indices, ref_raw_indices = topk_transform_512_reference(
        scores, seq_lens, page_tables, PAGE_SIZE
    )
    torch.cuda.synchronize()

    assert torch.equal(
        out_page_indices.sort(dim=1).values,
        ref_page_indices.sort(dim=1).values,
    )
    if out_raw_indices is not None:
        assert torch.equal(
            out_raw_indices.sort(dim=1).values,
            ref_raw_indices.sort(dim=1).values,
        )

        valid = out_raw_indices >= 0
        safe_raw_indices = out_raw_indices.clamp_min(0)
        page_slots = torch.div(
            safe_raw_indices, PAGE_SIZE, rounding_mode="floor"
        ).to(torch.long)
        page_offsets = safe_raw_indices.remainder(PAGE_SIZE)
        transformed = (
            torch.gather(page_tables, 1, page_slots) * PAGE_SIZE + page_offsets
        )
        transformed.masked_fill_(~valid, -1)
        assert torch.equal(out_page_indices, transformed)


def check_invalid_page_size():
    scores, seq_lens, page_tables = make_inputs([TOPK], row_padding=0)
    out_page_indices = torch.empty(1, TOPK, dtype=torch.int32, device="cuda")
    try:
        topk_transform_512(
            scores,
            seq_lens,
            page_tables,
            out_page_indices,
            63,
        )
    except RuntimeError as error:
        assert "power of two" in str(error)
        return
    raise AssertionError("non-power-of-two page_size should be rejected")


def benchmark(callable_fn, warmup=10, repeat=100):
    for _ in range(warmup):
        callable_fn()
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(repeat):
        callable_fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) * 1e6 / repeat


def run_performance(batch_size=4, seq_len=8192):
    scores, seq_lens, page_tables = make_inputs([seq_len] * batch_size)
    out_page_indices = torch.empty(
        batch_size, TOPK, dtype=torch.int32, device="cuda"
    )
    out_raw_indices = torch.empty_like(out_page_indices)

    def run_lightop():
        topk_transform_512(
            scores,
            seq_lens,
            page_tables,
            out_page_indices,
            PAGE_SIZE,
            out_raw_indices,
        )

    def run_reference():
        topk_transform_512_reference(scores, seq_lens, page_tables, PAGE_SIZE)

    reference_us = benchmark(run_reference)
    lightop_us = benchmark(run_lightop)
    logical_bytes = (
        scores.numel() * scores.element_size()
        + seq_lens.numel() * seq_lens.element_size()
        + batch_size * TOPK * page_tables.element_size()
        + out_page_indices.numel() * out_page_indices.element_size()
        + out_raw_indices.numel() * out_raw_indices.element_size()
    )
    effective_gbps = logical_bytes / (lightop_us * 1e-6) / 1e9

    print("\nPerformance (10 warmup, 100 iterations)")
    print(f"shape: scores=[{batch_size}, {seq_len}], dtype=float32")
    print(f"PyTorch reference: {reference_us:.2f} us")
    print(f"LightOp:           {lightop_us:.2f} us")
    print(f"Speedup:           {reference_us / lightop_us:.2f}x")
    print(f"Effective BW:      {effective_gbps:.2f} GB/s")


def test_topk_transform_512():
    assert torch.cuda.is_available(), "CUDA/ROCm device is required"
    torch.manual_seed(2026)

    seq_lens_cases = [
        [0, 1, 127, 511],
        [512, 513, 1024, 4096],
        [777, 2049, 8192],
        [65536, 131072, MAX_CONTEXT_LEN - 1, MAX_CONTEXT_LEN],
    ]
    for seq_lens_host in seq_lens_cases:
        for return_raw_indices in (False, True):
            check_case(seq_lens_host, return_raw_indices)
            print(
                f"PASS seq_lens={seq_lens_host}, "
                f"return_raw_indices={return_raw_indices}"
            )

    check_invalid_page_size()
    print("PASS invalid page_size check")
    run_performance()


if __name__ == "__main__":
    test_topk_transform_512()
