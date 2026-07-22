import importlib.util
import os
import sys
import time
from pathlib import Path
from typing import Optional

import torch

from lightop import fast_kpool_topk_transform_fused, op


POOL_SIZE = 16
TOKEN_TOPK = 2048
GROUP_TOPK = TOKEN_TOPK // POOL_SIZE
OUTPUT_COLS = TOKEN_TOPK + POOL_SIZE - 1


def _load_sglang_torch_fallback():
    default_root = Path(__file__).resolve().parents[2] / "sglang-model"
    sglang_root = Path(os.environ.get("SGLANG_CODE_ROOT", default_root))
    kernels_path = (
        sglang_root
        / "python/sglang/srt/layers/attention/nsa/kpool/kernels.py"
    )
    if not kernels_path.is_file():
        raise FileNotFoundError(
            f"Cannot find the SGLang kpool fallback at {kernels_path}; "
            "set SGLANG_CODE_ROOT to the SGLang repository root."
        )

    spec = importlib.util.spec_from_file_location(
        "_sglang_kpool_kernels_golden", kernels_path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module._torch_topk_pooled_history


SGLANG_TORCH_FALLBACK = _load_sglang_torch_fallback()


def reference(
    score: torch.Tensor,
    lengths: torch.Tensor,
    seq_lens: torch.Tensor,
    page_table: Optional[torch.Tensor] = None,
    topk_indices_offset: Optional[torch.Tensor] = None,
    row_starts: Optional[torch.Tensor] = None,
    out_rows: Optional[int] = None,
    page_table_row_index: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    rows, cols = score.shape
    output_rows = rows if out_rows is None else out_rows
    output = torch.full(
        (output_rows, OUTPUT_COLS),
        -1,
        dtype=torch.int32,
        device=score.device,
    )
    if rows == 0:
        return output

    starts = (
        torch.zeros(rows, dtype=torch.int32, device=score.device)
        if row_starts is None
        else row_starts
    )
    positions = torch.arange(cols, dtype=torch.int32, device=score.device)
    valid = (positions.unsqueeze(0) >= starts.unsqueeze(1)) & (
        positions.unsqueeze(0) < (starts + lengths).unsqueeze(1)
    )
    masked = torch.where(
        valid,
        score,
        torch.full_like(score, float("-inf")),
    )
    k = min(GROUP_TOPK, cols)
    if k:
        group_ids = torch.topk(masked, k=k, dim=1).indices.to(torch.int32)
        group_ids = group_ids - starts.unsqueeze(1)
    else:
        group_ids = torch.empty((rows, 0), dtype=torch.int32, device=score.device)
    if k < GROUP_TOPK:
        group_ids = torch.cat(
            [
                group_ids,
                torch.zeros(
                    (rows, GROUP_TOPK - k),
                    dtype=torch.int32,
                    device=score.device,
                ),
            ],
            dim=1,
        )

    group_rank = torch.arange(GROUP_TOPK, device=score.device)
    group_valid = group_rank.unsqueeze(0) < lengths.clamp(
        min=0, max=GROUP_TOPK
    ).unsqueeze(1)
    token_offset = torch.arange(
        POOL_SIZE, dtype=torch.int32, device=score.device
    )
    raw = (
        group_ids.unsqueeze(-1) * POOL_SIZE + token_offset
    ).reshape(rows, TOKEN_TOPK)
    expanded_valid = (
        group_valid.unsqueeze(-1)
        .expand(-1, -1, POOL_SIZE)
        .reshape(rows, TOKEN_TOPK)
    )

    if page_table is not None:
        table = page_table
        if page_table_row_index is not None:
            table = table.index_select(0, page_table_row_index.to(torch.int64))
        safe = raw.clamp(min=0, max=table.shape[1] - 1).to(torch.int64)
        expanded = torch.gather(table, 1, safe)
    elif topk_indices_offset is not None:
        offsets = topk_indices_offset
        if offsets.ndim == 2:
            offsets = offsets.squeeze(1)
        expanded = raw + offsets.unsqueeze(1)
    else:
        expanded = raw
    expanded = torch.where(
        expanded_valid,
        expanded.to(torch.int32),
        torch.full_like(raw, -1),
    )

    for row in range(rows):
        pool_len = max(0, min(int(lengths[row].item()), cols - int(starts[row].item())))
        history_len = min(pool_len * POOL_SIZE, TOKEN_TOPK)
        output[row, :history_len] = expanded[row, :history_len]

        tail_count = int(seq_lens[row].item()) % POOL_SIZE
        tail_raw = torch.arange(
            pool_len * POOL_SIZE,
            pool_len * POOL_SIZE + tail_count,
            dtype=torch.int32,
            device=score.device,
        )
        if page_table is not None:
            table_row = (
                int(page_table_row_index[row].item())
                if page_table_row_index is not None
                else row
            )
            tail = page_table[table_row].index_select(0, tail_raw.to(torch.int64))
        elif topk_indices_offset is not None:
            offsets = topk_indices_offset.squeeze(1) if topk_indices_offset.ndim == 2 else topk_indices_offset
            tail = tail_raw + offsets[row]
        else:
            tail = tail_raw
        output[row, history_len : history_len + tail_count] = tail.to(torch.int32)

    return output


def sglang_reference(
    score: torch.Tensor,
    lengths: torch.Tensor,
    seq_lens: torch.Tensor,
    page_table: Optional[torch.Tensor] = None,
    topk_indices_offset: Optional[torch.Tensor] = None,
    row_starts: Optional[torch.Tensor] = None,
    out_rows: Optional[int] = None,
    page_table_row_index: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Golden result from SGLang's original Torch kpool fallback."""
    return SGLANG_TORCH_FALLBACK(
        logits=score,
        group_lengths=lengths,
        pool_size=POOL_SIZE,
        topk=TOKEN_TOPK,
        page_table=page_table,
        topk_offsets=topk_indices_offset,
        seq_lens=seq_lens,
        row_starts=row_starts,
        out_rows=out_rows,
        page_table_row_index=page_table_row_index,
    )


def make_scores(rows: int, cols: int, row_padding: int = 17) -> torch.Tensor:
    storage = torch.empty(
        (rows, cols + row_padding), dtype=torch.float32, device="cuda"
    )
    for row in range(rows):
        if cols:
            storage[row, :cols] = torch.randperm(
                cols, dtype=torch.int64, device="cuda"
            ).to(torch.float32)
    return storage[:, :cols]


def make_page_table(rows: int, cols: int) -> torch.Tensor:
    values = torch.arange(rows * cols, dtype=torch.int32, device="cuda")
    return values.reshape(rows, cols)


def assert_equivalent(
    name: str,
    score: torch.Tensor,
    lengths: torch.Tensor,
    seq_lens: torch.Tensor,
    page_table: Optional[torch.Tensor] = None,
    topk_indices_offset: Optional[torch.Tensor] = None,
    row_starts: Optional[torch.Tensor] = None,
    out_rows: Optional[int] = None,
    page_table_row_index: Optional[torch.Tensor] = None,
) -> None:
    actual = fast_kpool_topk_transform_fused(
        score=score,
        lengths=lengths,
        pool_size=POOL_SIZE,
        topk=TOKEN_TOPK,
        page_table=page_table,
        topk_indices_offset=topk_indices_offset,
        row_starts=row_starts,
        seq_lens=seq_lens,
        out_rows=out_rows,
        page_table_row_index=page_table_row_index,
    )
    expected = sglang_reference(
        score=score,
        lengths=lengths,
        seq_lens=seq_lens,
        page_table=page_table,
        topk_indices_offset=topk_indices_offset,
        row_starts=row_starts,
        out_rows=out_rows,
        page_table_row_index=page_table_row_index,
    )
    independent = reference(
        score=score,
        lengths=lengths,
        seq_lens=seq_lens,
        page_table=page_table,
        topk_indices_offset=topk_indices_offset,
        row_starts=row_starts,
        out_rows=out_rows,
        page_table_row_index=page_table_row_index,
    )
    torch.cuda.synchronize()

    assert torch.equal(
        independent, expected
    ), f"{name}: independent/SGLang golden mismatch"

    expected_rows = score.shape[0] if out_rows is None else out_rows
    assert actual.shape == (expected_rows, OUTPUT_COLS), name
    assert actual.dtype == torch.int32, name

    rows = score.shape[0]
    starts = (
        torch.zeros(rows, dtype=torch.int32, device=score.device)
        if row_starts is None
        else row_starts
    )
    for row in range(rows):
        pool_len = max(
            0,
            min(
                int(lengths[row].item()),
                score.shape[1] - int(starts[row].item()),
            ),
        )
        history_len = min(pool_len * POOL_SIZE, TOKEN_TOPK)
        assert torch.equal(
            actual[row, :history_len].sort().values,
            expected[row, :history_len].sort().values,
        ), f"{name}: history mismatch in row {row}"
        assert torch.equal(
            actual[row, history_len:],
            expected[row, history_len:],
        ), f"{name}: tail/padding mismatch in row {row}"

    if actual.shape[0] > rows:
        assert torch.all(actual[rows:] == -1), f"{name}: padded rows are not -1"
    print(f"PASS {name}: score={tuple(score.shape)} output={tuple(actual.shape)}")


def run_correctness() -> None:
    torch.manual_seed(2026)

    lengths = torch.tensor([5496, 300, 128, 0], dtype=torch.int32, device="cuda")
    seq_lens = lengths * POOL_SIZE + torch.tensor(
        [7, 15, 3, 9], dtype=torch.int32, device="cuda"
    )
    assert_equivalent(
        "decode_paged_and_out_rows",
        make_scores(4, 5632),
        lengths,
        seq_lens,
        page_table=make_page_table(4, int(seq_lens.max().item()) + 1),
        out_rows=8,
    )

    row_starts = torch.tensor([5, 400, 0, 900], dtype=torch.int32, device="cuda")
    lengths = torch.tensor([500, 129, 0, 124], dtype=torch.int32, device="cuda")
    seq_lens = lengths * POOL_SIZE + torch.tensor(
        [1, 15, 7, 0], dtype=torch.int32, device="cuda"
    )
    assert_equivalent(
        "ragged_offsets_and_row_starts",
        make_scores(4, 1024),
        lengths,
        seq_lens,
        topk_indices_offset=torch.tensor(
            [[1000], [2000], [3000], [4000]],
            dtype=torch.int32,
            device="cuda",
        ),
        row_starts=row_starts,
    )

    row_starts = torch.tensor([0, 20, 100, 0], dtype=torch.int32, device="cuda")
    lengths = torch.tensor([300, 200, 129, 10], dtype=torch.int32, device="cuda")
    seq_lens = lengths * POOL_SIZE + torch.tensor(
        [2, 4, 6, 8], dtype=torch.int32, device="cuda"
    )
    assert_equivalent(
        "paged_row_index_and_row_starts",
        make_scores(4, 512),
        lengths,
        seq_lens,
        page_table=make_page_table(6, int(seq_lens.max().item()) + 1),
        row_starts=row_starts,
        page_table_row_index=torch.tensor(
            [4, 1, 4, 0], dtype=torch.int32, device="cuda"
        ),
        out_rows=6,
    )

    lengths = torch.tensor([250, 130, 7], dtype=torch.int32, device="cuda")
    seq_lens = lengths * POOL_SIZE + torch.tensor(
        [5, 9, 15], dtype=torch.int32, device="cuda"
    )
    expanded_table = make_page_table(
        1, int(seq_lens.max().item()) + 1
    ).expand(3, -1)
    assert expanded_table.stride(0) == 0
    assert_equivalent(
        "paged_stride_zero",
        make_scores(3, 300),
        lengths,
        seq_lens,
        page_table=expanded_table,
    )

    lengths = torch.tensor(
        [0, 1, 127, 128, 129, 300], dtype=torch.int32, device="cuda"
    )
    seq_lens = lengths * POOL_SIZE + torch.tensor(
        [0, 1, 15, 3, 7, 11], dtype=torch.int32, device="cuda"
    )
    assert_equivalent(
        "topk_cutoff_and_1d_offsets",
        make_scores(6, 320),
        lengths,
        seq_lens,
        topk_indices_offset=torch.arange(
            6, dtype=torch.int32, device="cuda"
        )
        * 10000,
    )

    assert_equivalent(
        "empty_batch_with_padded_rows",
        torch.empty((0, 0), dtype=torch.float32, device="cuda"),
        torch.empty((0,), dtype=torch.int32, device="cuda"),
        torch.empty((0,), dtype=torch.int32, device="cuda"),
        out_rows=3,
    )

    lengths = torch.zeros(3, dtype=torch.int32, device="cuda")
    seq_lens = torch.tensor([0, 1, 15], dtype=torch.int32, device="cuda")
    assert_equivalent(
        "zero_width_tail_only",
        torch.empty((3, 0), dtype=torch.float32, device="cuda"),
        lengths,
        seq_lens,
        page_table=make_page_table(3, 16),
    )

    lengths = torch.tensor([300, 3], dtype=torch.int32, device="cuda")
    seq_lens = lengths * POOL_SIZE + torch.tensor(
        [11, 2], dtype=torch.int32, device="cuda"
    )
    assert_equivalent(
        "logical_indices_no_mapping",
        make_scores(2, 320),
        lengths,
        seq_lens,
    )

    invalid_row_output = fast_kpool_topk_transform_fused(
        score=make_scores(2, 160),
        lengths=torch.tensor([129, 3], dtype=torch.int32, device="cuda"),
        pool_size=POOL_SIZE,
        topk=TOKEN_TOPK,
        page_table=make_page_table(2, 2080),
        seq_lens=torch.tensor([2065, 50], dtype=torch.int32, device="cuda"),
        page_table_row_index=torch.tensor(
            [-1, 2], dtype=torch.int32, device="cuda"
        ),
    )
    torch.cuda.synchronize()
    assert invalid_row_output.shape == (2, OUTPUT_COLS)
    assert invalid_row_output.dtype == torch.int32
    assert torch.all(invalid_row_output == -1)
    print(
        "PASS invalid_page_table_row_index: "
        f"output={tuple(invalid_row_output.shape)}"
    )


def run_hip_graph_replay() -> None:
    """Replay changing inputs and guard the exact 2063-column output extent."""
    rows = 4
    out_rows = 7
    score = make_scores(rows, 512)
    page_table = make_page_table(6, 8192)
    lengths = torch.tensor(
        [300, 129, 16, 0], dtype=torch.int32, device="cuda"
    )
    seq_lens = torch.tensor(
        [4807, 2073, 271, 7], dtype=torch.int32, device="cuda"
    )
    row_starts = torch.tensor(
        [3, 50, 100, 0], dtype=torch.int32, device="cuda"
    )
    page_table_row_index = torch.tensor(
        [5, 2, 5, 0], dtype=torch.int32, device="cuda"
    )

    # Offset the output by one int32 past a 16-byte boundary. This exercises
    # vectorized_write's unaligned prefix/remainder paths while keeping one
    # contiguous [out_rows, 2063] view between two canary regions.
    guard_words = 4097
    output_words = out_rows * OUTPUT_COLS
    storage = torch.empty(
        guard_words + output_words + guard_words,
        dtype=torch.int32,
        device="cuda",
    )
    left_canary = 0x13579BDF
    right_canary = 0x2468ACE
    storage[:guard_words].fill_(left_canary)
    storage[-guard_words:].fill_(right_canary)
    guarded_output = storage[
        guard_words : guard_words + output_words
    ].view(out_rows, OUTPUT_COLS)
    guarded_output.fill_(-777)
    assert guarded_output.is_contiguous()

    def native_run() -> None:
        op.fast_kpool_topk_transform_interface(
            score,
            lengths,
            guarded_output,
            POOL_SIZE,
            seq_lens,
            page_table,
            None,
            row_starts,
            page_table_row_index,
        )

    # Warm both the public wrapper and the native entry before capture so no
    # extension initialization or allocator setup occurs in the graph.
    warmup_output = fast_kpool_topk_transform_fused(
        score,
        lengths,
        POOL_SIZE,
        TOKEN_TOPK,
        page_table=page_table,
        row_starts=row_starts,
        seq_lens=seq_lens,
        out_rows=out_rows,
        page_table_row_index=page_table_row_index,
    )
    native_run()
    torch.cuda.synchronize()
    del warmup_output

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_output = fast_kpool_topk_transform_fused(
            score,
            lengths,
            POOL_SIZE,
            TOKEN_TOPK,
            page_table=page_table,
            row_starts=row_starts,
            seq_lens=seq_lens,
            out_rows=out_rows,
            page_table_row_index=page_table_row_index,
        )
        native_run()

    replay_cases = [
        ([300, 129, 16, 0], [4807, 2073, 271, 7], [3, 50, 100, 0], [5, 2, 5, 0]),
        ([0, 1, 127, 128], [7, 17, 2047, 2051], [0, 11, 70, 200], [1, 4, 0, 3]),
        ([129, 250, 3, 300], [2071, 4009, 50, 4811], [200, 0, 400, 100], [4, 4, 2, 1]),
        ([384, 64, 128, 129], [6159, 1024, 2055, 2073], [128, 448, 0, 300], [0, 5, 3, 2]),
    ]

    for replay_id, (lens, seqs, starts, page_rows) in enumerate(replay_cases):
        lengths.copy_(torch.tensor(lens, dtype=torch.int32, device="cuda"))
        seq_lens.copy_(torch.tensor(seqs, dtype=torch.int32, device="cuda"))
        row_starts.copy_(torch.tensor(starts, dtype=torch.int32, device="cuda"))
        page_table_row_index.copy_(
            torch.tensor(page_rows, dtype=torch.int32, device="cuda")
        )
        graph.replay()
        torch.cuda.synchronize()

        expected = sglang_reference(
            score=score,
            lengths=lengths,
            seq_lens=seq_lens,
            page_table=page_table,
            row_starts=row_starts,
            out_rows=out_rows,
            page_table_row_index=page_table_row_index,
        )
        torch.cuda.synchronize()

        for name, actual in (
            ("wrapper", graph_output),
            ("guarded_native", guarded_output),
        ):
            for row in range(rows):
                pool_len = max(
                    0,
                    min(
                        int(lengths[row].item()),
                        score.shape[1] - int(row_starts[row].item()),
                    ),
                )
                history_len = min(pool_len * POOL_SIZE, TOKEN_TOPK)
                assert torch.equal(
                    actual[row, :history_len].sort().values,
                    expected[row, :history_len].sort().values,
                ), f"graph replay {replay_id} {name}: history row {row}"
                assert torch.equal(
                    actual[row, history_len:], expected[row, history_len:]
                ), f"graph replay {replay_id} {name}: tail row {row}"
            assert torch.all(actual[rows:] == -1), (
                f"graph replay {replay_id} {name}: padded rows"
            )

        assert torch.all(storage[:guard_words] == left_canary), (
            f"graph replay {replay_id}: left canary corrupted"
        )
        assert torch.all(storage[-guard_words:] == right_canary), (
            f"graph replay {replay_id}: right canary corrupted"
        )

    print(
        "PASS hip_graph_replay_and_canary: "
        f"replays={len(replay_cases)} output={tuple(guarded_output.shape)}"
    )


def run_microbenchmark() -> None:
    score = make_scores(8, 5632)
    lengths = torch.full((8,), 5496, dtype=torch.int32, device="cuda")
    seq_lens = lengths * POOL_SIZE + 7
    page_table = make_page_table(8, int(seq_lens.max().item()) + 1)

    def run() -> None:
        fast_kpool_topk_transform_fused(
            score,
            lengths,
            POOL_SIZE,
            TOKEN_TOPK,
            page_table=page_table,
            seq_lens=seq_lens,
        )

    for _ in range(20):
        run()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(200):
        run()
    torch.cuda.synchronize()
    elapsed_us = (time.perf_counter() - start) * 1e6 / 200
    print(f"LightOp fused kpool topk: {elapsed_us:.2f} us/call")


if __name__ == "__main__":
    assert torch.cuda.is_available(), "ROCm/CUDA device is required"
    run_correctness()
    run_hip_graph_replay()
    run_microbenchmark()
