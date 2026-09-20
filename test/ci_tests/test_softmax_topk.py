"""Regression tests for top-k softmax with non-finite routing logits."""

import pytest
import torch

from lightop import op


NUM_EXPERTS = 512
TOPK = 10


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA/ROCm device is required"
)
@pytest.mark.parametrize("renormalize", [False, True])
def test_topk_softmax_sanitizes_nonfinite_rows(renormalize: bool) -> None:
    """Invalid rows produce safe routes without affecting valid candidates."""
    gating_output = torch.empty(
        (5, NUM_EXPERTS), device="cuda", dtype=torch.bfloat16
    )
    gating_output[0].fill_(float("nan"))
    gating_output[1].fill_(float("-inf"))
    gating_output[2].fill_(float("inf"))
    gating_output[3].fill_(float("nan"))
    gating_output[3, :5] = torch.tensor(
        [5.0, 4.0, 3.0, 2.0, 1.0],
        device="cuda",
        dtype=torch.bfloat16,
    )
    gating_output[4].fill_(-100.0)
    gating_output[4, :TOPK] = torch.arange(
        TOPK, 0, -1, device="cuda", dtype=torch.bfloat16
    )

    topk_weights = torch.empty(
        (gating_output.size(0), TOPK), device="cuda", dtype=torch.float32
    )
    topk_indices = torch.empty(
        (gating_output.size(0), TOPK), device="cuda", dtype=torch.int32
    )
    source_rows = torch.empty_like(topk_indices)

    op.topk_softmax(
        topk_weights,
        topk_indices,
        source_rows,
        gating_output,
        renormalize,
    )
    torch.cuda.synchronize()

    assert torch.all(torch.isfinite(topk_weights))
    assert torch.all((topk_indices >= 0) & (topk_indices < NUM_EXPERTS))

    # Rows without any finite candidate must use harmless, zero-weight routes.
    torch.testing.assert_close(
        topk_weights[:3], torch.zeros_like(topk_weights[:3])
    )

    # A partially valid row must preserve every finite candidate and fill the
    # remaining routes with zero weights instead of an out-of-range sentinel.
    valid_routes = topk_weights[3] > 0
    assert valid_routes.sum().item() == 5
    torch.testing.assert_close(
        topk_indices[3][valid_routes].sort().values,
        torch.arange(5, device="cuda", dtype=torch.int32),
    )
    torch.testing.assert_close(
        topk_weights[3].sum(),
        torch.tensor(1.0, device="cuda"),
        rtol=2e-6,
        atol=2e-6,
    )

    # The bookkeeping output remains valid for every row, including fallback
    # routes generated for non-finite input.
    rows = torch.arange(gating_output.size(0), device="cuda")[:, None]
    ranks = torch.arange(TOPK, device="cuda")[None, :]
    expected_source_rows = ranks * gating_output.size(0) + rows
    torch.testing.assert_close(
        source_rows, expected_source_rows.to(source_rows.dtype)
    )

    # A normal row must still select the same experts as torch.topk.
    expected_indices = torch.topk(gating_output[4].float(), TOPK).indices
    torch.testing.assert_close(
        topk_indices[4].to(torch.int64).sort().values,
        expected_indices.sort().values,
    )
