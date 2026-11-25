"""Correctness tests for the power-of-two lightop.topk fast path."""

from __future__ import annotations

import torch

from lightop import topk


def _check_topk(x: torch.Tensor, k: int, tag: str) -> None:
    values, indices = topk(x, k, dim=-1, largest=True, sorted=False)
    ref_values, _ = torch.topk(x, k, dim=-1, largest=True, sorted=True)
    gathered = torch.gather(x, -1, indices)

    atol = rtol = 1e-4 if x.dtype == torch.float32 else 1e-2
    torch.testing.assert_close(gathered.float(), values.float(), atol=atol, rtol=rtol, msg=tag)
    torch.testing.assert_close(
        torch.sort(values.float(), dim=-1, descending=True).values,
        ref_values.float(),
        atol=atol,
        rtol=rtol,
        msg=f"{tag} sorted values",
    )
    if k in (2, 4, 8):
        assert indices.dtype == torch.int64
    else:
        # k=16 uses the existing topk_opt fallback, whose index dtype is int32.
        assert indices.dtype in (torch.int32, torch.int64)


def test_topk_power2() -> None:
    assert torch.cuda.is_available()
    torch.manual_seed(13)

    # k=2 dedicated, multi-wave, and XL multi-wave dispatches.
    cases = {
        torch.float32: (1024, 16384, 65536),
        torch.float16: (1024, 32768, 131072),
        torch.bfloat16: (1024, 32768, 131072),
    }
    for dtype, dims in cases.items():
        for dim in dims:
            x = torch.randn(4, dim, device="cuda", dtype=dtype)
            _check_topk(x, 2, f"k2 dtype={dtype} dim={dim}")

    # k=4 and k=8 cover the regular and unrolled dispatches.
    for dtype in (torch.float32, torch.float16, torch.bfloat16):
        for k, dim in ((4, 2048), (4, 65536), (8, 2048), (8, 65536)):
            x = torch.randn(4, dim, device="cuda", dtype=dtype)
            _check_topk(x, k, f"k{k} dtype={dtype} dim={dim}")

    # k=16 intentionally uses the existing topk fallback from Python dispatch.
    x = torch.randn(8, 2048, device="cuda", dtype=torch.bfloat16)
    _check_topk(x, 16, "k16 fallback")


if __name__ == "__main__":
    test_topk_power2()
    print("All topk power2 checks passed.")
