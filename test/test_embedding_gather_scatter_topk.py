"""Accuracy: lightop embedding / gather / scatter / topk(power2) vs torch."""

from __future__ import annotations

import torch

from lightop import embedding, gather, scatter, topk


def _require_cuda() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA/HIP device required")


def _assert_close(
    a: torch.Tensor,
    b: torch.Tensor,
    atol: float | None = None,
    rtol: float | None = None,
) -> None:
    if atol is None or rtol is None:
        if a.dtype == torch.float32:
            atol = rtol = 1e-4
        else:
            atol = rtol = 1e-2
    assert torch.allclose(a.float(), b.float(), atol=atol, rtol=rtol), (
        f"max abs={(a.float() - b.float()).abs().max().item()} dtype={a.dtype}"
    )


def test_embedding_gather_scatter() -> None:
    _require_cuda()
    torch.manual_seed(3)
    for dtype in (torch.float16, torch.bfloat16, torch.float32):
        weight = torch.randn(1024, 768, device="cuda", dtype=dtype)
        index = torch.randint(0, 1024, (128,), device="cuda", dtype=torch.int64)
        dst = torch.empty(128, 768, device="cuda", dtype=dtype)
        embedding(dst, index, weight)
        _assert_close(dst, weight[index])

        # gather special / row-broadcast: index[:,0] selects whole rows
        src = torch.randn(256, 1024, device="cuda", dtype=dtype)  # 1024 % 8 == 0
        idx2 = torch.randint(0, 256, (32, 1), device="cuda", dtype=torch.int64).expand(32, 1024).contiguous()
        gdst = torch.empty(32, 1024, device="cuda", dtype=dtype)
        gather(gdst, idx2, src, special=True)
        _assert_close(gdst, src[idx2[:, 0]])

        sdst = torch.zeros(64, 1024, device="cuda", dtype=dtype)
        ssrc = torch.randn(16, 1024, device="cuda", dtype=dtype)
        sidx = torch.arange(16, device="cuda", dtype=torch.int64).view(16, 1).expand(16, 1024).contiguous()
        scatter(sdst, sidx, ssrc, special=True)
        _assert_close(sdst[:16], ssrc)

def test_topk_power2() -> None:
    _require_cuda()
    torch.manual_seed(4)
    for dtype in (torch.float16, torch.bfloat16, torch.float32):
        for k in (2, 4, 8, 16):
            # Use distinct-ish values to reduce tie ambiguity vs torch.topk
            x = torch.randn(32, 2048, device="cuda", dtype=dtype)
            values, indices = topk(x, k, dim=-1, largest=True, sorted=False)
            ref_v, _ = torch.topk(x, k, dim=-1, largest=True, sorted=True)
            # gathered values must match selected indices
            _assert_close(values, torch.gather(x, 1, indices))
            # top-k multiset of values should match torch (sort both)
            _assert_close(torch.sort(values, dim=-1, descending=True).values, ref_v)


if __name__ == "__main__":
    test_embedding_gather_scatter()
    test_topk_power2()
    print("PASS")
