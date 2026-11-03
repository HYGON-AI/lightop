"""BF16 MiniMax router projection on gfx938."""

import pytest
import torch


def on_gfx938():
    return (
        torch.version.hip is not None
        and torch.cuda.is_available()
        and torch.cuda.get_device_properties(0).gcnArchName.split(":")[0] == "gfx938"
    )


pytestmark = pytest.mark.skipif(not on_gfx938(), reason="requires gfx938")


def test_router_graph_with_changed_inputs():
    from lightop import _router_gemm

    gen = torch.Generator(device="cuda").manual_seed(42)
    x = torch.randn(64, 6144, dtype=torch.bfloat16, device="cuda", generator=gen)
    w = torch.randn(128, 6144, dtype=torch.bfloat16, device="cuda", generator=gen)
    _router_gemm.forward(x, w)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        logits = _router_gemm.forward(x, w)
    assert logits.shape == (64, 128)
    assert logits.dtype == torch.float32
    for _ in range(2):
        x.normal_(generator=gen)
        graph.replay()
        reference = (x.double() @ w.double().T).float()
        torch.testing.assert_close(logits, reference, atol=3e-4, rtol=1e-5)


def test_router_rejects_an_unsupported_batch():
    from lightop import _router_gemm

    x = torch.empty(63, 6144, device="cuda", dtype=torch.bfloat16)
    w = torch.empty(128, 6144, device="cuda", dtype=torch.bfloat16)
    with pytest.raises(RuntimeError, match="64,6144"):
        _router_gemm.forward(x, w)
