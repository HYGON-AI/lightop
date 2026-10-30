"""HCU mega-gate precision, DeepGEMM-compatible API, and graph lifecycle tests."""

import inspect
import subprocess
import sys

import pytest
import torch
from lightop import gemm_ops, op
from lightop.moe import bf16_mega_gate_hcu, moe_fused_gate_sqrtsoftplus_v41_hcu


def inputs(m, seed=123, experts=384, hidden=5120):
    torch.manual_seed(seed)
    x = torch.randn(m, hidden, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(experts, hidden, device="cuda", dtype=torch.bfloat16) * 0.03
    b = torch.randn(experts, device="cuda") * 0.05
    return x, w, b


def torch_gate_reference(
    logits,
    bias,
    topk,
    renormalize=True,
    routed_scaling_factor=1.0,
    apply_routed_scaling_factor_on_output=True,
    mask=None,
):
    """Independent reference retaining LightOp's ordered sum and NaN/ID rules."""
    scores = torch.nn.functional.softplus(logits, threshold=20).sqrt()
    ranking = logits if bias is None else scores + bias
    ranking = torch.where(torch.isnan(ranking), torch.inf, ranking)
    ids = ranking.argsort(dim=-1, descending=True, stable=True)[:, :topk]
    selected = scores.gather(1, ids)
    total = torch.zeros_like(selected[:, :1])
    for rank in range(topk):
        total = total + selected[:, rank : rank + 1]
    if renormalize:
        selected = selected / torch.where(total > 0, total, 1.0)
    if apply_routed_scaling_factor_on_output:
        selected = selected * routed_scaling_factor
    if mask is not None:
        selected = torch.where(mask[:, None], selected, 0.0)
        ids = torch.where(mask[:, None], ids, -1)
    return ids, selected


def baseline(x, w, b, topk=6, renormalize=True, scale=1.0, apply_scale=True, mask=None):
    z = gemm_ops.gemm(x, w, out_dtype=torch.float32)
    if (
        b is not None
        and w.shape[0] in (256, 384)
        and topk <= 16
        and scale > 0
        and mask is None
    ):
        weights, ids = moe_fused_gate_sqrtsoftplus_v41_hcu(
            z,
            b,
            topk=topk,
            renormalize=renormalize,
            routed_scaling_factor=scale,
            apply_routed_scaling_factor_on_output=apply_scale,
        )
        return z, (ids.long(), weights)
    return z, torch_gate_reference(z, b, topk, renormalize, scale, apply_scale, mask)


def public(x, w, b, topk=6, scale=1.0, mask=None, out=None):
    return bf16_mega_gate_hcu(
        x, w, topk, False, 0, scale, 0, bias=b, mask=mask, out=out
    )


def assert_result(actual, expected):
    assert actual[0].dtype == torch.int64
    assert actual[1].dtype == torch.float32
    torch.testing.assert_close(actual[0], expected[0], rtol=0, atol=0)
    torch.testing.assert_close(
        actual[1], expected[1], rtol=1e-5, atol=2e-6, equal_nan=True
    )


def check(x, w, b, topk=6, scale=1.5, mask=None):
    logits, expected = baseline(x, w, b, topk, scale=scale, mask=mask)
    assert_result(public(x, w, b, topk, scale, mask), expected)
    weights, ids, actual_logits = op._bf16_mega_gate_hcu(
        x, w, b, topk, True, scale, True, None, mask
    )
    assert_result((ids, weights), expected)
    torch.testing.assert_close(actual_logits, logits, rtol=0, atol=0, equal_nan=True)
    valid_ids = ids if mask is None else ids[mask]
    if valid_ids.numel():
        assert (valid_ids.sort(dim=-1).values.diff(dim=-1) != 0).all()


@pytest.mark.parametrize(
    "m",
    [
        0,
        1,
        2,
        4,
        8,
        15,
        16,
        17,
        31,
        32,
        33,
        42,
        48,
        84,
        64,
        96,
        128,
        160,
        192,
        256,
        511,
        512,
        513,
    ],
)
def test_shapes(m):
    check(*inputs(m))


@pytest.mark.parametrize(
    "experts,topk",
    [
        (e, k)
        for e in (4, 8, 32, 128, 132, 256, 260, 384, 388, 512)
        for k in range(1, min(e, 32) + 1)
    ],
)
def test_all_topk_independent_reference(experts, topk):
    # Sparse integer products are exact in FP32, independent of GEMM reduction.
    x, w, b = inputs(5, experts=experts, hidden=256)
    x.zero_()
    w.zero_()
    x[:, :4] = torch.arange(20, device=x.device).reshape(5, 4) % 3 - 1
    w[:, :4] = torch.arange(experts * 4, device=x.device).reshape(experts, 4) % 7 - 3
    b.copy_((torch.arange(experts, device=x.device) % 11) * 0.125)
    logits = x.float() @ w.float().T
    expected = torch_gate_reference(logits, b, topk, routed_scaling_factor=1.5)
    assert_result(public(x, w, b, topk, 1.5), expected)
    weights, ids, actual_logits = op._bf16_mega_gate_hcu(x, w, b, topk, True, 1.5, True)
    assert_result((ids, weights), expected)
    torch.testing.assert_close(actual_logits, logits, rtol=0, atol=0)


@pytest.mark.parametrize(
    "experts",
    [
        4,
        8,
        16,
        28,
        32,
        36,
        64,
        72,
        96,
        108,
        124,
        128,
        132,
        144,
        252,
        256,
        260,
        264,
        380,
        384,
        388,
        508,
        512,
    ],
)
@pytest.mark.parametrize("m", [0, 1, 15, 16, 17, 33])
def test_expert_tile_boundaries(experts, m):
    check(*inputs(m, experts=experts, hidden=768), topk=min(experts, 16))


@pytest.mark.parametrize("hidden", [256, 512, 768, 4096, 4352, 5120, 7168, 8192])
@pytest.mark.parametrize("experts", [128, 260, 384, 512])
def test_hidden_sizes(experts, hidden):
    check(*inputs(17, experts=experts, hidden=hidden), topk=8)


@pytest.mark.parametrize("renormalize", [False, True])
@pytest.mark.parametrize("apply_scale", [False, True])
@pytest.mark.parametrize("scale", [0.25, 1.5, 3.0])
def test_framework_route_parameters(renormalize, apply_scale, scale):
    # The internal framework adapter retains sglang's normalization options.
    x, w, b = inputs(17)
    logits, expected = baseline(
        x, w, b, renormalize=renormalize, scale=scale, apply_scale=apply_scale
    )
    weights, ids, actual_logits = op._bf16_mega_gate_hcu(
        x, w, b, 6, renormalize, scale, apply_scale
    )
    assert_result((ids, weights), expected)
    torch.testing.assert_close(actual_logits, logits, rtol=0, atol=0)


@pytest.mark.parametrize("experts", [4, 132, 260, 384, 388, 512])
@pytest.mark.parametrize(
    "kind",
    [
        "ties",
        "nan_bias",
        "infinite_bias",
        "negative_infinite_bias",
        "underflow",
        "threshold",
        "near_ties",
        "subnormal_rank",
        "signed_zero_rank",
        "nan_logits",
        "infinite_logits",
    ],
)
def test_numerical_edges(experts, kind):
    x, w, b = inputs(17, experts=experts, hidden=256)
    x.zero_()
    x[:, 0] = 1
    w.zero_()
    b.zero_()
    if kind == "nan_bias":
        b[::2] = torch.nan
    elif kind == "infinite_bias":
        b[::2] = torch.inf
    elif kind == "negative_infinite_bias":
        b.fill_(-torch.inf)
    elif kind == "nan_logits":
        w[::2, 0] = torch.nan
    elif kind == "infinite_logits":
        w[::2, 0] = torch.inf
        w[1::2, 0] = -torch.inf
    elif kind in ("underflow", "subnormal_rank", "signed_zero_rank"):
        w[:, 0] = -200
        if kind == "subnormal_rank":
            b[-1] = torch.finfo(torch.float32).tiny / 2
            b[-2] = -torch.finfo(torch.float32).tiny / 2
        elif kind == "signed_zero_rank":
            b[::2] = -0.0
    elif kind == "threshold":
        w[:, 0] = torch.linspace(19, 21, experts, device=x.device)
    elif kind == "near_ties":
        b.copy_(torch.arange(experts, device=x.device) * 1e-7)
    check(x, w, b, topk=min(experts, 32))


@pytest.mark.parametrize("m", [0, 1, 17])
@pytest.mark.parametrize("experts", [132, 384])
@pytest.mark.parametrize("with_bias", [False, True])
@pytest.mark.parametrize("mask_mode", ["none", "all", "mixed"])
def test_optional_bias_and_mask(m, experts, with_bias, mask_mode):
    x, w, b = inputs(m, experts=experts)
    if not with_bias:
        b = None
    mask = (
        None
        if mask_mode == "none"
        else torch.zeros(m, device=x.device, dtype=torch.bool)
    )
    if mask_mode == "mixed":
        mask[::2] = True
    check(x, w, b, topk=8, mask=mask)


@pytest.mark.parametrize("scale", [-2.0, 0.0, 0.25, 1.5])
def test_public_scaling(scale):
    check(*inputs(17), scale=scale)


def test_public_signature():
    from lightop import moe

    parameters = inspect.signature(bf16_mega_gate_hcu).parameters
    assert list(parameters) == [
        "x",
        "weight",
        "num_topk",
        "use_shared_as_routed",
        "num_shared_experts",
        "routed_scaling_factor",
        "ep_rank",
        "scoring_func",
        "mask",
        "bias",
        "image_bias",
        "image_token_mask",
        "fix_routing_mask",
        "to_physical_map",
        "logical_count",
        "unmapped_topk_idx",
        "force_random",
        "out",
        "return_logits",
        "renormalize",
        "apply_scale",
    ]
    for name in list(parameters)[:7]:
        assert parameters[name].default is inspect.Parameter.empty
    assert parameters["scoring_func"].default == "sqrtsoftplus"
    for name in list(parameters)[8:18]:
        assert parameters[name].default is None
    for name, default in (
        ("return_logits", False), ("renormalize", True), ("apply_scale", True)
    ):
        assert parameters[name].kind is inspect.Parameter.KEYWORD_ONLY
        assert parameters[name].default is default
    assert not hasattr(moe, "bf16_mega_gate_hcu_with_logits")
    assert not hasattr(moe, "bf16_mega_gate")
    assert not hasattr(moe, "create_bf16_mega_gate_hcu_workspace")
    assert not hasattr(op, "bf16_mega_gate_hcu_capture_id")
    x, w, b = inputs(5)
    # Exercise upstream's required positional arguments plus optional keywords.
    ids, weights = bf16_mega_gate_hcu(
        x=x,
        weight=w,
        num_topk=8,
        use_shared_as_routed=False,
        num_shared_experts=2,
        routed_scaling_factor=1.5,
        ep_rank=3,
        bias=b,
    )
    _, expected = baseline(x, w, b, topk=8, scale=1.5)
    assert_result((ids, weights), expected)
    assert_result(op.bf16_mega_gate_hcu(x, w, 8, False, 0, 1.5, 0, bias=b), expected)


@pytest.mark.parametrize(
    "option",
    [
        "use_shared_as_routed",
        "image_bias",
        "image_token_mask",
        "fix_routing_mask",
        "to_physical_map",
        "logical_count",
        "unmapped_topk_idx",
        "force_random",
    ],
)
def test_unsupported_options_raise(option):
    x, w, b = inputs(1)
    kwargs = dict(
        num_topk=6,
        use_shared_as_routed=False,
        num_shared_experts=0,
        routed_scaling_factor=1.0,
        ep_rank=0,
        bias=b,
    )
    kwargs[option] = (
        True if option == "use_shared_as_routed" else torch.empty(1, device=x.device)
    )
    with pytest.raises(NotImplementedError):
        bf16_mega_gate_hcu(x, w, **kwargs)


@pytest.mark.parametrize(
    "experts,hidden,topk",
    [
        (0, 256, 1),
        (2, 256, 1),
        (130, 256, 8),
        (516, 256, 8),
        (128, 0, 8),
        (128, 128, 8),
        (128, 257, 8),
        (4, 256, 8),
        (128, 256, 0),
        (128, 256, 33),
    ],
)
def test_invalid_shapes(experts, hidden, topk):
    with pytest.raises(RuntimeError):
        public(*inputs(1, experts=experts, hidden=hidden), topk=topk)


def test_validation_and_out():
    x, w, b = inputs(17)
    out = (
        torch.empty(17, 6, device=x.device, dtype=torch.int64),
        torch.empty(17, 6, device=x.device),
    )
    result = public(x, w, b, out=out)
    assert result[0] is out[0] and result[1] is out[1]
    _, expected = baseline(x, w, b)
    assert_result(result, expected)
    for bad_out in [
        (out[0].int(), out[1]),
        (out[0][:1], out[1]),
        (out[0], x.view(torch.float32).flatten()[:102].reshape(17, 6)),
        (out[0], out[1][:, ::2]),
    ]:
        with pytest.raises(RuntimeError):
            public(x, w, b, out=bad_out)
    for args in [
        (x.float(), w, b),
        (x, w.float(), b),
        (x, w, b.bfloat16()),
        (x[:, ::2], w, b),
        (x, w, b.repeat(17, 1)),
        (x, w.cpu(), b),
        (x, w, b.cpu()),
    ]:
        with pytest.raises(RuntimeError):
            public(*args)
    for mask in [
        torch.ones(17, device=x.device),
        torch.ones(1, device=x.device, dtype=torch.bool),
        torch.ones(17, dtype=torch.bool),
        torch.ones(34, device=x.device, dtype=torch.bool)[::2],
    ]:
        with pytest.raises(RuntimeError):
            public(x, w, b, mask=mask)
    for scale in [torch.inf, torch.nan]:
        with pytest.raises(RuntimeError):
            public(x, w, b, scale=scale)
    for kwargs in [dict(ep_rank=-1), dict(scoring_func="softmax")]:
        params = dict(
            num_topk=6,
            use_shared_as_routed=False,
            num_shared_experts=0,
            routed_scaling_factor=1.0,
            ep_rank=0,
            bias=b,
        )
        params.update(kwargs)
        with pytest.raises(RuntimeError):
            bf16_mega_gate_hcu(x, w, **params)
    with pytest.raises(TypeError):
        bf16_mega_gate_hcu(x, w, 6, False, 0, 1.0, 0, bias=b, workspace=None)


@pytest.mark.parametrize("same_stream", [True, False])
def test_graphs_and_streams(same_stream):
    streams = [torch.cuda.Stream() for _ in range(1 if same_stream else 2)]
    states = []
    for index in range(2):
        stream = streams[0 if same_stream else index]
        with torch.cuda.stream(stream):
            x, w, b = inputs(33, seed=index, experts=132, hidden=4352)
            mask = torch.ones(33, device=x.device, dtype=torch.bool)
            out = public(x, w, b, topk=8, mask=mask)
            stream.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                for _ in range(8):
                    public(x, w, b, topk=8, mask=mask, out=out)
            states.append((stream, x, w, b, mask, graph, out))
    for repeat in range(20):
        for stream, x, w, b, mask, graph, out in states:
            with torch.cuda.stream(stream):
                x.normal_()
                b.normal_(std=0.05)
                mask.fill_(True)
                mask[:repeat].fill_(False)
                graph.replay()
        for stream, x, w, b, mask, graph, out in states:
            with torch.cuda.stream(stream):
                _, expected = baseline(x, w, b, topk=8, mask=mask)
                assert_result(out, expected)
    for stream, x, w, b, mask, graph, out in states:
        with torch.cuda.stream(stream):
            big = torch.randn(1025, x.shape[1], device=x.device, dtype=x.dtype)
            public(big, w, b, topk=8)
            graph.replay()
            _, expected = baseline(x, w, b, topk=8, mask=mask)
            assert_result(out, expected)


def test_oversubscribed_publication():
    # More producer CTAs than can reside simultaneously; no global spin barrier.
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        x, w, b = inputs(1025)
        out = public(x, w, b)
        stream.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            for _ in range(8):
                public(x, w, b, out=out)
        for _ in range(20):
            x.normal_()
            b.normal_(std=0.05)
            graph.replay()
            _, expected = baseline(x, w, b)
            assert_result(out, expected)


def test_warmup_and_graph_lifetime():
    # A fresh process ensures counters have never been initialized.
    script = r"""
import gc
import torch
from lightop.moe import bf16_mega_gate_hcu
x = torch.randn(17, 5120, device="cuda", dtype=torch.bfloat16)
w = torch.randn(384, 5120, device="cuda", dtype=torch.bfloat16) * 0.03
b = torch.zeros(384, device="cuda")
def run():
    return bf16_mega_gate_hcu(x, w, 6, False, 0, 1.0, 0, bias=b)
stream = torch.cuda.Stream()
stream.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(stream):
    cold = torch.cuda.CUDAGraph()
    with torch.cuda.graph(cold, stream=stream):
        try:
            run()
        except RuntimeError as error:
            assert "warm up" in str(error), error
        else:
            raise AssertionError("cold capture must require stream warmup")
    del cold
    expected = run()
    stream.synchronize()
    gc.collect()
    before = torch.cuda.memory_allocated()
    for _ in range(5):
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            actual = run()
        for _ in range(10):
            graph.replay()
        stream.synchronize()
        for a, e in zip(actual, expected):
            torch.testing.assert_close(a, e, rtol=0, atol=0)
        del a, e, graph, actual
        gc.collect()
        assert torch.cuda.memory_allocated() == before
"""
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stdout + result.stderr
