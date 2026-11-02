# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Torch correctness and device-only three-way mHC statistics benchmark.

python test/ci_tests/test_hc_mix_stats.py --output /path/to/results
Requires gfx928, gfx936 or gfx938. Real model weights are optional via --model. --quick checks a
small random matrix; default covers seven weights when --model is supplied.
CI correctness: python -m pytest -v test/ci_tests/test_hc_mix_stats.py
"""

from lightop.gemm_ops import hc_mix_stats as categorized_hc_mix_stats
import argparse
import importlib.util
import json
import os
from pathlib import Path
import statistics

import pytest
import torch
import torch.nn.functional as F
import lightop

SUPPORTED_ARCHS = ("gfx928", "gfx936", "gfx938")
K, N, S = 20480, 24, 80
MS = [0, 1, 2, 10, 12, 15, 16, 17, 48, 128, 256, 512, 1024, 2048]


def reference(x, w, eps=1e-6):
    xf = x.float()
    return F.linear(xf, w) * torch.rsqrt(xf.square().mean(-1, keepdim=True) + eps)


def native(x, w, eps=1e-6):
    return categorized_hc_mix_stats(x, w, eps)


def metric(a, b):
    assert a.shape == b.shape, ("shape mismatch", a.shape, b.shape)
    a, b = a.detach().double(), b.detach().double().to(a.device)
    d = (a - b).abs()
    error_norm, reference_norm = d.norm().item(), b.norm().item()
    # Do not clamp small nonzero references: that hides relative error.
    relative_l2 = (
        error_norm / reference_norm if reference_norm != 0
        else (0.0 if error_norm == 0 else float("inf"))
    )
    nonzero = b != 0
    relative = d[nonzero] / b[nonzero].abs()
    finite_output = bool(torch.isfinite(a).all())
    finite_reference = bool(torch.isfinite(b).all())
    return dict(
        max_abs=d.max().item() if d.numel() else 0,
        relative_l2=relative_l2,
        max_relative_nonzero=relative.max().item() if relative.numel() else 0,
        relative_gt_1pct=int((relative > 0.01).sum()),
        nonzero_reference=int(nonzero.sum()),
        zero_reference_mismatches=int(((b == 0) & (d != 0)).sum()),
        violations=int((d > 1e-4 + 1e-4 * b.abs()).sum()),
        finite=finite_output and finite_reference,
        finite_output=finite_output,
        finite_reference=finite_reference,
    )


def check(a, b):
    result = metric(a, b)
    assert result["finite"] and result["violations"] == 0, result
    return result


def test_accuracy_checker():
    # NaN/Inf references previously bypassed the comparison (NaN > tol is false).
    for bad in (float("nan"), float("inf"), -float("inf")):
        with pytest.raises(AssertionError):
            check(torch.ones(1), torch.tensor([bad]))
        with pytest.raises(AssertionError):
            check(torch.tensor([bad]), torch.ones(1))
    with pytest.raises(AssertionError, match="shape mismatch"):
        check(torch.ones(1), torch.ones(2))
    # Keep the tiny reference representable when another runtime enables CPU
    # flush-to-zero: 1e-40 is subnormal in float32, but normal in float64.
    small = metric(
        torch.zeros(1, dtype=torch.float64),
        torch.tensor([1e-40], dtype=torch.float64),
    )
    assert small["relative_l2"] == 1.0
    assert small["max_relative_nonzero"] == 1.0
    assert small["relative_gt_1pct"] == 1
    assert metric(torch.zeros(1), torch.zeros(1))["relative_l2"] == 0
    zero = metric(torch.ones(1), torch.zeros(1))
    assert zero["relative_l2"] == float("inf")
    assert zero["zero_reference_mismatches"] == 1


def all_weights(model):
    torch.manual_seed(42)
    yield "random", torch.randn(N, K) / (K**0.5), None, None
    if model:
        from safetensors import safe_open

        index = json.loads((model / "model.safetensors.index.json").read_text())[
            "weight_map"
        ]
        for layer in [0, 20, 39]:
            for kind in ["attn", "ffn"]:
                name = f"layers.{layer}.hc_{kind}_fn"
                with safe_open(model / index[name], framework="pt", device="cpu") as f:
                    yield (
                        name,
                        f.get_tensor(name).float(),
                        f.get_tensor(name.replace("_fn", "_scale")).float(),
                        f.get_tensor(name.replace("_fn", "_base")).float(),
                    )


def accuracy(args):
    records = []
    ms = [0, 1, 12, 17, 128] if args.quick else MS
    for name, wcpu, scale, base in all_weights(None if args.quick else args.model):
        w = wcpu.cuda()
        for seed in range(1 if args.quick else 3):
            torch.manual_seed(1200 + seed)
            raw = torch.randn(max(ms), K, device="cuda")
            for pattern in ["normal", "tiny", "wide", "zero"]:
                x = raw
                if pattern == "tiny":
                    x = x * 1e-7
                elif pattern == "wide":
                    x = x * torch.exp2(torch.linspace(-12, 12, K, device="cuda"))
                elif pattern == "zero":
                    x = torch.zeros_like(x)
                x = x.to(torch.bfloat16)
                xd = x.cpu().double()
                wd = wcpu.double()
                ref64 = (xd @ wd.T) * torch.rsqrt(
                    xd.square().mean(-1, keepdim=True) + 1e-6
                )
                alone = native(x[:1], w)
                for m in ms:
                    y = native(x[:m], w)
                    r = dict(
                        weight=name,
                        seed=seed,
                        pattern=pattern,
                        m=m,
                        torch=check(y, reference(x[:m], w)),
                        fp64=check(y.cpu(), ref64[:m]),
                    )
                    if m:
                        assert torch.equal(y[0], alone[0]), (
                            "batch invariance",
                            name,
                            seed,
                            pattern,
                            m,
                        )
                    if args.sinkhorn and scale is not None and m == 12:
                        from sglang.kernels.ops.layernorm.mhc import hc_split_sinkhorn

                        sc, ba = scale.cuda(), base.cuda()
                        a = hc_split_sinkhorn(y.unsqueeze(1), sc, ba, 4, 20, 1e-6)
                        b = hc_split_sinkhorn(
                            reference(x[:m], w).unsqueeze(1), sc, ba, 4, 20, 1e-6
                        )
                        r["sinkhorn"] = [check(aa, bb) for aa, bb in zip(a, b)]
                    records.append(r)
                print("ACCURACY", name, seed, pattern, "passed", flush=True)
    return records


def components_and_edges():
    torch.manual_seed(17)
    x = torch.randn(17, K, device="cuda").bfloat16()
    w = torch.randn(N, K, device="cuda") / (K**0.5)
    before_x, before_w = x.clone(), w.clone()
    y = native(x, w)
    assert y.shape == (17, N) and y.dtype == torch.float32 and y.is_contiguous()
    result = {"output": check(y, reference(x, w))}
    assert torch.equal(x, before_x) and torch.equal(w, before_w)
    for package in (lightop, lightop.op):
        assert not hasattr(package, "hc_mix_stats_partial")
        assert not hasattr(package, "hc_mix_stats_reduce")
    # Inference parameters may retain requires_grad.
    for context in (torch.no_grad, torch.inference_mode):
        with context():
            check(native(x, w.clone().requires_grad_(True)), reference(x, w))
    # Allocation lifetime, current stream and graph replay with updated X AND W.
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        y = native(x, w)
    torch.cuda.current_stream().wait_stream(stream)
    check(y, reference(x, w))
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        gy = native(x, w)
    for i in range(8):
        x.copy_(torch.randn_like(x))
        w.copy_(torch.randn_like(w) / (K**0.5))
        g.replay()
        torch.cuda.synchronize()
        check(gy, reference(x, w))

    def unsupported(fn):
        before = torch.cuda.memory_allocated()
        assert fn() is None, "unsupported metadata must return None"
        assert torch.cuda.memory_allocated() == before

    for xx, ww in [
        (x.float(), w),
        (x, w.bfloat16()),
        (x[:, :-1], w),
        (x.unsqueeze(0), w),
        (x, w[:1].expand(N, K)),
        (x._neg_view(), w),
        (x[:, :0], w[:, :0]),
        (x.cpu(), w.cpu()),
        (x[:, ::2], w[:, ::2]),
    ]:
        unsupported(lambda: native(xx, ww))
    # Empty tensors express launch-limit violations without large allocations.
    huge_k = 65535 * 256 + 1
    unsupported(lambda: native(
        torch.empty(0, huge_k, device="cuda", dtype=torch.bfloat16),
        torch.empty(0, huge_k, device="cuda"),
    ))
    for eps in [0, -1, float("nan"), float("inf"), 1e-100, 1e100]:
        unsupported(lambda: native(x, w, eps))
    unaligned_w = torch.empty(N * K + 1, device="cuda")[1:].view(N, K)
    unaligned_w.copy_(w)
    check(native(x, unaligned_w), reference(x, w))
    unaligned_x = torch.empty(17 * K + 1, device="cuda", dtype=torch.bfloat16)[1:].view(
        17, K
    )
    unaligned_x.copy_(x)
    check(native(unaligned_x, w), reference(unaligned_x, w))
    if torch.cuda.device_count() > 1:
        other_w = w.to("cuda:1")
        unsupported(lambda: native(x, other_w))
        with torch.cuda.device(1):
            xx = x.to("cuda:1")
            ww = w.to("cuda:1")
            check(native(xx, ww), reference(xx, ww))
    return result


def shape_matrix():
    """Check the public output across all model-shape M boundaries."""
    torch.manual_seed(31)
    x = torch.randn(max(MS), K, device="cuda").bfloat16()
    w = torch.randn(N, K, device="cuda") / K**0.5
    rows = []
    for m in MS:
        y = native(x[:m], w)
        assert y.shape == (m, N) and y.dtype == torch.float32 and y.is_contiguous()
        rows.append(dict(m=m, mixes=check(y, reference(x[:m], w))))
    return rows


def general_accuracy():
    """Torch references for dynamic K/N, tails and empty axes."""
    from lightop.gemm_ops import hc_mix_stats

    assert hc_mix_stats is categorized_hc_mix_stats
    records = []
    ks = [
        1,
        3,
        15,
        16,
        63,
        64,
        255,
        256,
        257,
        511,
        512,
        1025,
        4096,
        7168,
        20479,
        20480,
        20481,
    ]
    ns = [0, 1, 7, 23, 24, 25, 31, 32, 33, 64, 65, 128]
    for i, k in enumerate(ks):
        for j, n in enumerate(ns):
            m = [0, 1, 2, 15, 16, 17, 48][(i + j) % 7]
            torch.manual_seed(3000 + i * 100 + j)
            x = torch.randn(m, k, device="cuda").bfloat16()
            w = torch.randn(n, k, device="cuda") / k**0.5
            # Half of shapes use misaligned but contiguous views.
            if j % 2 and m:
                xx = torch.empty(x.numel() + 1, device="cuda", dtype=x.dtype)[
                    1:
                ].view_as(x)
                ww = torch.empty(w.numel() + 1, device="cuda")[1:].view_as(w)
                xx.copy_(x)
                ww.copy_(w)
                x, w = xx, ww
            y = native(x, w)
            assert y.shape == (m, n) and y.dtype == torch.float32 and y.is_contiguous()
            records.append(dict(m=m, n=n, k=k, output=check(y, reference(x, w))))
            if m:
                assert torch.equal(y[0], native(x[:1], w)[0]), (m, n, k)
    # General-shape stream and graph lifecycle, including changed weights.
    x = torch.randn(17, 257, device="cuda").bfloat16()
    w = torch.randn(33, 257, device="cuda") / 257**0.5
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        y = native(x, w)
    torch.cuda.current_stream().wait_stream(stream)
    check(y, reference(x, w))
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        y = native(x, w)
    for _ in range(3):
        x.normal_()
        w.normal_(std=257**-0.5)
        g.replay()
        torch.cuda.synchronize()
        check(y, reference(x, w))
    return records


def general_benchmark():
    records = []
    for m, n, k in [
        (1, 1, 63),
        (17, 33, 257),
        (48, 64, 1024),
        (128, 128, 4096),
        (512, 64, 7168),
        (1, 24, 20479),
        (12, 32, 20480),
        (48, 48, 20480),
        (128, 64, 20481),
    ]:
        torch.manual_seed(42)
        x = torch.randn(m, k, device="cuda").bfloat16()
        w = torch.randn(n, k, device="cuda") / k**0.5
        check(native(x, w), reference(x, w))
        for backend, fn in [
            ("torch", lambda: reference(x, w)),
            ("lightop", lambda: native(x, w)),
        ]:
            records.append(dict(m=m, n=n, k=k, backend=backend, **bench(fn)))
    return records


def tile_boundary_accuracy():
    """M dispatch boundaries must not change the arithmetic of a fixed row."""
    rows = []
    for seed in [41, 42]:
        torch.manual_seed(seed)
        w = torch.randn(N, K, device="cuda") / K**0.5
        raw = torch.randn(513, K, device="cuda")
        for pattern in ["normal", "tiny", "wide"]:
            scale = 1e-7 if pattern == "tiny" else 1
            if pattern == "wide":
                scale = torch.exp2(torch.linspace(-12, 12, K, device="cuda"))
            x = (raw * scale).bfloat16()
            one_y = native(x[:1], w)
            for m in [
                31,
                32,
                33,
                63,
                64,
                65,
                127,
                128,
                129,
                255,
                256,
                257,
                511,
                512,
                513,
            ]:
                y = native(x[:m], w)
                assert torch.equal(y[0], one_y[0]), (m, seed, pattern)
                rows.append(
                    dict(
                        m=m,
                        seed=seed,
                        pattern=pattern,
                        output=check(y, reference(x[:m], w)),
                    )
                )
    return rows


def bench(fn, repeats=10):
    for _ in range(10):
        fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(100):
            y = fn()
    for _ in range(10):
        graph.replay()
    torch.cuda.synchronize()
    samples = []
    for _ in range(repeats):
        start, end = (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000 / 100)
    return dict(
        mean_us=statistics.mean(samples),
        p50_us=statistics.median(samples),
        p90_us=sorted(samples)[int(0.9 * (len(samples) - 1))],
        samples_us=samples,
    )


def benchmark(args):
    torch.manual_seed(42)
    w = next(all_weights(args.model))[1].cuda()
    if args.model:
        w = list(all_weights(args.model))[2][1].cuda()  # layer0 FFN
    rows = []
    for m in (
        [1, 12, 128] if args.quick else [1, 2, 10, 12, 48, 128, 256, 512, 1024, 2048]
    ):
        x = torch.randn(m, K, device="cuda").bfloat16()
        methods = {"torch": lambda: reference(x, w), "lightop": lambda: native(x, w)}
        expected = reference(x, w)
        for name, fn in methods.items():
            check(fn(), expected)
            r = dict(m=m, backend=name, **bench(fn))
            r["logical_GBps"] = (2 * m * K + 4 * N * K + 4 * m * N) / (
                r["mean_us"] * 1000
            )
            rows.append(r)
            print("BENCH", json.dumps(r), flush=True)
    return rows


def real_activations(directory):
    from sglang.kernels.ops.layernorm.mhc import hc_split_sinkhorn

    rows = []
    paths = sorted(directory.glob("*.pt"))
    assert paths, f"no activation captures in {directory}"
    for path in paths:
        data = torch.load(path, map_location="cpu", weights_only=True)
        x = data["x"].flatten(1).cuda()
        w = data["weight"].cuda()
        eps = data["eps"]
        y = native(x, w, eps)
        ref = reference(x, w, eps)
        xd = x.cpu().double()
        wd = w.cpu().double()
        ref64 = (xd @ wd.T) * torch.rsqrt(xd.square().mean(-1, keepdim=True) + eps)
        r = dict(
            file=path.name,
            m=x.shape[0],
            torch=check(y, ref),
            fp64=check(y.cpu(), ref64),
        )
        params = (
            data["scale"].cuda(),
            data["base"].cuda(),
            data["hc_mult"],
            data["iters"],
            data["hc_eps"],
        )
        actual = hc_split_sinkhorn(y.unsqueeze(1), *params)
        expected = hc_split_sinkhorn(ref.unsqueeze(1), *params)
        r["sinkhorn"] = [check(a, b) for a, b in zip(actual, expected)]
        r["service_coefficients"] = [
            check(a.squeeze(1), b.cuda()) for a, b in zip(actual, data["coefficients"])
        ]
        rows.append(r)
    return rows


@pytest.mark.parametrize(
    "suite",
    [components_and_edges, shape_matrix, general_accuracy, tile_boundary_accuracy],
    ids=["components_edges", "shapes", "general_shapes", "tile_boundaries"],
)
def test_hc_mix_stats(suite):
    """Run the native correctness suites under the repository's CI runner."""
    if (
        not torch.cuda.is_available()
        or torch.cuda.get_device_properties(0).gcnArchName.split(":")[0]
        not in SUPPORTED_ARCHS
    ):
        pytest.skip("Requires gfx928, gfx936 or gfx938")
    suite()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=Path("hc_mix_stats_results"))
    p.add_argument("--model", type=Path)

    p.add_argument(
        "--sinkhorn",
        action="store_true",
        help="Validate actual SGLang Sinkhorn (requires SGLang)",
    )

    p.add_argument("--general-only", action="store_true")
    p.add_argument("--activations", type=Path)
    p.add_argument("--quick", action="store_true")
    p.add_argument("--profile", action="store_true")
    p.add_argument("--profile-m", type=int, default=12)
    p.add_argument("--accuracy-only", action="store_true")
    p.add_argument("--benchmark-only", action="store_true")
    args = p.parse_args()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    if (
        not torch.cuda.is_available()
        or torch.cuda.get_device_properties(0).gcnArchName.split(":")[0]
        not in SUPPORTED_ARCHS
    ):
        raise SystemExit("Requires gfx928, gfx936 or gfx938")
    if args.profile:
        x = torch.randn(args.profile_m, K, device="cuda").bfloat16()
        w = torch.randn(N, K, device="cuda") / (K**0.5)
        check(native(x, w), reference(x, w))
        for _ in range(5):
            native(x, w)
        torch.cuda.synchronize()
        return
    args.output.mkdir(parents=True, exist_ok=True)
    result = {
        "torch": torch.__version__,
        "hip": torch.version.hip,
        "lightop": lightop.__file__,
        "extension": lightop.op.__file__,
        "device": str(torch.cuda.get_device_properties(0)),
        "HIP_VISIBLE_DEVICES": os.environ.get("HIP_VISIBLE_DEVICES"),
    }
    if not args.benchmark_only:
        result["components_edges"] = components_and_edges()
        result["shapes"] = shape_matrix()
        result["general_accuracy"] = general_accuracy()
        result["tile_boundary_accuracy"] = tile_boundary_accuracy()
        result["accuracy"] = [] if args.general_only else accuracy(args)
        if args.activations:
            result["real_activations"] = real_activations(args.activations)
        (args.output / "accuracy.json").write_text(json.dumps(result, indent=2))
    if not args.accuracy_only:
        result["performance"] = [] if args.general_only else benchmark(args)
        result["general_performance"] = general_benchmark()
    (args.output / "result.json").write_text(json.dumps(result, indent=2))
    print("PASS", args.output, flush=True)


if __name__ == "__main__":
    main()
