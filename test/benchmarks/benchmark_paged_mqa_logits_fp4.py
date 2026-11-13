# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Benchmark the existing FP4 paged MQA operator through its public API.

Explicit benchmark, not a pytest test.
Only public lightop.attention APIs are called. Run the same script with the
baseline or candidate package; --compare reads two saved results without a GPU.
"""

# 在已安装目标 LightOp 的 HIP PyTorch 环境中，从仓库根目录运行：
#   python3 test/benchmarks/benchmark_paged_mqa_logits_fp4.py \
#     --gpu 0 --output /tmp/fp4-candidate
# 默认覆盖 bs=1/4/7/8/16/32、decode 1行/VERIFY 6行、128K、偏移0/1023。
# graph 模式预热10次，捕图后再预热10次，采样9次、每次重放8次；
# 计时范围为 metadata + logits，不含输入准备、复制、捕图或校验。
#
# 基线复测：在基线 LightOp 环境运行同一脚本、相同参数，另存目录：
#   python3 test/benchmarks/benchmark_paged_mqa_logits_fp4.py \
#     --gpu 0 --output /tmp/fp4-baseline
# 离线对比（无需 GPU，不导入 torch/LightOp）：
#   python3 test/benchmarks/benchmark_paged_mqa_logits_fp4.py \
#     --compare /tmp/fp4-baseline/result.json /tmp/fp4-candidate/result.json \
#     --output /tmp/fp4-comparison
# 比较要求输入、输出、计时参数和设备一致，可用 --expect-commit 核对包版本。
#
# 其他形状可在运行命令后追加以下参数，并使用新的 --output 目录：
#   --batches 2,16 --rows 1,6 --input-lengths 4096,51200,65536 --offsets 512
#   --batches 3,16,32 --rows 5,6 --layout mixed --scale wide
#   --batches 1,16,32 --rows 1 --input-lengths 4096 --offsets 0 --table-block-size 8
# 默认仅用一张物理 GPU，分配上限6GiB；可用 --gpu / --memory-gib 修改。
# 输出目录须为新目录。result.csv 和 comparison.csv 仅展示平均耗时（us）及
# 对比变化比例，正值表示候选更快；JSON 保留原始采样和身份校验信息。

import argparse
import csv
import hashlib
import importlib
import itertools
import json
import os
from pathlib import Path
import statistics
import sys
import time


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def csv_ints(text):
    try:
        result = [int(x) for x in text.split(",")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from exc
    if not result or len(set(result)) != len(result):
        raise argparse.ArgumentTypeError("values must be nonempty and unique")
    return result


def cases(args):
    return [dict(batch=b, rows=r, input_len=n, offset=o, heads=args.heads,
                 table_block_size=args.table_block_size, layout=args.layout,
                 scale=args.scale)
            for b, r, n, o in itertools.product(
                args.batches, args.rows, args.input_lengths, args.offsets)]


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as f:
        json.dump(value, f, indent=2)
        f.write("\n")


def write_csv(path, rows):
    # BOM lets Excel/WPS recognize UTF-8; newline="" avoids blank Windows rows.
    with Path(path).open("x", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_result_csv(path, results, identity):
    rows = [dict(git_hash=identity["git_hash"], **result["case"],
                 mean_us=statistics.mean(result["raw_us"]))
            for result in results]
    write_csv(path, rows)


def tensors(case, seed, torch):
    """Nonzero random FP4 payloads, request-shared tables, independent row lengths.

    Use CPU RNG for identical inputs across wheel subprocesses. Full page tables
    also satisfy the older operator's shared-table contract. No captured model
    files or private experiment directories are required.
    """
    b, r, h = case["batch"], case["rows"], case["heads"]
    tb = case["table_block_size"]
    width = case["input_len"] + case["offset"] + r - 1
    blocks = (width + tb - 1) // tb
    units_per_page = 64 // tb
    pages = (b * blocks + units_per_page - 1) // units_per_page
    g = torch.Generator().manual_seed(seed)
    payload = torch.randint(0, 256, (pages, 4096), generator=g, dtype=torch.uint8)
    lo, hi = {"narrow": (124, 130), "wide": (115, 141), "uniform": (127, 128)}[case["scale"]]
    scales = torch.randint(lo, hi, (pages, 256), generator=g, dtype=torch.uint8)
    pool = torch.cat((payload, scales), dim=1).view(pages, 64, 1, 68)
    q = torch.randint(0, 256, (b * r, 1, h, 64), generator=g, dtype=torch.uint8)
    sf = torch.randint(124, 130, (b * r, 1, h, 4), generator=g,
                       dtype=torch.uint8).view(torch.int32).squeeze(-1)
    weights = torch.randn(b * r, h, generator=g)
    lengths = torch.tensor([
        (case["input_len"] if case["layout"] == "balanced" else
         [case["input_len"] // 8, case["input_len"] // 2, case["input_len"]][j % 3])
        + case["offset"] + k for j in range(b) for k in range(r)
    ], dtype=torch.int32)
    table = torch.randperm(b * blocks, generator=g).reshape(b, blocks)
    table = table.repeat_interleave(r, 0).to(torch.int32)
    cpu = [q, sf, pool, weights, lengths, table]
    digest = hashlib.sha256()
    for t in cpu:
        digest.update(str((str(t.dtype), tuple(t.shape))).encode())
        digest.update(t.contiguous().view(torch.uint8).numpy().tobytes())
    return [t.cuda() for t in cpu], width, digest.hexdigest()


def benchmark(args):
    import ctypes
    import torch
    import lightop
    from lightop.attention import (
        get_paged_mqa_logits_fp4_metadata, paged_mqa_logits_fp4,
    )

    package = Path(lightop.__file__).resolve().parent
    assert torch.cuda.device_count() == 1, "select one GPU with --gpu"
    props = torch.cuda.get_device_properties(0)
    torch.cuda.set_per_process_memory_fraction(args.memory_gib * 1024**3 / props.total_memory)
    version = importlib.import_module("lightop.version")
    git_hash = getattr(version, "git_hash", None)
    if args.expect_commit:
        assert git_hash and args.expect_commit.startswith(git_hash), (args.expect_commit, git_hash)
    pci = ctypes.create_string_buffer(32)
    assert ctypes.CDLL("libamdhip64.so").hipDeviceGetPCIBusId(pci, 32, 0) == 0
    extension = next(Path(lightop.__file__).parent.glob("op*.so"))
    identity = dict(pci=pci.value.decode(), arch=props.gcnArchName,
                    compute_units=props.multi_processor_count,
                    torch=torch.__version__, hip=torch.version.hip,
                    python=sys.version.split()[0], package=str(package),
                    git_hash=git_hash, version=getattr(version, "hcu_version", None),
                    dtk=getattr(version, "dtk", None), extension_sha256=sha256(extension),
                    script_sha256=sha256(__file__), physical_gpu=args.gpu)
    settings = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    write_json(args.output / "identity.json", identity)
    write_json(args.output / "config.json", dict(command=sys.argv, settings=settings, cases=cases(args)))
    results = []
    with (args.output / "samples.jsonl").open("x") as f:
        for case in cases(args):
            torch.cuda.reset_peak_memory_stats()
            inputs, width, input_sha = tensors(case, args.seed, torch)
            q, sf, pool, weights, lengths, table = inputs
            out = torch.full((case["batch"] * case["rows"], width), -12345., device="cuda")

            def score():
                meta = get_paged_mqa_logits_fp4_metadata(
                    lengths, 64, props.multi_processor_count, case["rows"])
                return paged_mqa_logits_fp4(
                    q, sf, pool, weights, lengths, table, schedule_meta=meta,
                    max_context_len=width, table_block_size=case["table_block_size"],
                    out=out, rows_per_request=case["rows"])

            for _ in range(args.warmup):
                score()
            graph = None
            if args.mode == "graph":
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    score()
                call = graph.replay
            else:
                call = score
            for _ in range(args.warmup):
                call()
            torch.cuda.synchronize()
            raw = []
            for _ in range(args.samples):
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                for _ in range(args.repeats):
                    call()
                end.record()
                end.synchronize()
                raw.append(start.elapsed_time(end) * 1000 / args.repeats)
            actual = out.cpu()
            assert torch.isfinite(actual).all(), case
            visible = torch.arange(width)[None, :] < lengths.cpu()[:, None]
            assert (actual[~visible] == -12345.).all(), "output tail was overwritten"
            row = dict(case=case, input_sha256=input_sha,
                       output_sha256=hashlib.sha256(actual.numpy().tobytes()).hexdigest(),
                       mean_us=statistics.mean(raw), median_us=statistics.median(raw),
                       min_us=min(raw), max_us=max(raw),
                       raw_us=raw, peak_allocated_mib=torch.cuda.max_memory_allocated() / 1024**2)
            results.append(row)
            f.write(json.dumps(row) + "\n")
            f.flush()
            print(case, row["mean_us"], flush=True)
            # Release graph-owned buffers before constructing the next case.
            del call, graph, score, inputs, q, sf, pool, weights, lengths, table, out, actual, visible
            torch.cuda.empty_cache()

    write_json(args.output / "result.json", dict(identity=identity, settings=settings,
               results=results, completed_at=time.time(),
               measurement="operator metadata + logits, not model throughput"))
    write_result_csv(args.output / "result.csv", results, identity)
    print(args.output / "result.json")
    print(args.output / "result.csv")


def compare(paths, output):
    base, candidate = [json.loads(path.read_text()) for path in paths]
    for key in ["pci", "arch", "compute_units", "torch", "hip", "script_sha256"]:
        assert base["identity"][key] == candidate["identity"][key], key
    for key in ["mode", "warmup", "samples", "repeats", "seed"]:
        assert base["settings"][key] == candidate["settings"][key], key
    assert len(base["results"]) == len(candidate["results"]), "incomplete cases"
    records = []
    lines = ["# FP4 paged MQA logits comparison", "",
             "| bs | rows | input | offset | baseline mean us | candidate mean us | reduction |",
             "|---:|---:|---:|---:|---:|---:|---:|"]
    for a, b in zip(base["results"], candidate["results"]):
        assert a["case"] == b["case"], "shape/order mismatch"
        assert a["input_sha256"] == b["input_sha256"], "inputs differ"
        assert a["output_sha256"] == b["output_sha256"], "outputs differ"
        baseline_us = statistics.mean(a["raw_us"])
        candidate_us = statistics.mean(b["raw_us"])
        reduction = (1 - candidate_us / baseline_us) * 100
        records.append(dict(case=a["case"], baseline_us=baseline_us,
                            candidate_us=candidate_us, latency_reduction_percent=reduction,
                            bitwise_equal=True))
        c = a["case"]
        lines.append(f'| {c["batch"]} | {c["rows"]} | {c["input_len"]} | {c["offset"]} | '
                     f'{baseline_us:.2f} | {candidate_us:.2f} | {reduction:+.2f}% |')
    write_json(output / "comparison.json", dict(
        baseline=base["identity"], candidate=candidate["identity"], results=records,
        metric="arithmetic_mean_us", all_outputs_equal=True))
    write_csv(output / "comparison.csv", [dict(
        baseline_commit=base["identity"]["git_hash"],
        candidate_commit=candidate["identity"]["git_hash"],
        **record["case"], baseline_mean_us=record["baseline_us"],
        candidate_mean_us=record["candidate_us"],
        latency_reduction_percent=record["latency_reduction_percent"],
        bitwise_equal=record["bitwise_equal"]) for record in records])
    (output / "comparison.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(output / "comparison.csv")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True, help="new result directory; never overwritten")
    p.add_argument("--expect-commit", help="optional source commit check against loaded package")
    p.add_argument("--compare", type=Path, nargs=2, metavar=("BASELINE_JSON", "CANDIDATE_JSON"),
                   help="compare saved result.json files without importing torch or LightOp")
    p.add_argument("--gpu", type=int, default=0, help="physical HIP GPU index")
    p.add_argument("--batches", type=csv_ints, default=csv_ints("1,4,7,8,16,32"))
    p.add_argument("--rows", type=csv_ints, default=csv_ints("1,6"))
    p.add_argument("--input-lengths", type=csv_ints, default=csv_ints("131072"))
    p.add_argument("--offsets", type=csv_ints, default=csv_ints("0,1023"))
    p.add_argument("--heads", type=int, choices=[16, 32, 64], default=32)
    p.add_argument("--table-block-size", type=int, choices=[8, 64], default=64)
    p.add_argument("--layout", choices=["balanced", "mixed"], default="balanced")
    p.add_argument("--scale", choices=["narrow", "wide", "uniform"], default="narrow")
    p.add_argument("--mode", choices=["graph", "eager"], default="graph")
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--samples", type=int, default=9)
    p.add_argument("--repeats", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--memory-gib", type=float, default=6)
    args = p.parse_args()
    if min(args.batches + args.rows + args.input_lengths) < 1 or min(args.offsets) < 0:
        p.error("batch, rows and input lengths must be positive; offsets nonnegative")
    if min(args.warmup, args.samples, args.repeats, args.memory_gib) <= 0 or args.gpu < 0:
        p.error("measurement counts and memory budget must be positive; GPU nonnegative")
    if max(args.rows) > 1 and (args.heads > 32 or args.table_block_size != 64):
        p.error("multiple rows per request require H<=32 and dense table block size 64")
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    if args.compare:
        compare(args.compare, args.output)
        return
    for name in ["ROCR_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES"]:
        os.environ.pop(name, None)
    os.environ["HIP_VISIBLE_DEVICES"] = str(args.gpu)
    benchmark(args)


if __name__ == "__main__":
    main()
