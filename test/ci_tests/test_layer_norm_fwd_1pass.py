# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

import argparse
import csv
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional

import torch
import triton
import triton.language as tl

from lightop.norm import layer_norm_fwd_1pass_opt


QWEN35_397B = {
    "dtype": "bfloat16",
    "image_group_size": 128,
    "rms_norm_eps": 1.0e-6,
    "activation": "swish",
}

DEFAULT_TOKENS = [
    1,
    16,
    32,
    64,
    128,
    256,
    512,
    1024,
    2048,
    4096,
    8192,
    9000,
    16384,
    32768,
    65536,
]


@dataclass(frozen=True)
class BenchCase:
    name: str
    n: int
    groups: int
    has_bias: bool = False
    has_z: bool = False
    norm_before_gate: bool = False
    is_rms_norm: bool = True
    activation: str = "none"
    rows_per_block: int = 1
    num_warps: int = 1


@triton.jit
def _layer_norm_fwd_1pass_triton_kernel(
    X,  # pointer to the input
    Y,  # pointer to the output
    W,  # pointer to the weights
    B,  # pointer to the biases
    Z,  # pointer to the other branch
    Mean,  # pointer to the mean
    Rstd,  # pointer to the 1/std
    stride_x_row,  # how much to increase the pointer when moving by 1 row
    stride_y_row,
    stride_z_row,
    M,  # number of rows in X
    N: tl.constexpr,  # number of columns in X
    eps,  # epsilon to avoid division by zero
    BLOCK_N: tl.constexpr,
    ROWS_PER_BLOCK: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    HAS_Z: tl.constexpr,
    NORM_BEFORE_GATE: tl.constexpr,
    IS_RMS_NORM: tl.constexpr,
    ACTIVATION: tl.constexpr,
):
    # Map the program id to the starting row of X and Y it should compute.
    row_start = tl.program_id(0) * ROWS_PER_BLOCK
    group = tl.program_id(1)

    # Create 2D tile: [ROWS_PER_BLOCK, BLOCK_N]
    rows = row_start + tl.arange(0, ROWS_PER_BLOCK)
    cols = tl.arange(0, BLOCK_N)

    # Compute offsets for 2D tile
    row_offsets = rows[:, None] * stride_x_row
    col_offsets = cols[None, :] + group * N

    # Base pointers
    X_base = X + row_offsets + col_offsets
    Y_base = Y + rows[:, None] * stride_y_row + col_offsets

    # Create mask for valid rows and columns
    row_mask = rows[:, None] < M
    col_mask = cols[None, :] < N
    mask = row_mask & col_mask

    # Load input data with 2D tile
    x = tl.load(X_base, mask=mask, other=0.0).to(tl.float32)

    if HAS_Z and not NORM_BEFORE_GATE:
        Z_base = Z + rows[:, None] * stride_z_row + col_offsets
        z = tl.load(Z_base, mask=mask, other=0.0).to(tl.float32)
        if ACTIVATION == "swish" or ACTIVATION == "silu":
            x *= z * tl.sigmoid(z)
        elif ACTIVATION == "sigmoid":
            x *= tl.sigmoid(z)

    # Compute mean and variance per row (reduce along axis 1)
    if not IS_RMS_NORM:
        mean = tl.sum(x, axis=1) / N  # Shape: [ROWS_PER_BLOCK]
        # Store mean for each row
        mean_offsets = group * M + rows
        mean_mask = rows < M
        tl.store(Mean + mean_offsets, mean, mask=mean_mask)
        # Broadcast mean back to 2D for subtraction
        xbar = tl.where(mask, x - mean[:, None], 0.0)
        var = tl.sum(xbar * xbar, axis=1) / N  # Shape: [ROWS_PER_BLOCK]
    else:
        xbar = tl.where(mask, x, 0.0)
        var = tl.sum(xbar * xbar, axis=1) / N  # Shape: [ROWS_PER_BLOCK]
        mean = 0.0  # Placeholder for RMS norm

    rstd = tl.rsqrt(var + eps)  # Shape: [ROWS_PER_BLOCK]

    # Store rstd for each row
    rstd_offsets = group * M + rows
    rstd_mask = rows < M
    tl.store(Rstd + rstd_offsets, rstd, mask=rstd_mask)

    # Load weights and biases (broadcast across rows)
    w_offsets = cols + group * N
    w_mask = cols < N
    w = tl.load(W + w_offsets, mask=w_mask, other=0.0).to(tl.float32)

    if HAS_BIAS:
        b = tl.load(B + w_offsets, mask=w_mask, other=0.0).to(tl.float32)

    # Normalize and apply linear transformation
    if not IS_RMS_NORM:
        x_hat = (x - mean[:, None]) * rstd[:, None]
    else:
        x_hat = x * rstd[:, None]

    y = x_hat * w[None, :] + b[None, :] if HAS_BIAS else x_hat * w[None, :]

    if HAS_Z and NORM_BEFORE_GATE:
        Z_base = Z + rows[:, None] * stride_z_row + col_offsets
        z = tl.load(Z_base, mask=mask, other=0.0).to(tl.float32)
        if ACTIVATION == "swish" or ACTIVATION == "silu":
            y *= z * tl.sigmoid(z)
        elif ACTIVATION == "sigmoid":
            y *= tl.sigmoid(z)

    # Write output
    tl.store(Y_base, y, mask=mask)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark lightop layer_norm_fwd_1pass_opt against Triton for the captured Qwen3.5-397B image-size shape."
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="bfloat16")
    parser.add_argument("--tokens", default=",".join(str(x) for x in DEFAULT_TOKENS))
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260518)
    parser.add_argument(
        "--atol",
        type=float,
        default=None,
        help="Absolute tolerance. Default is dtype-aware.",
    )
    parser.add_argument(
        "--rtol",
        type=float,
        default=None,
        help="Relative tolerance. Default is dtype-aware.",
    )
    parser.add_argument(
        "--csv",
        default="layer_norm_fwd_1pass_image_size.csv",
        help="CSV path for the full benchmark result.",
    )
    parser.add_argument(
        "--print",
        choices=["grouped", "full", "summary", "none"],
        default="grouped",
        help="Console output style. CSV always contains full rows.",
    )
    parser.add_argument(
        "--cases",
        default="all",
        help="Comma separated case names, or 'all'.",
    )
    parser.add_argument(
        "--bench-on-fail",
        action="store_true",
        help="Run timing even when correctness fails.",
    )
    return parser.parse_args()


def qwen_cases() -> List[BenchCase]:
    cfg = QWEN35_397B
    return [
        BenchCase(
            name="image_size_rms_swish_gate",
            n=cfg["image_group_size"],
            groups=1,
            has_z=True,
            norm_before_gate=True,
            is_rms_norm=True,
            activation=cfg["activation"],
            rows_per_block=1,
            num_warps=1,
        ),
    ]


def dtype_from_name(name: str) -> torch.dtype:
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[name]


def default_tolerances(dtype: torch.dtype) -> tuple:
    if dtype == torch.bfloat16:
        return 4.0e-2, 4.0e-2
    if dtype == torch.float16:
        return 5.0e-3, 5.0e-3
    return 1.0e-5, 1.0e-5


def normalize_cuda_device(device_arg: str) -> torch.device:
    if device_arg.isdigit():
        index = int(device_arg)
    else:
        device = torch.device(device_arg)
        if device.type != "cuda":
            raise ValueError(f"Expected a CUDA device, got: {device_arg}")
        index = torch.cuda.current_device() if device.index is None else device.index
    torch.cuda.set_device(index)
    return torch.device(f"cuda:{index}")


def next_power_of_2(x: int) -> int:
    return 1 << (x - 1).bit_length()


def randn_like_shape(shape, dtype: torch.dtype, device: torch.device, mean: float = 0.0, std: float = 1.0):
    tensor = torch.empty(shape, device=device, dtype=dtype)
    try:
        tensor.normal_(mean=mean, std=std)
    except RuntimeError:
        tmp = torch.empty(shape, device=device, dtype=torch.float32).normal_(mean=mean, std=std)
        tensor.copy_(tmp.to(dtype))
        del tmp
    return tensor


def make_tensors(case: BenchCase, m: int, dtype: torch.dtype, device: torch.device) -> Dict[str, torch.Tensor]:
    total_cols = case.n * case.groups
    x = randn_like_shape((m, total_cols), dtype, device, std=0.5)
    y_triton = torch.empty_like(x)
    y_cuda = torch.empty_like(x)
    w = randn_like_shape((total_cols,), dtype, device, mean=1.0, std=0.1)

    if case.has_bias:
        b = randn_like_shape((total_cols,), dtype, device, std=0.1)
    else:
        b = torch.empty((0,), device=device, dtype=dtype)

    if case.has_z:
        z = randn_like_shape((m, total_cols), dtype, device, std=0.5)
    else:
        z = torch.empty((0,), device=device, dtype=dtype)

    if case.is_rms_norm:
        mean_triton = torch.empty((0,), device=device, dtype=torch.float32)
        mean_cuda = torch.empty((0,), device=device, dtype=torch.float32)
    else:
        mean_triton = torch.empty((case.groups, m), device=device, dtype=torch.float32)
        mean_cuda = torch.empty((case.groups, m), device=device, dtype=torch.float32)

    stat_shape = (m,) if case.groups == 1 else (case.groups, m)
    rstd_triton = torch.empty(stat_shape, device=device, dtype=torch.float32)
    rstd_cuda = torch.empty(stat_shape, device=device, dtype=torch.float32)
    return {
        "x": x,
        "y_triton": y_triton,
        "y_cuda": y_cuda,
        "w": w,
        "b": b,
        "z": z,
        "mean_triton": mean_triton,
        "mean_cuda": mean_cuda,
        "rstd_triton": rstd_triton,
        "rstd_cuda": rstd_cuda,
    }


def launch_triton(case: BenchCase, tensors: Dict[str, torch.Tensor], m: int, eps: float):
    n = case.n
    total_cols = case.groups * n
    block_n = next_power_of_2(n)
    rows_per_block = case.rows_per_block
    grid = (triton.cdiv(m, rows_per_block), case.groups)
    _layer_norm_fwd_1pass_triton_kernel[grid](
        tensors["x"],
        tensors["y_triton"],
        tensors["w"],
        tensors["b"],
        tensors["z"],
        tensors["mean_triton"],
        tensors["rstd_triton"],
        total_cols,
        total_cols,
        total_cols,
        m,
        n,
        eps,
        BLOCK_N=block_n,
        ROWS_PER_BLOCK=rows_per_block,
        HAS_BIAS=case.has_bias,
        HAS_Z=case.has_z,
        NORM_BEFORE_GATE=case.norm_before_gate,
        IS_RMS_NORM=case.is_rms_norm,
        ACTIVATION=case.activation,
        num_warps=case.num_warps,
    )


def launch_cuda(case: BenchCase, tensors: Dict[str, torch.Tensor], m: int, eps: float):
    n = case.n
    total_cols = case.groups * n
    block_n = next_power_of_2(n)
    rows_per_block = case.rows_per_block
    layer_norm_fwd_1pass_opt(
        tensors["x"],
        tensors["y_cuda"],
        tensors["w"],
        tensors["b"],
        tensors["z"],
        tensors["mean_cuda"],
        tensors["rstd_cuda"],
        total_cols,
        total_cols,
        total_cols,
        m,
        n,
        eps,
        block_n,
        rows_per_block,
        case.has_bias,
        case.has_z,
        case.norm_before_gate,
        case.is_rms_norm,
        case.activation,
    )


def time_us(fn: Callable[[], None], warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) * 1000.0 / iters


def logical_bytes(case: BenchCase, m: int, dtype: torch.dtype) -> int:
    dtype_bytes = torch.empty((), dtype=dtype).element_size()
    elems = m * case.groups * case.n

    bytes_per_elem = dtype_bytes  # X read
    bytes_per_elem += dtype_bytes  # W read per row
    bytes_per_elem += dtype_bytes  # Y write
    if case.has_bias:
        bytes_per_elem += dtype_bytes
    if case.has_z:
        bytes_per_elem += dtype_bytes

    stats_bytes = m * case.groups * 4  # Rstd write
    if not case.is_rms_norm:
        stats_bytes += m * case.groups * 4
    return elems * bytes_per_elem + stats_bytes


def bandwidth_gbs(num_bytes: int, us: Optional[float]) -> Optional[float]:
    if us is None or us <= 0:
        return None
    return num_bytes / (us * 1000.0)


def compare_outputs(
    tensors: Dict[str, torch.Tensor],
    case: BenchCase,
    atol: float,
    rtol: float,
) -> Dict[str, object]:
    y_ref = tensors["y_triton"].float()
    y_out = tensors["y_cuda"].float()
    diff = (y_ref - y_out).abs()
    max_abs = diff.max().item()
    ref_abs = y_ref.abs()
    max_rel = (diff / torch.clamp(ref_abs, min=1.0e-6)).max().item()
    ok = torch.allclose(y_out, y_ref, atol=atol, rtol=rtol)

    rstd_diff = (tensors["rstd_triton"] - tensors["rstd_cuda"]).abs()
    rstd_max_abs = rstd_diff.max().item()
    ok = ok and torch.allclose(
        tensors["rstd_cuda"], tensors["rstd_triton"], atol=atol, rtol=rtol
    )

    mean_max_abs = 0.0
    if not case.is_rms_norm:
        mean_diff = (tensors["mean_triton"] - tensors["mean_cuda"]).abs()
        mean_max_abs = mean_diff.max().item()
        ok = ok and torch.allclose(
            tensors["mean_cuda"], tensors["mean_triton"], atol=atol, rtol=rtol
        )

    return {
        "ok": ok,
        "max_abs": max_abs,
        "max_rel": max_rel,
        "rstd_abs": rstd_max_abs,
        "mean_abs": mean_max_abs,
    }


def fmt_float(value: Optional[float], digits: int = 2) -> str:
    if value is None:
        return "-"
    if math.isnan(value):
        return "nan"
    return f"{value:.{digits}f}"


def markdown_table(rows: List[Dict[str, object]]) -> str:
    headers = [
        "case",
        "mode",
        "tokens",
        "N",
        "groups",
        "correct",
        "max_abs",
        "max_rel",
        "rstd_abs",
        "mean_abs",
        "triton_us",
        "cuda_us",
        "speedup",
        "triton_GB/s",
        "cuda_GB/s",
        "note",
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row[h]) for h in headers) + " |")
    return "\n".join(lines)


def compact_case_table(case: BenchCase, rows: List[Dict[str, object]]) -> str:
    headers = [
        "tok",
        "mode",
        "ok",
        "max_abs",
        "tri_us",
        "cu_us",
        "spd",
        "tri_GB/s",
        "cu_GB/s",
        "note",
    ]
    case_rows = [row for row in rows if row["case"] == case.name]
    lines = [
        f"\n[{case.name}] N={case.n} groups={case.groups} "
        f"bias={case.has_bias} z={case.has_z} rms={case.is_rms_norm} "
        f"gate={case.activation if case.has_z else 'none'} "
        f"rows_per_block={case.rows_per_block} num_warps={case.num_warps}",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in case_rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["tokens"]),
                    str(row["mode"]),
                    str(row["correct"]),
                    str(row["max_abs"]),
                    str(row["triton_us"]),
                    str(row["cuda_us"]),
                    str(row["speedup"]),
                    str(row["triton_GB/s"]),
                    str(row["cuda_GB/s"]),
                    str(row["note"]),
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def summary_table(cases: List[BenchCase], rows: List[Dict[str, object]]) -> str:
    headers = ["case", "N", "groups", "pass", "fail", "oom/err", "best_spd", "worst_spd"]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for case in cases:
        case_rows = [row for row in rows if row["case"] == case.name]
        passed = sum(row["correct"] == "PASS" for row in case_rows)
        failed = sum(row["correct"] == "FAIL" for row in case_rows)
        errored = sum(row["correct"] in ("OOM", "ERR") for row in case_rows)
        speedups = []
        for row in case_rows:
            try:
                speedups.append(float(row["speedup"]))
            except (TypeError, ValueError):
                pass
        best = max(speedups) if speedups else None
        worst = min(speedups) if speedups else None
        lines.append(
            "| "
            + " | ".join(
                [
                    case.name,
                    str(case.n),
                    str(case.groups),
                    str(passed),
                    str(failed),
                    str(errored),
                    fmt_float(best, 3),
                    fmt_float(worst, 3),
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def print_results(cases: List[BenchCase], rows: List[Dict[str, object]], style: str) -> None:
    if style == "none":
        return
    print("\nCorrectness/performance summary")
    print(summary_table(cases, rows))
    if style == "summary":
        return
    if style == "full":
        print()
        print(markdown_table(rows))
        return
    for case in cases:
        print(compact_case_table(case, rows))


def write_csv(path: str, rows: List[Dict[str, object]]) -> Path:
    csv_path = Path(path).expanduser()
    if csv_path.parent != Path("."):
        csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "case",
        "mode",
        "tokens",
        "N",
        "groups",
        "correct",
        "max_abs",
        "max_rel",
        "rstd_abs",
        "mean_abs",
        "triton_us",
        "cuda_us",
        "speedup",
        "triton_GB/s",
        "cuda_GB/s",
        "note",
    ]
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return csv_path


def run_one(
    case: BenchCase,
    m: int,
    dtype: torch.dtype,
    device: torch.device,
    warmup: int,
    iters: int,
    atol: float,
    rtol: float,
    bench_on_fail: bool,
) -> Dict[str, object]:
    mode = "decode" if m == 1 else "prefill"
    row = {
        "case": case.name,
        "mode": mode,
        "tokens": m,
        "N": case.n,
        "groups": case.groups,
        "correct": "NA",
        "max_abs": "-",
        "max_rel": "-",
        "rstd_abs": "-",
        "mean_abs": "-",
        "triton_us": "-",
        "cuda_us": "-",
        "speedup": "-",
        "triton_GB/s": "-",
        "cuda_GB/s": "-",
        "note": "",
    }

    try:
        tensors = make_tensors(case, m, dtype, device)
        launch_triton(case, tensors, m, QWEN35_397B["rms_norm_eps"])
        launch_cuda(case, tensors, m, QWEN35_397B["rms_norm_eps"])
        torch.cuda.synchronize()

        cmp_result = compare_outputs(tensors, case, atol, rtol)
        row["correct"] = "PASS" if cmp_result["ok"] else "FAIL"
        row["max_abs"] = fmt_float(cmp_result["max_abs"], 4)
        row["max_rel"] = fmt_float(cmp_result["max_rel"], 4)
        row["rstd_abs"] = fmt_float(cmp_result["rstd_abs"], 6)
        row["mean_abs"] = fmt_float(cmp_result["mean_abs"], 6)

        if cmp_result["ok"] or bench_on_fail:
            triton_us = time_us(
                lambda: launch_triton(case, tensors, m, QWEN35_397B["rms_norm_eps"]),
                warmup,
                iters,
            )
            cuda_us = time_us(
                lambda: launch_cuda(case, tensors, m, QWEN35_397B["rms_norm_eps"]),
                warmup,
                iters,
            )
            bytes_used = logical_bytes(case, m, dtype)
            row["triton_us"] = fmt_float(triton_us, 2)
            row["cuda_us"] = fmt_float(cuda_us, 2)
            row["speedup"] = fmt_float(triton_us / cuda_us if cuda_us > 0 else None, 3)
            row["triton_GB/s"] = fmt_float(bandwidth_gbs(bytes_used, triton_us), 2)
            row["cuda_GB/s"] = fmt_float(bandwidth_gbs(bytes_used, cuda_us), 2)
        else:
            row["note"] = "skip bench on correctness failure"

        del tensors
        torch.cuda.empty_cache()
    except torch.cuda.OutOfMemoryError as exc:
        torch.cuda.empty_cache()
        row["correct"] = "OOM"
        row["note"] = str(exc).splitlines()[0][:96]
    except RuntimeError as exc:
        torch.cuda.empty_cache()
        row["correct"] = "ERR"
        row["note"] = str(exc).splitlines()[0][:96]

    return row


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        print("CUDA/HIP device is not available through torch.cuda.", file=sys.stderr)
        return 1

    device = normalize_cuda_device(args.device)
    dtype = dtype_from_name(args.dtype)
    default_atol, default_rtol = default_tolerances(dtype)
    atol = default_atol if args.atol is None else args.atol
    rtol = default_rtol if args.rtol is None else args.rtol
    tokens = [int(x) for x in args.tokens.split(",") if x.strip()]
    torch.manual_seed(args.seed)

    cases = qwen_cases()
    if args.cases != "all":
        selected = set(x.strip() for x in args.cases.split(",") if x.strip())
        cases = [case for case in cases if case.name in selected]
        missing = selected - {case.name for case in cases}
        if missing:
            raise ValueError(f"Unknown case names: {sorted(missing)}")

    print("Qwen3.5-397B captured image-size layer_norm_fwd_1pass single-op benchmark")
    print(f"device={torch.cuda.get_device_name(device)} dtype={args.dtype}")
    print(f"warmup={args.warmup} iters={args.iters} unit=us")
    print(f"correctness tolerance: atol={atol} rtol={rtol}")
    print("bandwidth uses logical operator bytes, not profiler DRAM bytes")
    print()

    rows = []
    for case in cases:
        for m in tokens:
            print(
                f"[run] case={case.name} tokens={m} N={case.n} groups={case.groups}",
                flush=True,
            )
            row = run_one(
                case=case,
                m=m,
                dtype=dtype,
                device=device,
                warmup=args.warmup,
                iters=args.iters,
                atol=atol,
                rtol=rtol,
                bench_on_fail=args.bench_on_fail,
            )
            rows.append(row)

    print_results(cases, rows, args.print)
    csv_path = write_csv(args.csv, rows)
    print(f"\nCSV saved to: {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
