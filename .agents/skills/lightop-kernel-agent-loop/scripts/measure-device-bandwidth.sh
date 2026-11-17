#!/usr/bin/env bash
#
# Measure selected-device memory bandwidth before LightOp tuning.
#
# Direct mode:
#   bash measure-device-bandwidth.sh --output .lightop-kernel-agent/device-bandwidth.txt --hip-visible-devices 0
#
# Docker mode:
#   bash measure-device-bandwidth.sh --docker <container> --workdir /workspace/lightop \
#     --output .lightop-kernel-agent/device-bandwidth.txt --hip-visible-devices 0

set -euo pipefail

CONTAINER=""
WORKDIR=""
OUTPUT=".lightop-kernel-agent/device-bandwidth.txt"
HIP_VISIBLE=""
MAX_MIB="512"
MIN_MIB="16"
ITERS="80"
WARMUP="20"
DTYPE="float32"
PYTHON_BIN="${PYTHON:-python}"

usage() {
    sed -n '2,10p' "$0" >&2
    cat >&2 <<'EOF'

Options:
  --docker <container>          Run the measurement inside this Docker container.
  --workdir <path>              Container or local LightOp root. Required with --docker.
  --output <path>               Output path relative to workdir/cwd unless absolute.
  --hip-visible-devices <id>    Value for HIP_VISIBLE_DEVICES during measurement.
  --max-mib <n>                 Maximum bytes per buffer in MiB. Default: 512.
  --min-mib <n>                 Minimum bytes per buffer in MiB. Default: 16.
  --iters <n>                   Timed iterations. Default: 80.
  --warmup <n>                  Warmup iterations. Default: 20.
  --dtype <torch dtype>         float32, float16, bfloat16. Default: float32.
EOF
}

shell_quote() {
    printf '%q' "$1"
}

emit_python() {
    cat <<'PY'
import argparse
import datetime as _dt
import os
import platform
import sys
import time

import torch


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=".lightop-kernel-agent/device-bandwidth.txt")
    parser.add_argument("--max-mib", type=int, default=512)
    parser.add_argument("--min-mib", type=int, default=16)
    parser.add_argument("--iters", type=int, default=80)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--dtype", default="float32")
    return parser.parse_args()


def dtype_from_name(name):
    mapping = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "float": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "half": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    if name not in mapping:
        raise SystemExit(f"unsupported dtype: {name}")
    return mapping[name]


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("torch.cuda is not available in this environment")

    dtype = dtype_from_name(args.dtype)
    torch.cuda.init()
    device_index = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device_index)
    free, total = torch.cuda.mem_get_info()

    min_bytes = max(1, args.min_mib) << 20
    max_bytes = max(args.min_mib, args.max_mib) << 20
    bytes_per_buf = max(min_bytes, min(max_bytes, int(free // 5)))
    elem_size = torch.empty((), device="cuda", dtype=dtype).element_size()
    n = max(1, bytes_per_buf // elem_size)
    bytes_per_buf = n * elem_size

    a = torch.empty(n, device="cuda", dtype=dtype)
    b = torch.empty_like(a)
    c = torch.empty_like(a)
    a.fill_(1.0)
    b.fill_(2.0)
    c.zero_()
    torch.cuda.synchronize()

    def bench(name, fn, bytes_moved):
        for _ in range(args.warmup):
            fn()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(args.iters):
            fn()
        torch.cuda.synchronize()
        seconds = (time.perf_counter() - t0) / args.iters
        tbps = bytes_moved / seconds / 1e12
        return name, tbps, seconds, bytes_moved

    rows = [
        bench("write_fill", lambda: a.fill_(3.0), bytes_per_buf),
        bench("copy_read_write", lambda: c.copy_(a), bytes_per_buf * 2),
        bench("triad_2read_1write", lambda: torch.add(a, b, out=c), bytes_per_buf * 3),
        bench("read_reduce", lambda: torch.sum(a), bytes_per_buf),
    ]

    lines = [
        "device_bandwidth_calibration:",
        f"  timestamp_utc: {_dt.datetime.utcnow().isoformat(timespec='seconds')}Z",
        f"  host: {platform.node()}",
        f"  cwd: {os.getcwd()}",
        f"  python: {sys.version.split()[0]}",
        f"  torch: {torch.__version__}",
        f"  hip: {getattr(torch.version, 'hip', None)}",
        f"  hip_visible_devices: {os.environ.get('HIP_VISIBLE_DEVICES', '')}",
        f"  device_index: {device_index}",
        f"  device_name: {torch.cuda.get_device_name(device_index)}",
        f"  gcn_arch: {getattr(props, 'gcnArchName', '')}",
        f"  dtype: {args.dtype}",
        f"  buffer_bytes: {bytes_per_buf}",
        f"  total_mem_bytes: {total}",
        f"  free_mem_bytes_at_start: {free}",
        f"  warmup: {args.warmup}",
        f"  iters: {args.iters}",
        "  results:",
    ]
    for name, tbps, seconds, bytes_moved in rows:
        lines.extend([
            f"    {name}:",
            f"      tbps: {tbps:.6f}",
            f"      us_per_iter: {seconds * 1e6:.3f}",
            f"      bytes_moved: {bytes_moved}",
        ])

    text = "\n".join(lines) + "\n"
    print(text, end="")
    if args.output:
        output_dir = os.path.dirname(os.path.abspath(args.output))
        os.makedirs(output_dir, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as fh:
            fh.write(text)


if __name__ == "__main__":
    main()
PY
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --docker) CONTAINER="$2"; shift 2 ;;
        --workdir) WORKDIR="$2"; shift 2 ;;
        --output) OUTPUT="$2"; shift 2 ;;
        --hip-visible-devices) HIP_VISIBLE="$2"; shift 2 ;;
        --max-mib) MAX_MIB="$2"; shift 2 ;;
        --min-mib) MIN_MIB="$2"; shift 2 ;;
        --iters) ITERS="$2"; shift 2 ;;
        --warmup) WARMUP="$2"; shift 2 ;;
        --dtype) DTYPE="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Error: unknown argument: $1" >&2; usage; exit 1 ;;
    esac
done

if [[ -n "$CONTAINER" ]]; then
    if [[ -z "$WORKDIR" ]]; then
        echo "Error: --workdir is required with --docker" >&2
        exit 1
    fi
    q_workdir="$(shell_quote "$WORKDIR")"
    q_output="$(shell_quote "$OUTPUT")"
    q_max="$(shell_quote "$MAX_MIB")"
    q_min="$(shell_quote "$MIN_MIB")"
    q_iters="$(shell_quote "$ITERS")"
    q_warmup="$(shell_quote "$WARMUP")"
    q_dtype="$(shell_quote "$DTYPE")"
    hip_prefix=""
    if [[ -n "$HIP_VISIBLE" ]]; then
        hip_prefix="HIP_VISIBLE_DEVICES=$(shell_quote "$HIP_VISIBLE") "
    fi
    inner="cd $q_workdir && mkdir -p .lightop-kernel-agent && ${hip_prefix}$PYTHON_BIN - --output $q_output --max-mib $q_max --min-mib $q_min --iters $q_iters --warmup $q_warmup --dtype $q_dtype"
    emit_python | docker exec -i "$CONTAINER" bash -lc "$inner"
else
    if [[ -n "$WORKDIR" ]]; then
        cd "$WORKDIR"
    fi
    mkdir -p "$(dirname "$OUTPUT")"
    if [[ -n "$HIP_VISIBLE" ]]; then
        export HIP_VISIBLE_DEVICES="$HIP_VISIBLE"
    fi
    emit_python | "$PYTHON_BIN" - --output "$OUTPUT" --max-mib "$MAX_MIB" --min-mib "$MIN_MIB" --iters "$ITERS" --warmup "$WARMUP" --dtype "$DTYPE"
fi
