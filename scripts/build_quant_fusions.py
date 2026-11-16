#!/usr/bin/env python3
"""Build INT8/FP8 transforms and strided RMS quantization fusions from normal LightOp sources.

The regular setup.py/wheel build includes the same operation in lightop.op.
Use this incremental build to update an existing installation of the same ABI.
"""

import argparse
import importlib.util
import os
import shutil
import tempfile
from pathlib import Path

import torch
from torch.utils.cpp_extension import load

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--build-dir", type=Path, required=True)
parser.add_argument(
    "--op", choices=("transform-int8", "strided-rms"), default="transform-int8"
)
parser.add_argument("--install", action="store_true")
args = parser.parse_args()
root = Path(__file__).resolve().parents[1]
args.build_dir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MAX_JOBS", "2")
flags = ["-O3", "-ffp-contract=off"]
if not torch.version.hip:
    parser.error("these incremental extensions require a ROCm PyTorch build")
flags.append("-DUSE_ROCM=1")
torch_release = tuple(int(part) for part in torch.__version__.split(".")[:2])
if torch_release >= (1, 10):
    flags.append("-DTORCH_VERSION_OVER_110=1")
if torch_release >= (2, 9):
    flags.append("-DTORCH_VERSION_OVER_290=1")
if torch_release >= (2, 12):
    flags.append("-DTORCH_VERSION_OVER_2120=1")
strided = args.op == "strided-rms"
name = "_strided_rms_quant" if strided else "_transform_quant"
source = (
    "csrc/kernels/rmsnorm/rms_norm_dynamic_per_token_quant.cu"
    if strided
    else "csrc/kernels/per_token_quant/per_token_quant_int8.cu"
)
macro = "LIGHTOP_STRIDED_RMS_ONLY" if strided else "LIGHTOP_QUANT_FUSION_ONLY"
module = load(
    name=name,
    sources=[str(root / "csrc/pybind/export.cpp"), str(root / source)]
    + ([str(root / "csrc/kernels/rmsnorm/rms_norm_fp8_quant.cu")] if strided else []),
    extra_include_paths=[str(root / "csrc"), str(root / "csrc/include")],
    build_directory=str(args.build_dir),
    extra_cflags=["-O3", "-D" + macro + "=1"],
    extra_cuda_cflags=flags,
    verbose=True,
)
print("Built", module.__file__, flush=True)
if args.install:
    spec = importlib.util.find_spec("lightop")
    destination = Path(spec.origin).parent / Path(module.__file__).name
    # A running worker may still map the old extension. Replace its directory
    # entry only after copying, so existing mappings keep their original inode.
    with tempfile.NamedTemporaryFile(
        dir=destination.parent, prefix=destination.name + ".", delete=False
    ) as temporary:
        staging = Path(temporary.name)
    try:
        shutil.copy2(module.__file__, staging)
        os.replace(staging, destination)
    finally:
        staging.unlink(missing_ok=True)
    print("Installed", destination, flush=True)
