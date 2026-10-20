# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

import os
import subprocess
from pathlib import Path

import torch

ROOT_DIR = Path(__file__).parent.resolve()


def _run_cmd(cmd, shell=False):
    try:
        return subprocess.check_output(cmd, cwd=ROOT_DIR, stderr=subprocess.DEVNULL, shell=shell).decode("ascii").strip()
    except Exception:
        return None


def _get_version():
    if os.path.exists(ROOT_DIR / "version.txt"):
        with open(ROOT_DIR / "version.txt", "r") as f:
            version = f.read().strip()
    else:
        version = '0.6.0'
    if os.getenv("BUILD_VERSION"):
        version = os.getenv("BUILD_VERSION")
    return version


def _make_version_file(version, sha, abi, dtk, torch_version, branch):
    sha = "Unknown" if sha is None else sha
    torch_version = '.'.join(torch_version.split('.')[:2])
    
    if os.getenv("USE_LONG_VERSION", "false").lower() in ["1", "true"]:
        hcu_version = f"{version}+das1.1git{sha}.abi{abi}.dtk{dtk}.torch{torch_version}"
    else:
        hcu_version = f"{version}+das.dtk{dtk}"

    print("hcu_version")
    print(hcu_version)
    version_path = ROOT_DIR / "python" / "lightop" / "version.py"
    with open(version_path, "w") as f:
        f.write(f"version = '{version}'\n")
        f.write(f"git_hash = '{sha}'\n")
        f.write(f"git_branch = '{branch}'\n")
        f.write(f"abi = 'abi{abi}'\n")
        f.write(f"dtk = '{dtk}'\n")
        f.write(f"torch_version = '{torch_version}'\n")
        f.write(f"hcu_version = '{hcu_version}'\n")
    return hcu_version

def get_dtk_version(ROCM_HOME):
    dtk_version = _run_cmd(["cat", os.path.join(ROCM_HOME, '.info/rocm_version')])
    dtk_version = ''.join(dtk_version.split('.'))
    dtk_version = int(dtk_version)
    return dtk_version



def _get_pytorch_version():
    if "PYTORCH_VERSION" in os.environ:
        return f"{os.environ['PYTORCH_VERSION']}"
    return torch.__version__

def get_version(ROCM_HOME):
    sha = _run_cmd(["git", "rev-parse", "HEAD"])
    if sha is not None :
        sha = sha[:7]
    branch = _run_cmd(["git", "rev-parse", "--abbrev-ref", "HEAD"])
    tag = _run_cmd(["git", "describe", "--tags", "--exact-match", "@"])
    print("-- Git branch:", branch)
    print("-- Git SHA:", sha)
    print("-- Git tag:", tag)
    torch_version = _get_pytorch_version()
    print("-- PyTorch:", torch_version)
    version = _get_version()
    print("-- Building version", version)
    abi = _run_cmd(["echo '#include <string>' | gcc -x c++ -E -dM - | fgrep _GLIBCXX_USE_CXX11_ABI | awk '{print $3}'"], shell=True)
    print("-- _GLIBCXX_USE_CXX11_ABI:", abi)
    dtk = _run_cmd(["cat", os.path.join(ROCM_HOME, '.info/rocm_version')])
    dtk = ''.join(dtk.split('.'))
    print("-- DTK:", dtk)

    return _make_version_file(version, sha, abi, dtk, torch_version, branch)