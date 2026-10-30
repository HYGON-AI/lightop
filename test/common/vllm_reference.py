# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Load vLLM reference kernels, including its split stable-libtorch extensions."""

from importlib import import_module
from importlib.util import find_spec

import torch

from vllm import _custom_ops as vllm_ops

# Some platform plugins import only vllm._C. Recent vLLM wheels register
# the activation, quantization and cache references in separate libraries.
# Keep older wheels compatible, but propagate errors from libraries that exist.
for _module in ("vllm._C_stable_libtorch", "vllm._moe_C_stable_libtorch"):
    if find_spec(_module) is not None:
        import_module(_module)

del _module

# Newer vLLM wheels expose these reference kernels through torch.ops.
vllm_activation_ops = torch.ops._C
