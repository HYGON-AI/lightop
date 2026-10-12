# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

import inspect

import pytest
import torch

from lightop import gemm_ops
from lightop._lmslim_native.quantize import quant_tools


def test_triton_int8_gemm_helper_is_direct_public_binding():
    assert gemm_ops.triton_int8_gemm_helper is quant_tools.triton_int8_gemm_helper
    assert list(inspect.signature(gemm_ops.triton_int8_gemm_helper).parameters) == [
        "m",
        "n",
        "k",
        "per_token_act_quant",
        "per_out_channel_weight_quant",
        "use_bias",
        "out_dtype",
        "device",
        "best_config",
        "repeat",
    ]


def test_triton_int8_gemm_helper_runs_gpu_warmup():
    if not torch.cuda.is_available():
        pytest.skip("requires a CUDA/ROCm device")

    result = gemm_ops.triton_int8_gemm_helper(
        128,
        256,
        512,
        per_token_act_quant=True,
        per_out_channel_weight_quant=True,
        use_bias=True,
        out_dtype=torch.float16,
        device="cuda",
        best_config=None,
        repeat=1,
    )
    assert result is None
