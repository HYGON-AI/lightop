# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Old native packages must not prevent unrelated activation imports."""
import importlib.util
import sys
import types
from pathlib import Path

import pytest
import torch


def test_old_native_without_optional_extension(monkeypatch):
    package = types.ModuleType("lightop")
    package.__path__ = []
    package.op = types.SimpleNamespace()
    monkeypatch.setitem(sys.modules, "lightop", package)
    monkeypatch.delitem(sys.modules, "lightop.lightop_oai_int8_candidate", raising=False)
    path = Path(__file__).resolve().parents[2] / "python/lightop/activation/swiglu_oai_int8_quant.py"
    spec = importlib.util.spec_from_file_location("oai_int8_old_runtime_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module._native is None
    with pytest.raises(RuntimeError, match="rebuild LightOp"):
        module.swiglu_oai_int8_quant_contiguous(torch.empty((1, 16)))
