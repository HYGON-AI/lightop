# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Old native packages must not prevent unrelated activation imports."""
import importlib.util
import sys
import types

import pytest
import torch


def test_old_native_without_optional_extension(monkeypatch):
    # CI installs LightOp, then checks out only test/. Resolve the installed
    # module before replacing its parent package with the legacy-runtime stub.
    installed_spec = importlib.util.find_spec(
        "lightop.activation.swiglu_oai_int8_quant"
    )
    assert installed_spec is not None and installed_spec.origin is not None
    path = installed_spec.origin

    package = types.ModuleType("lightop")
    package.__path__ = []
    package.op = types.SimpleNamespace()
    monkeypatch.setitem(sys.modules, "lightop", package)
    monkeypatch.delitem(sys.modules, "lightop.lightop_oai_int8_candidate", raising=False)
    spec = importlib.util.spec_from_file_location("oai_int8_old_runtime_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module._native is None
    with pytest.raises(RuntimeError, match="rebuild LightOp"):
        module.swiglu_oai_int8_quant_contiguous(torch.empty((1, 16)))
