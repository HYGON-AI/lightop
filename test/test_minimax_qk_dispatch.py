# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Verify public rotary dispatch without initializing a GPU."""

import importlib.util
from pathlib import Path
from unittest.mock import Mock

import pytest


@pytest.mark.parametrize("rotary,expected", [(0, 5), (64, 6), (128, 5)])
def test_automatic_rotary_dispatch(monkeypatch, rotary, expected):
    path = (
        Path(__file__).resolve().parents[1] / "python/lightop/minimax_qk_norm_rope.py"
    )
    spec = importlib.util.spec_from_file_location("qk_dispatch_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    native = Mock()
    outputs = tuple(object() for _ in range(4))
    native.minimax_qk_norm_rope.return_value = outputs
    monkeypatch.setattr(module, "_native", lambda: native)
    monkeypatch.setenv("LIGHTOP_MINIMAX_QK_VARIANT", "1")
    inputs = [object() for _ in range(10)]
    actual = module.minimax_qk_norm_rope(*inputs, 1e-5, 128, rotary, True)
    assert actual == outputs
    native.minimax_qk_norm_rope.assert_called_once_with(
        *inputs, 1e-5, rotary, True, expected
    )
