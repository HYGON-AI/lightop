# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Verify public rotary dispatch without initializing a GPU."""

import importlib
from unittest.mock import Mock

import pytest


@pytest.mark.parametrize("rotary,expected", [(0, 5), (64, 6), (128, 5)])
def test_automatic_rotary_dispatch(monkeypatch, rotary, expected):
    module = importlib.import_module("lightop.minimax_qk_norm_rope")
    native = Mock()
    outputs = tuple(object() for _ in range(4))
    native.minimax_qk_norm_rope.return_value = outputs
    monkeypatch.setattr(module, "_native", lambda: native)
    inputs = [object() for _ in range(10)]
    actual = module.minimax_qk_norm_rope(*inputs, 1e-5, 128, rotary, True)
    assert actual == outputs
    native.minimax_qk_norm_rope.assert_called_once_with(
        *inputs, 1e-5, rotary, True, expected
    )
