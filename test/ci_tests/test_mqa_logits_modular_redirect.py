# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Verify modular split, op_modules.json config, and lazy redirection for lightop._mqa_logits."""

import json
from pathlib import Path
import subprocess
import sys

import lightop


def test_op_modules_json_declares_mqa_logits():
    cfg_path = Path(__file__).resolve().parents[2] / "configs" / "op_modules.json"
    assert cfg_path.is_file(), "configs/op_modules.json must exist"
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    assert "_mqa_logits" in cfg.get("modules", {})
    assert "csrc/mqa_logits/mqa_logits_pybind.cpp" in cfg["modules"]["_mqa_logits"]["sources"]


def test_mqa_logits_lazy_loaded_in_fresh_process():
    code = """
import sys
import lightop

assert "lightop._mqa_logits" not in sys.modules, "_mqa_logits should be lazy-loaded"
assert "mqa_logits" in dir(lightop.op), "dir(lightop.op) should advertise mqa_logits"
assert "mqa_logits" not in lightop.op.__dict__, "mqa_logits should not be cached before first access"

fn = lightop.op.mqa_logits
assert "lightop._mqa_logits" in sys.modules, "accessing op.mqa_logits should load _mqa_logits"
import lightop._mqa_logits as mqa_mod
assert fn is mqa_mod.mqa_logits, "op.mqa_logits must redirect to the native pybind function"
assert lightop.op.__dict__.get("mqa_logits") is mqa_mod.mqa_logits, "first access must cache into op.__dict__"
"""
    subprocess.run([sys.executable, "-c", code], check=True)


def test_mqa_logits_direct_and_redirected_identity():
    import lightop._mqa_logits as mqa_mod

    assert lightop.op.mqa_logits is mqa_mod.mqa_logits
    assert mqa_mod.__file__.endswith(".so")
    assert lightop.op.__file__ != mqa_mod.__file__


def test_mqa_logits_monkeypatch_delattr_behavior(monkeypatch):
    assert hasattr(lightop.op, "mqa_logits")
    monkeypatch.delattr(lightop.op, "mqa_logits")
    assert not hasattr(lightop.op, "mqa_logits")

    monkeypatch.undo()
    assert hasattr(lightop.op, "mqa_logits")
