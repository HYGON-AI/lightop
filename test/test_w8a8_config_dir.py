# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

import importlib
import sys
from pathlib import Path
from types import ModuleType


REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_PACKAGE_ROOT = REPO_ROOT / "python" / "lightop"
TEST_PACKAGE_NAME = "lightop_source_test"


def load_source_config_module():
    """Load config.py from this checkout without requiring an in-place op build."""
    package = ModuleType(TEST_PACKAGE_NAME)
    package.__path__ = [str(SOURCE_PACKAGE_ROOT)]
    sys.modules[TEST_PACKAGE_NAME] = package
    sys.modules.pop(f"{TEST_PACKAGE_NAME}.config", None)
    sys.modules.pop(f"{TEST_PACKAGE_NAME}.utils", None)
    return importlib.import_module(f"{TEST_PACKAGE_NAME}.config")


def test_get_w8a8_config_dir_uses_lightop_resource_path(monkeypatch):
    config = load_source_config_module()
    calls = []

    def fake_resource_path(*parts):
        calls.append(parts)
        return "/lightop/configs/w8a8"

    monkeypatch.setattr(config, "resource_path", fake_resource_path)

    assert config.get_w8a8_config_dir() == "/lightop/configs/w8a8"
    assert calls == [("configs", "w8a8")]


def test_get_w8a8_config_dir_resolves_checkout_resources():
    config = load_source_config_module()
    config_dir = Path(config.get_w8a8_config_dir())

    assert config_dir.samefile(REPO_ROOT / "configs" / "w8a8")
    assert any(config_dir.glob("W8A8_*.json"))
