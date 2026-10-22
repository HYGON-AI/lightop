# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""CPU regression tests for the gfx936 / 80-CU small-M configuration.

Load the source configuration and selector without importing the native extension.
These tests check lookup coverage, not GPU numerical correctness or performance.
"""

import ast
import functools
import importlib.metadata
import importlib.util
import os
from pathlib import Path
import unittest
from unittest.mock import patch


SOURCE_ROOT = Path(__file__).resolve().parents[2] / "python" / "lightop"
# PR CI checks out only test/ci_tests after installing this PR's build.
if not SOURCE_ROOT.is_dir():
    SOURCE_ROOT = Path(importlib.metadata.distribution("lightop").locate_file("lightop"))
WKV_CONFIG = (4, 16, 256, 1, False)
OBSERVED_CONFIGS = {
    (5, 1280, 5120): (4, 16, 256, 1, False),
    (5, 4608, 5120): (2, 16, 256, 1, True),
    (5, 5120, 2304): (2, 16, 256, 1, True),
    (5, 32768, 1280): (1, 32, 256, 1, False),
    (6, 1792, 5120): (2, 16, 256, 1, False),
    (6, 4096, 1280): (1, 16, 256, 1, False),
    (6, 4608, 5120): (2, 16, 256, 1, True),
    (6, 5120, 2304): (2, 16, 256, 1, True),
    (6, 25600, 6144): (16, 16, 256, 1, False),
    (6, 32768, 1280): (1, 32, 256, 1, False),
    (10, 1280, 5120): (4, 16, 256, 1, False),
    (10, 4608, 5120): (4, 32, 256, 1, False),
    (10, 5120, 2304): (2, 32, 256, 1, False),
    (10, 32768, 1280): (1, 64, 256, 1, False),
    (12, 1792, 5120): (4, 32, 256, 1, False),
    (12, 4096, 1280): (2, 32, 256, 1, False),
    (12, 4608, 5120): (4, 32, 256, 1, False),
    (12, 5120, 2304): (2, 32, 256, 1, False),
    (12, 25600, 6144): (1, 64, 256, 1, False),
    (12, 32768, 1280): (1, 64, 256, 1, False),
    (15, 1280, 5120): (4, 16, 256, 1, False),
    (15, 4608, 5120): (4, 32, 256, 1, False),
    (15, 5120, 2304): (2, 32, 256, 1, False),
    (15, 32768, 1280): (1, 64, 256, 1, True),
}
EXPECTED_CONFIGS = {
    **{(m, 512, 5120): WKV_CONFIG for m in range(1, 17)},
    **OBSERVED_CONFIGS,
}


class TestW8A8SmallMConfig(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location(
            "lightop_source_small_m_config", SOURCE_ROOT / "config_smooth_gemm.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        cls.config_class = module.gemm_w8a8_smooth_config
        # Exercise the real selector body/decorator without importing torch/op.
        path = SOURCE_ROOT / "gemmopt.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        selector = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                        and node.name == "w8a8_smooth_quant_config")
        namespace = {"functools": functools, "gemm_w8a8_smooth_config": cls.config_class}
        exec(compile(ast.Module(body=[selector], type_ignores=[]), str(path), "exec"), namespace)
        cls.selector = staticmethod(namespace[selector.name])

    def setUp(self):
        target = patch.dict(os.environ, {"LIGHTOP_GPU_TARGET": "gfx936", "LIGHTOP_GPU_CUS": "80"})
        target.start()
        self.addCleanup(target.stop)
        self.selector.cache_clear()
        self.addCleanup(self.selector.cache_clear)

    def test_all_40_exact_configurations(self):
        self.assertEqual(len(EXPECTED_CONFIGS), 40)
        table = self.config_class.get_w8a8_smooth_quant_config_dict()
        for (m, n, k), expected in EXPECTED_CONFIGS.items():
            with self.subTest(m=m, n=n, k=k):
                self.assertEqual(table.get(f"M{m}N{n}K{k}"), expected)

    def test_selector_resolves_all_40_shapes(self):
        for shape, expected in EXPECTED_CONFIGS.items():
            with self.subTest(shape=shape):
                self.assertEqual(self.selector(*shape), (*expected, True))
                self.assertEqual(self.selector(*shape), (*expected, True))  # Cached lookup.

    def test_existing_configuration_is_unchanged(self):
        self.assertEqual(self.selector(1, 2112, 7168), (2, 16, 256, 1, False, True))

    def test_unknown_configuration_still_reports_a_miss(self):
        self.assertEqual(self.selector(1, 17, 31), (-1, -1, -1, -1, False, False))


if __name__ == "__main__":
    unittest.main()
