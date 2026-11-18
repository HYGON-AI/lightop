# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""The GDN loader prefers lib/gdn/<arch>/ and keeps the flat fallback."""

import importlib.util
import pathlib
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _load_gdn():
    """Load gdn.py alone; only _candidates is exercised, so the package
    __init__ (which needs the compiled .op) does not have to run."""
    spec = importlib.util.spec_from_file_location(
        "_lightop_gdn_under_test", ROOT / "python" / "lightop" / "gdn.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


gdn = _load_gdn()


class GdnLoaderTest(unittest.TestCase):
    def test_arch_specific_path_comes_first(self):
        lib = pathlib.Path("/tmp/lightop-lib")
        paths = [str(p) for p in gdn._candidates(lib, "gfx938")]
        self.assertTrue(paths[0].startswith("/tmp/lightop-lib/gdn/gfx938/_gdn"))
        self.assertEqual(paths[-1].count("/gdn/"), 0)
        self.assertIn("/tmp/lightop-lib/_gdn", paths[-1])

    def test_no_arch_falls_back_to_the_flat_layout(self):
        lib = pathlib.Path("/tmp/lightop-lib")
        paths = [str(p) for p in gdn._candidates(lib, None)]
        self.assertTrue(paths)
        self.assertTrue(all("/gdn/" not in p for p in paths))


if __name__ == "__main__":
    unittest.main()
