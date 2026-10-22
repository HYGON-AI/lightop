# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Ensure migrated LightOp package code does not import the lmslim package."""

import ast
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[1] / "python" / "lightop"
TEST_ROOT = Path(__file__).resolve().parent
MOE_MARLIN_TEST_ROOT = TEST_ROOT / "moe_marlin"

MOE_CATEGORY_TESTS = [
    "test_moe_sum_opt.py",
    "test_fused_moe_gate.py",
    "test_kimi_k2_moe_fused_gate.py",
    "test_moe_align_block_size.py",
    "test_moe_fused_gate_sqrtsoftplus.py",
    "test_fuse_groupgemm_permute_quant.py",
    "moe_marlin/tuning/EP_tuning/fused_marlin_moe.py",
    "moe_marlin/tuning/EP_tuning/fused_marlin_moe_valid.py",
]


def test_lightop_package_has_no_lmslim_python_imports():
    violations = []
    for path in PACKAGE_ROOT.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module and (
                node.module == "lmslim" or node.module.startswith("lmslim.")
            ):
                violations.append(f"{path.relative_to(PACKAGE_ROOT)}:{node.lineno}")
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    # lmslimquant is the separately built native extension, not
                    # the lmslim Python package.
                    if alias.name == "lmslim" or alias.name.startswith("lmslim."):
                        violations.append(
                            f"{path.relative_to(PACKAGE_ROOT)}:{node.lineno}"
                        )

    assert not violations, "external lmslim imports remain: " + ", ".join(violations)


def test_moe_category_tests_do_not_import_top_level_op():
    violations = []
    for filename in MOE_CATEGORY_TESTS:
        path = TEST_ROOT / filename
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module == "lightop":
                if any(alias.name == "op" for alias in node.names):
                    violations.append(f"{filename}:{node.lineno}")
            elif isinstance(node, ast.Import):
                if any(alias.name == "lightop.op" for alias in node.names):
                    violations.append(f"{filename}:{node.lineno}")

    assert not violations, "top-level lightop.op imports remain: " + ", ".join(
        violations
    )


def test_moe_marlin_scripts_use_migrated_lightop_imports():
    violations = []
    for path in MOE_MARLIN_TEST_ROOT.rglob("*.py"):
        # A few historical tuning templates are intentionally incomplete, so
        # scan import statements without requiring every script to parse.
        for lineno, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            statement = line.strip()
            if statement.startswith("#"):
                continue
            imports_lmslim = statement.startswith(("from lmslim ", "from lmslim."))
            imports_lmslim = imports_lmslim or statement.startswith("import lmslim")
            imports_top_level_op = statement.startswith("import lightop.op")
            if statement.startswith("from lightop import"):
                imported_names = statement.partition("import")[2].split(",")
                imports_top_level_op = imports_top_level_op or any(
                    name.strip() and name.strip().split()[0] == "op"
                    for name in imported_names
                )
            if imports_lmslim or imports_top_level_op:
                violations.append(f"{path.relative_to(TEST_ROOT)}:{lineno}")

    assert not violations, "legacy MoE test imports remain: " + ", ".join(
        violations
    )
