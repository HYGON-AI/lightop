"""Ensure migrated LightOp package code does not import the lmslim package."""

import ast
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[1] / "python" / "lightop"
TEST_ROOT = Path(__file__).resolve().parent

MOE_CATEGORY_TESTS = [
    "test_moe_sum_opt.py",
    "test_fused_moe_gate.py",
    "test_kimi_k2_moe_fused_gate.py",
    "test_moe_align_block_size.py",
    "test_moe_fused_gate_sqrtsoftplus.py",
    "test_fuse_groupgemm_permute_quant.py",
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
