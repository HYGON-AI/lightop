"""Ensure migrated LightOp package code does not import the lmslim package."""

import ast
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[1] / "python" / "lightop"


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
