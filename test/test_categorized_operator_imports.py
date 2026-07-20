"""Prevent migrated tests from calling categorized APIs through lightop root."""

import ast
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = ROOT / "python" / "lightop"
TEST_ROOT = ROOT / "test"
CATEGORIES = [
    "activation",
    "attention",
    "gemm_ops",
    "kvcache",
    "loss",
    "moe",
    "norm",
    "quant",
    "sampling",
    "tensor",
    "vision",
]
PRIMARY_OVERRIDES = {"per_token_quant_int8": "quant"}


def _categorized_api_owners():
    owners = defaultdict(list)
    for category in CATEGORIES:
        path = PACKAGE_ROOT / category / "__init__.py"
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in tree.body:
            if isinstance(node, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id == "__all__"
                for target in node.targets
            ):
                for name in ast.literal_eval(node.value):
                    owners[name].append(category)
    for name, category in PRIMARY_OVERRIDES.items():
        owners[name] = [category]
    return owners


def test_migrated_tests_use_categorized_operator_imports():
    owners = _categorized_api_owners()
    violations = []
    for path in TEST_ROOT.rglob("*.py"):
        if path.name.endswith("_existing_api.py") or path == Path(__file__):
            continue
        try:
            tree = ast.parse(
                path.read_text(encoding="utf-8"), filename=str(path)
            )
        except SyntaxError:
            continue

        imports_lightop = any(
            isinstance(node, ast.Import)
            and any(
                alias.name == "lightop"
                and alias.asname in (None, "lightop")
                for alias in node.names
            )
            for node in ast.walk(tree)
        )
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module == "lightop":
                for alias in node.names:
                    if alias.name in owners:
                        violations.append(
                            f"{path.relative_to(TEST_ROOT)}:{node.lineno}:"
                            f"{alias.name}->{owners[alias.name]}"
                        )
            elif (
                imports_lightop
                and isinstance(node, ast.Attribute)
                and isinstance(node.value, ast.Name)
                and node.value.id == "lightop"
                and node.attr in owners
            ):
                violations.append(
                    f"{path.relative_to(TEST_ROOT)}:{node.lineno}:"
                    f"lightop.{node.attr}->{owners[node.attr]}"
                )

    assert not violations, "root LightOp operator imports remain: " + ", ".join(
        violations
    )
