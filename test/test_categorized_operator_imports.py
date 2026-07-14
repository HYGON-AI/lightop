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
ROOT_COMPATIBILITY_APIS = {
    "test_quant_blockwise_scaled_cast_transpose_api.py": (
        "blockwise_scaled_cast_transpose"
    ),
    "test_quant_destindex_copy_quantize_int8_cuda.py": (
        "destindex_copy_quantize_int8_cuda"
    ),
    "test_quant_destindex_dequantize_int8_cuda.py": (
        "destindex_dequantize_int8_cuda"
    ),
    "test_quant_fp8_to_16.py": "fp8_to_16",
    "test_quant_w4tow16.py": "w4tow16",
}


NATIVE_OPERATOR_REFERENCE_TESTS = {
    "test_fuse_rmsnorm_rope_qkv_no_quant.py",
    "test_native_categorized_attention_api.py",
    "test_native_categorized_norm_activation_api.py",
    "test_per_token_group_quant_fp8_api_alignment.py",
    "test_quant_per_token_group_quant_fp8.py",
    "test_quant_per_token_quant_fp8.py",
    "test_split_qkv_rms_rotary_embedding_fuse_with_kv_store_quant.py",
}
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
        if (
            path.name.endswith("_existing_api.py")
            or path.name in ROOT_COMPATIBILITY_APIS
            or path == Path(__file__)
        ):
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
        op_aliases = {
            alias.asname or "op"
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module == "lightop"
            for alias in node.names
            if alias.name == "op"
        }
        allows_native_reference = (
            path.name in NATIVE_OPERATOR_REFERENCE_TESTS
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
            elif (
                not allows_native_reference
                and isinstance(node, ast.Attribute)
                and isinstance(node.value, ast.Name)
                and node.value.id in op_aliases
                and node.attr in owners
            ):
                violations.append(
                    f"{path.relative_to(TEST_ROOT)}:{node.lineno}:"
                    f"{node.value.id}.{node.attr}->{owners[node.attr]}"
                )
            elif (
                not allows_native_reference
                and imports_lightop
                and isinstance(node, ast.Attribute)
                and isinstance(node.value, ast.Attribute)
                and isinstance(node.value.value, ast.Name)
                and node.value.value.id == "lightop"
                and node.value.attr == "op"
                and node.attr in owners
            ):
                violations.append(
                    f"{path.relative_to(TEST_ROOT)}:{node.lineno}:"
                    f"lightop.op.{node.attr}->{owners[node.attr]}"
                )

    assert not violations, "uncategorized LightOp operator calls remain: " + ", ".join(
        violations
    )


def _attribute_path(node):
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
        return tuple(reversed(parts))
    return ()


def test_root_compatibility_checks_compare_category_and_root_exports():
    for filename, api_name in ROOT_COMPATIBILITY_APIS.items():
        path = TEST_ROOT / filename
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        attribute_paths = {
            _attribute_path(node)
            for node in ast.walk(tree)
            if isinstance(node, ast.Attribute)
        }
        assert ("lightop", "quant", api_name) in attribute_paths
        assert ("lightop", api_name) in attribute_paths

        category_imports = sum(
            alias.name == "lightop.quant"
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            for alias in node.names
        )
        assert category_imports == 1
