#!/usr/bin/env python3
"""Generate compatibility manifests for the LightOp/lmslim migration."""

from __future__ import annotations

import argparse
import ast
import re
from pathlib import Path


PYBIND_DEF = re.compile(r'm\.def\(\s*"([^"]+)"')


def public_python_api(package_root: Path) -> list[str]:
    tree = ast.parse((package_root / "__init__.py").read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if alias.name != "*":
                    names.add(alias.asname or alias.name)
    return sorted(names)


def extension_symbols(csrc_root: Path) -> list[str]:
    names: set[str] = set()
    for path in sorted(csrc_root.rglob("*")):
        if not path.is_file() or path.suffix not in {".cpp", ".cu", ".cc"}:
            continue
        if "allreduce_fuse" in path.parts:
            continue
        names.update(PYBIND_DEF.findall(path.read_text(encoding="utf-8", errors="ignore")))
    return sorted(names)


def lmslim_requirements(lmslim_root: Path) -> list[str]:
    requirements: set[str] = set()
    package_root = lmslim_root / "lmslim"
    if package_root.is_dir():
        lmslim_root = package_root
    for path in sorted(lmslim_root.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        module_aliases: dict[str, str] = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                if node.module == "lightop" or node.module.startswith("lightop."):
                    for alias in node.names:
                        full_name = f"{node.module}.{alias.name}"
                        requirements.add(full_name)
                        module_aliases[alias.asname or alias.name] = full_name
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name == "lightop" or alias.name.startswith("lightop."):
                        requirements.add(alias.name)
                        module_aliases[alias.asname or alias.name.split(".")[0]] = alias.name

        for node in ast.walk(tree):
            if not isinstance(node, ast.Attribute) or not isinstance(node.value, ast.Name):
                continue
            base = module_aliases.get(node.value.id)
            if base:
                requirements.add(f"{base}.{node.attr}")
    return sorted(requirements)


def render(items: list[str]) -> str:
    return "".join(f"{item}\n" for item in items)


def update_or_check(path: Path, content: str, check: bool) -> None:
    if check:
        actual = path.read_text(encoding="utf-8") if path.exists() else ""
        if actual != content:
            raise SystemExit(f"compatibility manifest is stale: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--lmslim-root", type=Path, required=True)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    outputs = {
        "lightop_python_api.txt": public_python_api(repo_root / "lightop"),
        "lightop_op_symbols.txt": extension_symbols(repo_root / "lightop" / "csrc"),
        "lmslim_required_lightop_api.txt": lmslim_requirements(args.lmslim_root.resolve()),
    }
    for filename, items in outputs.items():
        update_or_check(repo_root / "compat" / filename, render(items), args.check)
        print(f"{filename}: {len(items)} entries")


if __name__ == "__main__":
    main()
