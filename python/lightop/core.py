# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Build metadata, static symbol registry generation, and lazy native-module routing.

Reading this file (including through runpy from setup.py) only loads JSON.
Native extensions are imported exclusively through load_ops() / preload_ops()
or on first attribute access on lightop.op.
"""
import importlib
import importlib.util
import json
import os
import re
import sys
from pathlib import Path
from types import ModuleType

DEFAULT_LIBRARIES = ["rocblas", "hipblas", "hipblaslt"]
_BINDING_SYMBOL_RE = re.compile(
    r'\b(?:m|module)\s*\.\s*(?:def|attr)\s*\(\s*"([A-Za-z_][A-Za-z0-9_]*)"'
)
_LOADED_SUBMODULES = set()


def _unique_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate registry key: {key!r}")
        result[key] = value
    return result


def load_registry(path=None):
    """Read module_registry.json, validate schema, and fill optional defaults."""
    path = Path(path) if path is not None else Path(__file__).with_name("module_registry.json")
    with path.open(encoding="utf-8") as stream:
        modules = json.load(stream, object_pairs_hook=_unique_keys)
    if not isinstance(modules, dict) or not modules:
        raise ValueError("Module registry must be a non-empty object")

    required = {"extension", "bindings", "sources"}
    allowed = required | {"include_dirs", "libraries", "extra_compile_args", "arch_profile"}
    extensions = set()

    for name, spec in modules.items():
        if not name.isidentifier() or name == "op" or not isinstance(spec, dict):
            raise ValueError(f"Invalid module definition: {name!r}")
        if required - set(spec) or set(spec) - allowed:
            raise ValueError(
                f"{name}: missing fields {sorted(required - set(spec))}; "
                f"unknown fields {sorted(set(spec) - allowed)}"
            )
        extension = spec["extension"]
        if not isinstance(extension, str) or not extension:
            raise ValueError(f"{name}.extension: expected a non-empty string")
        if extension in extensions:
            raise ValueError(f"{name}: duplicate extension name {extension!r}")
        extensions.add(extension)

        spec.setdefault("include_dirs", [])
        spec.setdefault("libraries", list(DEFAULT_LIBRARIES))
        spec.setdefault("extra_compile_args", {})
        spec.setdefault("arch_profile", "default")

        for field in ("bindings", "sources", "include_dirs", "libraries"):
            values = spec[field]
            if not isinstance(values, list) or not all(isinstance(v, str) and v for v in values):
                raise ValueError(f"{name}.{field}: expected a list of non-empty strings")
        if not spec["bindings"]:
            raise ValueError(f"{name}.bindings: at least one binding file is required")
        if not spec["sources"]:
            raise ValueError(f"{name}.sources: at least one source is required")

        flags = spec["extra_compile_args"]
        if not isinstance(flags, dict) or any(
            key not in ("cxx", "nvcc")
            or not isinstance(values, list)
            or not all(isinstance(v, str) and v for v in values)
            for key, values in flags.items()
        ):
            raise ValueError(f"{name}.extra_compile_args: expected cxx/nvcc lists of strings")

        if not isinstance(spec["arch_profile"], str) or not spec["arch_profile"]:
            raise ValueError(f"{name}.arch_profile: expected a non-empty string")

    return modules


MODULES = load_registry()


def select_build_targets(environ):
    """Parse LIGHTOP_MODULES.

    Returns (build_op: bool, build_standalone: bool, selected_split_modules: tuple[str, ...]).
    - Unset / empty: full modular build (remaining op.so + standalone exts + all split modules).
    - Non-empty: selective build of listed split modules (plus reserved token 'op' if requested).
    """
    raw = {
        name.strip()
        for name in environ.get("LIGHTOP_MODULES", "").split(",")
        if name.strip()
    }
    if not raw:
        return True, True, tuple(MODULES.keys())

    known = set(MODULES) | {"op"}
    unknown = sorted(raw - known)
    if unknown:
        raise ValueError(
            f"LIGHTOP_MODULES: unknown module(s) {unknown}; "
            f"known modules: {['op', *MODULES]}"
        )
    build_op = "op" in raw
    selected = tuple(name for name in MODULES if name in raw)
    return build_op, False, selected


def scan_binding_symbols(repo_root, modules=None):
    """Scan all registered binding files to produce {symbol_name: extension_name}.

    Raises ValueError on duplicate symbols across distinct split modules.
    """
    repo_root = Path(repo_root)
    modules = MODULES if modules is None else modules
    op_to_module = {}
    for mod_name, spec in modules.items():
        ext_name = spec["extension"]
        found_any = False
        for rel_path in spec["bindings"]:
            binding_path = repo_root / rel_path
            if not binding_path.is_file():
                raise FileNotFoundError(f"{mod_name}: binding file not found: {binding_path}")
            content = binding_path.read_text(encoding="utf-8")
            content = re.sub(r"//[^\n]*", "", content)
            for sym in _BINDING_SYMBOL_RE.findall(content):
                found_any = True
                prev = op_to_module.get(sym)
                if prev is not None and prev != ext_name:
                    raise ValueError(
                        f"Duplicate symbol {sym!r} exported by both {prev!r} and {ext_name!r}"
                    )
                op_to_module[sym] = ext_name
        if not found_any:
            raise ValueError(f"{mod_name}: no m.def/m.attr symbols found in {spec['bindings']}")
    return dict(sorted(op_to_module.items()))


def write_op_registry(repo_root, output_path):
    """Generate python/lightop/_op_registry.py at build time."""
    op_to_module = scan_binding_symbols(repo_root)
    lines = [
        "# Auto-generated by setup.py from module_registry.json bindings. Do not edit.",
        "OP_TO_MODULE = {",
    ]
    for sym, ext_name in op_to_module.items():
        lines.append(f"    {sym!r}: {ext_name!r},")
    lines.append("}")
    lines.append("")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    new_text = "\n".join(lines)
    if not output_path.is_file() or output_path.read_text(encoding="utf-8") != new_text:
        output_path.write_text(new_text, encoding="utf-8")
    return op_to_module


def _load_op_to_module():
    try:
        from ._op_registry import OP_TO_MODULE
        return OP_TO_MODULE
    except ImportError:
        return {}


def _mount_submodule(op_module, ext_name, *, ignore_missing=False):
    if ext_name in _LOADED_SUBMODULES:
        return
    try:
        submod = importlib.import_module(ext_name)
    except ModuleNotFoundError as exc:
        if ignore_missing and exc.name == ext_name:
            _LOADED_SUBMODULES.add(ext_name)
            return
        raise
    _LOADED_SUBMODULES.add(ext_name)
    for k, v in submod.__dict__.items():
        if not k.startswith("__"):
            setattr(op_module, k, v)


def _install_op_redirection(op_module, op_to_module):
    """Attach PEP 562 lazy router onto either native lightop.op or a facade ModuleType."""
    def _op_getattr(name: str):
        ext_name = op_to_module.get(name)
        if ext_name is not None and ext_name not in _LOADED_SUBMODULES:
            _mount_submodule(op_module, ext_name, ignore_missing=True)
            if name in op_module.__dict__:
                return op_module.__dict__[name]
        raise AttributeError(f"module {op_module.__name__!r} has no attribute {name!r}")

    def _op_dir():
        return sorted(set(op_module.__dict__.keys()) | set(op_to_module.keys()))

    op_module.__getattr__ = _op_getattr
    op_module.__dir__ = _op_dir
    return op_module


def preload_ops(op_module=None):
    """Eagerly load all built split extensions into op.__dict__ (e.g. before Graph capture)."""
    if op_module is None:
        op_module = sys.modules.get("lightop.op")
        if op_module is None:
            op_module = load_ops()
    for spec in MODULES.values():
        _mount_submodule(op_module, spec["extension"], ignore_missing=True)
    return op_module


def load_ops():
    """Load native lightop.op (with lazy split routing), or create a facade when absent."""
    op_to_module = _load_op_to_module()
    _LOADED_SUBMODULES.clear()

    op_module = None
    try:
        op_module = importlib.import_module("lightop.op")
    except ModuleNotFoundError as exc:
        if exc.name != "lightop.op":
            raise

    if op_module is None:
        has_any_split = any(
            importlib.util.find_spec(spec["extension"]) is not None
            for spec in MODULES.values()
        )
        if not has_any_split:
            raise ImportError(
                "No LightOp native extension found. Build the package or select "
                f"modules with LIGHTOP_MODULES={','.join(MODULES)}."
            )
        op_module = ModuleType("lightop.op")
        op_module.__file__ = __file__
        sys.modules["lightop.op"] = op_module

    _install_op_redirection(op_module, op_to_module)

    if os.environ.get("LIGHTOP_EAGER_LOAD", "0") == "1":
        preload_ops(op_module)

    return op_module
