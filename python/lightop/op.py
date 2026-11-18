# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Aggregator for LightOp's native extension modules.

Phase-0 pilot of AITER's "one operator backend = one extension module
(module_xxx.so)" layout.

Behaviour
---------
* Split enabled (build with ``LIGHTOP_ENABLE_NORM_MODULE=1`` and/or
  ``LIGHTOP_ENABLE_MQA_MODULE=1``): the monolithic extension is built as
  ``lightop.op_full`` and this module dispatches ``op.<name>`` to the
  specialised modules first (``lightop._ext.op_norm``,
  ``lightop._ext.op_mqa_logits``) and only falls back to ``lightop.op_full``
  for the remaining symbols. ``from . import op`` in the wrappers therefore
  keeps working unchanged.
* Selective build (``LIGHTOP_MODULES=mqa_logits[,norm]``): only the listed
  split modules are built; the monolith fallback is absent and symbols of
  unbuilt families raise AttributeError by design.
* Split disabled (default build): the ``lightop.op`` extension ``.so`` shadows
  this ``.py`` on the import path, so this file is never imported and the
  default behaviour is byte-for-byte unchanged.

Only torch itself is required here; no LightOp source is imported eagerly beyond
the native modules, matching the existing eager-load behaviour of the monolith.
"""
from __future__ import annotations

import importlib
from typing import Any

# Specialised modules first (highest priority), monolith last.
_SUBMODULES = (
    "lightop._ext.op_norm",  # Phase-0 pilot: norm family
    "lightop._ext.op_mqa_logits",  # Phase-1 pilot: MQA logits family
)
_FALLBACKS = (
    "lightop.op_full",  # full monolith when the split is enabled
)

_loaded: list[Any] = []
for _name in _SUBMODULES + _FALLBACKS:
    try:
        _loaded.append(importlib.import_module(_name))
    except ImportError:
        # A specialised module may be absent in a partial build; keep going so
        # the monolith (or another module) can still satisfy the symbol.
        continue

if not _loaded:
    raise ImportError(
        "lightop.op aggregator found no backing extension. Build either the "
        "monolith (LIGHTOP_ENABLE_NORM_MODULE=0) or the split modules "
        "(LIGHTOP_ENABLE_NORM_MODULE=1)."
    )


def __getattr__(name: str) -> Any:
    """Resolve ``op.<name>`` against the specialised modules, then the monolith."""
    for module in _loaded:
        try:
            value = getattr(module, name)
        except AttributeError:
            continue
        globals()[name] = value  # cache so repeated lookups are cheap
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    names = set(globals())
    for module in _loaded:
        names.update(dir(module))
    return sorted(names)
