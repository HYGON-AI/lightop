from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path


def cache_root() -> Path:
    base = os.environ.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")
    return Path(base).expanduser() / "lightop-kernel-agent"


@lru_cache(maxsize=1)
def knowledge_root() -> Path:
    override = os.environ.get("KERNEL_KNOWLEDGE_ROOT")
    if override:
        root = Path(override).expanduser().resolve()
        if not (root / "sources").is_dir() and not (root / "assets/index.jsonl").is_file():
            raise ValueError("KERNEL_KNOWLEDGE_ROOT must contain sources/ or assets/index.jsonl")
        return root
    return Path(__file__).resolve().parents[1]


def index_path() -> Path | None:
    root = knowledge_root()
    if (root / "sources").is_dir():
        return None
    return root / "assets/index.jsonl"


def relpath(path: Path) -> str:
    try:
        return path.resolve().relative_to(knowledge_root().resolve()).as_posix()
    except ValueError:
        return path.as_posix()
