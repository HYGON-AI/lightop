from __future__ import annotations
import hashlib, os, tarfile, tempfile
from functools import lru_cache
from pathlib import Path

EXPECTED = "2f8c35581ef307a46f0692abad9eb73e685c3e1d25c8c256589b8ede46db466b"

def cache_root() -> Path:
    base = os.environ.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")
    return Path(base).expanduser() / "lightop-kernel-agent"

@lru_cache(maxsize=1)
def knowledge_root() -> Path:
    override = os.environ.get("KERNEL_KNOWLEDGE_ROOT")
    if override:
        root = Path(override).expanduser().resolve()
        if not (root / "sources").is_dir():
            raise ValueError("KERNEL_KNOWLEDGE_ROOT must contain sources/")
        return root
    archive = Path(__file__).resolve().parents[1] / "assets/corpus.tar.gz"
    if hashlib.sha256(archive.read_bytes()).hexdigest() != EXPECTED:
        raise ValueError("Bundled knowledge corpus checksum mismatch")
    cache = cache_root()
    cache.mkdir(parents=True, exist_ok=True)
    root = cache / EXPECTED
    if not (root / ".ready").is_file():
        with tempfile.TemporaryDirectory(prefix="unpack-", dir=cache) as temporary:
            staging = Path(temporary) / "corpus"
            staging.mkdir()
            with tarfile.open(archive, "r:gz") as tar:
                members = tar.getmembers()
                total = 0
                for member in members:
                    path = Path(member.name)
                    if path.is_absolute() or ".." in path.parts or not (member.isdir() or member.isfile()):
                        raise ValueError("Unsafe bundled corpus member")
                    total += member.size
                if total > 64 * 1024 * 1024 or len(members) > 10000:
                    raise ValueError("Bundled corpus exceeds unpack limits")
                tar.extractall(staging, members=members)
            (staging / ".ready").write_text(EXPECTED)
            try:
                staging.rename(root)
            except OSError:
                if not (root / ".ready").is_file():
                    raise
    return root

def relpath(path: Path) -> str:
    root = knowledge_root()
    try:
        return path.resolve().relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()
