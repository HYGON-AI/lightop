from __future__ import annotations

import datetime
import json
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from _knowledge_root import index_path, knowledge_root, relpath


FRONTMATTER_RE = re.compile(r"\A---\n(.*?)\n---\n?", re.S)


@dataclass
class Page:
    path: Path
    relpath: str
    meta: dict[str, Any]
    body: str
    source: str | None = None
    frontmatter: str | None = None

    @property
    def id(self) -> str:
        return str(self.meta.get("id") or "")

    @property
    def title(self) -> str:
        return str(self.meta.get("title") or self.path.stem)

    @property
    def kind(self) -> str:
        if self.relpath.startswith("sources/"):
            return str(self.meta.get("source_type") or self.meta.get("source_category") or "source")
        return str(self.meta.get("type") or "page")


def restore_types(value: dict) -> Any:
    if set(value) == {"$date"}:
        return datetime.date.fromisoformat(value["$date"])
    if set(value) == {"$datetime"}:
        return datetime.datetime.fromisoformat(value["$datetime"])
    return value


@lru_cache(maxsize=1)
def load_index(path: Path) -> tuple[dict, dict[str, Page]]:
    root = path.parent.parent
    pages: dict[str, Page] = {}
    identifiers: set[str] = set()
    header = None
    with path.open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            try:
                row = json.loads(line, object_hook=restore_types)
                if number == 1:
                    if row.get("kind") != "catalog" or row.get("schema_version") != 1:
                        raise ValueError("Unsupported knowledge index schema")
                    header = row
                    continue
                name = row["path"]
                relative = Path(name)
                if relative.is_absolute() or ".." in relative.parts or name in pages or name in ("", "."):
                    raise ValueError("Invalid or duplicate index path")
                if row["kind"] == "page":
                    meta, body = row["meta"], row["body"]
                    if not isinstance(meta, dict) or not isinstance(body, str):
                        raise ValueError("Invalid index page")
                    prefix = row.get("frontmatter")
                    if prefix is not None and not isinstance(prefix, str):
                        raise ValueError("Invalid page frontmatter")
                    source = None
                    identifier = str(meta.get("id") or "")
                    if identifier and identifier in identifiers:
                        raise ValueError("Duplicate index page ID")
                    if identifier:
                        identifiers.add(identifier)
                elif row["kind"] == "file":
                    source = row["text"]
                    match = FRONTMATTER_RE.match(source)
                    meta = yaml.safe_load(match.group(1)) or {} if match else {}
                    body = source[match.end():] if match else source
                    prefix = None
                else:
                    raise ValueError("Unsupported index record kind")
                pages[name] = Page(root / relative, name, meta, body, source, prefix)
            except (ValueError, KeyError, TypeError) as exc:
                raise ValueError(f"Invalid knowledge index at line {number}: {exc}") from exc
    if header is None or sum(name.startswith(("sources/prs/", "sources/refs/")) and name.endswith(".md") for name in pages) != header.get("page_count"):
        raise ValueError("Knowledge index page count mismatch")
    return header, pages


def parse_markdown(path: Path) -> Page:
    catalog = index_path()
    if catalog is not None and relpath(path) in load_index(catalog)[1]:
        return load_index(catalog)[1][relpath(path)]
    text = path.read_text(encoding="utf-8")
    match = FRONTMATTER_RE.match(text)
    if not match:
        return Page(path, relpath(path), {}, text, text)
    meta = yaml.safe_load(match.group(1)) or {}
    body = text[match.end() :]
    return Page(path, relpath(path), meta, body, text)


def iter_pages() -> list[Page]:
    catalog = index_path()
    if catalog is not None:
        return [page for name, page in sorted(load_index(catalog)[1].items())
                if name.startswith(("sources/prs/", "sources/refs/")) and name.endswith(".md")]
    root = knowledge_root()
    bases = [root / "sources" / "prs", root / "sources" / "refs"]
    pages: list[Page] = []
    for base in bases:
        if not base.exists():
            continue
        for path in sorted(base.rglob("*.md")):
            pages.append(parse_markdown(path))
    return pages


def load_aliases() -> dict[str, str]:
    catalog = index_path()
    data = load_index(catalog)[0].get("aliases", {}) if catalog is not None else None
    path = knowledge_root() / "data/aliases.yaml"
    if data is None and not path.exists():
        return {}
    if data is None:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    aliases: dict[str, str] = {}
    for canonical, values in data.items():
        aliases[str(canonical).lower()] = str(canonical).lower()
        for value in values or []:
            aliases[str(value).lower()] = str(canonical).lower()
    if "hygon" in aliases:
        aliases["hcu"] = aliases["hygon"]
    return aliases


def normalize_term(value: str, aliases: dict[str, str]) -> str:
    key = value.lower().replace("_", "-")
    return aliases.get(key, aliases.get(value.lower(), key))


def as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    return [str(value)]


def page_text(page: Page) -> str:
    values = [page.title, page.id, page.relpath, page.body]
    for value in page.meta.values():
        if isinstance(value, (str, int, float)):
            values.append(str(value))
        elif isinstance(value, list):
            values.extend(str(item) for item in value)
    return "\n".join(values)


def id_map() -> dict[str, Page]:
    out: dict[str, Page] = {}
    for page in iter_pages():
        if page.id:
            out[page.id] = page
        out[page.relpath] = page
    return out
