#!/usr/bin/env python3
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Read a cached PR diff; --fetch explicitly permits one GitHub PR download."""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
import urllib.error
import urllib.parse
import urllib.request

from _kb import id_map, parse_markdown
from _knowledge_root import cache_root, knowledge_root

MAX_BYTES = 16 * 1024 * 1024


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def request(url, accept):
    parsed = urllib.parse.urlsplit(url)
    if parsed.scheme != "https" or parsed.hostname != "api.github.com" or parsed.username or parsed.password:
        raise ValueError("Evidence requests must use the GitHub HTTPS API")
    headers = {"Accept": accept, "User-Agent": "LightOp-kernel-skill"}
    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    if token:
        headers["Authorization"] = "Bearer " + token
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.build_opener(NoRedirect).open(req, timeout=20) as response:
            data = response.read(MAX_BYTES + 1)
    except urllib.error.HTTPError as exc:
        raise ValueError(f"GitHub evidence request failed: HTTP {exc.code}") from None
    except urllib.error.URLError:
        raise ValueError("GitHub evidence request failed: network unavailable") from None
    if len(data) > MAX_BYTES:
        raise ValueError("Evidence response exceeds 16 MiB; obtain it separately")
    return data


def github_snapshot(meta):
    parsed = urllib.parse.urlsplit(str(meta.get("url", "")))
    match = re.fullmatch(r"/([\w.-]+)/([\w.-]+)/pull/([1-9][0-9]*)/?", parsed.path)
    if parsed.scheme != "https" or parsed.hostname != "github.com" or parsed.username or parsed.password or parsed.query or parsed.fragment or not match:
        raise ValueError("Automatic fetch supports GitHub PR URLs; obtain other sources with an authorized tool")
    owner, repo, number = match.groups()
    endpoint = f"https://api.github.com/repos/{owner}/{repo}/pulls/{number}"
    before = json.loads(request(endpoint, "application/vnd.github+json"))
    base, head = before["base"]["sha"], before["head"]["sha"]
    if not all(re.fullmatch(r"[0-9a-f]{40}", value) for value in (base, head)):
        raise ValueError("Upstream PR lacks valid base/HEAD commit identities")
    indexed_head = meta.get("head_ref_oid")
    if indexed_head and indexed_head != head:
        raise ValueError("Upstream HEAD differs from the index; obtain the recorded revision or update the index explicitly")
    diff = request(endpoint, "application/vnd.github.diff")
    if not diff.startswith(b"diff --git "):
        raise ValueError("Upstream response is not a nonempty unified diff")
    after = json.loads(request(endpoint, "application/vnd.github+json"))
    if (after["base"]["sha"], after["head"]["sha"]) != (base, head):
        raise ValueError("PR changed during collection; no evidence was cached")
    return diff, {
        "api_url": endpoint, "base_sha": base, "head_sha": head,
        "indexed_head_sha": indexed_head,
        "index_head_matches": True if indexed_head else None,
        "historical_base_verified": False,
    }


def read_cached(directory, fingerprint):
    meta = json.loads((directory / "manifest.json").read_text())
    diff = directory / "review.diff"
    if meta.get("index_fingerprint") != fingerprint:
        raise ValueError("Evidence cache belongs to a different index entry")
    if hashlib.sha256(diff.read_bytes()).hexdigest() != meta.get("diff_sha256"):
        raise ValueError("Evidence cache checksum mismatch; inspect the cached files")
    return diff, meta


def load_or_fetch(page, *, fetch=False):
    root = knowledge_root().resolve()
    artifact = page.meta.get("artifact_dir")
    if artifact:
        directory = (root / str(artifact)).resolve()
        if not directory.is_relative_to(root):
            raise ValueError("Evidence artifact path escapes the knowledge root")
        diff = directory / "review.diff"
        if diff.is_file():
            if not diff.resolve().is_relative_to(root):
                raise ValueError("Evidence diff escapes the knowledge root")
            return diff, {"location": "full-corpus", "diff_sha256": hashlib.sha256(diff.read_bytes()).hexdigest()}
    encoded = json.dumps(page.meta, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str).encode()
    fingerprint = hashlib.sha256(encoded).hexdigest()
    identifier = re.sub(r"[^a-zA-Z0-9_-]", "_", page.id or page.relpath)[:120]
    directory = cache_root() / "evidence" / identifier / fingerprint
    if directory.exists():
        return read_cached(directory, fingerprint)
    if not fetch:
        raise FileNotFoundError("Evidence is not cached. Use --fetch to allow network access, or provide a full offline corpus")
    diff, revision = github_snapshot(page.meta)
    manifest = {
        "schema_version": 1, "page_id": page.id,
        "source_url": str(page.meta.get("url", "")),
        "index_fingerprint": fingerprint, "revision": revision,
        "diff_sha256": hashlib.sha256(diff).hexdigest(), "bytes": len(diff),
        "captured_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    directory.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="fetch-", dir=directory.parent) as temporary:
        staging = Path(temporary) / "evidence"
        staging.mkdir()
        (staging / "review.diff").write_bytes(diff)
        (staging / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
        try:
            staging.rename(directory)
        except OSError:
            if not directory.is_dir():
                raise
    return read_cached(directory, fingerprint)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("page", help="Index page ID or a path within the knowledge root")
    parser.add_argument("--fetch", action="store_true", help="Allow network access only when the diff is not cached")
    args = parser.parse_args()
    try:
        page = id_map().get(args.page)
        if page is None:
            root = knowledge_root().resolve()
            path = (root / args.page).resolve()
            if not path.is_relative_to(root) or not path.is_file():
                raise ValueError("Index page not found within the knowledge root")
            page = parse_markdown(path)
        path, metadata = load_or_fetch(page, fetch=args.fetch)
        print(json.dumps({"diff": str(path), "metadata": metadata}, ensure_ascii=False, indent=2))
        return 0
    except (ValueError, OSError, KeyError, TypeError) as exc:
        parser.exit(1, f"Evidence unavailable: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
