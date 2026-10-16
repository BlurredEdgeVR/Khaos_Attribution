"""Which code an engine checkout is: the HEAD of a git checkout, read with every variable that would point git at
another repository stripped. Never an exception, never a guess: None when it cannot be read."""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
# Environment variables that make git answer about another repository no matter what `-C <dir>` says.
GIT_REPO_ENV_OVERRIDES = ("GIT_DIR", "GIT_WORK_TREE", "GIT_COMMON_DIR", "GIT_INDEX_FILE", "GIT_OBJECT_DIRECTORY",
                          "GIT_CEILING_DIRECTORIES")


def git_env() -> dict:
    return {k: v for k, v in os.environ.items() if k not in GIT_REPO_ENV_OVERRIDES}


def checkout_commit(root) -> str | None:
    """The 40-character HEAD of the checkout at *root*, or None (not a checkout, no git, a slow git, no full SHA)."""
    if root is None:
        return None
    root = str(root)
    try:
        if not (Path(root) / ".git").exists():
            return None
    except OSError:
        return None
    try:
        proc = subprocess.run(["git", "-C", root, "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5,
                              env=git_env())
    except Exception:  # noqa: BLE001 — git missing, a timeout, an unreadable cwd: all None
        return None
    if proc.returncode != 0:
        return None
    sha = (proc.stdout or "").strip()
    return sha if _SHA_RE.match(sha) else None


def is_sha(value) -> bool:
    return isinstance(value, str) and bool(_SHA_RE.match(value))
