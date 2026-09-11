"""Three version numbers disagreed on 2026-09-11 (pyproject 0.18.0, the
fallback __version__ 0.17.0, the newest tag v0.17.0), and `requires-python
= ">=3.9"` was false: `X | None` in an annotation raises at import on 3.9
unless the module defers annotations. Both are pinned here.
"""
from __future__ import annotations

import ast
import re
from pathlib import Path

import khaos_attribution

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "src" / "khaos_attribution"


def _pyproject_version() -> str:
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    return re.search(r'^version = "([^"]+)"', text, re.M).group(1)


def test_the_fallback_version_is_the_pyproject_version():
    """The fallback is what a checkout reports; a stale one is a lie the
    rooms would pin."""
    source = (PACKAGE / "__init__.py").read_text(encoding="utf-8")
    fallback = re.search(r'__version__ = "([^"]+)"', source).group(1)
    assert fallback == _pyproject_version()
    # The installed metadata is whatever `pip install -e` last recorded; a
    # stale editable install reported 0.13.2 against a 0.19.0 checkout on
    # 2026-09-11. That is an install to refresh, not a source to pin, so the
    # runtime value is only required to be a version string.
    assert re.fullmatch(r"\d+\.\d+\.\d+", khaos_attribution.__version__)


def _uses_pep604_union(tree: ast.AST) -> bool:
    for node in ast.walk(tree):
        ann = None
        if isinstance(node, ast.arg):
            ann = node.annotation
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            ann = node.returns
        elif isinstance(node, ast.AnnAssign):
            ann = node.annotation
        if ann is None:
            continue
        for sub in ast.walk(ann):
            if isinstance(sub, ast.BinOp) and isinstance(sub.op, ast.BitOr):
                return True
    return False


def test_every_module_with_union_annotations_defers_them():
    """`int | None` is evaluated at definition time on 3.9 unless the module
    starts with `from __future__ import annotations` — the claim in
    pyproject's requires-python depends on it."""
    offenders = []
    for path in sorted(PACKAGE.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        if not _uses_pep604_union(tree):
            continue
        deferred = any(isinstance(n, ast.ImportFrom) and n.module == "__future__"
                       and any(a.name == "annotations" for a in n.names)
                       for n in tree.body)
        if not deferred:
            offenders.append(path.name)
    assert offenders == [], f"union annotations without the future import: {offenders}"
