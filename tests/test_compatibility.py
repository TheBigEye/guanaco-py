"""Guards for the oldest interpreter the automation still supports.

The build matrix starts at Python 3.9, and the test workflow runs 3.9, 3.13
and 3.14. Two mistakes are invisible on a newer development machine and only
surface as a failure on one runner out of six:

* ``Path.read_text`` and ``Path.write_text`` learned about ``newline`` in
  Python 3.10, so passing it raises ``TypeError`` on 3.9;
* ``list[str]`` and ``str | None`` are evaluated when the module is imported
  unless it opts into postponed evaluation.

Both are checked by reading the sources, so the failure names the cause
instead of appearing as a ``TypeError`` inside an unrelated test.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SOURCES = sorted(
    [
        *(ROOT / "guanaco").glob("*.py"),
        *(ROOT / "tests").glob("*.py"),
        ROOT / "docker" / "fetch_release.py",
    ]
)
POSTPONED = "from __future__ import annotations"
TEXT_HELPERS = ("read_text", "write_text")


def _helpers_called_with_newline(tree: ast.AST) -> list[int]:
    """Return the lines that pass ``newline`` to a ``Path`` text helper."""
    return [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr in TEXT_HELPERS
        and any(keyword.arg == "newline" for keyword in node.keywords)
    ]


@pytest.mark.parametrize("path", SOURCES, ids=lambda path: path.name)
def test_text_helpers_do_not_take_a_newline_argument(path: Path):
    """``newline`` reached the text helpers in Python 3.10; open the file instead."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    offenders = _helpers_called_with_newline(tree)
    assert offenders == [], f"{path.name}: use path.open(..., newline=...) on line(s) {offenders}"


@pytest.mark.parametrize("path", SOURCES, ids=lambda path: path.name)
def test_annotations_are_postponed(path: Path):
    """Builtin generics and unions need postponed evaluation on Python 3.9."""
    text = path.read_text(encoding="utf-8")
    if not any(keyword in text for keyword in ("def ", "class ", ": list[", ": dict[")):
        pytest.skip(f"{path.name} declares nothing that is evaluated at import time")
    assert POSTPONED in text, f"{path.name} must start with '{POSTPONED}'"
