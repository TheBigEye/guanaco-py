"""Replace only a known generated site, with rollback and stale-page removal."""

from __future__ import annotations

import contextlib
import json
import re
import tempfile
from pathlib import Path

from release_common import write_json

MARKER = ".guanaco-wheel-index.json"
GENERATOR = "guanaco-wheel-index-v1"
PAGE_PATH = re.compile(
    r"(?:index\.html|whl/index\.html|whl/(?:cpu|avx2|cu[0-9]+)/(?:guanaco-py/)?index\.html)"
)


def check_ownership(output: Path) -> None:
    if output.is_symlink() or (output.exists() and not output.is_dir()):
        raise ValueError("Index output must be a directory, not a symlink")
    if not output.exists() or not any(output.iterdir()):
        return
    marker = output / MARKER
    if not marker.is_file() or marker.is_symlink():
        raise ValueError(
            "Refusing to replace an unowned index directory; use a fresh output directory"
        )
    state = json.loads(marker.read_text(encoding="utf-8"))
    files = state.get("files", [])
    if (
        state.get("generator") != GENERATOR
        or not files
        or any(not isinstance(name, str) or not PAGE_PATH.fullmatch(name) for name in files)
    ):
        raise ValueError("Invalid generated-site ownership record")
    allowed = {MARKER, *files}
    for name in files:
        allowed.update(parent.as_posix() for parent in Path(name).parents if str(parent) != ".")
    if any(
        path.is_symlink() or path.relative_to(output).as_posix() not in allowed
        for path in output.rglob("*")
    ):
        raise ValueError("Index output contains unowned files or symlinks; nothing was deleted")


@contextlib.contextmanager
def generated_site(output: Path):
    """Build next to the old site, then swap it only after generation succeeds."""
    check_ownership(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".wheel-index-", dir=output.parent) as name:
        workspace = Path(name)
        next_site, previous = workspace / "next", workspace / "previous"
        next_site.mkdir()
        yield next_site
        files = sorted(
            path.relative_to(next_site).as_posix()
            for path in next_site.rglob("*")
            if path.is_file()
        )
        write_json(next_site / MARKER, {"generator": GENERATOR, "files": files})
        if output.exists():
            output.replace(previous)
        try:
            next_site.replace(output)
        except OSError:
            if previous.exists():
                previous.replace(output)
            raise
