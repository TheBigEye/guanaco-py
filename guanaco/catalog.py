"""The PEP 503 wheel index served from GitHub Pages.

The generated site is what makes ``pip install guanaco-py --index-url ...`` work.
It is plain static HTML with the stylesheet and the icon embedded, so the site
has no external assets at all.

Because the generator *replaces* the previous site, it is careful: it builds the
new one next to the old one, only swaps them after generation succeeds, and
refuses to touch a directory it did not create (see :func:`generated_site`).
"""

from __future__ import annotations

import base64
import contextlib
import html
import json
import re
import tempfile
from pathlib import Path

from .models import (
    Channel,
    Release,
    expected_asset_names,
    is_complete,
    wheel_prefix,
    write_json,
)
from .settings import Settings

MARKER_FILE = ".guanaco-wheel-index.json"
GENERATOR = "guanaco-wheel-index-v1"
PAGE_PATH = re.compile(
    r"(?:index\.html|whl/index\.html|whl/(?:cpu|avx2|cu[0-9]+)/(?:[^/]+/)?index\.html)"
)

PAGE_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>{title}</title>
    <style>{css}</style>
</head>
<body>
    <main>
{back_link}        <header>
{icon}            <div class="eyebrow">{owner} &middot; {package}</div>
            <h1>{title}</h1>
            <p>{intro}</p>
        </header>
        {body}
        <footer>
            PEP 503 compatible index &middot; <code>pip install {package}</code>
        </footer>
    </main>
</body>
</html>
"""


class IndexError(ValueError):
    """Raised when the index cannot be generated or the output is not ours."""


def check_ownership(output: Path) -> None:
    """Refuse to replace a directory we did not generate.

    Raises:
        IndexError: If `output` holds anything other than a site this generator
            produced. Deleting unrelated files is never acceptable.
    """
    if output.is_symlink() or (output.exists() and not output.is_dir()):
        raise IndexError("Index output must be a directory, not a symlink")
    if not output.exists() or not any(output.iterdir()):
        return
    marker = output / MARKER_FILE
    if not marker.is_file() or marker.is_symlink():
        raise IndexError(
            "Refusing to replace an unowned index directory; use a fresh output directory"
        )
    state = json.loads(marker.read_text(encoding="utf-8"))
    files = state.get("files", [])
    if (
        state.get("generator") != GENERATOR
        or not files
        or any(not isinstance(name, str) or not PAGE_PATH.fullmatch(name) for name in files)
    ):
        raise IndexError("Invalid generated-site ownership record")
    allowed = {MARKER_FILE, *files}
    for name in files:
        allowed.update(parent.as_posix() for parent in Path(name).parents if str(parent) != ".")
    if any(
        path.is_symlink() or path.relative_to(output).as_posix() not in allowed
        for path in output.rglob("*")
    ):
        raise IndexError("Index output contains unowned files or symlinks; nothing was deleted")


@contextlib.contextmanager
def generated_site(output: Path):
    """Yield a scratch directory that replaces `output` once generation succeeds.

    If the block raises, the previous site stays exactly where it was, and stale
    channel pages from an older release cannot linger.
    """
    output = Path(output)
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
        write_json(next_site / MARKER_FILE, {"generator": GENERATOR, "files": files})
        if output.exists():
            output.replace(previous)
        try:
            next_site.replace(output)
        except OSError:
            if previous.exists():
                previous.replace(output)
            raise


class IndexGenerator:
    """Renders the wheel index from the releases that are already published."""

    def __init__(self, settings: Settings) -> None:
        """Load the stylesheet and the icon from the repository's docs folder."""
        self.settings = settings
        self.package = settings.package
        docs = settings.docs_dir
        self.stylesheet = (docs / "wheel-index.css").read_text(encoding="utf-8")
        self.icon = "data:image/svg+xml;base64," + base64.b64encode(
            (docs / "icon.svg").read_bytes()
        ).decode("ascii")

    def generate(self, releases: list[Release], output: Path) -> Path:
        """Write the index for `releases` into `output` and return the path.

        Only complete, provenance-carrying releases are indexed. Drafts, partial
        uploads and legacy personal-fork wheels are skipped, so a half-finished
        version never becomes installable.
        """
        with generated_site(output) as site:
            self._render(self._collect(releases), site)
        return Path(output)

    # -- Reading -------------------------------------------------------------

    def _collect(self, releases: list[Release]) -> dict[str, set[tuple[str, str]]]:
        """Gather the downloadable wheels of every complete channel."""
        channels: dict[str, set[tuple[str, str]]] = {"cpu": set(), "avx2": set()}
        prefix = f"https://github.com/{self.settings.repository}/releases/download/"
        for release in releases:
            channel = self._channel_of(release)
            if channel is None:
                continue
            state = release.provenance(self.settings.upstream, self.settings.package)
            if not is_complete(
                release, state, self.settings.package, self.settings.python_versions
            ):
                continue
            allowed = self._allowed_assets(release, channel, state)
            for asset in release.assets:
                if asset.name not in allowed or asset.url != prefix + f"{release.tag}/{asset.name}":
                    continue
                url = asset.url.split("#", 1)[0]
                if re.fullmatch(r"sha256:[a-f0-9]{64}", asset.digest):
                    url += "#" + asset.digest.replace(":", "=", 1)
                channels.setdefault(channel, set()).add((asset.name, url))
        return channels

    def _channel_of(self, release: Release) -> str | None:
        """Return the channel a release tag belongs to, or ``None``."""
        version = release.version
        if version is None:
            return None
        match = re.fullmatch(rf"v{re.escape(str(version))}(?:-(avx2|cu[0-9]+))?", release.tag)
        return (match.group(1) or "cpu") if match else None

    def _allowed_assets(self, release: Release, channel: str, state) -> set[str]:
        """Return the asset names a complete release of this channel may carry."""
        return expected_asset_names(
            self.settings.package,
            state.version,
            Channel(channel),
            state.python_versions,
        )

    # -- Rendering -----------------------------------------------------------

    def _render(self, channels: dict[str, set[tuple[str, str]]], output: Path) -> None:
        """Write every page of the site."""
        root = output / "whl"
        root.mkdir(parents=True, exist_ok=True)
        owner = self.settings.repository.split("/")[0]
        cards: list[str] = []
        total = 0
        for channel in sorted(channels, key=self.order):
            assets = sorted(channels[channel], reverse=True)
            assets.sort(key=lambda item: self._version_key(item[0]), reverse=True)
            total += len(assets)
            label = self._label(channel)
            cards.append(self._card(f"{channel}/", label, f"{len(assets)} wheel(s)"))

            channel_dir = root / channel
            project_dir = channel_dir / self.package
            project_dir.mkdir(parents=True, exist_ok=True)
            # This anchor is the normalized project link a PEP 503 root requires.
            (channel_dir / "index.html").write_text(
                self._page(
                    label,
                    f"Distribution channel for {self.package}.",
                    self._grid([self._card(f"{self.package}/", self.package, "View wheels")]),
                    owner=owner,
                    parent="../",
                ),
                encoding="utf-8",
            )
            # Direct anchors to wheel files are what pip consumes on this page.
            items = (
                "\n            ".join(
                    f'<li><a class="wheel" href="{html.escape(url, quote=True)}">{html.escape(name)}</a></li>'
                    for name, url in assets
                )
                or '<li class="meta">No wheels published yet.</li>'
            )
            (project_dir / "index.html").write_text(
                self._page(
                    f"{self.package} · {label}",
                    f"{len(assets)} file(s) published.",
                    f'<ul class="wheels">\n            {items}\n        </ul>',
                    owner=owner,
                    parent="../",
                ),
                encoding="utf-8",
            )

        (root / "index.html").write_text(
            self._page(
                "Wheel index",
                "Choose CPU (portable), CPU (AVX2) or a CUDA build.",
                self._grid(cards),
                owner=owner,
                parent="../",
            ),
            encoding="utf-8",
        )
        # Landing page at the site root, so the Pages URL is not a 404.
        home = [
            self._card("whl/", "Wheel index", f"{total} wheel(s) · {len(channels)} channel(s)"),
            self._card(
                f"https://github.com/{self.settings.repository}",
                "Build repository",
                "Guanaco distribution automation",
            ),
            self._card(
                f"https://github.com/{self.settings.upstream}",
                "Upstream",
                "Bindings, documentation and wiki",
            ),
            self._card(
                f"https://github.com/{self.settings.repository}#installation",
                "Installation",
                "Setup instructions",
            ),
            self._card(
                f"https://github.com/{self.settings.repository}/releases",
                "Releases",
                "Changelog &amp; downloads",
            ),
        ]
        (output / "index.html").write_text(
            self._page(
                self.package,
                f"Prebuilt CPU (portable), CPU (AVX2) and CUDA wheels for {self.settings.upstream_package}.",
                self._grid(home),
                owner=owner,
                icon=True,
            ),
            encoding="utf-8",
        )

    def _page(
        self,
        title: str,
        intro: str,
        body: str,
        *,
        owner: str,
        parent: str | None = None,
        icon: bool = False,
    ) -> str:
        """Render one HTML page."""
        back_link = (
            f'        <nav><a href="{html.escape(parent)}">&larr; Back</a></nav>\n'
            if parent
            else ""
        )
        return PAGE_TEMPLATE.format(
            title=html.escape(title),
            css=self.stylesheet,
            back_link=back_link,
            icon=f'            <img class="icon" src="{self.icon}" alt="">\n' if icon else "",
            owner=html.escape(owner),
            package=html.escape(self.package),
            intro=html.escape(intro),
            body=body,
        )

    @staticmethod
    def _card(href: str, title: str, subtitle: str) -> str:
        """Render one clickable card. `title` and `subtitle` may contain HTML."""
        return (
            f'<a class="card" href="{html.escape(href, quote=True)}">'
            f"<strong>{html.escape(title)}</strong>"
            f"<span>{subtitle}</span>"
            f"</a>"
        )

    @staticmethod
    def _grid(cards: list[str]) -> str:
        """Render a grid of cards."""
        joined = "\n            ".join(cards)
        return f'<div class="grid">\n            {joined}\n        </div>'

    @staticmethod
    def order(channel: str) -> tuple[int, int]:
        """Sort channels: portable CPU, AVX2, then CUDA by toolkit version."""
        if channel == "cpu":
            return (0, 0)
        if channel == "avx2":
            return (1, 0)
        match = re.fullmatch(r"cu(\d+)", channel)
        return (2, int(match.group(1)) if match else 9999)

    def _version_key(self, name: str) -> tuple[int, ...]:
        """Extract the numeric version of a wheel filename, for sorting.

        Wheel filenames follow ``{distribution}-{version}-...`` (PEP 427), so the
        version is the segment right after the package name.
        """
        match = re.match(rf"{re.escape(wheel_prefix(self.package))}-([^-]+)-", name)
        version = match.group(1) if match else ""
        return tuple(int(part) for part in re.findall(r"\d+", version)) or (0,)

    def _label(self, channel: str) -> str:
        """Human readable channel label, e.g. ``cu124`` -> ``CUDA 12.4``."""
        if channel == "cpu":
            return "CPU (portable)"
        if channel == "avx2":
            return "CPU (AVX2)"
        cuda = self.settings.cuda.get(channel)
        return cuda.label if cuda else channel


__all__ = ["IndexError", "IndexGenerator", "check_ownership", "generated_site"]
