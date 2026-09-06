#!/usr/bin/env python3
"""Generate a HTML index compatible with PEP 503/pip"""

from __future__ import annotations

import argparse
import base64
import html
import json
import os
import pathlib
import re

from release_common import CONFIG, UPSTREAM, complete, expected_assets, provenance, repository_name
from site_utils import generated_site

PACKAGE = "guanaco-py"
WHEEL_PREFIX = "guanaco_py"
ICON_URL = "data:image/svg+xml;base64," + base64.b64encode(
    (pathlib.Path(__file__).resolve().parents[2] / "docs/icon.svg").read_bytes()
).decode("ascii")
TAG = re.compile(r"^v[^-]+(?:-(cu\d+|avx2))?$")

# The stylesheet is embedded so the generated site has no external assets.
CSS = (pathlib.Path(__file__).resolve().parents[2] / "docs/wheel-index.css").read_text(
    encoding="utf-8"
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
{icon}            <div class="eyebrow">TheBigEye &middot; guanaco-py</div>
            <h1>{title}</h1>
            <p>{intro}</p>
        </header>
        {body}
        <footer>
            PEP 503 compatible index &middot; <code>pip install guanaco-py</code>
        </footer>
    </main>
</body>
</html>
"""


def page(
    title: str, intro: str, body: str, parent: str | None = None, show_icon: bool = False
) -> str:
    back_link = (
        f'        <nav><a href="{html.escape(parent)}">&larr; Back</a></nav>\n' if parent else ""
    )
    icon = f'            <img class="icon" src="{ICON_URL}" alt="">\n' if show_icon else ""
    return PAGE_TEMPLATE.format(
        title=html.escape(title),
        css=CSS,
        back_link=back_link,
        icon=icon,
        intro=html.escape(intro),
        body=body,
    )


def card(href: str, title: str, subtitle: str) -> str:
    return (
        f'<a class="card" href="{html.escape(href, quote=True)}">'
        f"<strong>{html.escape(title)}</strong>"
        f"<span>{subtitle}</span>"
        f"</a>"
    )


def grid(cards: list[str]) -> str:
    items = "\n            ".join(cards)
    return f'<div class="grid">\n            {items}\n        </div>'


def order(channel: str) -> tuple[int, int]:
    """Sort channels: CPU (portable), then CPU (AVX2), then CUDA by version."""
    if channel == "cpu":
        return (0, 0)
    if channel == "avx2":
        return (1, 0)
    match = re.fullmatch(r"cu(\d+)", channel)
    return (2, int(match.group(1)) if match else 9999)


def wheel_version_key(name: str) -> tuple[int, ...]:
    """Extract the numeric version from a wheel filename for sorting newest-first.

    Wheel filenames follow {distribution}-{version}-... (PEP 427), so the
    version is the segment right after the package name.
    """
    match = re.match(rf"{re.escape(WHEEL_PREFIX)}-([^-]+)-", name)
    version = match.group(1) if match else ""
    return tuple(int(part) for part in re.findall(r"\d+", version)) or (0,)


def generate(
    source: pathlib.Path, output: pathlib.Path, repository: str = "TheBigEye/guanaco-py"
) -> None:
    repo_url = "https://github.com/" + repository_name(repository)
    raw = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("Release input must be a list or a list of API pages")
    pages = raw if not raw or isinstance(raw[0], list) else [raw]
    if any(
        not isinstance(batch, list) or any(not isinstance(item, dict) for item in batch)
        for batch in pages
    ):
        raise ValueError("Malformed release API pages")
    channels: dict[str, set[tuple[str, str]]] = {"cpu": set(), "avx2": set()}
    for release in (item for batch in pages for item in batch):
        match = TAG.fullmatch(release.get("tag_name", ""))
        if not match:
            continue
        state = provenance(release)
        if not complete(release, state, CONFIG["python_versions"]):
            continue  # drafts, prereleases, partial uploads and legacy personal-fork wheels
        channel = match.group(1) or "cpu"
        allowed = expected_assets(state["version"], channel, state["python_versions"])
        prefix = f"{repo_url}/releases/download/{release['tag_name']}/"
        for asset in release.get("assets", []):
            name, url = asset.get("name", ""), asset.get("browser_download_url", "")
            if name in allowed and name.endswith(".whl") and url == prefix + name:
                digest = asset.get("digest") or ""
                if re.fullmatch(r"sha256:[a-f0-9]{64}", digest):
                    url = url.split("#", 1)[0] + "#" + digest.replace(":", "=", 1)
                channels.setdefault(channel, set()).add((name, url))

    with generated_site(output) as temporary:
        render(channels, temporary, repo_url)


def render(channels: dict, output: pathlib.Path, repo_url: str) -> None:
    output.mkdir(parents=True, exist_ok=True)
    root = output / "whl"
    root.mkdir(parents=True, exist_ok=True)

    whl_cards: list[str] = []
    total_wheels = 0
    for channel in sorted(channels, key=order):
        # Alphabetical first for a stable secondary order (e.g. platform tag),
        # then re-sort by version descending so the newest release is on top.
        assets = sorted(channels[channel])
        assets.sort(key=lambda item: wheel_version_key(item[0]), reverse=True)
        total_wheels += len(assets)
        if channel == "cpu":
            label = "CPU (portable)"
        elif channel == "avx2":
            label = "CPU (AVX2)"
        else:
            label = f"CUDA {channel[2:-1]}.{channel[-1]}"
        whl_cards.append(card(f"{channel}/", label, f"{len(assets)} wheel(s)"))

        channel_dir, project_dir = root / channel, root / channel / PACKAGE
        project_dir.mkdir(parents=True, exist_ok=True)

        # This anchor is the normalized project link required by a PEP 503 repository root.
        channel_body = grid([card(f"{PACKAGE}/", PACKAGE, "View wheels")])
        (channel_dir / "index.html").write_text(
            page(label, "Distribution channel for guanaco-py.", channel_body, "../"),
            encoding="utf-8",
        )

        # Direct anchors to wheel files are what pip consumes on the project page.
        if assets:
            items = "\n            ".join(
                f'<li><a class="wheel" href="{html.escape(url, quote=True)}">{html.escape(name)}</a></li>'
                for name, url in assets
            )
        else:
            items = '<li class="meta">No wheels published yet.</li>'
        project_body = f'<ul class="wheels">\n            {items}\n        </ul>'
        (project_dir / "index.html").write_text(
            page(
                f"{PACKAGE} \u00b7 {label}",
                f"{len(assets)} file(s) published.",
                project_body,
                "../",
            ),
            encoding="utf-8",
        )

    (root / "index.html").write_text(
        page(
            "Wheel index",
            "Choose CPU (portable), CPU (AVX2) or a CUDA build.",
            grid(whl_cards),
            "../",
        ),
        encoding="utf-8",
    )

    # Landing page at the site root (was previously missing, causing a 404).
    home_cards = [
        card("whl/", "Wheel index", f"{total_wheels} wheel(s) &middot; {len(channels)} channel(s)"),
        card(repo_url, "Build repository", "Guanaco distribution automation"),
        card(f"https://github.com/{UPSTREAM}", "Upstream", "Bindings, documentation &amp; wiki"),
        card(f"{repo_url}#installation", "Installation", "Setup instructions"),
        card(f"{repo_url}/releases", "Releases", "Changelog &amp; downloads"),
    ]
    (output / "index.html").write_text(
        page(
            "guanaco-py",
            "Prebuilt CPU (portable), CPU (AVX2) and CUDA wheels for llama.cpp Python bindings.",
            grid(home_cards),
            show_icon=True,
        ),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("releases", type=pathlib.Path)
    parser.add_argument("output", type=pathlib.Path)
    parser.add_argument(
        "--repository", default=os.getenv("GITHUB_REPOSITORY", "TheBigEye/guanaco-py")
    )
    args = parser.parse_args()
    generate(args.releases, args.output, args.repository)


if __name__ == "__main__":
    main()
