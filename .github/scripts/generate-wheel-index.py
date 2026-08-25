#!/usr/bin/env python3
""" Generate a HTML index compatible with PEP 503/pip """

from __future__ import annotations

import argparse
import html
import json
import pathlib
import re

PACKAGE = "guanaco-py"
WHEEL_PREFIX = "guanaco_py"
REPO_URL = "https://github.com/TheBigEye/guanaco-py"
ICON_URL = "https://raw.githubusercontent.com/TheBigEye/guanaco-py/main/docs/icon.svg"
TAG = re.compile(r"^v[^-]+(?:-(cu\d+|avx2))?$")

# Clean :D
CSS = """
:root {
    color-scheme: light;
    --bg: #ffffff;
    --panel: #f7f7f8;
    --line: #e2e2e5;
    --text: #1c1c1f;
    --muted: #6b6b72;
    --accent: #2f5fb3;
}

* {
    box-sizing: border-box;
}

body {
    margin: 0;
    min-height: 100vh;
    font: 16px/1.6 -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
    color: var(--text);
    background: var(--bg);
}

main {
    width: min(760px, calc(100% - 40px));
    margin: auto;
    padding: 56px 0 72px;
}

nav a {
    color: var(--accent);
    text-decoration: none;
    font-size: .9rem;
}

nav a:hover {
    text-decoration: underline;
}

.eyebrow {
    color: var(--muted);
    font-size: .78rem;
    font-weight: 600;
    letter-spacing: .08em;
    text-transform: uppercase;
}

.icon {
    display: block;
    width: 12rem;
    height: 12rem;
    margin: 0 auto 20px;
}

h1 {
    margin: .4rem 0 .5rem;
    font-size: 1.9rem;
    line-height: 1.25;
    font-weight: 700;
}

p {
    color: var(--muted);
    margin: .2rem 0 0;
}

.meta {
    color: var(--muted);
    font-size: .92rem;
}

.grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
    gap: 12px;
    margin-top: 28px;
}

.card {
    display: block;
    padding: 16px 18px;
    color: var(--text);
    text-decoration: none;
    background: var(--panel);
    border: 1px solid var(--line);
    border-radius: 8px;
}

.card:hover {
    border-color: var(--accent);
}

.card strong {
    display: block;
    font-size: 1.05rem;
    font-weight: 600;
}

.card span {
    display: block;
    color: var(--muted);
    font-size: .85rem;
    margin-top: .15rem;
}

.wheels {
    display: grid;
    gap: 8px;
    padding: 0;
    list-style: none;
    margin-top: 24px;
}

.wheel {
    display: block;
    padding: 11px 14px;
    color: var(--text);
    text-decoration: none;
    background: var(--panel);
    border: 1px solid var(--line);
    border-radius: 6px;
    font-family: ui-monospace, SFMono-Regular, Consolas, Menlo, monospace;
    font-size: .85rem;
    overflow-wrap: anywhere;
}

.wheel:hover {
    border-color: var(--accent);
}

footer {
    margin-top: 48px;
    padding-top: 16px;
    border-top: 1px solid var(--line);
    color: var(--muted);
    font-size: .85rem;
}

code {
    color: var(--accent);
    background: var(--panel);
    padding: .1em .35em;
    border-radius: 4px;
    font-size: .9em;
}
"""

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


def page(title: str, intro: str, body: str, parent: str | None = None, show_icon: bool = False) -> str:
    back_link = (
        f'        <nav><a href="{html.escape(parent)}">&larr; Back</a></nav>\n'
        if parent
        else ""
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


def generate(source: pathlib.Path, output: pathlib.Path) -> None:
    raw = json.loads(source.read_text(encoding="utf-8"))
    pages = raw if not raw or isinstance(raw[0], list) else [raw]
    channels: dict[str, set[tuple[str, str]]] = {"cpu": set(), "avx2": set()}
    for release in (item for batch in pages for item in batch):
        match = TAG.fullmatch(release.get("tag_name", ""))
        if not match:
            continue
        channel = match.group(1) or "cpu"
        for asset in release.get("assets", []):
            name, url = asset.get("name", ""), asset.get("browser_download_url", "")
            if name.startswith(WHEEL_PREFIX) and name.endswith(".whl") and url:
                channels.setdefault(channel, set()).add((name, url))

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
            page(f"{PACKAGE} \u00b7 {label}", f"{len(assets)} file(s) published.", project_body, "../"),
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
        card(REPO_URL, "Source code", "GitHub repository"),
        card(f"{REPO_URL}#installation", "Installation", "Setup instructions"),
        card(f"{REPO_URL}/releases", "Releases", "Changelog &amp; downloads"),
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
    args = parser.parse_args()
    generate(args.releases, args.output)


if __name__ == "__main__":
    main()
