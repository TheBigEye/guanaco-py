#!/usr/bin/env python3
"""Generate a styled HTML index that remains compatible with PEP 503/pip."""

from __future__ import annotations

import argparse
import html
import json
import pathlib
import re

PACKAGE = "guanaco-py"
WHEEL_PREFIX = "guanaco_py"
TAG = re.compile(r"^v[^-]+(?:-(cu\d+))?$")

CSS = """
:root{color-scheme:dark;--bg:#07111f;--panel:#102036;--line:#29415e;--text:#eef5ff;--muted:#9fb2ca;--gold:#efb84f;--mint:#59d7b5}
*{box-sizing:border-box}body{margin:0;min-height:100vh;font:16px/1.55 system-ui,sans-serif;color:var(--text);background:radial-gradient(circle at 15% 0,#1a3b61 0,transparent 35%),var(--bg)}
main{width:min(1000px,calc(100% - 32px));margin:auto;padding:64px 0 80px}.eyebrow{color:var(--mint);font-size:.78rem;font-weight:800;letter-spacing:.14em;text-transform:uppercase}
h1{margin:.35rem 0 .5rem;font-size:clamp(2rem,6vw,3.8rem);line-height:1}p,.meta,footer{color:var(--muted)}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(230px,1fr));gap:16px}
.card,.wheel{display:block;padding:19px;color:var(--text);text-decoration:none;background:linear-gradient(145deg,#142943,#0c192a);border:1px solid var(--line);border-radius:15px;box-shadow:0 18px 50px #0004}
.card:hover,.wheel:hover{border-color:var(--gold);transform:translateY(-2px)}.card strong{display:block;color:var(--gold);font-size:1.18rem}.card span{color:var(--muted);font-size:.88rem}
.wheels{display:grid;gap:10px;padding:0;list-style:none}.wheel{padding:13px 15px;overflow-wrap:anywhere;font-family:ui-monospace,Consolas,monospace}nav a{color:var(--mint)}footer{margin-top:35px;font-size:.85rem}code{color:var(--gold)}
"""


def page(title: str, intro: str, body: str, parent: str | None = None) -> str:
    back = f'<nav><a href="{html.escape(parent)}">← Volver</a></nav>' if parent else ""
    return f'''<!doctype html>
<html lang="es"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>{html.escape(title)}</title><style>{CSS}</style></head><body><main>{back}
<header><div class="eyebrow">TheBigEye · Guanaco</div><h1>{html.escape(title)}</h1><p>{html.escape(intro)}</p></header>
{body}<footer>Índice compatible con PEP 503 · <code>pip install guanaco-py</code></footer></main></body></html>\n'''


def order(channel: str) -> tuple[int, int]:
    match = re.fullmatch(r"cu(\d+)", channel)
    return (0, 0) if channel == "cpu" else (1, int(match.group(1)) if match else 9999)


def generate(source: pathlib.Path, output: pathlib.Path) -> None:
    raw = json.loads(source.read_text(encoding="utf-8"))
    pages = raw if not raw or isinstance(raw[0], list) else [raw]
    channels: dict[str, set[tuple[str, str]]] = {"cpu": set()}
    for release in (item for batch in pages for item in batch):
        match = TAG.fullmatch(release.get("tag_name", ""))
        if not match:
            continue
        channel = match.group(1) or "cpu"
        for asset in release.get("assets", []):
            name, url = asset.get("name", ""), asset.get("browser_download_url", "")
            if name.startswith(WHEEL_PREFIX) and name.endswith(".whl") and url:
                channels.setdefault(channel, set()).add((name, url))

    root = output / "whl"
    root.mkdir(parents=True, exist_ok=True)
    cards: list[str] = []
    for channel in sorted(channels, key=order):
        assets = sorted(channels[channel])
        label = "CPU" if channel == "cpu" else f"CUDA {channel[2:-1]}.{channel[-1]}"
        cards.append(f'<a class="card" href="{channel}/"><strong>{label}</strong><span>{len(assets)} wheels</span></a>')
        channel_dir, project_dir = root / channel, root / channel / PACKAGE
        project_dir.mkdir(parents=True, exist_ok=True)

        # This anchor is the normalized project link required by a PEP 503 repository root.
        (channel_dir / "index.html").write_text(page(
            label, "Canal de distribución de guanaco-py.",
            '<div class="grid"><a class="card" href="guanaco-py/"><strong>guanaco-py</strong><span>Ver wheels</span></a></div>', "../"), encoding="utf-8")

        # Direct anchors to wheel files are what pip consumes on the project page.
        links = "\n".join(f'<li><a class="wheel" href="{html.escape(url, quote=True)}">{html.escape(name)}</a></li>' for name, url in assets)
        if not links:
            links = '<li class="meta">Todavía no hay wheels publicados.</li>'
        (project_dir / "index.html").write_text(page(
            f"{PACKAGE} · {label}", f"{len(assets)} archivos publicados.",
            f'<ul class="wheels">{links}</ul>', "../"), encoding="utf-8")

    (root / "index.html").write_text(page(
        "Guanaco wheel index", "Elegí CPU o una versión de CUDA.",
        f'<div class="grid">{"".join(cards)}</div>'), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("releases", type=pathlib.Path)
    parser.add_argument("output", type=pathlib.Path)
    args = parser.parse_args()
    generate(args.releases, args.output)


if __name__ == "__main__":
    main()
