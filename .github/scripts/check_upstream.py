"""Select a stable upstream version; freeze its source and plan missing channels."""

from __future__ import annotations

import argparse
import copy
import os
import re
import urllib.parse
from pathlib import Path

from release_common import (
    CONFIG,
    UPSTREAM,
    GitHub,
    complete,
    outputs,
    provenance,
    repository_name,
    source_notes,
    validate_build_matrix,
    version_from_tag,
    version_key,
    write_json,
)


def select_release(releases: list[dict], requested: str = "") -> dict:
    if requested:
        version_key(requested)
    eligible = []
    for release in releases:
        version = version_from_tag(release.get("tag_name"))
        if (
            release.get("draft")
            or release.get("prerelease")
            or not release.get("published_at")
            or not version
        ):
            continue
        if not requested or version == requested:
            eligible.append(release)
    if not eligible:
        raise ValueError(f"No published stable upstream release found for {requested or 'latest'}")
    return max(
        eligible,
        key=lambda r: (
            version_key(version_from_tag(r["tag_name"])),
            r["published_at"],
            r["tag_name"],
        ),
    )


def family_releases(releases: list[dict], version: str) -> dict[str, dict]:
    """Never let a duplicate draft silently overwrite another release in a dict."""
    pattern = re.compile(rf"v{re.escape(version)}(?:-(avx2|cu[0-9]+))?")
    family = {}
    for release in releases:
        match = pattern.fullmatch(release.get("tag_name", ""))
        if not match:
            continue
        channel = match.group(1) or "cpu"
        if channel in family:
            raise ValueError(f"Duplicate releases for {release['tag_name']}; resolve the ambiguity")
        family[channel] = release
    return family


def fresh_origin(api: GitHub, release: dict) -> dict:
    commit = api.commit(UPSTREAM, release["tag_name"])
    return {
        "repository": UPSTREAM,
        "release_id": release["id"],
        "tag": release["tag_name"],
        "commit": commit,
        "release_url": release["html_url"],
        "release_name": release.get("name") or release["tag_name"],
        "published_at": release["published_at"],
        "body": release.get("body") or "",
        "zip_url": f"https://codeload.github.com/{UPSTREAM}/zip/{commit}",
    }


def legacy_snapshot(state: dict, release: dict, upstream_releases: list[dict]) -> dict:
    """Compatibility with the original v1 markers, which lacked a full snapshot."""
    body = source_notes(release.get("body") or "")
    original = next((r for r in upstream_releases if r["id"] == state["upstream_release_id"]), {})
    tag, commit = state["upstream_tag"], state["upstream_commit"]
    origin = {
        "repository": UPSTREAM,
        "release_id": state["upstream_release_id"],
        "tag": tag,
        "commit": commit,
        "release_url": f"https://github.com/{UPSTREAM}/releases/tag/{urllib.parse.quote(tag, safe='')}",
        "release_name": original.get("name") or tag,
        "published_at": original.get("published_at"),
        "body": body,
        "zip_url": f"https://codeload.github.com/{UPSTREAM}/zip/{commit}",
    }
    print("Legacy v1 marker: using its pinned source/Python versions and the current CUDA matrix")
    return {
        "upstream": origin,
        "python_versions": state["python_versions"],
        "channels": ["cpu", "avx2", *CONFIG["cuda"]],
        "cuda": copy.deepcopy(CONFIG["cuda"]),
    }


def frozen_family(
    family: dict[str, dict], upstream_releases: list[dict]
) -> tuple[dict | None, dict]:
    states = {}
    snapshots = []
    for channel, release in family.items():
        state = provenance(release)
        if state is None:
            raise ValueError(
                f"{release['tag_name']} already exists without Guanaco provenance; refusing to overwrite a legacy/manual release"
            )
        if not release.get("draft") and not complete(release, state, state["python_versions"]):
            raise ValueError(
                f"Published release {release['tag_name']} is incomplete; refusing automatic replacement"
            )
        states[channel] = state
        snapshots.append(
            state.get("snapshot") or legacy_snapshot(state, release, upstream_releases)
        )
    if len({state["upstream_commit"] for state in states.values()}) > 1:
        raise ValueError(
            "Mixed upstream commits in this release family; manual investigation required"
        )
    if snapshots and any(snapshot != snapshots[0] for snapshot in snapshots[1:]):
        raise ValueError("Mixed source/notes/build matrices in the release family")
    return (copy.deepcopy(snapshots[0]) if snapshots else None), states


def make_plan(api: GitHub, repository: str, requested: str = "") -> dict:
    repository_name(repository)
    upstream_releases = api.releases(UPSTREAM)
    selected = select_release(upstream_releases, requested)
    latest = version_from_tag(select_release(upstream_releases)["tag_name"])
    version = version_from_tag(selected["tag_name"])
    family = family_releases(api.releases(repository), version)
    snapshot, states = frozen_family(family, upstream_releases)
    if snapshot is None:
        snapshot = {
            "upstream": fresh_origin(api, selected),
            "python_versions": copy.deepcopy(CONFIG["python_versions"]),
            "channels": ["cpu", "avx2", *CONFIG["cuda"]],
            "cuda": copy.deepcopy(CONFIG["cuda"]),
        }
    validate_build_matrix(snapshot)
    missing = [
        channel
        for channel in snapshot["channels"]
        if not complete(family.get(channel, {}), states.get(channel), snapshot["python_versions"])
    ]
    return {
        "schema": 1,
        "repository": repository,
        "version": version,
        **snapshot,
        "missing_channels": missing,
        "build": bool(missing),
        "promote_latest": version == latest,
        "recipe_commit": os.getenv("GITHUB_SHA", "local-working-tree"),
        "run_url": f"https://github.com/{repository}/actions/runs/{os.environ['GITHUB_RUN_ID']}"
        if os.getenv("GITHUB_RUN_ID")
        else None,
    }


def write_summary(plan: dict) -> None:
    missing = ", ".join(plan["missing_channels"]) or "none; no rebuild"
    summary = (
        f"Upstream {plan['version']} @ {plan['upstream']['commit']}\nMissing channels: {missing}\n"
    )
    print(summary)
    if os.getenv("GITHUB_STEP_SUMMARY"):
        with Path(os.environ["GITHUB_STEP_SUMMARY"]).open("a", encoding="utf-8") as stream:
            stream.write("## Upstream release check\n\n" + summary.replace("\n", "  \n"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repository", default=os.getenv("GITHUB_REPOSITORY", "TheBigEye/guanaco-py")
    )
    parser.add_argument(
        "--version", default="", help="Stable X.Y.Z; empty means latest (no historical backfill)"
    )
    parser.add_argument("--output", type=Path, default=Path("work/plan.json"))
    parser.add_argument("--github-output", default=os.getenv("GITHUB_OUTPUT"))
    args = parser.parse_args()
    plan = make_plan(GitHub(), args.repository, args.version)
    write_json(args.output, plan)
    missing = plan["missing_channels"]
    outputs(
        args.github_output,
        build=bool(missing),
        version=plan["version"],
        cpu="cpu" in missing,
        avx2="avx2" in missing,
        cuda=[channel for channel in missing if channel.startswith("cu")],
        promote_latest=plan["promote_latest"],
        publish_matrix={
            "include": [
                {
                    "channel": channel,
                    "artifact_pattern": f"guanaco-py-{channel}-*"
                    if channel in ("cpu", "avx2")
                    else f"guanaco-py-cuda-*-{channel}-py*",
                }
                for channel in missing
            ]
        },
    )
    write_summary(plan)


if __name__ == "__main__":
    main()
