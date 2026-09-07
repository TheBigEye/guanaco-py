"""Stage validated wheels and publish immutable channel releases. Dry-run by default."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import shutil
import subprocess
from pathlib import Path

from archive_utils import empty_destination
from release_common import (
    PROVENANCE_SEPARATOR,
    SHA,
    UPSTREAM,
    GitHub,
    build_snapshot,
    complete,
    expected_assets,
    outputs,
    provenance,
    release_tag,
    sha256,
    write_json,
)
from validate_receipts import artifact_specs, matching_manifest
from verify_wheels import verify_directory


def artifact_channel(name: str) -> tuple[str, str]:
    match = re.fullmatch(r"guanaco-py-(cpu|avx2)-(linux|windows)-x64", name)
    if match:
        return match[1], match[2]
    match = re.fullmatch(r"guanaco-py-cuda-(linux|windows)-x64-(cu[0-9]+)-py(3\.[0-9]+)", name)
    if match:
        return match[2], match[1]
    raise ValueError(f"Unexpected build artifact: {name}")


def link_or_copy(source: Path, destination: Path) -> None:
    """Avoid a second multi-GB wheel copy when staging on the same filesystem."""
    try:
        os.link(source, destination)
    except OSError:
        shutil.copyfile(source, destination)


def verify_prepared(plan: dict, prepared: Path) -> dict:
    manifest = json.loads((prepared / "build-manifest.json").read_text(encoding="utf-8"))
    matching_manifest(plan, manifest)
    for name, key in (
        ("source.tar.gz", "source_archive_sha256"),
        ("packaging.patch", "packaging_patch_sha256"),
    ):
        if sha256(prepared / name) != manifest[key]:
            raise ValueError(f"Prepared artifact checksum mismatch: {name}")
    return manifest


def verify_gate(plan: dict, manifest: dict, gate: dict) -> None:
    if (
        not isinstance(gate, dict)
        or type(gate.get("schema")) is not int
        or gate.get("schema") != 1
        or gate.get("plan") != plan
        or gate.get("source_archive_sha256") != manifest["source_archive_sha256"]
    ):
        raise ValueError("Global validation gate does not match the prepared build")
    inventory = gate.get("channels", {})
    if set(inventory) != set(plan["missing_channels"]):
        raise ValueError("Global validation gate is missing requested channels")
    for channel, records in inventory.items():
        expected = {
            name
            for name in expected_assets(plan["version"], channel, plan["python_versions"])
            if name.endswith(".whl")
        }
        if set(records) != expected:
            raise ValueError(f"Global validation gate has an incomplete matrix: {channel}")


def collect_wheels(plan: dict, artifacts: Path, temporary: Path, channels: list[str]) -> None:
    specs = artifact_specs(plan)
    for wheel in sorted(artifacts.rglob("*.whl")):
        relative = wheel.relative_to(artifacts)
        if len(relative.parts) != 2 or wheel.is_symlink() or wheel.parent.is_symlink():
            raise ValueError(f"Unexpected wheel artifact layout: {relative}")
        artifact = relative.parts[0]
        if artifact not in specs:
            raise ValueError(f"Unexpected or unrequested build artifact: {artifact}")
        spec = specs[artifact]
        channel, platform = spec["channel"], spec["platform"]
        if channel not in channels:
            raise ValueError(f"Unrequested build channel: {channel}")
        if channel.startswith("cu") and wheel.name.split("-")[2] != "cp" + spec["python_versions"][
            0
        ].replace(".", ""):
            raise ValueError(f"CUDA artifact/wheel Python mismatch: {artifact}")
        destination = temporary / "validation" / channel / platform / wheel.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            raise ValueError(f"Duplicate wheel artifact: {wheel.name}")
        link_or_copy(wheel, destination)


def write_inventory(
    plan: dict, manifest: dict, channel: str, folder: Path, gate: dict | None
) -> None:
    hashes = {path.name: sha256(path) for path in sorted(folder.iterdir())}
    if gate is not None:
        for path in folder.glob("*.whl"):
            expected = gate["channels"][channel][path.name]
            if expected != {"size": path.stat().st_size, "sha256": hashes[path.name]}:
                raise ValueError(f"Wheel differs from the globally validated receipt: {path.name}")
    write_json(
        folder / "guanaco-build.json",
        {
            **manifest,
            "channel": channel,
            "release_tag": release_tag(plan["version"], channel),
            "assets_sha256": hashes,
        },
    )
    hashes["guanaco-build.json"] = sha256(folder / "guanaco-build.json")
    with (folder / "SHA256SUMS").open("w", encoding="utf-8", newline="\n") as stream:
        stream.write("".join(f"{digest}  {name}\n" for name, digest in sorted(hashes.items())))
    if {path.name for path in folder.iterdir()} != expected_assets(
        plan["version"], channel, plan["python_versions"]
    ):
        raise ValueError("Staged release inventory does not match the expected matrix")


def stage(
    plan: dict,
    prepared: Path,
    artifacts: Path,
    output: Path,
    *,
    channel: str | None = None,
    gate: dict | None = None,
) -> tuple[dict, dict[str, Path]]:
    manifest = verify_prepared(plan, prepared)
    if channel is not None and channel not in plan["missing_channels"]:
        raise ValueError("Cannot stage a channel absent from the release plan")
    channels = [channel] if channel else plan["missing_channels"]
    if gate is not None:
        verify_gate(plan, manifest, gate)
    with empty_destination(output) as temporary:
        collect_wheels(plan, artifacts, temporary, channels)
        for selected in channels:
            folder = temporary / "releases" / selected
            folder.mkdir(parents=True)
            for platform in ("linux", "windows"):
                wheels = verify_directory(
                    temporary / "validation" / selected / platform, manifest, selected, platform
                )
                for wheel in wheels:
                    wheel.replace(folder / wheel.name)
            if selected == "cpu":
                link_or_copy(
                    prepared / "source.tar.gz", folder / f"guanaco-source-{plan['version']}.tar.gz"
                )
                link_or_copy(prepared / "packaging.patch", folder / "packaging.patch")
            write_inventory(plan, manifest, selected, folder, gate)
    return manifest, {selected: output / "releases" / selected for selected in channels}


def state_for(plan: dict, channel: str, finished: bool) -> dict:
    return {
        "version": plan["version"],
        "channel": channel,
        "tag": release_tag(plan["version"], channel),
        "upstream_repository": UPSTREAM,
        "upstream_commit": plan["upstream"]["commit"],
        "upstream_tag": plan["upstream"]["tag"],
        "upstream_release_id": plan["upstream"]["release_id"],
        "recipe_commit": plan["recipe_commit"],
        "python_versions": copy.deepcopy(plan["python_versions"]),
        "snapshot": build_snapshot(plan),
        "complete": finished,
    }


def release_body(plan: dict, channel: str, finished: bool) -> str:
    origin = plan["upstream"]
    state = state_for(plan, channel, finished)
    # The notes are already at the top of the body. Bind their hash instead of
    # duplicating a potentially long changelog inside the hidden JSON marker.
    frozen_origin = state["snapshot"]["upstream"]
    notes = frozen_origin.pop("body")
    frozen_origin["body_sha256"] = hashlib.sha256(notes.encode("utf-8")).hexdigest()
    marker = json.dumps(state, sort_keys=True, separators=(",", ":")).replace("-->", "--\\u003e")
    return (
        origin["body"]
        + PROVENANCE_SEPARATOR
        + (
            f"Rebuilt as **guanaco-py {plan['version']} · {channel}** from [{UPSTREAM} / {origin['tag']}]({origin['release_url']}). "
            "The upstream release notes above are preserved; these are Guanaco's binaries, not upstream's wheel assets.\n\n"
            f"- Upstream source commit: [`{origin['commit']}`](https://github.com/{UPSTREAM}/commit/{origin['commit']})\n"
            f"- Build recipe: [`{plan['recipe_commit']}`](https://github.com/{plan['repository']}/commit/{plan['recipe_commit']})\n"
            f"- Build run: {plan.get('run_url') or 'local'}\n"
            "- See `guanaco-build.json` for pinned submodules, source/code hashes and wheel checksums; `SHA256SUMS` covers the downloadable assets.\n\n"
            "Only distribution/build metadata was adapted. The Python bindings are unchanged.\n\n"
            f"<!-- guanaco-upstream-build-v1\n{marker}\n-->"
        )
    )


def preflight(api: GitHub, plan: dict, channels: list[str]) -> dict[str, dict | None]:
    """Inspect EVERY requested destination before creating or modifying a release."""
    existing = {}
    for channel in channels:
        tag = release_tag(plan["version"], channel)
        release = api.release(plan["repository"], tag)
        expected = state_for(plan, channel, False)
        state = provenance(release) if release else None
        if release:
            if not state or any(
                state[key] != expected[key]
                for key in (
                    "upstream_commit",
                    "upstream_tag",
                    "upstream_release_id",
                    "python_versions",
                )
            ):
                raise ValueError(f"Refusing to replace unrelated release {tag}")
            if state.get("snapshot") and state["snapshot"] != build_snapshot(plan):
                raise ValueError(f"Frozen source/notes/matrix mismatch: {tag}")
            if not release.get("draft") and not complete(release, state, plan["python_versions"]):
                raise ValueError(
                    f"Published release {tag} is incomplete; refusing to make it a draft or replace it"
                )
        target = api.tag_commit(plan["repository"], tag)
        expected_recipe = (
            state["recipe_commit"] if state and not release.get("draft") else plan["recipe_commit"]
        )
        if target is not None and target != expected_recipe:
            raise ValueError(
                f"Existing Git tag {tag} points to another build recipe; tags are never moved automatically"
            )
        if release and release.get("draft") and state["recipe_commit"] != plan["recipe_commit"]:
            raise ValueError(
                f"Draft {tag} belongs to another build recipe; resolve it before retrying"
            )
        existing[channel] = release
    return existing


def check_uploaded(release: dict, files: list[Path]) -> None:
    assets = release["assets"]
    by_name = {asset["name"]: asset for asset in assets}
    if len(by_name) != len(assets) or set(by_name) != {path.name for path in files}:
        raise ValueError("Unexpected or duplicate release assets; release remains a draft")
    for path in files:
        asset = by_name[path.name]
        if (
            asset.get("state") != "uploaded"
            or type(asset.get("size")) is not int
            or asset["size"] != path.stat().st_size
        ):
            raise ValueError(f"Incomplete upload: {path.name}; release remains a draft")
        if asset.get("digest") and asset["digest"] != "sha256:" + sha256(path):
            raise ValueError(f"GitHub asset digest mismatch: {path.name}")


def publish(api: GitHub, plan: dict, folders: dict[str, Path], uploader=None) -> None:
    if not SHA.fullmatch(plan["recipe_commit"]):
        raise ValueError("Publishing requires an immutable automation commit (GITHUB_SHA)")
    repo = plan["repository"]
    existing = preflight(api, plan, list(folders))

    def upload(tag, files):
        subprocess.run(
            ["gh", "release", "upload", tag, *map(str, files), "--repo", repo, "--clobber"],
            check=True,
        )

    uploader = uploader or upload
    for channel, folder in folders.items():
        tag = release_tag(plan["version"], channel)
        release = existing[channel]
        if release and not release.get("draft"):
            print(f"{tag} already complete; keeping its published binaries")
            continue
        body = release_body(plan, channel, False)
        if release:
            release = api.request(
                f"/repos/{repo}/releases/{release['id']}",
                method="PATCH",
                data={"draft": True, "body": body},
            )
        else:
            release = api.request(
                f"/repos/{repo}/releases",
                method="POST",
                data={
                    "tag_name": tag,
                    "target_commitish": plan["recipe_commit"],
                    "name": tag,
                    "body": body,
                    "draft": True,
                    "prerelease": False,
                    "make_latest": "false",
                },
            )
        files = sorted(folder.iterdir())
        uploader(tag, files)
        refreshed = api.request(f"/repos/{repo}/releases/{release['id']}")
        check_uploaded(refreshed, files)
        api.request(
            f"/repos/{repo}/releases/{release['id']}",
            method="PATCH",
            data={
                "draft": False,
                "prerelease": False,
                "body": release_body(plan, channel, True),
                "make_latest": "true" if channel == "cpu" and plan["promote_latest"] else "false",
            },
        )
        print(f"Published {tag} with {len(files)} verified assets")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--prepared", type=Path)
    parser.add_argument("--artifacts", type=Path)
    parser.add_argument("--output", type=Path, default=Path("work/publish"))
    parser.add_argument("--channel", help="Stage one channel only (bounded CI disk usage)")
    parser.add_argument(
        "--gate", type=Path, help="Global matrix validation produced by validate_receipts.py"
    )
    parser.add_argument(
        "--preflight",
        action="store_true",
        help="Read-only inspection of all release/tag destinations",
    )
    parser.add_argument(
        "--publish", action="store_true", help="Allow GitHub writes; otherwise only validate/stage"
    )
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    if args.preflight and args.publish:
        parser.error("--preflight cannot be combined with --publish")
    if args.preflight:
        preflight(GitHub(), plan, plan["missing_channels"])
        print("Release/tag preflight passed; no GitHub writes")
        return
    if args.prepared is None or args.artifacts is None:
        parser.error("Staging requires --prepared and --artifacts")
    gate = json.loads(args.gate.read_text(encoding="utf-8")) if args.gate else None
    if args.publish and args.channel and gate is None:
        parser.error("Publishing a single channel requires the global --gate")
    _, folders = stage(
        plan, args.prepared, args.artifacts, args.output, channel=args.channel, gate=gate
    )
    if args.publish:
        if os.getenv("GITHUB_REPOSITORY") != plan["repository"]:
            raise ValueError("Publishing repository does not match GITHUB_REPOSITORY")
        publish(GitHub(writable=True), plan, folders)
        outputs(os.getenv("GITHUB_OUTPUT"), version=plan["version"], published=True)
    else:
        print("DRY RUN: validated channel artifacts; no GitHub writes")


if __name__ == "__main__":
    main()
