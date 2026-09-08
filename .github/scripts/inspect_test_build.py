"""Small downloadable diagnostics for artifact-only builds; never publish anything."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tarfile
import zipfile
from pathlib import Path

from archive_utils import MAX_MEMBERS, portable_path
from configure_build import cpu_options, cuda_options
from release_common import build_platforms, is_test_build, sha256, write_json
from validate_receipts import artifact_specs, matching_manifest, validate_receipts

PATCHES = Path(__file__).resolve().parents[1] / "patches"
MAX_TEXT_BYTES = 4 * 1024**2


def test_input(path: Path) -> dict:
    plan = json.loads(path.read_text(encoding="utf-8"))
    if not is_test_build(plan):
        raise ValueError("This report is only for test-build plans")
    build_platforms(plan)
    return plan


def markdown(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(text)


def source_report(
    plan: dict, prepared: Path, output: Path, log: Path | None = None, *, patches: Path = PATCHES
) -> None:
    """Keep patch inputs/logs even when source preparation fails."""
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "test-plan.json", plan)
    inputs = []
    for patch in sorted(patches.glob("*.patch")):
        destination = output / "patches" / patch.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(patch, destination)
        inputs.append({"name": patch.name, "sha256": sha256(patch)})
    write_json(output / "patch-inputs.json", {"patches": inputs})
    if log and log.is_file():
        shutil.copyfile(log, output / "preparation.log")
    summary = "# Test build source — NOT a release\n\n"
    manifest_path = prepared / "build-manifest.json"
    if not manifest_path.is_file():
        markdown(
            output / "README.md",
            summary
            + "Source preparation did not produce a manifest. Inspect `preparation.log` and the source job; no valid source snapshot is implied.\n",
        )
        return
    manifest = test_input(manifest_path)
    matching_manifest(plan, manifest)
    for name, field in (
        ("source.tar.gz", "source_archive_sha256"),
        ("packaging.patch", "packaging_patch_sha256"),
    ):
        if sha256(prepared / name) != manifest[field]:
            raise ValueError(f"Prepared test artifact checksum mismatch: {name}")
    shutil.copyfile(manifest_path, output / "build-manifest.json")
    shutil.copyfile(prepared / "packaging.patch", output / "packaging.patch")
    records = manifest.get("applied_patches", [])
    write_json(output / "applied-patches.json", {"applied_patches": records})
    targets = {name for record in records for name in record["files"]}
    skipped = []
    # Copy only the final files mentioned by patches, not the multi-MB native
    # source tree. The full prepared tarball remains a separate Actions artifact.
    with tarfile.open(prepared / "source.tar.gz", "r:gz") as archive:
        for name in sorted(targets):
            relative = portable_path(name)
            member = archive.getmember(name)
            if not member.isfile() or member.size > MAX_TEXT_BYTES:
                skipped.append(name)
                continue
            destination = output / "patched-source" / str(relative)
            destination.parent.mkdir(parents=True, exist_ok=True)
            with archive.extractfile(member) as original, destination.open("wb") as target:
                shutil.copyfileobj(original, target)
    settings = {}
    for channel in plan["missing_channels"]:
        settings[channel] = (
            {
                platform: cpu_options(manifest, channel, platform)
                for platform in build_platforms(plan)
            }
            if channel in ("cpu", "avx2")
            else cuda_options(manifest, channel)
        )
    write_json(output / "build-options.json", settings)
    summary += (
        f"- Version: `{plan['version']}`\n- Upstream tag: `{plan['upstream']['tag']}`\n"
        f"- Upstream SHA: `{plan['upstream']['commit']}`\n"
        f"- Source archive SHA256: `{manifest['source_archive_sha256']}`\n"
        f"- Patch records: **{len(records)}**\n\n"
        "`patches/` contains the recipe's input diffs; `patched-source/` contains the final affected files. "
        "`applied-patches.json` and the manifest preserve the existing patch mechanism's report. "
        "The source archive already contains those changes; do not apply the patches a second time.\n\n"
        "Build flags/selectors are in `build-options.json`. These files are diagnostics, not publication approval.\n"
    )
    if skipped:
        summary += (
            "\nLarge/nonregular targets omitted from this small report (see source tarball): "
            + ", ".join(skipped)
            + "\n"
        )
    markdown(output / "README.md", summary)


def wheel_report(
    manifest: dict, directory: Path, output: Path, channel: str, platform: str, verification: str
) -> None:
    """Read ZIP metadata/file inventories only; no wheel is installed or executed."""
    output.mkdir(parents=True, exist_ok=True)
    records = []
    for wheel in sorted(directory.glob("*.whl")):
        record = {"name": wheel.name, "size": wheel.stat().st_size, "sha256": sha256(wheel)}
        try:
            with zipfile.ZipFile(wheel) as archive:
                members = archive.infolist()
                if len(members) > MAX_MEMBERS:
                    raise ValueError("Excessive wheel member count")
                record["files"] = [
                    {
                        "path": member.filename,
                        "size": member.file_size,
                        "compressed_size": member.compress_size,
                    }
                    for member in members
                ]
                texts = {}
                for kind in ("METADATA", "WHEEL"):
                    found = [
                        member
                        for member in members
                        if member.filename.endswith(f".dist-info/{kind}")
                    ]
                    if len(found) != 1 or found[0].file_size > MAX_TEXT_BYTES:
                        raise ValueError(f"Missing/ambiguous/oversized {kind}")
                    texts[kind] = archive.read(found[0]).decode("utf-8")
                record["metadata"] = texts
        except (ValueError, OSError, zipfile.BadZipFile, RuntimeError) as error:
            record["inspection_error"] = str(error)
        records.append(record)
    write_json(
        output / "wheels.json",
        {
            "test_only": True,
            "version": manifest["version"],
            "channel": channel,
            "platform": platform,
            "verification_step": verification,
            "wheels": records,
        },
    )
    markdown(
        output / "README.md",
        (
            f"# Test wheels — {channel} / {platform}\n\n"
            f"Verification step: **{verification}**. Wheels present: **{len(records)}**.\n\n"
            "`wheels.json` contains SHA256, sizes, METADATA/WHEEL text and the ZIP member inventory. "
            "A downloadable test wheel is NOT automatically a validated wheel: partial/failed test jobs may also retain binaries for debugging. "
            "Check the final `test-build-report` and the job logs before installing in an isolated environment.\n"
        ),
    )


def result_report(
    plan: dict,
    prepared: Path,
    receipts: Path,
    output: Path,
    jobs: dict,
    downloads: dict | None = None,
) -> bool:
    output.mkdir(parents=True, exist_ok=True)
    specs = artifact_specs(plan)
    required = ["source"]
    required += [channel for channel in ("cpu", "avx2") if channel in plan["missing_channels"]]
    if any(channel.startswith("cu") for channel in plan["missing_channels"]):
        required.append("cuda")
    states = {name: jobs.get(name, {}).get("result", "missing") for name in required}
    error = None
    gate = None
    try:
        if downloads is not None and states.get("source") == "success":
            for name in ("source", "receipts"):
                if downloads.get(name, {}).get("outcome") != "success":
                    raise ValueError(f"Report artifact download failed or was skipped: {name}")
        manifest = test_input(prepared / "build-manifest.json")
        gate = validate_receipts(plan, manifest, receipts)
    except (ValueError, OSError) as failure:
        error = str(failure)
    success = gate is not None and all(state == "success" for state in states.values())
    expected_wheels = sum(len(spec["python_versions"]) for spec in specs.values())
    available = [name for name in specs if (receipts / (name + ".json")).is_file()]
    result = {
        "test_only": True,
        "success": success,
        "version": plan["version"],
        "jobs": states,
        "expected_wheels": expected_wheels,
        "expected_receipts": len(specs),
        "received_receipts": len(available),
        "validation_error": error,
        "artifacts": specs,
    }
    write_json(output / "result.json", result)
    write_json(output / "test-plan.json", plan)
    if gate is not None:
        write_json(output / "validated-test-build.json", gate)
    summary = (
        f"# Test build: {'PASS' if success else 'FAIL'} — no release created\n\n"
        f"- Upstream: `{plan['version']}` / `{plan['upstream']['tag']}`\n"
        f"- Source SHA: `{plan['upstream']['commit']}`\n"
        f"- Requested: **{expected_wheels} wheels**\n"
        f"- Receipts found: **{len(available)}/{len(specs)}**\n"
        f"- Jobs: {', '.join(f'{name}={state}' for name, state in states.items())}\n\n"
        "Download **test-guanaco-py-*** artifacts for wheels; **inspection-test-guanaco-py-*** for ZIP inventories/metadata; "
        "**test-build-inspection** for patches, preparation logs and affected source files; **guanaco-source** for the complete prepared archive.\n\n"
        "Artifacts from failed/partial jobs are diagnostic only. PASS means the selected build/verification matrix succeeded, "
        "not that every model/GPU/integration was tested. No release, tag, Pages deployment or container was published.\n"
    )
    if error:
        summary += "\n## Validation error\n\n```text\n" + error.replace("```", "'''") + "\n```\n"
    summary += (
        "\n## Expected wheel artifacts\n\n| Artifact | Python | Receipt found |\n|---|---|---|\n"
    )
    for name, spec in specs.items():
        summary += f"| `{name}` | {', '.join(spec['python_versions'])} | {'yes' if name in available else 'no'} |\n"
    markdown(output / "README.md", summary)
    if os.getenv("GITHUB_STEP_SUMMARY"):
        with Path(os.environ["GITHUB_STEP_SUMMARY"]).open(
            "a", encoding="utf-8", newline="\n"
        ) as stream:
            stream.write(summary)
    print(summary)
    return success


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="kind", required=True)
    source = sub.add_parser("source")
    source.add_argument("--plan", type=Path, required=True)
    source.add_argument("--prepared", type=Path, required=True)
    source.add_argument("--log", type=Path)
    source.add_argument("--output", type=Path, required=True)
    wheels = sub.add_parser("wheels")
    wheels.add_argument("--manifest", type=Path, required=True)
    wheels.add_argument("--directory", type=Path, required=True)
    wheels.add_argument("--output", type=Path, required=True)
    wheels.add_argument("--channel", required=True)
    wheels.add_argument("--platform", choices=["linux", "windows"], required=True)
    wheels.add_argument(
        "--verification", choices=["success", "failure", "skipped", "cancelled"], required=True
    )
    result = sub.add_parser("result")
    result.add_argument("--plan", type=Path, required=True)
    result.add_argument("--prepared", type=Path, required=True)
    result.add_argument("--receipts", type=Path, required=True)
    result.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.kind == "source":
        source_report(test_input(args.plan), args.prepared, args.output, args.log)
    elif args.kind == "wheels":
        wheel_report(
            test_input(args.manifest),
            args.directory,
            args.output,
            args.channel,
            args.platform,
            args.verification,
        )
    else:
        if not result_report(
            test_input(args.plan),
            args.prepared,
            args.receipts,
            args.output,
            json.loads(os.getenv("BUILD_JOBS", "{}")),
            json.loads(os.environ["ARTIFACT_DOWNLOADS"])
            if os.getenv("ARTIFACT_DOWNLOADS")
            else None,
        ):
            raise SystemExit(1)


if __name__ == "__main__":
    main()
