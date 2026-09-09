"""Downloadable diagnostics for manual, artifact-only test builds.

A rehearsal produces no release, so the only way to understand what happened is
to read what the job left behind. These reports collect exactly that -- the
patches that went in, the files that came out, the wheels that were built and
whether the requested matrix was actually complete -- without installing or
executing anything.
"""

from __future__ import annotations

import json
import os
import shutil
import tarfile
import zipfile
from pathlib import Path

from .models import Plan, SourceManifest, read_json, write_json
from .releases import ReceiptCollector, artifact_specs
from .settings import Settings
from .toolchain import Toolchain
from .transfer import MAX_MEMBERS, portable_path, sha256

MAX_TEXT_BYTES = 4 * 1024**2


class ReportError(ValueError):
    """Raised when a report is asked for something it cannot describe."""


def require_test_plan(path: Path) -> Plan:
    """Load a plan and insist it is a rehearsal, not a release."""
    plan = Plan.from_mapping(read_json(path))
    if not plan.test_only:
        raise ReportError("This report is only for test-build plans")
    plan.build_platforms()
    return plan


def _write_markdown(path: Path, text: str) -> None:
    """Write a Markdown file with stable, Unix line endings.

    ``Path.write_text`` only learned about ``newline`` in Python 3.10, and the
    supported matrix starts at 3.9, so the file is opened explicitly instead.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(text)


class ReportWriter:
    """Writes the three diagnostics a test build can produce."""

    def __init__(self, settings: Settings) -> None:
        """Remember the configuration and build a toolchain for the reports."""
        self.settings = settings
        self.toolchain = Toolchain(settings)

    # -- Source diagnostics ---------------------------------------------------

    def source(self, plan: Plan, prepared: Path, output: Path, log: Path | None = None) -> None:
        """Describe the prepared source, even when preparation failed.

        Args:
            plan: The test plan.
            prepared: Directory holding (or missing) the prepared snapshot.
            output: Where to write the report.
            log: Optional preparation log to include.
        """
        output = Path(output)
        output.mkdir(parents=True, exist_ok=True)
        write_json(output / "test-plan.json", plan.to_mapping())
        self._copy_patches(output)

        if log is not None and Path(log).is_file():
            shutil.copyfile(log, output / "preparation.log")

        summary = "# Test build source — NOT a release\n\n"
        manifest_path = Path(prepared) / "build-manifest.json"
        if not manifest_path.is_file():
            _write_markdown(
                output / "README.md",
                summary
                + "Source preparation did not produce a manifest. Inspect `preparation.log` "
                "and the source job; no valid source snapshot is implied.\n",
            )
            return

        manifest = SourceManifest.from_mapping(read_json(manifest_path))
        ReceiptCollector.check_plan(plan, manifest)
        for name, expected in (
            ("source.tar.gz", manifest.source_archive_sha256),
            ("packaging.patch", manifest.packaging_patch_sha256),
        ):
            if sha256(Path(prepared) / name) != expected:
                raise ReportError(f"Prepared test artifact checksum mismatch: {name}")

        shutil.copyfile(manifest_path, output / "build-manifest.json")
        shutil.copyfile(Path(prepared) / "packaging.patch", output / "packaging.patch")
        write_json(
            output / "applied-patches.json",
            {"applied_patches": [record.to_mapping() for record in manifest.applied_patches]},
        )
        skipped = self._extract_patched_files(Path(prepared) / "source.tar.gz", manifest, output)
        write_json(output / "build-options.json", self._build_options(plan, manifest))

        summary += (
            f"- Version: `{plan.version}`\n"
            f"- Upstream tag: `{plan.origin.tag}`\n"
            f"- Upstream SHA: `{plan.origin.commit}`\n"
            f"- Source archive SHA256: `{manifest.source_archive_sha256}`\n"
            f"- Patch records: **{len(manifest.applied_patches)}**\n\n"
            "`patches/` contains the recipe's input diffs; `patched-source/` contains the final "
            "affected files. `applied-patches.json` and the manifest preserve the existing patch "
            "mechanism's report. The source archive already contains those changes; do not apply "
            "the patches a second time.\n\n"
            "Build flags/selectors are in `build-options.json`. These files are diagnostics, "
            "not publication approval.\n"
        )
        if skipped:
            summary += (
                "\nLarge/nonregular targets omitted from this small report "
                f"(see source tarball): {', '.join(skipped)}\n"
            )
        _write_markdown(output / "README.md", summary)

    def _copy_patches(self, output: Path) -> None:
        """Copy the recipe's input patches and their checksums."""
        patches_dir = self.settings.patches_dir
        inputs = []
        for patch in sorted(patches_dir.glob("*.patch")) if patches_dir.is_dir() else []:
            destination = output / "patches" / patch.name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(patch, destination)
            inputs.append({"name": patch.name, "sha256": sha256(patch)})
        write_json(output / "patch-inputs.json", {"patches": inputs})

    @staticmethod
    def _extract_patched_files(archive: Path, manifest: SourceManifest, output: Path) -> list[str]:
        """Copy only the files a patch touched, not the whole native tree.

        Returns:
            The target names that were too large or not regular files to copy.
        """
        skipped = []
        targets = {name for record in manifest.applied_patches for name in record.files}
        with tarfile.open(archive, "r:gz") as source:
            for name in sorted(targets):
                member = source.getmember(str(portable_path(name)))
                if not member.isfile() or member.size > MAX_TEXT_BYTES:
                    skipped.append(name)
                    continue
                destination = output / "patched-source" / name
                destination.parent.mkdir(parents=True, exist_ok=True)
                with source.extractfile(member) as original, destination.open("wb") as target:
                    shutil.copyfileobj(original, target)
        return skipped

    def _build_options(self, plan: Plan, manifest: SourceManifest) -> dict:
        """Record the exact compiler flags every requested channel would use."""
        options: dict = {}
        for channel in plan.missing_channels:
            if channel.is_cpu_variant:
                options[channel.name] = {
                    platform.value: self.toolchain.cpu(
                        manifest.plan, channel, platform
                    ).to_mapping()
                    for platform in plan.build_platforms()
                }
            else:
                options[channel.name] = self.toolchain.cuda(manifest.plan, channel).to_mapping()
        return options

    # -- Wheel diagnostics ----------------------------------------------------

    def wheels(
        self,
        manifest: SourceManifest,
        directory: Path,
        output: Path,
        channel: str,
        platform: str,
        verification: str,
    ) -> None:
        """Describe the wheels a build job produced.

        Only ZIP metadata and file inventories are read: no wheel is installed
        or executed, so a broken binary cannot damage the runner.
        """
        output = Path(output)
        output.mkdir(parents=True, exist_ok=True)
        records = []
        for wheel in sorted(Path(directory).glob("*.whl")):
            record: dict = {
                "name": wheel.name,
                "size": wheel.stat().st_size,
                "sha256": sha256(wheel),
            }
            try:
                self._describe_wheel(wheel, record)
            except (ValueError, OSError, zipfile.BadZipFile, RuntimeError) as error:
                record["inspection_error"] = str(error)
            records.append(record)
        write_json(
            output / "wheels.json",
            {
                "test_only": True,
                "version": manifest.version,
                "channel": channel,
                "platform": platform,
                "verification_step": verification,
                "wheels": records,
            },
        )
        _write_markdown(
            output / "README.md",
            f"# Test wheels — {channel} / {platform}\n\n"
            f"Verification step: **{verification}**. Wheels present: **{len(records)}**.\n\n"
            "`wheels.json` contains SHA256, sizes, METADATA/WHEEL text and the ZIP member "
            "inventory. A downloadable test wheel is NOT automatically a validated wheel: "
            "partial/failed test jobs may also retain binaries for debugging. Check the final "
            "`test-build-report` and the job logs before installing in an isolated environment.\n",
        )

    @staticmethod
    def _describe_wheel(wheel: Path, record: dict) -> None:
        """Add the file inventory and metadata text of one wheel to `record`."""
        with zipfile.ZipFile(wheel) as archive:
            members = archive.infolist()
            if len(members) > MAX_MEMBERS:
                raise ReportError("Excessive wheel member count")
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
                    member for member in members if member.filename.endswith(f".dist-info/{kind}")
                ]
                if len(found) != 1 or found[0].file_size > MAX_TEXT_BYTES:
                    raise ReportError(f"Missing/ambiguous/oversized {kind}")
                texts[kind] = archive.read(found[0]).decode("utf-8")
            record["metadata"] = texts

    # -- Final report ---------------------------------------------------------

    def result(
        self,
        plan: Plan,
        prepared: Path,
        receipts: Path,
        output: Path,
        jobs: dict,
        downloads: dict | None = None,
    ) -> bool:
        """Summarise the whole rehearsal and return whether it passed.

        Args:
            plan: The test plan.
            prepared: Directory holding the prepared snapshot.
            receipts: Directory holding every downloaded receipt.
            output: Where to write the report.
            jobs: Mapping of job name to its outcome, as reported by Actions.
            downloads: Artifact download outcomes, when available.

        Returns:
            ``True`` only if every required job succeeded *and* every requested
            receipt validated. Partial success is reported as a failure.
        """
        output = Path(output)
        output.mkdir(parents=True, exist_ok=True)
        specs = artifact_specs(plan, self.settings.package)
        required = ["source"]
        required += [name for name in ("cpu", "avx2") if name in self._requested(plan)]
        if any(name.startswith("cu") for name in self._requested(plan)):
            required.append("cuda")
        states = {name: jobs.get(name, {}).get("result", "missing") for name in required}

        error = None
        gate = None
        try:
            if downloads is not None and states.get("source") == "success":
                for name in ("source", "receipts"):
                    if downloads.get(name, {}).get("outcome") != "success":
                        raise ReportError(f"Report artifact download failed or was skipped: {name}")
            manifest = SourceManifest.from_mapping(
                read_json(Path(prepared) / "build-manifest.json")
            )
            if not manifest.plan.test_only:
                raise ReportError("Prepared source is not from a test build")
            gate = ReceiptCollector(self.settings).collect(plan, manifest, receipts)
        except (ValueError, OSError) as failure:
            error = str(failure)

        success = gate is not None and all(state == "success" for state in states.values())
        available = [name for name in specs if (Path(receipts) / f"{name}.json").is_file()]
        write_json(
            output / "result.json",
            {
                "test_only": True,
                "success": success,
                "version": plan.version,
                "jobs": states,
                "expected_wheels": sum(len(spec.python_versions) for spec in specs.values()),
                "expected_receipts": len(specs),
                "received_receipts": len(available),
                "validation_error": error,
                "artifacts": {name: spec.python_versions for name, spec in specs.items()},
            },
        )
        write_json(output / "test-plan.json", plan.to_mapping())
        if gate is not None:
            write_json(output / "validated-test-build.json", gate.to_mapping())

        summary = (
            f"# Test build: {'PASS' if success else 'FAIL'} — no release created\n\n"
            f"- Upstream: `{plan.version}` / `{plan.origin.tag}`\n"
            f"- Source SHA: `{plan.origin.commit}`\n"
            f"- Requested: **{sum(len(spec.python_versions) for spec in specs.values())} wheels**\n"
            f"- Receipts found: **{len(available)}/{len(specs)}**\n"
            f"- Jobs: {', '.join(f'{name}={state}' for name, state in states.items())}\n\n"
            f"Download **test-{self.settings.package}-*** artifacts for wheels; "
            f"**inspection-test-{self.settings.package}-*** for ZIP inventories/metadata; "
            "**test-build-inspection** for patches, preparation logs and affected source files; "
            "**guanaco-source** for the complete prepared archive.\n\n"
            "Artifacts from failed/partial jobs are diagnostic only. PASS means the selected "
            "build/verification matrix succeeded, not that every model/GPU/integration was "
            "tested. No release, tag, Pages deployment or container was published.\n"
        )
        if error:
            summary += (
                "\n## Validation error\n\n```text\n" + error.replace("```", "'''") + "\n```\n"
            )
        summary += "\n## Expected wheel artifacts\n\n| Artifact | Python | Receipt found |\n|---|---|---|\n"
        for name, spec in specs.items():
            found = "yes" if name in available else "no"
            summary += f"| `{name}` | {', '.join(spec.python_versions)} | {found} |\n"
        _write_markdown(output / "README.md", summary)
        if os.getenv("GITHUB_STEP_SUMMARY"):
            with Path(os.environ["GITHUB_STEP_SUMMARY"]).open(
                "a", encoding="utf-8", newline="\n"
            ) as stream:
                stream.write(summary)
        print(summary)
        return success

    @staticmethod
    def _requested(plan: Plan) -> tuple[str, ...]:
        """Return the channel names this plan asked to build."""
        return tuple(channel.name for channel in plan.missing_channels)

    @staticmethod
    def read_json_env(name: str) -> dict:
        """Decode a JSON object passed through an environment variable."""
        text = os.getenv(name)
        return json.loads(text) if text else {}


__all__ = ["ReportError", "ReportWriter", "require_test_plan"]
