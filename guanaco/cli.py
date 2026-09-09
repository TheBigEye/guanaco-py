"""One command line entry point for the whole build system.

Everything CI needs is a subcommand of ``python -m guanaco``:

.. code-block:: text

    python -m guanaco plan              discover the upstream version and plan it
    python -m guanaco plan-test         plan a manual, artifact-only rehearsal
    python -m guanaco prepare-source    download, patch and package the source
    python -m guanaco unpack-source     verify and unpack a prepared snapshot
    python -m guanaco configure ...     resolve compiler flags and job settings
    python -m guanaco verify-wheels     validate built wheels and write a receipt
    python -m guanaco validate-receipts prove the whole matrix was built
    python -m guanaco publish           preflight, stage and publish releases
    python -m guanaco build-index       render the PEP 503 wheel index
    python -m guanaco inspect ...       write downloadable test-build diagnostics
    python -m guanaco explain           print the resolved configuration

Every command reads the same configuration (see :mod:`guanaco.settings`) and
writes GitHub Actions outputs through :func:`guanaco.models.write_outputs`.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from .catalog import IndexGenerator
from .github_api import GithubClient
from .models import Channel, Plan, Platform, SourceManifest, read_json, write_json
from .models import write_outputs as emit
from .releases import (
    ReceiptCollector,
    ReleasePlanner,
    ReleasePublisher,
    TestBuildPlanner,
)
from .reports import ReportWriter, require_test_plan
from .settings import Settings
from .source import SourceArchive, SourcePreparer
from .toolchain import Toolchain
from .transfer import Downloader
from .wheels import WheelValidator


class CommandError(ValueError):
    """Raised when a command cannot honour the arguments it was given."""


def _summary(text: str) -> None:
    """Append a Markdown block to the Actions step summary, when there is one."""
    target = os.getenv("GITHUB_STEP_SUMMARY")
    if target:
        with Path(target).open("a", encoding="utf-8") as stream:
            stream.write(text.replace("\n", "  \n"))


def _publish_matrix(settings: Settings, plan: Plan) -> dict:
    """Return the per-channel publication matrix the workflow iterates over."""
    prefix = "test-" if plan.test_only else ""
    include = []
    for channel in plan.missing_channels:
        if channel.is_cpu_variant:
            pattern = f"{prefix}{settings.package}-{channel}-*"
        else:
            pattern = f"{prefix}{settings.package}-cuda-*-{channel}-py*"
        include.append({"channel": channel.name, "artifact_pattern": pattern})
    return {"include": include}


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


def command_plan(arguments: argparse.Namespace, settings: Settings) -> int:
    """Discover the upstream version and write the frozen release plan."""
    client = GithubClient()
    plan = ReleasePlanner(settings, client).plan(arguments.version)
    write_json(arguments.output, plan.to_mapping())
    missing = plan.missing_channels
    emit(
        os.getenv("GITHUB_OUTPUT"),
        build=plan.needs_build,
        version=plan.version,
        cpu=Channel("cpu") in missing,
        avx2=Channel("avx2") in missing,
        cuda=[channel.name for channel in missing if channel.is_cuda],
        promote_latest=plan.promote_latest,
        publish_matrix=_publish_matrix(settings, plan),
    )
    text = (
        f"Upstream {plan.version} @ {plan.origin.commit}\n"
        f"Missing channels: {', '.join(c.name for c in missing) or 'none; no rebuild'}\n"
    )
    print(text)
    _summary("## Upstream release check\n\n" + text)
    return 0


def command_plan_test(arguments: argparse.Namespace, settings: Settings) -> int:
    """Plan a manual rehearsal that never touches published releases."""
    client = GithubClient()
    plan = TestBuildPlanner(settings, client).plan(
        version=arguments.version,
        cpu=arguments.cpu,
        avx2=arguments.avx2,
        cuda=arguments.cuda,
        cuda_channels=arguments.cuda_channels,
        python_versions=arguments.python_versions,
        systems=arguments.systems,
    )
    write_json(arguments.output, plan.to_mapping())
    missing = plan.missing_channels
    emit(
        os.getenv("GITHUB_OUTPUT"),
        build=True,
        version=plan.version,
        cpu=Channel("cpu") in missing,
        avx2=Channel("avx2") in missing,
        cuda=[channel.name for channel in missing if channel.is_cuda],
    )
    print(
        "## Manual test build - NOT a release\n\n"
        f"- Upstream: `{plan.version}` / `{plan.origin.tag}`\n"
        f"- Source SHA: `{plan.origin.commit}`\n"
        f"- Channels: {', '.join(c.name for c in missing)}\n"
        f"- Python: {', '.join(plan.python_versions)}\n"
        f"- Systems: {', '.join(p.value for p in plan.build_platforms())} (x86-64)\n"
        f"- Expected: **{_expected_wheel_count(plan)} wheels**, "
        f"{len(_publish_matrix(settings, plan)['include'])} build receipts\n\n"
        "Only Actions artifacts will be uploaded. No releases, tags, Pages or Docker publication.\n"
    )
    return 0


def _expected_wheel_count(plan: Plan) -> int:
    """Return how many wheels a plan is going to produce."""
    per_channel = len(plan.python_versions) * len(plan.build_platforms())
    return per_channel * len(plan.missing_channels)


def command_prepare_source(arguments: argparse.Namespace, settings: Settings) -> int:
    """Download the pinned upstream source, patch it and package it."""
    plan = Plan.from_mapping(read_json(arguments.plan))
    preparer = SourcePreparer(settings, GithubClient(), Downloader())
    preparer.prepare(plan, arguments.output)
    return 0


def command_unpack_source(arguments: argparse.Namespace, settings: Settings) -> int:
    """Verify and unpack a prepared snapshot into a build directory."""
    del settings  # unpacking only needs the artifact itself
    SourceArchive.extract(arguments.artifact, arguments.destination, arguments.version)
    return 0


def command_configure(arguments: argparse.Namespace, settings: Settings) -> int:
    """Resolve the compiler flags and job settings of one build."""
    toolchain = Toolchain(settings)
    if arguments.kind == "docker":
        image = toolchain.image(arguments.version, arguments.promote_latest)
        values = image.to_mapping()
    else:
        manifest = SourceManifest.from_mapping(read_json(arguments.manifest))
        plan = manifest.plan
        if arguments.version and arguments.version != plan.version:
            raise CommandError("Prepared manifest version does not match the workflow input")
        if arguments.kind == "matrix":
            values = {
                "matrix": {"include": [row.to_mapping() for row in toolchain.platform_matrix(plan)]}
            }
        elif arguments.kind == "cpu":
            values = toolchain.cpu(
                plan, Channel(arguments.channel), Platform.parse(arguments.platform)
            ).to_mapping()
        else:
            values = toolchain.cuda(plan, Channel(arguments.channel)).to_mapping()
    emit(os.getenv("GITHUB_OUTPUT"), **values)
    print(json.dumps(values, indent=2))
    return 0


def command_verify_wheels(arguments: argparse.Namespace, settings: Settings) -> int:
    """Validate built wheels and optionally write a small receipt."""
    validator = WheelValidator(settings)
    manifest = SourceManifest.from_mapping(read_json(arguments.manifest))
    platform = Platform.parse(arguments.platform)
    channel = Channel(arguments.channel)

    if arguments.selectors:
        print("build=" + Toolchain(settings).build_selector(manifest.plan, platform))
        return 0
    if arguments.directory is None:
        raise CommandError("--directory is required to validate wheels")

    if arguments.single:
        wheels = sorted(Path(arguments.directory).glob("*.whl"))
        if len(wheels) != 1:
            raise CommandError("Expected exactly one wheel in this CUDA job")
        if arguments.python and wheels[0].name.split("-")[2] != "cp" + arguments.python.replace(
            ".", ""
        ):
            raise CommandError("CUDA job Python version does not match its wheel")
        validator.verify(
            wheels[0],
            manifest,
            channel,
            platform,
            allow_unrepaired=arguments.unrepaired,
        )
    else:
        wheels = validator.verify_directory(
            Path(arguments.directory),
            manifest,
            channel,
            platform,
            allow_unrepaired=arguments.unrepaired,
        )
    if arguments.receipt:
        write_json(
            arguments.receipt,
            validator.receipt(wheels, manifest, channel, platform).to_mapping(),
        )
    return 0


def command_validate_receipts(arguments: argparse.Namespace, settings: Settings) -> int:
    """Require every receipt of the requested matrix and emit the gate."""
    plan = Plan.from_mapping(read_json(arguments.plan))
    manifest = SourceManifest.from_mapping(read_json(arguments.prepared / "build-manifest.json"))
    gate = ReceiptCollector(settings).collect(plan, manifest, arguments.receipts)
    write_json(arguments.output, gate.to_mapping())
    total = sum(len(records) for records in gate.channels.values())
    print(f"Validated {total} wheel receipts across {len(gate.channels)} channels")
    return 0


def command_publish(arguments: argparse.Namespace, settings: Settings) -> int:
    """Preflight destinations, or stage and publish validated channels."""
    plan = Plan.from_mapping(read_json(arguments.plan))
    publisher = ReleasePublisher(settings, GithubClient(writable=arguments.publish))
    if arguments.preflight:
        publisher.preflight(plan)
        print("Release/tag preflight passed; no GitHub writes")
        return 0
    if arguments.prepared is None or arguments.artifacts is None:
        raise CommandError("Staging requires --prepared and --artifacts")
    gate = None
    if arguments.gate:
        gate = read_gate(arguments.gate)
    _, folders = publisher.stage(
        plan,
        arguments.prepared,
        arguments.artifacts,
        arguments.output,
        channel=Channel(arguments.channel) if arguments.channel else None,
        gate=gate,
    )
    if arguments.publish:
        publisher.publish(plan, folders)
        emit(os.getenv("GITHUB_OUTPUT"), version=plan.version, published=True)
    else:
        print("DRY RUN: validated channel artifacts; no GitHub writes")
    return 0


def read_gate(path: Path):
    """Load the publication gate produced by ``validate-receipts``."""
    from .models import PublicationGate

    return PublicationGate.from_mapping(read_json(path))


def command_build_index(arguments: argparse.Namespace, settings: Settings) -> int:
    """Render the PEP 503 index from the releases that are already published."""
    payload = json.loads(Path(arguments.releases).read_text(encoding="utf-8"))
    pages = payload if not payload or isinstance(payload[0], list) else [payload]
    from .models import Release

    releases = [Release.from_mapping(item) for page in pages for item in page]
    generator = IndexGenerator(settings)
    generator.generate(releases, arguments.output)
    print(f"Generated wheel index with {len(releases)} release(s) at {arguments.output}")
    return 0


def command_inspect(arguments: argparse.Namespace, settings: Settings) -> int:
    """Write the downloadable diagnostics of a test build."""
    writer = ReportWriter(settings)
    if arguments.kind == "source":
        writer.source(
            require_test_plan(arguments.plan), arguments.prepared, arguments.output, arguments.log
        )
    elif arguments.kind == "wheels":
        writer.wheels(
            SourceManifest.from_mapping(read_json(arguments.manifest)),
            arguments.directory,
            arguments.output,
            arguments.channel,
            arguments.platform,
            arguments.verification,
        )
    else:
        downloads = (
            json.loads(os.environ["ARTIFACT_DOWNLOADS"])
            if os.getenv("ARTIFACT_DOWNLOADS")
            else None
        )
        jobs = json.loads(os.getenv("BUILD_JOBS", "{}"))
        if not writer.result(
            require_test_plan(arguments.plan),
            arguments.prepared,
            arguments.receipts,
            arguments.output,
            jobs,
            downloads,
        ):
            return 1
    return 0


def command_explain(arguments: argparse.Namespace, settings: Settings) -> int:
    """Print the resolved configuration, so renaming things is verifiable."""
    del arguments
    print(settings.describe())
    return 0


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for every subcommand."""
    parser = argparse.ArgumentParser(
        prog="python -m guanaco",
        description="Build, validate and publish the wheel distribution described by build-matrix.json.",
    )
    parser.add_argument(
        "--matrix",
        type=Path,
        default=None,
        help="Path to build-matrix.json (default: $GUANACO_MATRIX or the repository's own)",
    )
    parser.add_argument(
        "--repository",
        default=None,
        help="Repository we publish to, e.g. owner/name (default: $GUANACO_REPOSITORY)",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    plan = commands.add_parser("plan", help="discover the upstream version and plan the release")
    plan.add_argument("--version", default="", help="Upstream X.Y.Z; empty means newest stable")
    plan.add_argument("--output", type=Path, default=Path("work/plan.json"))
    plan.set_defaults(handler=command_plan)

    test = commands.add_parser("plan-test", help="plan a manual test build (no release)")
    test.add_argument("--version", default="")
    test.add_argument("--cpu", default=True, type=_boolean)
    test.add_argument("--avx2", default=False, type=_boolean)
    test.add_argument("--cuda", default=False, type=_boolean)
    test.add_argument("--cuda-channels", default="")
    test.add_argument("--python-versions", default="3.13")
    test.add_argument("--systems", choices=["linux", "windows", "both"], default="both")
    test.add_argument("--output", type=Path, default=Path("work/test-plan.json"))
    test.set_defaults(handler=command_plan_test)

    prepare = commands.add_parser("prepare-source", help="download, patch and package the source")
    prepare.add_argument("--plan", type=Path, required=True)
    prepare.add_argument("--output", type=Path, default=Path("work/prepared"))
    prepare.set_defaults(handler=command_prepare_source)

    unpack = commands.add_parser("unpack-source", help="verify and unpack a prepared snapshot")
    unpack.add_argument("artifact", type=Path)
    unpack.add_argument("destination", type=Path)
    unpack.add_argument("--version")
    unpack.set_defaults(handler=command_unpack_source)

    configure = commands.add_parser("configure", help="resolve compiler flags and job settings")
    kinds = configure.add_subparsers(dest="kind", required=True)
    for name in ("cpu", "cuda", "matrix"):
        sub = kinds.add_parser(name)
        sub.add_argument("--manifest", type=Path, required=True)
        sub.add_argument("--channel", required=(name != "matrix"))
        sub.add_argument("--platform", choices=["linux", "windows"])
        sub.add_argument("--version", default=os.getenv("VERSION", ""))
    docker = kinds.add_parser("docker")
    docker.add_argument("--version", default=os.getenv("VERSION", ""))
    docker.add_argument("--repository", default=os.getenv("GITHUB_REPOSITORY", ""))
    docker.add_argument(
        "--promote-latest", action="store_true", default=os.getenv("PROMOTE_LATEST") == "true"
    )
    configure.set_defaults(handler=command_configure)

    verify = commands.add_parser("verify-wheels", help="validate wheels and write a receipt")
    verify.add_argument("--manifest", type=Path, required=True)
    verify.add_argument("--directory", type=Path)
    verify.add_argument("--channel", default="cpu")
    verify.add_argument("--platform", choices=["linux", "windows"], required=True)
    verify.add_argument("--selectors", action="store_true")
    verify.add_argument(
        "--single", action="store_true", help="One CUDA/Python build, not a channel"
    )
    verify.add_argument("--python", help="Expected Python version for a single CUDA job")
    verify.add_argument("--receipt", type=Path)
    verify.add_argument(
        "--unrepaired",
        action="store_true",
        help="Local --single testing only; NOT manylinux certification",
    )
    verify.set_defaults(handler=command_verify_wheels)

    gate = commands.add_parser("validate-receipts", help="prove the whole matrix was built")
    gate.add_argument("--plan", type=Path, required=True)
    gate.add_argument("--prepared", type=Path, required=True)
    gate.add_argument("--receipts", type=Path, required=True)
    gate.add_argument("--output", type=Path, required=True)
    gate.set_defaults(handler=command_validate_receipts)

    publish = commands.add_parser("publish", help="preflight, stage and publish releases")
    publish.add_argument("--plan", type=Path, required=True)
    publish.add_argument("--prepared", type=Path)
    publish.add_argument("--artifacts", type=Path)
    publish.add_argument("--output", type=Path, default=Path("work/publish"))
    publish.add_argument("--channel", help="Stage one channel only (bounded CI disk usage)")
    publish.add_argument("--gate", type=Path)
    publish.add_argument("--preflight", action="store_true")
    publish.add_argument("--publish", action="store_true", help="Allow GitHub writes")
    publish.set_defaults(handler=command_publish)

    index = commands.add_parser("build-index", help="render the PEP 503 wheel index")
    index.add_argument("releases", type=Path)
    index.add_argument("output", type=Path)
    index.set_defaults(handler=command_build_index)

    inspect = commands.add_parser("inspect", help="write test-build diagnostics")
    kinds = inspect.add_subparsers(dest="kind", required=True)
    source_report = kinds.add_parser("source")
    source_report.add_argument("--plan", type=Path, required=True)
    source_report.add_argument("--prepared", type=Path, required=True)
    source_report.add_argument("--log", type=Path)
    source_report.add_argument("--output", type=Path, required=True)
    wheels_report = kinds.add_parser("wheels")
    wheels_report.add_argument("--manifest", type=Path, required=True)
    wheels_report.add_argument("--directory", type=Path, required=True)
    wheels_report.add_argument("--output", type=Path, required=True)
    wheels_report.add_argument("--channel", required=True)
    wheels_report.add_argument("--platform", choices=["linux", "windows"], required=True)
    wheels_report.add_argument(
        "--verification", choices=["success", "failure", "skipped", "cancelled"], required=True
    )
    result = kinds.add_parser("result")
    result.add_argument("--plan", type=Path, required=True)
    result.add_argument("--prepared", type=Path, required=True)
    result.add_argument("--receipts", type=Path, required=True)
    result.add_argument("--output", type=Path, required=True)
    inspect.set_defaults(handler=command_inspect)

    explain = commands.add_parser("explain", help="print the resolved configuration")
    explain.set_defaults(handler=command_explain)
    return parser


def _boolean(text: str) -> bool:
    """Parse a ``true``/``false`` command line flag."""
    if str(text).lower() not in ("true", "false"):
        raise argparse.ArgumentTypeError("Expected true or false")
    return str(text).lower() == "true"


def main(argv: list[str] | None = None) -> int:
    """Run the command line interface and return the process exit code."""
    parser = build_parser()
    arguments = parser.parse_args(argv)
    if arguments.command == "configure" and arguments.kind in ("cpu", "cuda"):
        if arguments.channel is None:
            parser.error(f"configure {arguments.kind} requires --channel")
    if arguments.command == "verify-wheels":
        if arguments.unrepaired and (not arguments.single or arguments.receipt):
            parser.error("--unrepaired requires --single and cannot produce a publication receipt")
        if not arguments.selectors and arguments.directory is None:
            parser.error("--directory is required to validate wheels")
    if arguments.command == "publish" and arguments.preflight and arguments.publish:
        parser.error("--preflight cannot be combined with --publish")
    try:
        settings = Settings.load(matrix=arguments.matrix, repository=arguments.repository)
        return arguments.handler(arguments, settings)
    except (ValueError, OSError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


__all__ = ["build_parser", "main"]
