"""Plan a manual, artifact-only build, even for an already published upstream version."""

from __future__ import annotations

import argparse
import copy
import os
from pathlib import Path

from check_upstream import fresh_origin, select_release
from release_common import (
    CONFIG,
    UPSTREAM,
    GitHub,
    outputs,
    repository_name,
    validate_build_matrix,
    version_from_tag,
    version_key,
    write_json,
)
from validate_receipts import artifact_specs


def selection(text: str, allowed: list[str], label: str) -> list[str]:
    values = [value.strip().lower() for value in text.split(",")]
    if values == ["all"]:
        return allowed.copy()
    if not values or any(not value or value not in allowed for value in values):
        raise ValueError(
            f"Invalid {label}: use a comma-separated selection of {', '.join(allowed)}, or all"
        )
    if len(values) != len(set(values)):
        raise ValueError(f"Duplicate {label} selection")
    return [value for value in allowed if value in values]


def boolean(text: str) -> bool:
    if text.lower() not in ("true", "false"):
        raise argparse.ArgumentTypeError("Expected true or false")
    return text.lower() == "true"


def make_test_plan(
    api: GitHub,
    repository: str,
    version: str = "",
    *,
    cpu=True,
    avx2=False,
    cuda=False,
    cuda_channels="cu124",
    python_versions="3.13",
    systems="both",
) -> dict:
    repository_name(repository)
    version = version.strip()
    if version:
        version_key(version)
    if any(type(value) is not bool for value in (cpu, avx2, cuda)):
        raise ValueError("Channel switches must be booleans")
    if not (cpu or avx2 or cuda):
        raise ValueError("Select at least one of CPU, AVX2 or CUDA")
    if systems not in ("linux", "windows", "both"):
        raise ValueError("Systems must be linux, windows or both")
    versions = selection(python_versions, CONFIG["python_versions"], "Python versions")
    channels = (["cpu"] if cpu else []) + (["avx2"] if avx2 else [])
    if cuda:
        channels += selection(cuda_channels, list(CONFIG["cuda"]), "CUDA channels")
    # Do not read Guanaco releases/drafts: this is a fresh test of the current
    # recipe/patches, not the release checker's missing-channel/resume policy.
    release = select_release(api.releases(UPSTREAM), version.strip())
    origin = fresh_origin(api, release)
    plan = {
        "schema": 1,
        "test_only": True,
        "repository": repository,
        "version": version_from_tag(origin["tag"]),
        "upstream": origin,
        "channels": ["cpu", "avx2", *CONFIG["cuda"]],
        "cuda": copy.deepcopy(CONFIG["cuda"]),
        "python_versions": versions,
        "platforms": ["linux", "windows"] if systems == "both" else [systems],
        "missing_channels": channels,
        "build": True,
        "promote_latest": False,
        "recipe_commit": os.getenv("GITHUB_SHA", "local-working-tree"),
        "run_url": f"https://github.com/{repository}/actions/runs/{os.environ['GITHUB_RUN_ID']}"
        if os.getenv("GITHUB_RUN_ID")
        else None,
    }
    validate_build_matrix(plan)
    return plan


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repository", default=os.getenv("GITHUB_REPOSITORY", "TheBigEye/guanaco-py")
    )
    parser.add_argument("--version", default="")
    parser.add_argument("--cpu", type=boolean, default=True)
    parser.add_argument("--avx2", type=boolean, default=False)
    parser.add_argument("--cuda", type=boolean, default=False)
    parser.add_argument("--cuda-channels", default="cu124")
    parser.add_argument("--python-versions", default="3.13")
    parser.add_argument("--systems", choices=["linux", "windows", "both"], default="both")
    parser.add_argument("--output", type=Path, default=Path("work/test-plan.json"))
    args = parser.parse_args()
    plan = make_test_plan(
        GitHub(),
        args.repository,
        args.version,
        cpu=args.cpu,
        avx2=args.avx2,
        cuda=args.cuda,
        cuda_channels=args.cuda_channels,
        python_versions=args.python_versions,
        systems=args.systems,
    )
    write_json(args.output, plan)
    specs = artifact_specs(plan)
    total = sum(len(spec["python_versions"]) for spec in specs.values())
    outputs(
        os.getenv("GITHUB_OUTPUT"),
        version=plan["version"],
        cpu="cpu" in plan["missing_channels"],
        avx2="avx2" in plan["missing_channels"],
        cuda=[c for c in plan["missing_channels"] if c.startswith("cu")],
    )
    summary = (
        f"## Manual test build — NOT a release\n\n"
        f"- Upstream: `{plan['version']}` / `{plan['upstream']['tag']}`\n"
        f"- Source SHA: `{plan['upstream']['commit']}`\n"
        f"- Channels: {', '.join(plan['missing_channels'])}\n"
        f"- Python: {', '.join(plan['python_versions'])}\n"
        f"- Systems: {', '.join(plan['platforms'])} (x86-64)\n"
        f"- Expected: **{total} wheels**, {len(specs)} build receipts\n\n"
        "Only Actions artifacts will be uploaded. No releases, tags, Pages or Docker publication.\n"
    )
    print(summary)
    if os.getenv("GITHUB_STEP_SUMMARY"):
        with Path(os.environ["GITHUB_STEP_SUMMARY"]).open(
            "a", encoding="utf-8", newline="\n"
        ) as stream:
            stream.write(summary)


if __name__ == "__main__":
    main()
