"""Validate the entire build matrix using small receipts instead of downloading all wheels."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from release_common import expected_assets, write_json


def artifact_specs(plan: dict) -> dict[str, dict]:
    result = {}
    for channel in plan["missing_channels"]:
        for platform in ("linux", "windows"):
            if channel in ("cpu", "avx2"):
                name = f"guanaco-py-{channel}-{platform}-x64"
                result[name] = {
                    "channel": channel,
                    "platform": platform,
                    "python_versions": plan["python_versions"],
                }
            else:
                for python in plan["python_versions"]:
                    name = f"guanaco-py-cuda-{platform}-x64-{channel}-py{python}"
                    result[name] = {
                        "channel": channel,
                        "platform": platform,
                        "python_versions": [python],
                    }
    return result


def matching_manifest(plan: dict, manifest: dict) -> None:
    for key, value in plan.items():
        if key not in manifest or manifest[key] != value:
            raise ValueError(f"Prepared source does not match the release plan: {key}")


def validate_receipts(plan: dict, manifest: dict, directory: Path) -> dict:
    matching_manifest(plan, manifest)
    inventory = {channel: {} for channel in plan["missing_channels"]}
    specs = artifact_specs(plan)
    for artifact, spec in specs.items():
        path = directory / f"{artifact}.json"
        if not path.is_file():
            raise ValueError(f"Missing validation receipt: {artifact}")
        data = json.loads(path.read_text(encoding="utf-8"))
        expected_identity = {
            "schema": 1,
            "version": plan["version"],
            "recipe_commit": plan["recipe_commit"],
            "source_archive_sha256": manifest["source_archive_sha256"],
            "channel": spec["channel"],
            "platform": spec["platform"],
        }
        if (
            not isinstance(data, dict)
            or type(data.get("schema")) is not int
            or any(data.get(key) != value for key, value in expected_identity.items())
        ):
            raise ValueError(f"Receipt identity mismatch: {artifact}")
        allowed = expected_assets(plan["version"], spec["channel"], spec["python_versions"])
        allowed = {
            name
            for name in allowed
            if name.endswith(".whl") and ("win_amd64" in name) == (spec["platform"] == "windows")
        }
        records = data.get("wheels", [])
        if len(records) != len(allowed) or {record.get("name") for record in records} != allowed:
            raise ValueError(f"Incomplete wheel matrix in receipt: {artifact}")
        for record in records:
            if (
                type(record.get("size")) is not int
                or record["size"] <= 0
                or not re.fullmatch(r"[a-f0-9]{64}", record.get("sha256", ""))
            ):
                raise ValueError(f"Invalid wheel checksum/size in receipt: {artifact}")
            inventory[spec["channel"]][record["name"]] = {
                "size": record["size"],
                "sha256": record["sha256"],
            }
    return {
        "schema": 1,
        "plan": plan,
        "source_archive_sha256": manifest["source_archive_sha256"],
        "channels": inventory,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--receipts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    manifest = json.loads((args.prepared / "build-manifest.json").read_text(encoding="utf-8"))
    result = validate_receipts(plan, manifest, args.receipts)
    write_json(args.output, result)
    print(
        f"Validated {sum(len(records) for records in result['channels'].values())} wheel receipts across {len(result['channels'])} channels"
    )


if __name__ == "__main__":
    main()
