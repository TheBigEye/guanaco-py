"""Verify a prepared artifact and extract it without leaving partial source trees."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from archive_utils import extract_tar
from download_utils import sha256


def unpack(artifact: Path, destination: Path, version: str | None = None) -> None:
    manifest = json.loads((artifact / "build-manifest.json").read_text(encoding="utf-8"))
    if version and manifest["version"] != version:
        raise ValueError("Source artifact version does not match the requested build")
    archive = artifact / "source.tar.gz"
    if sha256(archive) != manifest["source_archive_sha256"]:
        raise ValueError("Prepared source SHA256 mismatch")
    extract_tar(archive, destination)
    print(f"Unpacked guanaco-py {manifest['version']} from verified source archive")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--version")
    args = parser.parse_args()
    unpack(args.artifact, args.destination, args.version)


if __name__ == "__main__":
    main()
