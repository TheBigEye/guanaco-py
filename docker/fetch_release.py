"""Install a pinned release wheel, or fetch its checksummed reconstructed source."""

from __future__ import annotations

import argparse
import platform
import re
import subprocess
import sys
from pathlib import Path

# Docker copies the shared helper beside this file; local use finds the repo.
if (Path(__file__).resolve().parents[1] / ".github/scripts").is_dir():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / ".github/scripts"))

from download_utils import download

VERSION = re.compile(r"(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)")


def release_base(repository: str, version: str, channel: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repository) or any(
        part in (".", "..") for part in repository.split("/")
    ):
        raise ValueError("Expected a GitHub owner/repository")
    if not VERSION.fullmatch(version):
        raise ValueError("Expected an explicit stable X.Y.Z version")
    if not re.fullmatch(r"cpu|avx2|cu[0-9]+", channel):
        raise ValueError("Invalid build channel")
    tag = f"v{version}" + ("" if channel == "cpu" else f"-{channel}")
    return f"https://github.com/{repository}/releases/download/{tag}"


def read_checksums(path: Path) -> dict[str, str]:
    hashes = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        match = re.fullmatch(r"([a-f0-9]{64})  ([A-Za-z0-9_.-]+)", line)
        if not match or match[2] in (".", "..") or match[2] in hashes:
            raise ValueError("Malformed or duplicate release checksum entry")
        hashes[match[2]] = match[1]
    if not hashes:
        raise ValueError("Empty release checksum inventory")
    return hashes


def checked_asset(base: str, name: str, hashes: dict, destination: Path) -> None:
    expected = hashes.get(name)
    if not isinstance(expected, str) or not re.fullmatch(r"[a-f0-9]{64}", expected):
        raise ValueError(f"No release checksum for {name}")
    download(f"{base}/{name}", destination, expected_sha256=expected)


def wheel_name(version: str, channel: str) -> str:
    if platform.machine().lower() not in ("x86_64", "amd64") or sys.platform != "linux":
        raise ValueError("These Docker images require linux/amd64")
    cp = f"cp{sys.version_info.major}{sys.version_info.minor}"
    policy = "manylinux_2_34_x86_64" if channel in ("cpu", "avx2") else "linux_x86_64"
    return f"guanaco_py-{version}-{cp}-{cp}-{policy}.whl"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["wheel", "source"])
    parser.add_argument("--repository", default="TheBigEye/guanaco-py")
    parser.add_argument("--version", required=True)
    parser.add_argument("--channel", default="cpu")
    parser.add_argument("--directory", type=Path, default=Path("/tmp/guanaco-release"))
    args = parser.parse_args()
    # Reconstructed source belongs to the CPU release, regardless of build backend.
    channel = "cpu" if args.mode == "source" else args.channel
    base = release_base(args.repository, args.version, channel)
    name = wheel_name(args.version, channel) if args.mode == "wheel" else None
    checksums = args.directory / "SHA256SUMS"
    download(base + "/SHA256SUMS", checksums, max_bytes=1024**2)
    hashes = read_checksums(checksums)
    if args.mode == "source":
        checked_asset(
            base, f"guanaco-source-{args.version}.tar.gz", hashes, args.directory / "source.tar.gz"
        )
        checked_asset(base, "guanaco-build.json", hashes, args.directory / "build-manifest.json")
        return
    wheel = args.directory / name
    checked_asset(base, name, hashes, wheel)
    # Install the local Guanaco wheel; only dependencies come from PyPI.
    # No Pages propagation race, fallback source build or upstream-name alias.
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "--no-cache-dir", f"{wheel}[server]"], check=True
    )


if __name__ == "__main__":
    main()
