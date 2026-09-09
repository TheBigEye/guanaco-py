"""Install a pinned release wheel, or fetch its checksummed reconstructed source.

This helper runs inside the Docker images, so it only needs the standard
library plus the :mod:`guanaco` package itself, which the Dockerfile copies
beside this file. It reuses the package's own naming rules, so renaming the
distribution or the repository cannot silently break an image build.
"""

from __future__ import annotations

import argparse
import platform
import re
import subprocess
import sys
from pathlib import Path

# Docker copies the package beside this file; local use finds the repository.
ROOT = Path(__file__).resolve().parents[1]
if (ROOT / "guanaco").is_dir():
    sys.path.insert(0, str(ROOT))

from guanaco.channels import Channel, Platform  # noqa: E402  (path set up above)
from guanaco.models import Version, wheel_prefix  # noqa: E402  (path set up above)
from guanaco.transfer import Downloader  # noqa: E402  (path set up above)

CHECKSUM_LINE = re.compile(r"([a-f0-9]{64})  ([A-Za-z0-9_.-]+)")
CHECKSUM_LIMIT = 1024**2
DEFAULT_REPOSITORY = "TheBigEye/guanaco-py"


def release_base(repository: str, version: str, channel: Channel) -> str:
    """Return the download base URL of one channel's release."""
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repository) or any(
        part in (".", "..") for part in repository.split("/")
    ):
        raise ValueError("Expected a GitHub owner/repository")
    Version.parse(version)
    return f"https://github.com/{repository}/releases/download/{channel.release_tag(version)}"


def read_checksums(path: Path) -> dict[str, str]:
    """Parse a ``SHA256SUMS`` file into ``{filename: digest}``."""
    hashes: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        match = CHECKSUM_LINE.fullmatch(line)
        if not match or match[2] in hashes:
            raise ValueError("Malformed or duplicate release checksum entry")
        hashes[match[2]] = match[1]
    if not hashes:
        raise ValueError("Empty release checksum inventory")
    return hashes


def checked_asset(base: str, name: str, hashes: dict, destination: Path) -> None:
    """Download one release asset and verify it against `hashes`."""
    expected = hashes.get(name)
    if not expected:
        raise ValueError(f"No release checksum for {name}")
    Downloader().fetch(f"{base}/{name}", destination, expected_sha256=expected)


def wheel_name(package: str, version: str, channel: Channel) -> str:
    """Return the wheel filename for this interpreter, as the verifier expects."""
    if sys.platform != "linux" or platform.machine().lower() not in ("x86_64", "amd64"):
        raise ValueError("These Docker images require linux/amd64")
    python = f"cp{sys.version_info.major}{sys.version_info.minor}"
    policy = channel.wheel_platform(Platform.LINUX)
    return f"{wheel_prefix(package)}-{version}-{python}-{python}-{policy}.whl"


def install_wheel(wheel: Path) -> None:
    """Install the downloaded wheel; only its dependencies come from PyPI."""
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "--no-cache-dir", f"{wheel}[server]"],
        check=True,
    )


def main() -> None:
    """Fetch and install a release wheel, or fetch its reconstructed source."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["wheel", "source"])
    parser.add_argument("--repository", default=DEFAULT_REPOSITORY)
    parser.add_argument("--package", help="Distribution name (default: from --repository)")
    parser.add_argument("--version", required=True)
    parser.add_argument("--channel", default=Channel("cpu"), type=Channel)
    parser.add_argument("--directory", type=Path, default=Path("/tmp/guanaco-release"))
    arguments = parser.parse_args()

    package = arguments.package or arguments.repository.rsplit("/", 1)[-1].casefold()
    # The reconstructed source always belongs to the CPU release.
    channel = Channel("cpu") if arguments.mode == "source" else arguments.channel
    base = release_base(arguments.repository, arguments.version, channel)
    arguments.directory.mkdir(parents=True, exist_ok=True)
    checksums = arguments.directory / "SHA256SUMS"
    Downloader(max_bytes=CHECKSUM_LIMIT).fetch(base + "/SHA256SUMS", checksums)
    hashes = read_checksums(checksums)

    if arguments.mode == "source":
        checked_asset(
            base,
            f"guanaco-source-{arguments.version}.tar.gz",
            hashes,
            arguments.directory / "source.tar.gz",
        )
        checked_asset(
            base, "guanaco-build.json", hashes, arguments.directory / "build-manifest.json"
        )
        return

    wheel = arguments.directory / wheel_name(package, arguments.version, channel)
    checked_asset(base, wheel.name, hashes, wheel)
    install_wheel(wheel)


if __name__ == "__main__":
    main()
