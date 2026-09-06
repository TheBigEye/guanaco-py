"""Check wheel structure, RECORD integrity, native architecture and upstream code."""

from __future__ import annotations

import argparse
import ast
import base64
import csv
import email.parser
import hashlib
import io
import json
import re
import stat
import struct
import zipfile
from dataclasses import dataclass
from pathlib import Path

from archive_utils import MAX_MEMBERS, ArchivePaths
from download_utils import CHUNK_SIZE, sha256
from release_common import validate_python_versions, version_key, write_json

MAX_WHEEL_BYTES = 8 * 1024**3
RECORD_ALGORITHMS = {"sha256", "sha384", "sha512"}


@dataclass(frozen=True)
class WheelIdentity:
    version: str
    python: str
    platform: str

    @property
    def dist_info(self) -> str:
        return f"guanaco_py-{self.version}.dist-info"

    @property
    def tag(self) -> str:
        return f"{self.python}-{self.python}-{self.platform}"


def selector(python_versions: list[str], platform: str) -> str:
    validate_python_versions(python_versions)
    suffix = "manylinux_x86_64" if platform == "linux" else "win_amd64"
    return " ".join(f"cp{version.replace('.', '')}-{suffix}" for version in python_versions)


def wheel_identity(
    wheel: Path, manifest: dict, channel: str, platform: str, allow_unrepaired: bool
) -> WheelIdentity:
    version = manifest["version"]
    version_key(version)
    validate_python_versions(manifest["python_versions"])
    if channel not in manifest["channels"] or platform not in ("linux", "windows"):
        raise ValueError("Wheel channel/platform is not in the build plan")
    linux = "manylinux_2_34_x86_64" if channel in ("cpu", "avx2") else "linux_x86_64"
    if allow_unrepaired:
        linux = "linux_x86_64"
    platform_tag = linux if platform == "linux" else "win_amd64"
    match = re.fullmatch(
        rf"guanaco_py-{re.escape(version)}-(cp[0-9]+)-\1-{platform_tag}\.whl", wheel.name
    )
    allowed = {"cp" + v.replace(".", "") for v in manifest["python_versions"]}
    if not match or match[1] not in allowed:
        raise ValueError(f"Unexpected wheel filename for {channel}/{platform}: {wheel.name}")
    return WheelIdentity(version, match[1], platform_tag)


def archive_files(archive: zipfile.ZipFile) -> set[str]:
    members = archive.infolist()
    if len(members) > MAX_MEMBERS or sum(item.file_size for item in members) > MAX_WHEEL_BYTES:
        raise ValueError("Wheel exceeds validation size limits")
    seen = ArchivePaths()
    files = set()
    for member in members:
        path = seen.add(member.filename, directory=member.is_dir())
        if (
            len(path.parts) >= 3
            and path.parts[0].endswith(".data")
            and path.parts[1] in ("purelib", "platlib")
        ):
            raise ValueError("Wheel installation scheme could override the verified runtime")
        if member.filename.endswith((".pth", ".pyc", ".pyo")):
            raise ValueError("Wheel contains an unexpected Python startup/bytecode file")
        kind = stat.S_IFMT(member.external_attr >> 16)
        if kind not in (0, stat.S_IFREG, stat.S_IFDIR) or member.flag_bits & 1:
            raise ValueError("Unsupported link/device/encryption in wheel")
        if member.filename != member.orig_filename:
            raise ValueError("NUL byte in wheel path")
        if (
            member.filename.endswith(("/METADATA", "/WHEEL", ".py"))
            and member.file_size > 4 * 1024**2
        ):
            raise ValueError("Wheel metadata or Python file is unexpectedly large")
        if not member.is_dir():
            files.add(member.filename)
    return files


def single_header(message, key: str) -> str:
    values = message.get_all(key, [])
    if len(values) != 1:
        raise ValueError(f"Missing or repeated wheel header: {key}")
    return values[0].strip()


def check_metadata(archive: zipfile.ZipFile, files: set[str], identity: WheelIdentity) -> None:
    metadata_path = f"{identity.dist_info}/METADATA"
    if {name for name in files if name.endswith(".dist-info/METADATA")} != {metadata_path}:
        raise ValueError("Wheel filename was renamed without rebuilding distribution metadata")
    metadata = email.parser.BytesParser().parsebytes(archive.read(metadata_path))
    name = re.sub(r"[-_.]+", "-", single_header(metadata, "Name")).lower()
    if name != "guanaco-py" or single_header(metadata, "Version") != identity.version:
        raise ValueError("Wheel distribution name/version mismatch")
    for requirement in metadata.get_all("Requires-Dist", []):
        if re.match(r"llama[-_.]+cpp[-_.]+python(?:\b|\[)", requirement, re.I):
            raise ValueError("Wheel still depends on the upstream distribution itself")
    wheel_path = f"{identity.dist_info}/WHEEL"
    if wheel_path not in files:
        raise ValueError("Wheel is missing its WHEEL metadata")
    wheel = email.parser.BytesParser().parsebytes(archive.read(wheel_path))
    if single_header(wheel, "Wheel-Version") != "1.0":
        raise ValueError("Unsupported Wheel-Version")
    if single_header(wheel, "Root-Is-Purelib").lower() != "false":
        raise ValueError("Native wheel incorrectly declares a pure-Python layout")
    if set(wheel.get_all("Tag", [])) != {identity.tag}:
        raise ValueError("WHEEL compatibility tags do not match the filename")


def check_record(archive: zipfile.ZipFile, files: set[str], identity: WheelIdentity) -> None:
    """Stream every recorded file, including large native libraries, through its hash."""
    record_path = f"{identity.dist_info}/RECORD"
    if record_path not in files:
        raise ValueError("Wheel is missing RECORD")
    rows = {}
    with (
        archive.open(record_path) as raw,
        io.TextIOWrapper(raw, encoding="utf-8", newline="") as text,
    ):
        for row in csv.reader(text):
            if len(row) != 3 or row[0] in rows:
                raise ValueError("Malformed or duplicate RECORD entry")
            rows[row[0]] = row[1:]
    signatures = {f"{identity.dist_info}/RECORD.jws", f"{identity.dist_info}/RECORD.p7s"}
    if set(rows) != files - signatures or rows.get(record_path) != ["", ""]:
        raise ValueError("RECORD does not describe the wheel contents")
    for name in files & signatures:
        # Signatures are deliberately excluded from RECORD by the wheel spec,
        # but still read them fully so malformed ZIP CRCs cannot go unnoticed.
        with archive.open(name) as stream:
            for _ in iter(lambda: stream.read(CHUNK_SIZE), b""):
                pass
    for name, (digest, size) in rows.items():
        if name == record_path:
            continue
        algorithm, separator, expected = digest.partition("=")
        if not separator or algorithm not in RECORD_ALGORITHMS or not size.isdecimal():
            raise ValueError(f"Invalid RECORD hash/size: {name}")
        hashed = hashlib.new(algorithm)
        length = 0
        with archive.open(name) as stream:
            for chunk in iter(lambda: stream.read(CHUNK_SIZE), b""):
                length += len(chunk)
                hashed.update(chunk)
        encoded = base64.urlsafe_b64encode(hashed.digest()).decode("ascii").rstrip("=")
        if encoded != expected or length != int(size):
            raise ValueError(f"RECORD integrity mismatch: {name}")


def check_runtime(archive: zipfile.ZipFile, files: set[str], manifest: dict) -> None:
    expected = manifest["runtime_sha256"]
    actual = {name for name in files if name.endswith((".py", ".pyi", "/py.typed"))}
    expected_python = {name for name in expected if name.endswith((".py", ".pyi", "/py.typed"))}
    if actual != expected_python or not set(expected) <= files:
        raise ValueError("Wheel contains missing or additional Python binding files")
    for name, digest in expected.items():
        if hashlib.sha256(archive.read(name)).hexdigest() != digest:
            raise ValueError(f"Upstream binding changed inside wheel: {name}")
    assignments = []
    for node in ast.parse(archive.read("llama_cpp/__init__.py")).body:
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant):
            if any(
                isinstance(target, ast.Name) and target.id == "__version__"
                for target in node.targets
            ):
                assignments.append(node.value.value)
    if assignments != [manifest["version"]]:
        raise ValueError("llama_cpp.__version__ does not match the upstream release")


def check_native(archive: zipfile.ZipFile, files: set[str], platform: str) -> None:
    pattern = r".+\.dll" if platform == "windows" else r".+\.so(?:\.[A-Za-z0-9_.-]+)?"
    native = [
        name for name in files if name.startswith("llama_cpp/lib/") and re.fullmatch(pattern, name)
    ]
    main_libraries = (
        {"llama_cpp/lib/llama.dll", "llama_cpp/lib/libllama.dll"}
        if platform == "windows"
        else {"llama_cpp/lib/libllama.so"}
    )
    if not native or not (main_libraries & files):
        raise ValueError("Wheel contains no native runtime libraries")
    for name in native:
        with archive.open(name) as stream:
            header = stream.read(64)
            if platform == "linux":
                valid = (
                    len(header) == 64
                    and header[:7] == b"\x7fELF\x02\x01\x01"
                    and struct.unpack_from("<HH", header, 16) == (3, 62)
                )
            else:
                valid = len(header) == 64 and header[:2] == b"MZ"
                if valid:
                    offset = struct.unpack_from("<I", header, 60)[0]
                    valid = 64 <= offset <= 8 * 1024**2
                    if valid:
                        stream.seek(offset)
                        valid = stream.read(6) == b"PE\x00\x00\x64\x86"
            if not valid:
                raise ValueError(f"Invalid x86-64 native library header: {name}")


def verify(
    wheel: Path, manifest: dict, channel: str, platform: str, *, allow_unrepaired=False
) -> None:
    identity = wheel_identity(wheel, manifest, channel, platform, allow_unrepaired)
    try:
        with zipfile.ZipFile(wheel) as archive:
            files = archive_files(archive)
            check_metadata(archive, files, identity)
            check_record(archive, files, identity)
            check_runtime(archive, files, manifest)
            check_native(archive, files, platform)
            if not any(
                "license" in name.lower() and name.startswith(identity.dist_info + "/")
                for name in files
            ):
                raise ValueError("Wheel is missing its license notice")
    except (zipfile.BadZipFile, KeyError, UnicodeDecodeError) as error:
        raise ValueError(f"Invalid wheel archive: {wheel.name}") from error
    print(f"Verified {wheel.name}: metadata, RECORD, native architecture and upstream code")


def verify_directory(directory: Path, manifest: dict, channel: str, platform: str) -> list[Path]:
    wheels = sorted(directory.glob("*.whl"))
    if len(wheels) != len(manifest["python_versions"]):
        raise ValueError(
            f"Expected {len(manifest['python_versions'])} wheels in {directory}, found {len(wheels)}"
        )
    seen = set()
    for wheel in wheels:
        verify(wheel, manifest, channel, platform)
        python = wheel.name.split("-")[2]
        if python in seen:
            raise ValueError("Duplicate Python build in matrix")
        seen.add(python)
    return wheels


def receipt(wheels: list[Path], manifest: dict, channel: str, platform: str) -> dict:
    return {
        "schema": 1,
        "version": manifest["version"],
        "channel": channel,
        "platform": platform,
        "recipe_commit": manifest["recipe_commit"],
        "source_archive_sha256": manifest["source_archive_sha256"],
        "wheels": [
            {"name": wheel.name, "size": wheel.stat().st_size, "sha256": sha256(wheel)}
            for wheel in wheels
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--directory", type=Path)
    parser.add_argument("--channel", default="cpu")
    parser.add_argument("--platform", choices=["linux", "windows"], required=True)
    parser.add_argument("--selectors", action="store_true")
    parser.add_argument(
        "--single", action="store_true", help="One CUDA/Python build, not a complete channel"
    )
    parser.add_argument("--python", help="Expected Python version for a single CUDA job")
    parser.add_argument(
        "--receipt", type=Path, help="Write a small validation receipt for the global publish gate"
    )
    parser.add_argument(
        "--unrepaired",
        action="store_true",
        help="Local --single testing only; NOT manylinux certification",
    )
    args = parser.parse_args()
    if args.unrepaired and (not args.single or args.receipt):
        parser.error("--unrepaired requires --single and cannot produce a publication receipt")
    if not args.selectors and args.directory is None:
        parser.error("--directory is required to validate wheels")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if args.selectors:
        print("build=" + selector(manifest["python_versions"], args.platform))
        return
    if args.single:
        wheels = list(args.directory.glob("*.whl"))
        if len(wheels) != 1:
            raise ValueError("Expected exactly one wheel in this CUDA job")
        if args.python and wheels[0].name.split("-")[2] != "cp" + args.python.replace(".", ""):
            raise ValueError("CUDA job Python version does not match its wheel")
        verify(wheels[0], manifest, args.channel, args.platform, allow_unrepaired=args.unrepaired)
    else:
        wheels = verify_directory(args.directory, manifest, args.channel, args.platform)
    if args.receipt:
        write_json(args.receipt, receipt(wheels, manifest, args.channel, args.platform))


if __name__ == "__main__":
    main()
