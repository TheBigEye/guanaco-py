"""Wheel validation: identity, integrity, contents and receipts.

A wheel is only allowed to reach a release if it passes every check here. The
checks are deliberately independent of the build that produced the wheel: they
re-derive what the file *must* look like from the source manifest and compare,
so a misconfigured builder cannot quietly ship a wrong binary.

Checked for every wheel:

* the filename, ``METADATA`` and ``WHEEL`` agree on distribution, version and tags;
* every file listed in ``RECORD`` hashes to what the archive actually contains;
* the Python bindings are byte-identical to the prepared source snapshot;
* the native libraries are really x86-64 ELF or PE files;
* an upstream license notice is present.
"""

from __future__ import annotations

import ast
import base64
import csv
import email.parser
import hashlib
import io
import re
import stat
import struct
import zipfile
from dataclasses import dataclass
from pathlib import Path

from .models import (
    Channel,
    Platform,
    Receipt,
    SourceManifest,
    WheelRecord,
    wheel_prefix,
)
from .settings import Settings
from .transfer import MAX_MEMBERS, ArchivePaths, TransferError, sha256

MAX_WHEEL_BYTES = 8 * 1024**3
MAX_TEXT_BYTES = 4 * 1024**2
RECORD_ALGORITHMS = {"sha256", "sha384", "sha512"}
CHUNK_SIZE = 1024 * 1024

# Anything that would let the wheel install code outside the verified tree.
FORBIDDEN_SUFFIXES = (".pth", ".pyc", ".pyo")


class WheelError(ValueError):
    """Raised when a wheel is missing, malformed or not what we expected."""


@dataclass(frozen=True)
class WheelIdentity:
    """What a wheel's filename promises about its contents."""

    package: str
    version: str
    python: str
    platform: str

    @property
    def prefix(self) -> str:
        """Normalised wheel filename prefix, e.g. ``guanaco_py``."""
        return wheel_prefix(self.package)

    @property
    def dist_info(self) -> str:
        """Name of the ``.dist-info`` directory inside the wheel."""
        return f"{self.prefix}-{self.version}.dist-info"

    @property
    def tag(self) -> str:
        """The single compatibility tag the ``WHEEL`` file must declare."""
        return f"{self.python}-{self.python}-{self.platform}"

    @property
    def filename(self) -> str:
        """The exact filename this wheel must have."""
        return f"{self.prefix}-{self.version}-{self.python}-{self.python}-{self.platform}.whl"


class WheelValidator:
    """Validates wheels against a source manifest and emits receipts."""

    def __init__(self, settings: Settings) -> None:
        """Remember the distribution name the wheels must declare."""
        self.settings = settings

    # -- Public API ---------------------------------------------------------

    def identity(
        self,
        wheel: Path,
        manifest: SourceManifest,
        channel: Channel,
        platform: Platform,
        *,
        allow_unrepaired: bool = False,
    ) -> WheelIdentity:
        """Derive the identity a wheel claims, validating its filename.

        Args:
            wheel: The wheel file.
            manifest: The prepared source manifest.
            channel: Channel the wheel belongs to.
            platform: Platform it was built on.
            allow_unrepaired: Accept a ``linux_x86_64`` tag for a CPU channel.
                Only valid for local experiments, never for a release.
        """
        wheel = Path(wheel)
        expected_platform = channel.wheel_platform(platform, repaired=not allow_unrepaired)
        pattern = (
            rf"{re.escape(wheel_prefix(manifest.package))}-"
            rf"{re.escape(manifest.version)}-(cp[0-9]+)-\1-{re.escape(expected_platform)}\.whl"
        )
        match = re.fullmatch(pattern, wheel.name)
        allowed = {"cp" + v.replace(".", "") for v in manifest.python_versions}
        if not match or match[1] not in allowed:
            raise WheelError(f"Unexpected wheel filename for {channel}/{platform}: {wheel.name}")
        return WheelIdentity(
            package=manifest.package,
            version=manifest.version,
            python=match[1],
            platform=expected_platform,
        )


    def verify(
        self,
        wheel: Path,
        manifest: SourceManifest,
        channel: Channel,
        platform: Platform,
        *,
        allow_unrepaired: bool = False,
    ) -> WheelIdentity:
        """Validate one wheel completely.

        Raises:
            WheelError: On any mismatch. The message names the offending file or
                header, so a failed CI job points straight at the cause.
        """
        wheel = Path(wheel)
        identity = self.identity(
            wheel, manifest, channel, platform, allow_unrepaired=allow_unrepaired
        )
        try:
            with zipfile.ZipFile(wheel) as archive:
                files = self._members(archive)
                self._check_metadata(archive, files, identity)
                self._check_record(archive, files, identity)
                self._check_runtime(archive, files, manifest)
                self._check_native(archive, files, platform)
                self._check_license(files, identity)
        except (zipfile.BadZipFile, KeyError, UnicodeDecodeError, TransferError) as error:
            raise WheelError(f"Invalid wheel archive: {wheel.name}") from error
        print(f"Verified {wheel.name}: metadata, RECORD, native architecture and upstream code")
        return identity


    def verify_directory(
        self,
        directory: Path,
        manifest: SourceManifest,
        channel: Channel,
        platform: Platform,
        *,
        allow_unrepaired: bool = False,
    ) -> list[Path]:
        """Validate a complete per-platform matrix of wheels.

        Args:
            directory: Directory holding the wheels of one build job.
            manifest: The prepared source manifest.
            channel: Channel being validated.
            platform: Platform being validated.
            allow_unrepaired: Accept unrepaired CPU wheels (local testing only).

        Raises:
            WheelError: If the number of wheels or the Python versions do not
                cover the planned matrix exactly.
        """
        wheels = sorted(Path(directory).glob("*.whl"))
        if len(wheels) != len(manifest.python_versions):
            raise WheelError(
                f"Expected {len(manifest.python_versions)} wheels in {directory}, found {len(wheels)}"
            )
        seen: set[str] = set()
        for wheel in wheels:
            identity = self.verify(
                wheel, manifest, channel, platform, allow_unrepaired=allow_unrepaired
            )
            if identity.python in seen:
                raise WheelError("Duplicate Python build in matrix")
            seen.add(identity.python)
        return wheels


    def receipt(
        self,
        wheels: list[Path],
        manifest: SourceManifest,
        channel: Channel,
        platform: Platform,
    ) -> Receipt:
        """Summarise validated wheels in a small, portable receipt."""
        return Receipt(
            version=manifest.version,
            channel=channel,
            platform=platform,
            recipe_commit=manifest.plan.recipe_commit,
            source_archive_sha256=manifest.source_archive_sha256,
            wheels=tuple(
                WheelRecord(name=wheel.name, size=wheel.stat().st_size, sha256=sha256(wheel))
                for wheel in wheels
            ),
        )

    # -- Individual checks ---------------------------------------------------

    def _members(self, archive: zipfile.ZipFile) -> set[str]:
        """List every file in the wheel, rejecting anything unsafe."""
        members = archive.infolist()
        if len(members) > MAX_MEMBERS or sum(item.file_size for item in members) > MAX_WHEEL_BYTES:
            raise WheelError("Wheel exceeds validation size limits")
        inventory = ArchivePaths()
        files: set[str] = set()
        for member in members:
            path = inventory.add(member.filename, directory=member.is_dir())
            if (
                len(path.parts) >= 3
                and path.parts[0].endswith(".data")
                and path.parts[1] in ("purelib", "platlib")
            ):
                raise WheelError("Wheel installation scheme could override the verified runtime")
            if member.filename.endswith(FORBIDDEN_SUFFIXES):
                raise WheelError("Wheel contains an unexpected Python startup/bytecode file")
            kind = stat.S_IFMT(member.external_attr >> 16)
            if kind not in (0, stat.S_IFREG, stat.S_IFDIR) or member.flag_bits & 1:
                raise WheelError("Unsupported link/device/encryption in wheel")
            if member.filename != member.orig_filename:
                raise WheelError("NUL byte in wheel path")
            if (
                member.filename.endswith(("/METADATA", "/WHEEL", ".py"))
                and member.file_size > MAX_TEXT_BYTES
            ):
                raise WheelError("Wheel metadata or Python file is unexpectedly large")
            if not member.is_dir():
                files.add(member.filename)
        return files


    def _check_metadata(self, archive: zipfile.ZipFile, files: set[str], identity: WheelIdentity) -> None:
        """ Check ``METADATA`` and ``WHEEL`` against the filename's promise. """

        metadata_path = f"{identity.dist_info}/METADATA"
        if {name for name in files if name.endswith(".dist-info/METADATA")} != {metadata_path}:
            raise WheelError("Wheel filename was renamed without rebuilding distribution metadata")

        metadata = email.parser.BytesParser().parsebytes(archive.read(metadata_path))
        name = re.sub(r"[-_.]+", "-", self._single(metadata, "Name")).lower()
        if (
            name != self.settings.package.lower()
            or self._single(metadata, "Version") != identity.version
        ):
            raise WheelError("Wheel distribution name/version mismatch")
        for requirement in metadata.get_all("Requires-Dist", []):
            if re.match(r"llama[-_.]+cpp[-_.]+python(?:\b|\[)", requirement, re.IGNORECASE):
                raise WheelError("Wheel still depends on the upstream distribution itself")

        wheel_path = f"{identity.dist_info}/WHEEL"
        if wheel_path not in files:
            raise WheelError("Wheel is missing its WHEEL metadata")

        wheel = email.parser.BytesParser().parsebytes(archive.read(wheel_path))
        if self._single(wheel, "Wheel-Version") != "1.0":
            raise WheelError("Unsupported Wheel-Version")

        if self._single(wheel, "Root-Is-Purelib").lower() != "false":
            raise WheelError("Native wheel incorrectly declares a pure-Python layout")

        if set(wheel.get_all("Tag", [])) != {identity.tag}:
            raise WheelError("WHEEL compatibility tags do not match the filename")


    def _check_record(self, archive: zipfile.ZipFile, files: set[str], identity: WheelIdentity) -> None:
        "" "Re-hash every file the ``RECORD`` claims, streaming large libraries. """

        record_path = f"{identity.dist_info}/RECORD"
        if record_path not in files:
            raise WheelError("Wheel is missing RECORD")
        rows: dict[str, list[str]] = {}
        with (
            archive.open(record_path) as raw,
            io.TextIOWrapper(raw, encoding="utf-8", newline="") as text,
        ):
            for row in csv.reader(text):
                if len(row) != 3 or row[0] in rows:
                    raise WheelError("Malformed or duplicate RECORD entry")
                rows[row[0]] = row[1:]
        signatures = {f"{identity.dist_info}/RECORD.jws", f"{identity.dist_info}/RECORD.p7s"}
        if set(rows) != files - signatures or rows.get(record_path) != ["", ""]:
            raise WheelError("RECORD does not describe the wheel contents")
        for name in files & signatures:
            # Signatures are excluded from RECORD by the wheel spec, but read them
            # fully so a malformed ZIP CRC cannot go unnoticed.
            with archive.open(name) as stream:
                for _ in iter(lambda: stream.read(CHUNK_SIZE), b""):
                    pass
        for name, (digest, size) in rows.items():
            if name == record_path:
                continue
            algorithm, separator, expected = digest.partition("=")
            if not separator or algorithm not in RECORD_ALGORITHMS or not size.isdecimal():
                raise WheelError(f"Invalid RECORD hash/size: {name}")
            hashed = hashlib.new(algorithm)
            length = 0
            with archive.open(name) as stream:
                for chunk in iter(lambda: stream.read(CHUNK_SIZE), b""):
                    length += len(chunk)
                    hashed.update(chunk)
            encoded = base64.urlsafe_b64encode(hashed.digest()).decode("ascii").rstrip("=")
            if encoded != expected or length != int(size):
                raise WheelError(f"RECORD integrity mismatch: {name}")


    def _check_runtime(self, archive: zipfile.ZipFile, files: set[str], manifest: SourceManifest) -> None:
        """ Check that the bindings are exactly the prepared source, byte for byte. """

        expected = manifest.runtime_sha256
        actual = {name for name in files if name.endswith((".py", ".pyi", "/py.typed"))}
        expected_python = {name for name in expected if name.endswith((".py", ".pyi", "/py.typed"))}
        if actual != expected_python or not set(expected) <= files:
            raise WheelError("Wheel contains missing or additional Python binding files")
        for name, digest in expected.items():
            if hashlib.sha256(archive.read(name)).hexdigest() != digest:
                raise WheelError(f"Upstream binding changed inside wheel: {name}")
        assignments = []
        for node in ast.parse(archive.read("llama_cpp/__init__.py")).body:
            if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant):
                if any(
                    isinstance(target, ast.Name) and target.id == "__version__"
                    for target in node.targets
                ):
                    assignments.append(node.value.value)
        if assignments != [manifest.version]:
            raise WheelError("llama_cpp.__version__ does not match the upstream release")


    def _check_native(self, archive: zipfile.ZipFile, files: set[str], platform: Platform) -> None:
        """ Check that the bundled native libraries are really x86-64 binaries. """
        pattern = r".+\.dll" if platform.is_windows else r".+\.so(?:[.][A-Za-z0-9_.-]+)?"
        native = [
            name
            for name in files
            if name.startswith("llama_cpp/lib/") and re.fullmatch(pattern, name)
        ]
        main_libraries = (
            {"llama_cpp/lib/llama.dll", "llama_cpp/lib/libllama.dll"}
            if platform.is_windows
            else {"llama_cpp/lib/libllama.so"}
        )
        if not native or not (main_libraries & files):
            raise WheelError("Wheel contains no native runtime libraries")
        for name in native:
            with archive.open(name) as stream:
                valid = (
                    self._is_elf(stream)
                    if platform is not Platform.WINDOWS
                    else self._is_pe(stream)
                )
            if not valid:
                raise WheelError(f"Invalid x86-64 native library header: {name}")

    @staticmethod
    def _is_elf(stream) -> bool:
        """Whether the stream starts with a little-endian 64-bit x86-64 ELF header."""
        header = stream.read(64)
        return (
            len(header) == 64
            and header[:7] == b"\x7fELF\x02\x01\x01"
            and struct.unpack_from("<HH", header, 16) == (3, 62)
        )

    @staticmethod
    def _is_pe(stream) -> bool:
        """Whether the stream starts with a 64-bit Windows PE header."""
        header = stream.read(64)
        if len(header) != 64 or header[:2] != b"MZ":
            return False
        offset = struct.unpack_from("<I", header, 60)[0]
        if not 64 <= offset <= 8 * 1024**2:
            return False
        stream.seek(offset)
        return stream.read(6) == b"PE\x00\x00\x64\x86"

    @staticmethod
    def _check_license(files: set[str], identity: WheelIdentity) -> None:
        """Require an upstream license notice inside the wheel."""
        if not any(
            "license" in name.lower() and name.startswith(identity.dist_info + "/")
            for name in files
        ):
            raise WheelError("Wheel is missing its license notice")

    @staticmethod
    def _single(message, key: str) -> str:
        """Return a header that must appear exactly once."""
        values = message.get_all(key, [])
        if len(values) != 1:
            raise WheelError(f"Missing or repeated wheel header: {key}")
        return values[0].strip()


__all__ = ["WheelError", "WheelIdentity", "WheelValidator"]
