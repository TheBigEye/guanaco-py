"""Portable archive paths and transactional extraction (Python 3.9+)."""

from __future__ import annotations

import contextlib
import shutil
import stat
import tarfile
import tempfile
import zipfile
from pathlib import Path, PurePosixPath

MAX_MEMBERS = 50_000
MAX_UNPACKED_BYTES = 2 * 1024**3
WINDOWS_DEVICES = {"con", "prn", "aux", "nul"} | {
    f"{prefix}{number}" for prefix in ("com", "lpt") for number in range(1, 10)
}


def portable_path(name: str) -> PurePosixPath:
    """Reject traversal, Windows aliases/ADS/devices and Git control directories."""
    if not isinstance(name, str) or not name or name.startswith("/"):
        raise ValueError(f"Unsafe archive path: {name!r}")
    parts = name.removesuffix("/").split("/")
    for part in parts:
        if (
            not part
            or part in (".", "..")
            or part.casefold() == ".git"
            or part.endswith((" ", "."))
            or part.split(".", 1)[0].casefold() in WINDOWS_DEVICES
            or any(ord(char) < 32 or char in '\\<>:"|?*' for char in part)
        ):
            raise ValueError(f"Unsafe archive path: {name!r}")
    return PurePosixPath(*parts)


class ArchivePaths:
    """Track implicit directories as well as explicit members on case-folding systems."""

    def __init__(self):
        self.members: set[str] = set()
        self.files: set[str] = set()
        self.spellings: dict[str, str] = {}

    def add(self, name: str, *, directory: bool = False) -> PurePosixPath:
        path = portable_path(name)
        key = str(path).casefold()
        if key in self.members:
            raise ValueError(f"Duplicate or case-colliding archive member: {name}")
        if not directory and key in self.spellings:
            raise ValueError(f"File/directory collision in archive: {name}")
        for component in (path, *path.parents):
            text = str(component)
            if text == ".":
                continue
            folded = text.casefold()
            if folded in self.files:
                raise ValueError(f"File/directory collision in archive: {name}")
            if folded in self.spellings and self.spellings[folded] != text:
                raise ValueError(f"Case-colliding archive directory: {name}")
            self.spellings[folded] = text
        self.members.add(key)
        if not directory:
            self.files.add(key)
        return path


@contextlib.contextmanager
def empty_destination(destination: Path):
    """A failed extraction leaves no partial destination and can be retried."""
    if destination.is_symlink() or (
        destination.exists() and (not destination.is_dir() or any(destination.iterdir()))
    ):
        raise ValueError("Source destination must be empty and must not be a symlink")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".extract-", dir=destination.parent) as name:
        temporary = Path(name)
        yield temporary
        if destination.exists():
            destination.rmdir()
        temporary.replace(destination)


def extract_zip(archive: Path, destination: Path) -> None:
    """Strip exactly one GitHub ZIP root; validate all paths before writing."""
    with zipfile.ZipFile(archive) as source:
        members = source.infolist()
        if not members or len(members) > MAX_MEMBERS:
            raise ValueError("Empty ZIP or excessive member count")
        if sum(member.file_size for member in members) > MAX_UNPACKED_BYTES:
            raise ValueError("Source ZIP exceeds extraction limits")
        roots = {portable_path(member.filename).parts[0] for member in members}
        if len(roots) != 1:
            raise ValueError("Expected a single GitHub archive root")
        seen = ArchivePaths()
        entries = []
        for member in members:
            path = portable_path(member.filename)
            kind = stat.S_IFMT(member.external_attr >> 16)
            if kind not in (0, stat.S_IFREG, stat.S_IFDIR):
                raise ValueError(f"Unsupported link/device in ZIP: {member.filename}")
            if len(path.parts) == 1:
                if not member.is_dir():
                    raise ValueError("A source ZIP must contain a root directory")
                continue
            relative = seen.add(str(PurePosixPath(*path.parts[1:])), directory=member.is_dir())
            entries.append((member, relative))
        with empty_destination(destination) as temporary:
            for member, relative in entries:
                target = temporary.joinpath(*relative.parts)
                if member.is_dir() or stat.S_IFMT(member.external_attr >> 16) == stat.S_IFDIR:
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                with source.open(member) as input_file, target.open("wb") as output:
                    shutil.copyfileobj(input_file, output)
                target.chmod(0o755 if (member.external_attr >> 16) & 0o111 else 0o644)


def extract_tar(archive: Path, destination: Path) -> None:
    with tarfile.open(archive, "r:gz") as source:
        members = []
        for member in source:
            members.append(member)
            if len(members) > MAX_MEMBERS:
                raise ValueError("Excessive tar member count")
        if not members or len(members) > MAX_MEMBERS:
            raise ValueError("Empty tar or excessive member count")
        if sum(member.size for member in members) > MAX_UNPACKED_BYTES:
            raise ValueError("Source tar exceeds extraction limits")
        seen = ArchivePaths()
        entries = []
        for member in members:
            path = seen.add(member.name, directory=member.isdir())
            if not (member.isfile() or member.isdir()):
                raise ValueError(f"Unsafe source member: {member.name}")
            entries.append((member, path))
        with empty_destination(destination) as temporary:
            for member, path in entries:
                target = temporary.joinpath(*path.parts)
                if member.isdir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                with source.extractfile(member) as input_file, target.open("wb") as output:
                    shutil.copyfileobj(input_file, output)
                target.chmod(0o755 if member.mode & 0o111 else 0o644)
