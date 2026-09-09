"""Moving bytes in and out of the build: downloads and archive extraction.

Two jobs live here, and both are about *not trusting the input*:

* :class:`Downloader` fetches a file over HTTPS with a bounded retry budget, a
  size limit and an optional checksum, writing to a temporary file so a failed
  download never clobbers a good one.
* :class:`Archive` extracts a source archive into a directory, validating every
  member path first and swapping the directory into place only on success, so a
  failed extraction cannot leave a half-built source tree behind.
"""

from __future__ import annotations

import contextlib
import hashlib
import http.client
import os
import shutil
import stat
import tarfile
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath

CHUNK_SIZE = 1024 * 1024
DEFAULT_ATTEMPTS = 4
MAX_DOWNLOAD_BYTES = 2 * 1024**3
MAX_MEMBERS = 50_000
MAX_UNPACKED_BYTES = 2 * 1024**3
TRANSIENT_STATUS = {429, 500, 502, 503, 504}
NETWORK_ERRORS = (urllib.error.URLError, TimeoutError, ConnectionError, http.client.IncompleteRead)

# Reserved Windows device names. A member called "NUL" or "COM1" would be
# harmless on Linux and destructive on the Windows builders.
WINDOWS_DEVICES = {"con", "prn", "aux", "nul"} | {
    f"{prefix}{number}" for prefix in ("com", "lpt") for number in range(1, 10)
}


class TransferError(ValueError):
    """Raised when a download or an archive cannot be handled safely."""


def sha256(path: Path) -> str:
    """Return the SHA-256 of a file, streamed in chunks."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


class SafeRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Allow HTTPS redirects, but never let a token follow one to another host."""

    def redirect_request(self, request, response, code, message, headers, new_url):
        """Return the redirected request, or refuse the redirect."""
        old = urllib.parse.urlsplit(request.full_url)
        new = urllib.parse.urlsplit(new_url)
        if new.scheme != "https" or new.username or new.password:
            raise TransferError("Refusing an unsafe HTTP redirect")
        if request.has_header("Authorization") and (new.hostname, new.port) != (
            old.hostname,
            old.port,
        ):
            raise TransferError("Refusing an authenticated redirect to another host")
        return super().redirect_request(request, response, code, message, headers, new_url)


def open_url(request: urllib.request.Request, timeout: int = 60):
    """Open a URL with the safe redirect policy applied."""
    return urllib.request.build_opener(SafeRedirectHandler()).open(request, timeout=timeout)


def is_retryable(error: Exception) -> bool:
    """Whether a failed request is worth retrying."""
    if isinstance(error, urllib.error.HTTPError):
        return error.code in TRANSIENT_STATUS or (
            error.code == 403 and error.headers.get("Retry-After") is not None
        )
    return isinstance(error, NETWORK_ERRORS)


def pause_before_retry(attempt: int, error: Exception) -> None:
    """Sleep with a capped exponential backoff, honouring ``Retry-After``."""
    delay = 2**attempt
    if isinstance(error, urllib.error.HTTPError):
        try:
            delay = max(delay, int(error.headers.get("Retry-After", "0")))
        except ValueError:
            pass
    time.sleep(min(delay, 30))


class Downloader:
    """Fetches files over HTTPS with retries, size limits and checksums.

    Attributes:
        attempts: How many times to try a transient failure before giving up.
        max_bytes: Hard limit on a single download.
        user_agent: Sent with every request.
    """

    def __init__(
        self,
        *,
        attempts: int = DEFAULT_ATTEMPTS,
        max_bytes: int = MAX_DOWNLOAD_BYTES,
        user_agent: str = "guanaco-py-builder",
    ) -> None:
        """Store the download policy. Nothing is fetched until :meth:`fetch`."""
        self.attempts = attempts
        self.max_bytes = max_bytes
        self.user_agent = user_agent

    def fetch(self, url: str, destination: Path, *, expected_sha256: str | None = None) -> str:
        """Download `url` to `destination` and return its SHA-256.

        The file is streamed into a temporary file next to the destination and
        moved into place only after the length and checksum check out, so an
        interrupted download cannot corrupt the build.

        Args:
            url: A public HTTPS URL without embedded credentials.
            destination: Where to put the file.
            expected_sha256: Optional digest the file must match.

        Raises:
            TransferError: If the URL, the size or the checksum is rejected.
        """
        parsed = urllib.parse.urlsplit(url)
        if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password:
            raise TransferError("Downloads must use a public HTTPS URL without credentials")
        destination = Path(destination)
        if destination.is_symlink() or destination.is_dir():
            raise TransferError("Download destination must be a regular file path")
        destination.parent.mkdir(parents=True, exist_ok=True)
        request = urllib.request.Request(url, headers={"User-Agent": self.user_agent})
        for attempt in range(self.attempts):
            handle, temporary_name = tempfile.mkstemp(prefix=".download-", dir=destination.parent)
            temporary = Path(temporary_name)
            try:
                with os.fdopen(handle, "wb") as output, open_url(request, timeout=120) as response:
                    self._copy(response, output)
                digest = sha256(temporary)
                if expected_sha256 is not None and digest != expected_sha256:
                    raise TransferError(f"Checksum mismatch for {destination.name}")
                temporary.replace(destination)
                return digest
            except Exception as error:
                if attempt + 1 == self.attempts or not is_retryable(error):
                    raise
                pause_before_retry(attempt, error)
            finally:
                temporary.unlink(missing_ok=True)
        raise TransferError("Download retry loop exhausted")

    def _copy(self, response, output) -> None:
        """Stream the response body to `output`, enforcing the size limit."""
        length = response.headers.get("Content-Length")
        expected = int(length) if length is not None else None
        if expected is not None and (expected < 0 or expected > self.max_bytes):
            raise TransferError("Download exceeds the allowed size")
        total = 0
        for chunk in iter(lambda: response.read(CHUNK_SIZE), b""):
            total += len(chunk)
            if total > self.max_bytes:
                raise TransferError("Download exceeds the allowed size")
            output.write(chunk)
        if expected is not None and total != expected:
            raise http.client.IncompleteRead(b"", expected - total)


def portable_path(name: str) -> PurePosixPath:
    """Validate an archive member name and return it as a POSIX path.

    Rejects absolute paths, traversal, Git control directories, Windows device
    names, alternate data streams and control characters.

    Raises:
        TransferError: If the name is not a safe relative path.
    """
    if not isinstance(name, str) or not name or name.startswith("/"):
        raise TransferError(f"Unsafe archive path: {name!r}")
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
            raise TransferError(f"Unsafe archive path: {name!r}")
    return PurePosixPath(*parts)


class ArchivePaths:
    """Tracks archive members, catching duplicates and case collisions.

    Case-insensitive filesystems (Windows, macOS) would silently merge
    ``README.md`` and ``readme.md``; refusing the collision here means the
    source tree is identical everywhere we build it.
    """

    def __init__(self) -> None:
        """Start with an empty inventory."""
        self.members: set[str] = set()
        self.files: set[str] = set()
        self.spellings: dict[str, str] = {}

    def add(self, name: str, *, directory: bool = False) -> PurePosixPath:
        """Record a member and return its validated path.

        Raises:
            TransferError: On a duplicate, a case collision, or a file that
                lands where a directory already exists.
        """
        path = portable_path(name)
        key = str(path).casefold()
        if key in self.members:
            raise TransferError(f"Duplicate or case-colliding archive member: {name}")
        if not directory and key in self.spellings:
            raise TransferError(f"File/directory collision in archive: {name}")
        for component in (path, *path.parents):
            text = str(component)
            if text == ".":
                continue
            folded = text.casefold()
            if folded in self.files:
                raise TransferError(f"File/directory collision in archive: {name}")
            if folded in self.spellings and self.spellings[folded] != text:
                raise TransferError(f"Case-colliding archive directory: {name}")
            self.spellings[folded] = text
        self.members.add(key)
        if not directory:
            self.files.add(key)
        return path


@contextlib.contextmanager
def empty_destination(destination: Path):
    """Yield a temporary directory that replaces `destination` on success.

    If the block raises, `destination` is left exactly as it was, which makes
    every extraction safe to retry.
    """
    destination = Path(destination)
    if destination.is_symlink() or (
        destination.exists() and (not destination.is_dir() or any(destination.iterdir()))
    ):
        raise TransferError("Archive destination must be empty and must not be a symlink")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".extract-", dir=destination.parent) as name:
        temporary = Path(name)
        yield temporary
        if destination.exists():
            destination.rmdir()
        temporary.replace(destination)


def _check_budget(member_count: int, total_bytes: int) -> None:
    """Refuse archives that are empty or implausibly large."""
    if not member_count or member_count > MAX_MEMBERS:
        raise TransferError("Empty archive or excessive member count")
    if total_bytes > MAX_UNPACKED_BYTES:
        raise TransferError("Archive exceeds extraction limits")


class Archive:
    """Extracts a source archive, validating every member before writing it."""

    @staticmethod
    def extract(archive: Path, destination: Path) -> None:
        """Extract `archive` into `destination`, choosing by file extension.

        Args:
            archive: A ``.zip`` or ``.tar.gz`` file.
            destination: Directory to create. It must not exist or be empty.

        Raises:
            TransferError: If the format is unknown or the archive is unsafe.
        """
        archive = Path(archive)
        name = archive.name.casefold()
        if name.endswith(".zip"):
            Archive.extract_zip(archive, destination)
        elif name.endswith((".tar.gz", ".tgz")):
            Archive.extract_tar(archive, destination)
        else:
            raise TransferError(f"Unsupported archive format: {archive.name}")

    @staticmethod
    def extract_zip(archive: Path, destination: Path) -> None:
        """Extract a GitHub source ZIP, stripping its single root directory.

        ``ZipInfo.filename`` normalises Windows backslashes and truncates NUL
        bytes, so the *original* spelling from the archive is validated first.
        """
        with zipfile.ZipFile(archive) as source:
            members = source.infolist()
            _check_budget(len(members), sum(member.file_size for member in members))
            roots = {portable_path(member.orig_filename).parts[0] for member in members}
            if len(roots) != 1:
                raise TransferError("Expected a single GitHub archive root")
            inventory = ArchivePaths()
            entries = []
            for member in members:
                path = portable_path(member.orig_filename)
                kind = stat.S_IFMT(member.external_attr >> 16)
                if kind not in (0, stat.S_IFREG, stat.S_IFDIR):
                    raise TransferError(f"Unsupported link/device in ZIP: {member.filename}")
                if len(path.parts) == 1:
                    if not member.is_dir():
                        raise TransferError("A source ZIP must contain a root directory")
                    continue
                relative = inventory.add(
                    str(PurePosixPath(*path.parts[1:])), directory=member.is_dir()
                )
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

    @staticmethod
    def extract_tar(archive: Path, destination: Path) -> None:
        """Extract a ``.tar.gz`` source archive."""
        with tarfile.open(archive, "r:gz") as source:
            members = []
            for member in source:
                members.append(member)
                if len(members) > MAX_MEMBERS:
                    raise TransferError("Excessive tar member count")
            _check_budget(len(members), sum(member.size for member in members))
            inventory = ArchivePaths()
            entries = []
            for member in members:
                path = inventory.add(member.name, directory=member.isdir())
                if not (member.isfile() or member.isdir()):
                    raise TransferError(f"Unsafe source member: {member.name}")
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


__all__ = [
    "Archive",
    "ArchivePaths",
    "Downloader",
    "MAX_MEMBERS",
    "SafeRedirectHandler",
    "TransferError",
    "empty_destination",
    "is_retryable",
    "open_url",
    "pause_before_retry",
    "portable_path",
    "sha256",
]
