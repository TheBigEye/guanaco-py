"""Bounded, retryable HTTPS reads; credentials never follow an off-host redirect."""

from __future__ import annotations

import hashlib
import http.client
import os
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

CHUNK_SIZE = 1024 * 1024
ATTEMPTS = 4
MAX_DOWNLOAD_BYTES = 2 * 1024**3
TRANSIENT_STATUS = {429, 500, 502, 503, 504}
NETWORK_ERRORS = (urllib.error.URLError, TimeoutError, ConnectionError, http.client.IncompleteRead)


class SafeRedirect(urllib.request.HTTPRedirectHandler):
    """Allow HTTPS redirects; an authenticated request must keep the same host."""

    def redirect_request(self, request, response, code, message, headers, new_url):
        old = urllib.parse.urlsplit(request.full_url)
        new = urllib.parse.urlsplit(new_url)
        if new.scheme != "https" or new.username or new.password:
            raise ValueError("Refusing an unsafe HTTP redirect")
        if request.has_header("Authorization") and (new.hostname, new.port) != (
            old.hostname,
            old.port,
        ):
            raise ValueError("Refusing an authenticated redirect to another host")
        return super().redirect_request(request, response, code, message, headers, new_url)


def open_url(request: urllib.request.Request, timeout: int = 60):
    return urllib.request.build_opener(SafeRedirect()).open(request, timeout=timeout)


def retryable(error: Exception) -> bool:
    if isinstance(error, urllib.error.HTTPError):
        return error.code in TRANSIENT_STATUS or (
            error.code == 403 and error.headers.get("Retry-After") is not None
        )
    return isinstance(error, NETWORK_ERRORS)


def pause_before_retry(attempt: int, error: Exception) -> None:
    delay = 2**attempt
    if isinstance(error, urllib.error.HTTPError):
        try:
            delay = max(delay, int(error.headers.get("Retry-After", "0")))
        except ValueError:
            pass
    time.sleep(min(delay, 30))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(
    url: str,
    destination: Path,
    *,
    expected_sha256: str | None = None,
    max_bytes: int = MAX_DOWNLOAD_BYTES,
) -> str:
    """Verify in a temporary file, then replace the destination after success only."""
    parsed = urllib.parse.urlsplit(url)
    if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password:
        raise ValueError("Downloads must use a public HTTPS URL without credentials")
    if destination.is_symlink() or destination.is_dir():
        raise ValueError("Download destination must be a regular file path")
    destination.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": "guanaco-py-builder"})
    for attempt in range(ATTEMPTS):
        fd, name = tempfile.mkstemp(prefix=".download-", dir=destination.parent)
        temporary = Path(name)
        try:
            with os.fdopen(fd, "wb") as output, open_url(request, timeout=120) as response:
                length_header = response.headers.get("Content-Length")
                expected_size = int(length_header) if length_header is not None else None
                if expected_size is not None and (expected_size < 0 or expected_size > max_bytes):
                    raise ValueError("Download exceeds the allowed size")
                total = 0
                for chunk in iter(lambda: response.read(CHUNK_SIZE), b""):
                    total += len(chunk)
                    if total > max_bytes:
                        raise ValueError("Download exceeds the allowed size")
                    output.write(chunk)
                if expected_size is not None and total != expected_size:
                    raise http.client.IncompleteRead(b"", expected_size - total)
            digest = sha256(temporary)
            if expected_sha256 is not None and digest != expected_sha256:
                raise ValueError(f"Checksum mismatch for {destination.name}")
            temporary.replace(destination)
            return digest
        except Exception as error:
            if attempt + 1 == ATTEMPTS or not retryable(error):
                raise
            pause_before_retry(attempt, error)
        finally:
            temporary.unlink(missing_ok=True)
    raise RuntimeError("Download retry loop exhausted")
