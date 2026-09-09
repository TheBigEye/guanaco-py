"""Downloads and archive extraction: retries, limits and path safety."""

from __future__ import annotations

import http.client
import io
import tarfile
import urllib.error
import zipfile

import pytest
from helpers import archive_member

from guanaco.transfer import (
    Archive,
    ArchivePaths,
    Downloader,
    TransferError,
    empty_destination,
    is_retryable,
    pause_before_retry,
    portable_path,
    sha256,
)


class FakeResponse(io.BytesIO):
    """A response-like object: bytes plus the headers a download expects."""

    def __init__(self, payload: bytes, headers: dict | None = None) -> None:
        """Store the payload and an optional Content-Length."""
        super().__init__(payload)
        self.headers = headers or {}


class TestDownloader:
    def test_refuses_a_non_https_url(self, tmp_path):
        with pytest.raises(TransferError):
            Downloader().fetch("http://example.test/file", tmp_path / "f")

    def test_refuses_credentials_in_the_url(self, tmp_path):
        with pytest.raises(TransferError):
            Downloader().fetch("https://user:pw@example.test/f", tmp_path / "f")

    def test_refuses_a_directory_destination(self, tmp_path):
        with pytest.raises(TransferError):
            Downloader().fetch("https://example.test/f", tmp_path)

    def test_writes_and_hashes_the_body(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "guanaco.transfer.open_url",
            lambda request, timeout=60: FakeResponse(b"payload"),
        )
        destination = tmp_path / "file.bin"
        digest = Downloader().fetch("https://example.test/file.bin", destination)
        assert destination.read_bytes() == b"payload"
        assert digest == sha256(destination)

    def test_checks_the_expected_digest(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "guanaco.transfer.open_url",
            lambda request, timeout=60: FakeResponse(b"payload"),
        )
        with pytest.raises(TransferError):
            Downloader(attempts=1).fetch(
                "https://example.test/f", tmp_path / "f", expected_sha256="0" * 64
            )

    def test_refuses_a_download_over_the_limit(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "guanaco.transfer.open_url",
            lambda request, timeout=60: FakeResponse(b"x" * 100),
        )
        with pytest.raises(TransferError):
            Downloader(attempts=1, max_bytes=10).fetch("https://example.test/f", tmp_path / "f")

    def test_detects_a_truncated_body(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "guanaco.transfer.open_url",
            lambda request, timeout=60: FakeResponse(b"short", {"Content-Length": "50"}),
        )
        with pytest.raises((TransferError, http.client.IncompleteRead)):
            Downloader(attempts=1).fetch("https://example.test/f", tmp_path / "f")


class TestRetryPolicy:
    def test_transient_statuses_are_retryable(self):
        for code in (429, 500, 502, 503, 504):
            assert is_retryable(urllib.error.HTTPError("u", code, "m", {}, None))

    def test_a_permanent_status_is_not(self):
        assert not is_retryable(urllib.error.HTTPError("u", 404, "m", {}, None))

    def test_network_errors_are_retryable(self):
        assert is_retryable(ConnectionError("boom"))

    def test_backoff_is_capped(self, monkeypatch):
        sleeps = []
        monkeypatch.setattr("guanaco.transfer.time.sleep", lambda value: sleeps.append(value))
        pause_before_retry(20, ConnectionError("boom"))
        assert sleeps == [30]

    def test_backoff_honours_retry_after(self, monkeypatch):
        sleeps = []
        monkeypatch.setattr("guanaco.transfer.time.sleep", lambda value: sleeps.append(value))
        error = urllib.error.HTTPError("u", 503, "m", {"Retry-After": "7"}, None)
        pause_before_retry(0, error)
        assert sleeps == [7]

    def test_a_bad_retry_after_is_ignored(self, monkeypatch):
        sleeps = []
        monkeypatch.setattr("guanaco.transfer.time.sleep", lambda value: sleeps.append(value))
        error = urllib.error.HTTPError("u", 503, "m", {"Retry-After": "soon"}, None)
        pause_before_retry(2, error)
        assert sleeps == [4]


class TestPortablePath:
    @pytest.mark.parametrize(
        "name",
        [
            "../escape",
            "/absolute",
            "a/../../b",
            "dir/.git/config",
            "CON",
            "nul.txt",
            "com1",
            "lpt9",
            "trailing ",
            "trailing.",
            "bad\\slash",
            "with\nnewline",
            "",
        ],
    )
    def test_rejects_unsafe_names(self, name):
        with pytest.raises(TransferError):
            portable_path(name)

    def test_accepts_a_normal_relative_path(self):
        assert portable_path("a/b/c.txt").as_posix() == "a/b/c.txt"

    def test_rejects_a_non_string(self):
        with pytest.raises(TransferError):
            portable_path(None)


class TestArchivePaths:
    def test_detects_duplicates(self):
        inventory = ArchivePaths()
        inventory.add("a.txt")
        with pytest.raises(TransferError):
            inventory.add("a.txt")

    def test_detects_case_collisions(self):
        inventory = ArchivePaths()
        inventory.add("Readme.md")
        with pytest.raises(TransferError):
            inventory.add("readme.md")

    def test_detects_a_file_used_as_a_directory(self):
        inventory = ArchivePaths()
        inventory.add("thing")
        with pytest.raises(TransferError):
            inventory.add("thing/inside.txt")

    def test_detects_a_directory_used_as_a_file(self):
        inventory = ArchivePaths()
        inventory.add("thing/inside.txt")
        with pytest.raises(TransferError):
            inventory.add("thing")


class TestEmptyDestination:
    def test_refuses_a_non_empty_destination(self, tmp_path):
        (tmp_path / "file").write_text("x", encoding="utf-8")
        with pytest.raises(TransferError):
            with empty_destination(tmp_path):
                pass

    def test_leaves_nothing_behind_on_failure(self, tmp_path):
        with pytest.raises(RuntimeError):
            with empty_destination(tmp_path / "out") as scratch:
                (scratch / "partial.txt").write_text("x", encoding="utf-8")
                raise RuntimeError("boom")
        assert not (tmp_path / "out").exists()

    def test_swaps_on_success(self, tmp_path):
        with empty_destination(tmp_path / "out") as scratch:
            (scratch / "file.txt").write_text("ok", encoding="utf-8")
        assert (tmp_path / "out" / "file.txt").read_text(encoding="utf-8") == "ok"


class TestArchiveExtraction:
    def test_strips_the_zip_root(self, tmp_path):
        archive = tmp_path / "source.zip"
        with zipfile.ZipFile(archive, "w") as zipped:
            zipped.writestr("root/", "")
            zipped.writestr("root/hello.txt", "hi")
        Archive.extract(archive, tmp_path / "out")
        assert (tmp_path / "out" / "hello.txt").read_text(encoding="utf-8") == "hi"

    def test_rejects_several_roots(self, tmp_path):
        archive = tmp_path / "source.zip"
        with zipfile.ZipFile(archive, "w") as zipped:
            zipped.writestr("one/", "")
            zipped.writestr("two/", "")
        with pytest.raises(TransferError):
            Archive.extract(archive, tmp_path / "out")

    def test_rejects_an_unknown_format(self, tmp_path):
        archive = tmp_path / "source.rar"
        archive.write_bytes(b"nope")
        with pytest.raises(TransferError):
            Archive.extract(archive, tmp_path / "out")

    def test_extracts_a_tarball(self, tmp_path):
        source = tmp_path / "src"
        (source / "inner").mkdir(parents=True)
        (source / "inner" / "file.txt").write_text("hi", encoding="utf-8")
        archive = tmp_path / "source.tar.gz"
        with tarfile.open(archive, "w:gz") as tar:
            tar.add(source / "inner", arcname="inner")
        Archive.extract(archive, tmp_path / "out")
        assert (tmp_path / "out" / "inner" / "file.txt").read_text(encoding="utf-8") == "hi"

    def test_rejects_a_traversal_member(self, tmp_path):
        archive = tmp_path / "evil.zip"
        with zipfile.ZipFile(archive, "w") as zipped:
            zipped.writestr(archive_member("root/"), "")
            zipped.writestr(archive_member("root/../escape.txt"), "nope")
        with pytest.raises(TransferError):
            Archive.extract(archive, tmp_path / "out")
        assert not (tmp_path / "escape.txt").exists()

    def test_rejects_an_empty_archive(self, tmp_path):
        archive = tmp_path / "empty.zip"
        with zipfile.ZipFile(archive, "w"):
            pass
        with pytest.raises(TransferError):
            Archive.extract(archive, tmp_path / "out")
