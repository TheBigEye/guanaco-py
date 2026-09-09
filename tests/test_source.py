"""Preparing the frozen upstream source: patches, metadata and archives."""

from __future__ import annotations

import shutil
from dataclasses import replace
from pathlib import Path

import pytest
from helpers import SHA_A, SHA_B, make_plan
from source_helpers import GITMODULES, fixture_source, zip_source

from guanaco.source import (
    MetadataAdapter,
    PatchSet,
    SourceArchive,
    SourceError,
    SourcePreparer,
    submodule_repositories,
)
from guanaco.transfer import TransferError, sha256

PATCH_TEXT = (
    "--- a/llama_cpp/__init__.py\n"
    "+++ b/llama_cpp/__init__.py\n"
    "@@ -1 +1 @@\n"
    '-__version__ = "0.3.49"\n'
    '+__version__ = "0.3.49"\n'
    " "
)


class StubDownloader:
    """A downloader that writes pre-built archives instead of using the network."""

    def __init__(self, payloads: dict) -> None:
        """Index payloads by the repository they belong to."""
        self.payloads = payloads
        self.requested: list[str] = []

    def fetch(self, url: str, archive: Path, expected_sha256: str | None = None) -> str:
        """Write the archive registered for the repository named in `url`."""
        del expected_sha256
        self.requested.append(url)
        repository = next(name for name in self.payloads if name in url)
        archive.parent.mkdir(parents=True, exist_ok=True)
        zip_source(archive, self.payloads[repository])
        return sha256(archive)


class StubClient:
    """A GitHub client that only knows about submodule gitlinks."""

    def __init__(self, entries: dict) -> None:
        """Store the gitlinks keyed by ``repository/sha``."""
        self.entries = entries

    def tree(self, repository: str, sha: str, recursive: bool = True) -> list[dict]:
        """Return the gitlinks recorded for this repository."""
        del recursive
        return self.entries.get(f"{repository}/{sha}", [])


def gitlink(path: str = "vendor/llama.cpp", commit: str = SHA_B) -> dict:
    """Return one gitlink entry as the Git tree API reports it."""
    return {"path": path, "type": "commit", "sha": commit, "mode": "160000"}


def add_patch(directory: Path, name: str = "0001-example.patch", text: str = PATCH_TEXT) -> Path:
    """Drop a patch into the patch directory so it gets applied."""
    directory.mkdir(parents=True, exist_ok=True)
    patch = directory / name
    patch.write_text(text, encoding="utf-8")
    return patch


# ---------------------------------------------------------------------------
# Patches
# ---------------------------------------------------------------------------


class TestPatchSet:
    def _source(self, tmp_path) -> Path:
        source = tmp_path / "src"
        source.mkdir(parents=True)
        (source / "file.txt").write_text("one\ntwo\n", encoding="utf-8")
        return source

    def test_applies_with_git_when_available(self, tmp_path):
        source = self._source(tmp_path)
        add_patch(
            tmp_path / "patches",
            text="--- a/file.txt\n+++ b/file.txt\n@@ -1,2 +1,2 @@\n one\n-two\n+TWO\n",
        )
        records = PatchSet(tmp_path / "patches").apply(source)
        assert (source / "file.txt").read_text(encoding="utf-8") == "one\nTWO\n"
        assert records[0].files == ("file.txt",)

    def test_falls_back_to_python_when_git_is_absent(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            shutil, "which", lambda name: None if name == "git" else "/usr/bin/" + name
        )
        source = self._source(tmp_path)
        add_patch(
            tmp_path / "patches",
            text="--- a/file.txt\n+++ b/file.txt\n@@ -1,2 +1,2 @@\n one\n-two\n+TWO\n",
        )
        PatchSet(tmp_path / "patches").apply(source)
        assert (source / "file.txt").read_text(encoding="utf-8") == "one\nTWO\n"

    def test_a_missing_patch_directory_is_fine(self, tmp_path):
        assert PatchSet(tmp_path / "absent").apply(self._source(tmp_path)) == ()

    def test_an_empty_patch_is_rejected(self, tmp_path):
        source = self._source(tmp_path)
        directory = tmp_path / "patches"
        directory.mkdir()
        (directory / "empty.patch").write_text("", encoding="utf-8")
        with pytest.raises(SourceError):
            PatchSet(directory).apply(source)

    def test_a_patch_that_does_not_apply_is_rejected(self, tmp_path):
        source = self._source(tmp_path)
        add_patch(
            tmp_path / "patches",
            text="--- a/file.txt\n+++ b/file.txt\n@@ -1,2 +1,2 @@\n NOPE\n-two\n+TWO\n",
        )
        with pytest.raises(SourceError, match="no longer applies"):
            PatchSet(tmp_path / "patches").apply(source)

    def test_a_patch_targeting_an_absent_file_is_rejected(self, tmp_path):
        source = self._source(tmp_path)
        add_patch(
            tmp_path / "patches",
            text="--- a/absent.txt\n+++ b/absent.txt\n@@ -1 +1 @@\n-a\n+A\n",
        )
        with pytest.raises(SourceError, match="targets a file upstream no longer has"):
            PatchSet(tmp_path / "patches").apply(source)

    def test_a_patch_without_a_target_is_rejected(self, tmp_path):
        source = self._source(tmp_path)
        add_patch(tmp_path / "patches", text="not a patch at all\n")
        with pytest.raises(SourceError, match="declares no target file"):
            PatchSet(tmp_path / "patches").apply(source)


# ---------------------------------------------------------------------------
# Submodules
# ---------------------------------------------------------------------------


class TestSubmodules:
    def test_reads_gitmodules(self, tmp_path):
        source = fixture_source(tmp_path)
        (source / ".gitmodules").write_text(GITMODULES, encoding="utf-8")
        assert submodule_repositories(source) == {"vendor/llama.cpp": "ggml-org/llama.cpp"}

    def test_missing_file_gives_an_empty_mapping(self, tmp_path):
        assert submodule_repositories(fixture_source(tmp_path)) == {}

    def test_rejects_a_weird_url(self, tmp_path):
        source = fixture_source(tmp_path)
        (source / ".gitmodules").write_text(
            '[submodule "x"]\n\tpath = x\n\turl = ../relative\n', encoding="utf-8"
        )
        with pytest.raises(SourceError):
            submodule_repositories(source)


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------


class TestMetadataAdapter:
    def adapt(self, settings, tmp_path, package="guanaco-py"):
        """Run the adapter over the fixture source and return the results."""
        plan = make_plan(settings)
        source = fixture_source(tmp_path)
        patch = MetadataAdapter(package, settings.upstream_package).adapt(source, plan, SHA_A)
        return source, plan, patch

    def test_targets_our_package(self, settings, tmp_path):
        source, _, patch = self.adapt(settings, tmp_path)
        text = (source / "pyproject.toml").read_text(encoding="utf-8")
        assert 'name = "guanaco-py"' in text
        assert 'name = "llama_cpp_python"' not in text
        assert patch.startswith("--- a/pyproject.toml")

    def test_the_description_credits_the_upstream_owner(self, settings, tmp_path):
        source, plan, _ = self.adapt(settings, tmp_path)
        text = (source / "pyproject.toml").read_text(encoding="utf-8")
        owner = plan.origin.repository.split("/")[0]
        assert f"CPU and CUDA builds of {owner}'s" in text
        assert "distributed as guanaco-py" in text

    def test_keeps_the_upstream_version(self, settings, tmp_path):
        source, plan, _ = self.adapt(settings, tmp_path)
        assert (
            f'__version__ = "{plan.version}"' in (source / "llama_cpp" / "__init__.py").read_text()
        )

    def test_internal_extras_and_urls_are_rewritten(self, settings, tmp_path):
        source, plan, _ = self.adapt(settings, tmp_path)
        text = (source / "pyproject.toml").read_text(encoding="utf-8")
        assert "guanaco-py[server]" in text
        assert f"github.com/{plan.repository}" in text
        assert f"github.com/{plan.upstream}/blob/main/docs/wiki/index.md" in text

    def test_build_identity_and_licenses_are_pinned(self, settings, tmp_path):
        source, _, _ = self.adapt(settings, tmp_path)
        text = (source / "pyproject.toml").read_text(encoding="utf-8")
        assert f"LLAMA_BUILD_COMMIT = {SHA_A!r}".replace("'", '"') in text
        assert "license-files" in text
        assert "vendor/llama.cpp/LICENSE*" in text

    def test_identifiers_are_normalised_for_the_backend(self, settings, tmp_path):
        source, _, _ = self.adapt(settings, tmp_path, "guanaco_py")
        text = (source / "pyproject.toml").read_text(encoding="utf-8")
        assert 'name = "guanaco_py"' in text
        assert 'wheel.packages = ["llama_cpp"]' in text

    def test_requirement_checks_detect_the_upstream_name(self, settings):
        adapter = MetadataAdapter("guanaco-py", settings.upstream_package)
        assert adapter.check_requirement("llama_cpp_python==0.3.49")
        assert adapter.check_requirement("llama-cpp-python[server]")
        assert not adapter.check_requirement("guanaco-py")
        assert not adapter.check_requirement("diskcache>=5.6")

    def test_a_foreign_project_is_rejected(self, settings, tmp_path):
        source = fixture_source(tmp_path)
        (source / "pyproject.toml").write_text(
            '[project]\nname = "something-else"\n', encoding="utf-8"
        )
        with pytest.raises(SourceError, match="Unexpected upstream distribution name"):
            MetadataAdapter("guanaco-py", settings.upstream_package).adapt(
                source, make_plan(settings), SHA_A
            )

    def test_a_version_mismatch_is_rejected(self, settings, tmp_path):
        source = fixture_source(tmp_path, version="9.9.9")
        with pytest.raises(SourceError, match="does not match __version__"):
            MetadataAdapter("guanaco-py", settings.upstream_package).adapt(
                source, make_plan(settings), SHA_A
            )

    def test_adapting_twice_is_rejected(self, settings, tmp_path):
        source, plan, _ = self.adapt(settings, tmp_path)
        adapter = MetadataAdapter("guanaco-py", settings.upstream_package)
        with pytest.raises(SourceError):
            adapter.adapt(source, plan, SHA_A)


# ---------------------------------------------------------------------------
# Preparation
# ---------------------------------------------------------------------------


def run_preparation(
    tmp_path,
    settings,
    *,
    include_submodule: bool = True,
    output_name: str = "prepared",
    patch: bool = False,
    gitmodules: bool = True,
):
    """Run :class:`SourcePreparer` over the synthetic upstream layout."""
    if patch:
        add_patch(settings.patches_dir)
    plan = make_plan(settings)
    upstream_files = {
        "pyproject.toml": (
            fixture_source(tmp_path / f"fixture-{output_name}") / "pyproject.toml"
        ).read_text(),
        "llama_cpp/__init__.py": '__version__ = "0.3.49"\n',
        "LICENSE.md": "MIT - original attribution",
    }
    if gitmodules:
        upstream_files[".gitmodules"] = GITMODULES
    payloads = {
        settings.upstream: upstream_files,
        "ggml-org/llama.cpp": {"LICENSE": "ggml"},
    }
    entries = {f"{settings.upstream}/{SHA_A}": [gitlink()] if include_submodule else []}
    preparer = SourcePreparer(settings, StubClient(entries), StubDownloader(payloads))
    output = tmp_path / output_name
    return plan, preparer.prepare(plan, output), output


class TestSourcePreparer:
    def test_prepare_freezes_both_snapshots(self, tmp_path, settings):
        plan, manifest, output = run_preparation(tmp_path, settings)
        assert manifest.version == plan.version
        assert manifest.native_commit == SHA_B
        assert [s.repository for s in manifest.snapshots] == [
            settings.upstream,
            "ggml-org/llama.cpp",
        ]
        assert manifest.snapshots[0].zip_sha256
        assert (output / "source.tar.gz").is_file()
        assert (output / "build-manifest.json").is_file()

    def test_prepare_records_applied_patches(self, tmp_path, settings):
        _, manifest, _ = run_preparation(tmp_path, settings, patch=True)
        assert len(manifest.applied_patches) == 1
        assert manifest.applied_patches[0].files == ("llama_cpp/__init__.py",)

    def test_prepare_without_patches_reports_none(self, tmp_path, settings):
        _, manifest, _ = run_preparation(tmp_path, settings)
        assert manifest.applied_patches == ()

    def test_prepare_is_deterministic(self, tmp_path, settings):
        _, first, _ = run_preparation(tmp_path, settings, output_name="first")
        _, second, _ = run_preparation(tmp_path, settings, output_name="second")
        assert first.to_mapping() == second.to_mapping()

    def test_prepare_rejects_a_missing_native_snapshot(self, tmp_path, settings):
        with pytest.raises(SourceError, match="expected llama.cpp submodule"):
            run_preparation(tmp_path, settings, include_submodule=False, gitmodules=False)

    def test_prepare_rejects_a_tree_that_disagrees_with_gitmodules(self, tmp_path, settings):
        with pytest.raises(SourceError, match="do not agree"):
            run_preparation(tmp_path, settings, include_submodule=False)

    def test_prepare_refuses_a_foreign_upstream(self, tmp_path, settings):
        plan = replace(
            make_plan(settings),
            origin=replace(make_plan(settings).origin, repository="someone/else"),
        )
        preparer = SourcePreparer(settings, StubClient({}), StubDownloader({}))
        with pytest.raises(SourceError, match="Unexpected upstream repository"):
            preparer.prepare(plan, tmp_path / "out")

    def test_a_stale_destination_is_refused(self, tmp_path, settings):
        output = tmp_path / "prepared"
        output.mkdir()
        (output / "old").write_text("x", encoding="utf-8")
        preparer = SourcePreparer(settings, StubClient({}), StubDownloader({}))
        with pytest.raises(TransferError):
            preparer.prepare(make_plan(settings), output)

    def test_a_non_pinned_commit_is_refused(self, tmp_path, settings):
        plan = replace(
            make_plan(settings), origin=replace(make_plan(settings).origin, commit="short")
        )
        preparer = SourcePreparer(settings, StubClient({}), StubDownloader({}))
        with pytest.raises(SourceError):
            preparer.prepare(plan, tmp_path / "out")

    def test_nested_submodules_are_downloaded_recursively(self, tmp_path, settings):
        nested = {
            f"{settings.upstream}/{SHA_A}": [gitlink()],
            f"ggml-org/llama.cpp/{SHA_B}": [gitlink("vendor/ggml", SHA_A)],
            f"ggml-org/ggml/{SHA_A}": [],
        }
        payloads = {
            settings.upstream: {".gitmodules": GITMODULES, "LICENSE.md": "MIT"},
            "ggml-org/llama.cpp": {
                ".gitmodules": '[submodule "vendor/ggml"]\n\tpath = vendor/ggml\n\turl = https://github.com/ggml-org/ggml\n'
            },
            "ggml-org/ggml": {"LICENSE": "ggml"},
        }
        preparer = SourcePreparer(settings, StubClient(nested), StubDownloader(payloads))
        with pytest.raises(SourceError, match="Missing upstream runtime"):
            # No Python runtime in this fixture, but the recursion already happened.
            preparer.prepare(make_plan(settings), tmp_path / "out")
        assert len(preparer.downloader.requested) == 3


# ---------------------------------------------------------------------------
# Travelling artifact
# ---------------------------------------------------------------------------


def empty_prepared(tmp_path) -> Path:
    """A prepared directory holding only the two files the archive needs."""
    prepared = tmp_path / "prepared"
    prepared.mkdir(parents=True, exist_ok=True)
    (prepared / "source.tar.gz").write_bytes(b"placeholder")
    (prepared / "packaging.patch").write_text("patch", encoding="utf-8")
    return prepared


class TestSourceArchive:
    def test_verify_returns_the_manifest(self, tmp_path, settings):
        plan, _, prepared = run_preparation(tmp_path, settings)
        manifest = SourceArchive.verify(prepared)
        assert manifest.version == plan.version
        assert manifest.source_archive_sha256 == sha256(prepared / "source.tar.gz")
        assert manifest.packaging_patch_sha256 == sha256(prepared / "packaging.patch")

    def test_verify_rejects_a_tampered_archive(self, tmp_path, settings):
        _, _, prepared = run_preparation(tmp_path, settings)
        (prepared / "source.tar.gz").write_bytes(b"tampered")
        with pytest.raises(SourceError, match="checksum mismatch"):
            SourceArchive.verify(prepared)

    def test_verify_rejects_a_missing_archive(self, tmp_path, settings):
        _, _, prepared = run_preparation(tmp_path, settings)
        (prepared / "source.tar.gz").unlink()
        with pytest.raises(SourceError):
            SourceArchive.verify(prepared)

    def test_verify_rejects_a_missing_patch(self, tmp_path, settings):
        _, _, prepared = run_preparation(tmp_path, settings)
        (prepared / "packaging.patch").unlink()
        with pytest.raises(SourceError):
            SourceArchive.verify(prepared)

    def test_extract_unpacks_the_snapshot(self, tmp_path, settings):
        _, _, prepared = run_preparation(tmp_path, settings)
        manifest = SourceArchive.extract(prepared, tmp_path / "unpacked")
        assert (tmp_path / "unpacked" / "llama_cpp" / "__init__.py").is_file()
        assert manifest.package == "guanaco-py"

    def test_extract_checks_the_version(self, tmp_path, settings):
        _, _, prepared = run_preparation(tmp_path, settings)
        with pytest.raises(SourceError, match="does not match the requested build"):
            SourceArchive.extract(prepared, tmp_path / "unpacked", "9.9.9")

    def test_extract_rejects_a_missing_manifest(self, tmp_path, settings):
        prepared = empty_prepared(tmp_path)
        with pytest.raises(SourceError):
            SourceArchive.extract(prepared, tmp_path / "out", None)
