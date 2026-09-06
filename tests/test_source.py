import io
import json
import stat
import tarfile
from types import SimpleNamespace

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib
import zipfile

import pytest
from helpers import SHA_A, SHA_B, plan
from prepare_source import adapt_metadata, extract_zip, materialize, runtime_hashes, source_tar
from release_common import sha256
from source_helpers import METADATA, fixture_source, raw_zip_member, zip_source
from unpack_source import unpack


def test_metadata_rebrand_preserves_runtime_version_authors_and_build_layout(tmp_path):
    source = fixture_source(tmp_path)
    original = runtime_hashes(source)
    before = tomllib.loads(METADATA)
    patch = adapt_metadata(source, plan(), SHA_A)
    result = tomllib.loads((source / "pyproject.toml").read_text())
    assert set(result) == set(before)  # regression: no stray top-level [cmake.define]
    assert result["project"]["name"] == "guanaco-py"
    assert result["project"]["optional-dependencies"]["all"] == ["guanaco-py[server]"]
    for key in ("authors", "license", "dependencies", "requires-python", "dynamic"):
        assert result["project"][key] == before["project"][key]
    build = result["tool"]["scikit-build"]
    assert build["cmake"]["minimum-version"] == "3.21"
    assert build["cmake"]["define"] == {"LLAMA_BUILD_COMMIT": SHA_A, "LLAMA_BUILD_NUMBER": "0"}
    assert build["minimum-version"] == "0.5.1"
    assert build["metadata"] == before["tool"]["scikit-build"]["metadata"]
    assert build["sdist"] == before["tool"]["scikit-build"]["sdist"]
    assert build["wheel"]["packages"] == ["llama_cpp"]
    assert "vendor/llama.cpp/LICENSE*" in build["wheel"]["license-files"]
    assert runtime_hashes(source) == original
    assert patch.startswith("--- a/pyproject.toml\n+++ b/pyproject.toml\n")


def test_source_version_mismatch_fails_before_writing(tmp_path):
    source = fixture_source(tmp_path, "0.3.50")
    with pytest.raises(ValueError, match="does not match __version__"):
        adapt_metadata(source, plan(), SHA_A)
    assert (source / "pyproject.toml").read_text() == METADATA


def test_recursive_submodules_use_the_gitlink_commit_not_main(tmp_path):
    calls = []

    class API:
        def tree(self, repo, commit):
            return (
                [{"path": "vendor/llama.cpp", "mode": "160000", "sha": SHA_B}]
                if repo == "JamePeng/llama-cpp-python"
                else []
            )

    def download(repo, commit, target):
        calls.append((repo, commit))
        files = (
            {
                ".gitmodules": '[submodule "vendor/llama.cpp"]\npath = vendor/llama.cpp\nurl = https://github.com/ggml-org/llama.cpp.git\n',
                "pyproject.toml": METADATA,
            }
            if len(calls) == 1
            else {"LICENSE": "MIT", "CMakeLists.txt": "# native fixture"}
        )
        zip_source(target, files)
        return sha256(target)

    snapshots = materialize(
        API(), "JamePeng/llama-cpp-python", SHA_A, tmp_path / "prepared", downloader=download
    )
    assert calls == [("JamePeng/llama-cpp-python", SHA_A), ("ggml-org/llama.cpp", SHA_B)]
    assert snapshots[1]["path"] == "vendor/llama.cpp"
    assert (tmp_path / "prepared/vendor/llama.cpp/LICENSE").read_text() == "MIT"


@pytest.mark.parametrize(
    "path", ["root/../../escape", "/absolute/file", "root/C:/evil", "root/a\\b", "root/.git/config"]
)
def test_zip_rejects_unsafe_paths(tmp_path, path):
    archive = tmp_path / "source.zip"
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr(raw_zip_member(path), "bad")
    with zipfile.ZipFile(archive) as z:
        assert z.infolist()[0].orig_filename == path
    with pytest.raises(ValueError, match="Unsafe"):
        extract_zip(archive, tmp_path / "dest")


def test_zip_rejects_symlinks(tmp_path):
    archive = tmp_path / "source.zip"
    with zipfile.ZipFile(archive, "w") as z:
        info = zipfile.ZipInfo("root/link")
        info.external_attr = (stat.S_IFLNK | 0o777) << 16
        z.writestr(info, "../outside")
    with pytest.raises(ValueError, match="link/device"):
        extract_zip(archive, tmp_path / "dest")


def test_submodule_declaration_without_gitlink_fails(tmp_path):
    class API:
        def tree(self, repo, commit):
            return []

    def download(repo, commit, target):
        zip_source(
            target, {".gitmodules": '[submodule "x"]\npath=x\nurl=https://github.com/test/x.git\n'}
        )
        return sha256(target)

    with pytest.raises(ValueError, match="do not agree"):
        materialize(API(), "test/main", SHA_A, tmp_path / "dest", downloader=download)


def test_source_archive_is_deterministic_and_checked_when_unpacked(tmp_path):
    source = fixture_source(tmp_path)
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    source_tar(source, artifact / "source.tar.gz")
    source_tar(source, artifact / "again.tar.gz")
    assert sha256(artifact / "source.tar.gz") == sha256(artifact / "again.tar.gz")
    (artifact / "build-manifest.json").write_text(
        json.dumps(
            {"version": "0.3.49", "source_archive_sha256": sha256(artifact / "source.tar.gz")}
        )
    )
    unpack(artifact, tmp_path / "extracted", "0.3.49")
    assert (tmp_path / "extracted/pyproject.toml").read_text() == METADATA
    with pytest.raises(ValueError, match="must be empty"):
        unpack(artifact, tmp_path / "extracted")
    with pytest.raises(ValueError, match="does not match"):
        unpack(artifact, tmp_path / "wrong-version", "9.9.9")
    (artifact / "source.tar.gz").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        unpack(artifact, tmp_path / "bad")


def test_tar_rejects_traversal_even_with_a_matching_checksum(tmp_path):
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    with tarfile.open(artifact / "source.tar.gz", "w:gz") as tar:
        info = tarfile.TarInfo("../escape")
        info.size = 1
        tar.addfile(info, io.BytesIO(b"x"))
    (artifact / "build-manifest.json").write_text(
        json.dumps(
            {"version": "0.3.49", "source_archive_sha256": sha256(artifact / "source.tar.gz")}
        )
    )
    with pytest.raises(ValueError, match="Unsafe"):
        unpack(artifact, tmp_path / "output")


@pytest.fixture(params=["/", "\\"], ids=["posix-names", "windows-names"])
def zip_name_rules(request, monkeypatch):
    """Exercise ZipInfo's host-specific normalization, not a Windows OS emulation."""
    proxy = SimpleNamespace(**vars(zipfile.os))
    proxy.sep = request.param
    monkeypatch.setattr(zipfile, "os", proxy)
    return request.param


@pytest.mark.parametrize("name", [r"root/a\b", "root/a\x00hidden"])
def test_zip_checks_original_names_before_normalization(tmp_path, zip_name_rules, name):
    archive = tmp_path / "source.zip"
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr(raw_zip_member(name), b"payload")
    with zipfile.ZipFile(archive) as z:
        member = z.infolist()[0]
        assert member.orig_filename == name
        assert member.filename == name.split("\x00", 1)[0].replace(zip_name_rules, "/")
    destination = tmp_path / "dest"
    with pytest.raises(ValueError, match="Unsafe"):
        extract_zip(archive, destination)
    assert not destination.exists()


def test_zip_keeps_safe_nested_paths_with_either_name_rule(tmp_path, zip_name_rules):
    archive = tmp_path / "source.zip"
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr("root/a/b.txt", b"safe contents")
    destination = tmp_path / "dest"
    extract_zip(archive, destination)
    assert (destination / "a/b.txt").read_bytes() == b"safe contents"
