"""Exercise the local-patch mechanism: strict, auditable, fails closed on drift."""

import difflib

import prepare_source
import pytest
from prepare_source import apply_patches, patch_targets, runtime_hashes
from source_helpers import fixture_source


def write_patch(path, before: str, after: str, target="llama_cpp/__init__.py"):
    diff = "".join(
        difflib.unified_diff(
            before.splitlines(keepends=True),
            after.splitlines(keepends=True),
            fromfile=f"a/{target}",
            tofile=f"b/{target}",
        )
    )
    path.write_text(diff, encoding="utf-8")
    return diff


def test_no_patches_directory_means_no_patches(tmp_path):
    source = fixture_source(tmp_path)
    assert apply_patches(source, tmp_path / "does-not-exist") == []


def test_empty_patches_directory_means_no_patches(tmp_path):
    source = fixture_source(tmp_path)
    patches_dir = tmp_path / "patches"
    patches_dir.mkdir()
    assert apply_patches(source, patches_dir) == []


def test_a_clean_patch_is_applied_and_its_provenance_is_recorded(tmp_path):
    source = fixture_source(tmp_path)
    before = (source / "llama_cpp/__init__.py").read_text()
    after = before + "# patched: restore a fix upstream dropped\n"
    patches_dir = tmp_path / "patches"
    patches_dir.mkdir()
    write_patch(patches_dir / "0001-example.patch", before, after)

    applied = apply_patches(source, patches_dir)

    assert (source / "llama_cpp/__init__.py").read_text() == after
    assert len(applied) == 1
    record = applied[0]
    assert record["patch"] == "0001-example.patch"
    assert record["files"] == ["llama_cpp/__init__.py"]
    assert record["pre_sha256"] != record["post_sha256"]
    assert record["post_sha256"] == runtime_hashes(source)


def test_patches_apply_in_filename_order(tmp_path):
    source = fixture_source(tmp_path)
    patches_dir = tmp_path / "patches"
    patches_dir.mkdir()
    step1 = (source / "llama_cpp/__init__.py").read_text()
    step2 = step1 + "# step one\n"
    step3 = step2 + "# step two\n"
    write_patch(patches_dir / "0002-second.patch", step2, step3)
    write_patch(patches_dir / "0001-first.patch", step1, step2)

    applied = apply_patches(source, patches_dir)

    assert [record["patch"] for record in applied] == ["0001-first.patch", "0002-second.patch"]
    assert (source / "llama_cpp/__init__.py").read_text() == step3


def test_a_patch_that_no_longer_applies_fails_closed_without_touching_the_file(tmp_path):
    source = fixture_source(tmp_path)
    before = (source / "llama_cpp/__init__.py").read_text()
    after = before + "# patched\n"
    patches_dir = tmp_path / "patches"
    patches_dir.mkdir()
    write_patch(patches_dir / "0001-example.patch", before, after)

    # Simulate upstream drifting: the context the patch expects is now gone.
    drifted = "# an unrelated upstream change\n" + before
    (source / "llama_cpp/__init__.py").write_text(drifted)

    with pytest.raises(ValueError, match="no longer applies cleanly"):
        apply_patches(source, patches_dir)

    # Fails closed: nothing was written, not even a partial/fuzzy match.
    assert (source / "llama_cpp/__init__.py").read_text() == drifted


def test_a_patch_targeting_a_missing_file_fails_closed(tmp_path):
    source = fixture_source(tmp_path)
    patches_dir = tmp_path / "patches"
    patches_dir.mkdir()
    write_patch(patches_dir / "0001-example.patch", "old\n", "new\n", target="llama_cpp/missing.py")
    with pytest.raises(ValueError, match="no longer has"):
        apply_patches(source, patches_dir)


def test_a_patch_touching_the_same_file_twice_is_rejected(tmp_path):
    patch_path = tmp_path / "0001-duplicate.patch"
    patch_path.write_text(
        "--- a/llama_cpp/x.py\n+++ b/llama_cpp/x.py\n@@ -1 +1 @@\n-a\n+b\n"
        "--- a/llama_cpp/x.py\n+++ b/llama_cpp/x.py\n@@ -1 +1 @@\n-b\n+c\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="more than once"):
        patch_targets(patch_path)


def test_a_patch_declaring_no_target_is_rejected(tmp_path):
    patch_path = tmp_path / "0001-empty.patch"
    patch_path.write_text("not a real diff\n", encoding="utf-8")
    with pytest.raises(ValueError, match="declares no target file"):
        patch_targets(patch_path)


def test_prepare_records_both_upstream_and_shipped_runtime_hashes(tmp_path, monkeypatch):
    """End-to-end: prepare() reports the patched hash as runtime_sha256 (what
    verify_wheels.py checks the wheel against) and keeps the pristine upstream
    hash separately for audit, while still catching accidental drift in the
    metadata-adaptation step.
    """
    from helpers import plan

    def fake_materialize(api, repository, commit, destination):
        source = fixture_source(destination.parent)
        if source != destination:
            source.replace(destination)
        return [
            {"path": "vendor/llama.cpp", "repository": "ggml-org/llama.cpp", "commit": "a" * 40}
        ]

    monkeypatch.setattr(prepare_source, "materialize", fake_materialize)

    patches_dir = tmp_path / "patches"
    patches_dir.mkdir()
    monkeypatch.setattr(prepare_source, "PATCHES_DIR", patches_dir)

    # Write the patch against the exact fixture content prepare() will see.
    probe = fixture_source(tmp_path / "probe")
    before = (probe / "llama_cpp/__init__.py").read_text()
    after = before + "# patched: restore a fix upstream dropped\n"
    write_patch(patches_dir / "0001-example.patch", before, after)

    manifest = prepare_source.prepare(object(), plan(), tmp_path / "prepared")

    assert manifest["applied_patches"][0]["files"] == ["llama_cpp/__init__.py"]
    assert manifest["runtime_sha256"] != manifest["upstream_runtime_sha256"]
    assert (
        manifest["runtime_sha256"]["llama_cpp/__init__.py"]
        == manifest["applied_patches"][0]["post_sha256"]["llama_cpp/__init__.py"]
    )
