"""Regressions for bugs reproduced in the first migration deliverable."""

import copy
import io
import json
import tarfile
import zipfile

import check_upstream
import pytest
import release_common as common
import wheel_index
from archive_utils import extract_tar, extract_zip, portable_path
from helpers import (
    FakeGitHub,
    PublishingAPI,
    manifest_for,
    owned,
    plan,
    prepared_build,
    upstream,
    wheel_data,
    write_wheel,
    zip_contents,
)
from publish_release import preflight, publish, release_body, stage
from site_utils import generated_site
from verify_wheels import verify


@pytest.mark.parametrize("missing", ["WHEEL", "RECORD"])
def test_wheel_without_mandatory_installation_metadata_is_rejected(tmp_path, missing):
    manifest, runtime = manifest_for(plan())
    wheel = write_wheel(tmp_path, manifest, runtime, "cpu", "linux")
    data = wheel_data(wheel)
    data.pop(f"guanaco_py-{manifest['version']}.dist-info/{missing}")
    zip_contents(wheel, data, record=False)
    with pytest.raises(ValueError, match=missing):
        verify(wheel, manifest, "cpu", "linux")


@pytest.mark.parametrize("platform", ["linux", "windows"])
def test_fake_native_library_with_valid_record_is_still_rejected(tmp_path, platform):
    manifest, runtime = manifest_for(plan())
    wheel = write_wheel(tmp_path, manifest, runtime, "cpu", platform)
    data = wheel_data(wheel)
    native = next(name for name in data if name.startswith("llama_cpp/lib/"))
    data[native] = b"not a native library"
    zip_contents(wheel, data)
    with pytest.raises(ValueError, match="native library header"):
        verify(wheel, manifest, "cpu", platform)


def test_library_payload_is_hashed_not_just_the_header(tmp_path):
    manifest, runtime = manifest_for(plan())
    wheel = write_wheel(tmp_path, manifest, runtime, "cpu", "linux")
    data = wheel_data(wheel)
    data["llama_cpp/lib/libllama.so"] += b"damaged body"
    zip_contents(wheel, data, record=False)  # ZIP CRC is valid; RECORD hash is not.
    with pytest.raises(ValueError, match="RECORD integrity"):
        verify(wheel, manifest, "cpu", "linux")


def test_zip_crc_damage_in_a_native_binary_is_detected(tmp_path):
    manifest, runtime = manifest_for(plan())
    wheel = write_wheel(tmp_path, manifest, runtime, "cpu", "linux")
    raw = bytearray(wheel.read_bytes())
    offset = raw.index(b"\x7fELF")
    raw[offset + 55] ^= 1
    wheel.write_bytes(raw)
    with pytest.raises(ValueError, match="Invalid wheel archive"):
        verify(wheel, manifest, "cpu", "linux")


def test_string_false_cannot_mark_a_release_complete():
    release = owned("cpu")
    state = common.provenance(release)
    state["complete"] = "false"
    assert not common.complete(release, state, state["python_versions"])
    release["body"] = release["body"].replace('"complete":true', '"complete":"false"')
    with pytest.raises(ValueError, match="boolean"):
        common.provenance(release)


@pytest.mark.parametrize(
    "field,value",
    [
        ("recipe_commit", "c" * 40),
        ("python_versions", ["3.9"]),
        ("run_url", "https://example.com/another-run"),
    ],
)
def test_prepared_manifest_must_match_the_entire_plan(tmp_path, field, value):
    p, prepared, artifacts = prepared_build(tmp_path)
    path = prepared / "build-manifest.json"
    manifest = json.loads(path.read_text())
    manifest[field] = value
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match=f"release plan: {field}"):
        stage(p, prepared, artifacts, tmp_path / "publish")
    assert not (tmp_path / "publish").exists()


def test_regenerating_index_removes_stale_channel_pages(tmp_path):
    source = tmp_path / "releases.json"
    site = tmp_path / "site"
    source.write_text(json.dumps([owned("cpu"), owned("cu124")]))
    wheel_index.generate(source, site)
    assert (site / "whl/cu124/guanaco-py/index.html").exists()
    source.write_text(json.dumps([owned("cpu")]))
    wheel_index.generate(source, site)
    assert not (site / "whl/cu124").exists()
    assert (site / "whl/cpu/guanaco-py/index.html").exists()


def test_generator_does_not_delete_unowned_files_or_directories(tmp_path):
    source = tmp_path / "releases.json"
    source.write_text("[]")
    site = tmp_path / "site"
    site.mkdir()
    notes = site / "do-not-delete.txt"
    notes.write_text("private notes")
    with pytest.raises(ValueError, match="unowned"):
        wheel_index.generate(source, site)
    assert notes.read_text() == "private notes"
    notes.unlink()
    wheel_index.generate(source, site)
    notes.write_text("still mine")
    with pytest.raises(ValueError, match="unowned"):
        wheel_index.generate(source, site)
    assert notes.read_text() == "still mine"


def test_generator_failure_leaves_previous_site_intact(tmp_path):
    source = tmp_path / "releases.json"
    source.write_text("[]")
    site = tmp_path / "site"
    wheel_index.generate(source, site)
    before = (site / "index.html").read_bytes()
    with pytest.raises(RuntimeError), generated_site(site) as staging:
        (staging / "index.html").write_text("partial")
        raise RuntimeError("rendering interrupted")
    assert (site / "index.html").read_bytes() == before


def test_index_repository_does_not_leak_between_calls(tmp_path):
    source = tmp_path / "releases.json"
    source.write_text("[]")
    wheel_index.generate(source, tmp_path / "first", "example/one")
    wheel_index.generate(source, tmp_path / "second", "example/two")
    assert "https://github.com/example/one" in (tmp_path / "first/index.html").read_text()
    assert "example/one" not in (tmp_path / "second/index.html").read_text()


def test_failed_tar_extraction_does_not_leave_a_partial_destination(tmp_path):
    archive = tmp_path / "source.tar.gz"
    destination = tmp_path / "source"
    with tarfile.open(archive, "w:gz") as tar:
        for name in ("valid.txt", "../escape"):
            info = tarfile.TarInfo(name)
            info.size = 1
            tar.addfile(info, io.BytesIO(b"x"))
    with pytest.raises(ValueError, match="Unsafe"):
        extract_tar(archive, destination)
    assert not destination.exists() and not (tmp_path / "escape").exists()
    with tarfile.open(archive, "w:gz") as tar:
        info = tarfile.TarInfo("valid.txt")
        info.size = 1
        tar.addfile(info, io.BytesIO(b"x"))
    extract_tar(archive, destination)
    assert (destination / "valid.txt").read_bytes() == b"x"


@pytest.mark.parametrize(
    "path",
    [
        "NUL.txt",
        "con",
        "COM1.log",
        "data:secret",
        "a/../b",
        "a//b",
        "a/./b",
        "a/b.",
        "a/b ",
        "a/\x00b",
        ".GIT/config",
        "a\\b",
        "LPT9",
        "a/<bad>",
    ],
)
def test_archive_paths_are_portable_on_windows_and_linux(path):
    with pytest.raises(ValueError, match="Unsafe"):
        portable_path(path)


@pytest.mark.parametrize(
    "members", [["a", "A"], ["Dir/a", "dir/b"], ["file", "file/nested"], ["file/nested", "file"]]
)
def test_zip_rejects_duplicates_aliases_and_file_directory_collisions(tmp_path, members):
    archive = tmp_path / "source.zip"
    with zipfile.ZipFile(archive, "w") as zipped:
        for name in members:
            zipped.writestr("root/" + name, b"x")
    with pytest.raises(ValueError, match="[Cc]ollision|[Cc]olliding"):
        extract_zip(archive, tmp_path / "source")
    assert not (tmp_path / "source").exists()


def test_partial_family_freezes_notes_python_versions_and_cuda_matrix(monkeypatch):
    p = plan()
    p["python_versions"] = ["3.12"]
    original = owned("cpu", base=p)
    monkeypatch.setitem(common.CONFIG, "python_versions", ["3.14"])
    changed = copy.deepcopy(common.CONFIG["cuda"])
    changed["cu124"]["architectures"] = "90"
    monkeypatch.setitem(common.CONFIG, "cuda", changed)
    api = FakeGitHub([upstream(body="Edited upstream changelog")], [original])
    frozen = check_upstream.make_plan(api, p["repository"])
    assert frozen["upstream"]["body"] == p["upstream"]["body"]
    assert frozen["python_versions"] == p["python_versions"]
    assert frozen["cuda"] == p["cuda"]
    assert not api.commit_calls


def test_legacy_markers_keep_their_own_upstream_notes_on_retry():
    p = plan()
    release = owned("cpu")
    state = common.provenance(release)
    state.pop("snapshot")
    release["body"] = (
        release["body"].split("<!-- guanaco-upstream-build-v1", 1)[0]
        + "<!-- guanaco-upstream-build-v1\n"
        + json.dumps(state)
        + "\n-->"
    )
    result = check_upstream.make_plan(
        FakeGitHub([upstream(body="Changed!")], [release]), p["repository"]
    )
    assert result["upstream"]["body"] == p["upstream"]["body"]


def test_release_body_stores_notes_once_and_detects_note_edits():
    p = plan()
    p["upstream"]["body"] = "Unique upstream changelog" * 500
    body = release_body(p, "cpu", True)
    assert body.count(p["upstream"]["body"]) == 1
    assert (
        common.provenance({"tag_name": "v0.3.49", "body": body})["snapshot"]["upstream"]["body"]
        == p["upstream"]["body"]
    )
    with pytest.raises(ValueError, match="snapshot checksum"):
        common.provenance({"tag_name": "v0.3.49", "body": body.replace("Unique", "Modified", 1)})


def test_duplicate_drafts_are_not_silently_collapsed():
    draft = owned("cpu", draft=True, finished=False)
    with pytest.raises(ValueError, match="Duplicate releases"):
        check_upstream.make_plan(
            FakeGitHub(existing=[draft, copy.deepcopy(draft)]), plan()["repository"]
        )


def test_publisher_checks_later_channels_before_any_mutation(tmp_path):
    p, prepared, artifacts = prepared_build(tmp_path)
    _, folders = stage(p, prepared, artifacts, tmp_path / "publish")
    api = PublishingAPI()
    api.items[1] = {"id": 1, "tag_name": "v0.3.49-avx2", "body": "unrelated manual release"}
    with pytest.raises(ValueError, match="unrelated"):
        publish(api, p, folders, uploader=lambda *_: pytest.fail("Unexpected upload"))
    assert not api.calls and len(api.items) == 1


def test_published_damaged_release_is_never_reopened_or_clobbered():
    p = plan(["cpu"])
    damaged = owned("cpu")
    damaged["assets"].pop()
    api = PublishingAPI()
    api.items[1] = damaged
    with pytest.raises(ValueError, match="Published release.*incomplete"):
        preflight(api, p, ["cpu"])
    assert not api.calls and not damaged["draft"]
    with pytest.raises(ValueError, match="Published release.*incomplete"):
        check_upstream.make_plan(FakeGitHub(existing=[damaged]), p["repository"])


def test_existing_tag_cannot_point_to_another_recipe(monkeypatch):
    api = PublishingAPI()
    monkeypatch.setattr(api, "tag_commit", lambda *_: "c" * 40)
    with pytest.raises(ValueError, match="another build recipe"):
        preflight(api, plan(["cpu"]), ["cpu"])
    assert not api.calls


def test_draft_from_another_recipe_is_not_reused():
    p = plan(["cpu"])
    api = PublishingAPI()
    previous = copy.deepcopy(p)
    previous["recipe_commit"] = "c" * 40
    api.items[1] = owned("cpu", draft=True, finished=False, base=previous)
    with pytest.raises(ValueError, match="another build recipe"):
        preflight(api, p, ["cpu"])
    assert not api.calls


@pytest.mark.parametrize(
    "name",
    [
        "guanaco_py-0.3.49.data/purelib/llama_cpp/override.py",
        "other.data/platlib/llama_cpp/__init__.py",
        "startup.pth",
        "llama_cpp/hidden.pyc",
        "outside_package.py",
    ],
)
def test_additional_python_cannot_bypass_the_source_hash_check(tmp_path, name):
    manifest, runtime = manifest_for(plan())
    wheel = write_wheel(tmp_path, manifest, runtime, "cpu", "linux")
    data = wheel_data(wheel)
    data[name] = b"import unexpected_runtime_code"
    zip_contents(wheel, data)
    with pytest.raises(ValueError, match="installation scheme|Python"):
        verify(wheel, manifest, "cpu", "linux")


def test_an_upstream_quoted_marker_is_not_our_own_provenance():
    p = plan()
    p["upstream"]["body"] = (
        "Notes quote this comment:\n<!-- guanaco-upstream-build-v1\nnot metadata\n-->"
    )
    release = {"tag_name": "v0.3.49", "body": release_body(p, "cpu", True)}
    assert common.provenance(release)["snapshot"]["upstream"]["body"] == p["upstream"]["body"]


def test_mixed_note_and_footer_line_endings_are_preserved():
    p = plan()
    p["upstream"]["body"] += common.PROVENANCE_SEPARATOR + "This separator belongs to upstream."
    body = release_body(p, "cpu", True)
    notes = p["upstream"]["body"]
    body = notes + body[len(notes) :].replace("\n", "\r\n")
    assert (
        common.provenance({"tag_name": "v0.3.49", "body": body})["snapshot"]["upstream"]["body"]
        == notes
    )
