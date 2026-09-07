import pytest
from helpers import PublishingAPI, manifest_for, plan, prepared_build, write_wheel
from publish_release import artifact_channel, publish, stage
from release_common import provenance, sha256
from verify_wheels import selector, verify, verify_directory


def test_correct_wheel_metadata_and_unchanged_bindings(tmp_path):
    m, runtime = manifest_for(plan())
    wheel = write_wheel(tmp_path, m, runtime, "cpu", "linux")
    verify(wheel, m, "cpu", "linux")


@pytest.mark.parametrize(
    "kwargs,message",
    [
        ({"metadata_name": "llama-cpp-python"}, "name/version"),
        ({"native": False}, "no native"),
        ({"altered": True}, "binding changed"),
        ({"extra": True}, "additional Python"),
    ],
)
def test_rejects_broken_or_merely_renamed_wheels(tmp_path, kwargs, message):
    m, runtime = manifest_for(plan())
    wheel = write_wheel(tmp_path, m, runtime, "cpu", "linux", **kwargs)
    with pytest.raises(ValueError, match=message):
        verify(wheel, m, "cpu", "linux")


def test_cpu_release_validation_rejects_unrepaired_linux_wheel(tmp_path):
    m, runtime = manifest_for(plan())
    wheel = write_wheel(tmp_path, m, runtime, "cpu", "linux", raw=True)
    with pytest.raises(ValueError, match="filename"):
        verify(wheel, m, "cpu", "linux")
    verify(wheel, m, "cpu", "linux", allow_unrepaired=True)


def test_missing_python_matrix_entry_fails(tmp_path):
    m, runtime = manifest_for(plan())
    write_wheel(tmp_path, m, runtime, "cpu", "linux")
    with pytest.raises(ValueError, match="Expected 6 wheels"):
        verify_directory(tmp_path, m, "cpu", "linux")


def test_python_selectors_and_artifact_channels():
    assert selector(["3.9", "3.13"], "linux") == "cp39-manylinux_x86_64 cp313-manylinux_x86_64"
    assert artifact_channel("guanaco-py-cuda-windows-x64-cu124-py3.13") == ("cu124", "windows")
    assert artifact_channel("guanaco-py-avx2-linux-x64") == ("avx2", "linux")
    with pytest.raises(ValueError):
        artifact_channel("something-else")


def test_all_matrices_verified_before_any_publication(tmp_path):
    p, prepared, artifacts = prepared_build(tmp_path, missing=True)
    with pytest.raises(ValueError, match="Expected 2 wheels"):
        stage(p, prepared, artifacts, tmp_path / "publish")


def test_stages_checksums_and_source_only_in_cpu_release(tmp_path):
    p, prepared, artifacts = prepared_build(tmp_path)
    _, folders = stage(p, prepared, artifacts, tmp_path / "publish")
    assert (folders["cpu"] / "guanaco-source-0.3.49.tar.gz").exists()
    assert not (folders["avx2"] / "guanaco-source-0.3.49.tar.gz").exists()
    for folder in folders.values():
        for line in (folder / "SHA256SUMS").read_text().splitlines():
            digest, name = line.split("  ", 1)
            assert sha256(folder / name) == digest


def test_publish_starts_as_drafts_and_only_publishes_verified_assets(tmp_path):
    p, prepared, artifacts = prepared_build(tmp_path)
    _, folders = stage(p, prepared, artifacts, tmp_path / "publish")
    api = PublishingAPI()
    publish(api, p, folders, uploader=api.upload)
    assert len(api.items) == 2
    assert all(not r["draft"] and provenance(r)["complete"] for r in api.items.values())
    assert all(data["draft"] for method, _, data in api.calls if method == "POST")
    assert api.items[1]["make_latest"] == "true"
    assert api.items[2]["make_latest"] == "false"
    before = len(api.calls)
    publish(
        api, p, folders, uploader=lambda *_: pytest.fail("Must not upload complete releases again")
    )
    assert len(api.calls) == before


def test_upload_failure_leaves_release_draft_and_retryable(tmp_path):
    p, prepared, artifacts = prepared_build(tmp_path)
    _, folders = stage(p, prepared, artifacts, tmp_path / "publish")
    api = PublishingAPI()

    def failed_upload(*args):
        raise RuntimeError("upload interrupted")

    with pytest.raises(RuntimeError, match="interrupted"):
        publish(api, p, folders, uploader=failed_upload)
    assert api.items[1]["draft"]
    assert not provenance(api.items[1])["complete"]
    publish(api, p, folders, uploader=api.upload)
    assert all(not r["draft"] for r in api.items.values())
