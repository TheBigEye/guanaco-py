"""Exercise entry points and late failure paths without executing remote writes."""

import json
import sys
import zipfile
from pathlib import Path

import archive_utils
import check_upstream
import configure_build
import prepare_source
import publish_release
import pytest
import release_common as common
import unpack_source
import validate_receipts
import verify_wheels
import wheel_index
from helpers import (
    SHA_A,
    SHA_B,
    FakeGitHub,
    PublishingAPI,
    manifest_for,
    owned,
    plan,
    prepared_build,
    receipts_for,
    wheel_data,
    write_wheel,
    zip_contents,
)
from source_helpers import fixture_source


def argv(monkeypatch, script, *arguments):
    monkeypatch.setattr(sys, "argv", [script, *map(str, arguments)])


def test_discovery_cli_writes_plan_summary_and_safe_matrix_outputs(tmp_path, monkeypatch):
    monkeypatch.setattr(check_upstream, "GitHub", FakeGitHub)
    monkeypatch.setenv("GITHUB_SHA", SHA_B)
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(tmp_path / "summary"))
    argv(
        monkeypatch,
        "check_upstream",
        "--output",
        tmp_path / "plan.json",
        "--github-output",
        tmp_path / "outputs",
    )
    check_upstream.main()
    result = json.loads((tmp_path / "plan.json").read_text())
    assert result["recipe_commit"] == SHA_B
    assert "Missing channels" in (tmp_path / "summary").read_text()
    values = dict(line.split("=", 1) for line in (tmp_path / "outputs").read_text().splitlines())
    matrix = json.loads(values["publish_matrix"])
    assert len(matrix["include"]) == 9
    assert matrix["include"][0] == {"channel": "cpu", "artifact_pattern": "guanaco-py-cpu-*"}
    assert matrix["include"][-1]["artifact_pattern"] == "guanaco-py-cuda-*-cu131-py*"


@pytest.mark.parametrize("kind", ["cpu", "cuda", "docker"])
def test_build_configuration_cli(kind, tmp_path, monkeypatch):
    path = tmp_path / "manifest.json"
    common.write_json(path, plan())
    monkeypatch.setenv("GITHUB_OUTPUT", str(tmp_path / "outputs"))
    if kind == "docker":
        argv(
            monkeypatch,
            "configure_build",
            kind,
            "--repository",
            "test/repo",
            "--version",
            "0.3.49",
            "--promote-latest",
        )
    else:
        argv(
            monkeypatch,
            "configure_build",
            kind,
            "--manifest",
            path,
            "--channel",
            "cpu" if kind == "cpu" else "cu124",
            "--platform",
            "linux",
            "--version",
            "0.3.49",
        )
    configure_build.main()
    assert (tmp_path / "outputs").read_text()


def test_configuration_cli_missing_arguments_and_version_mismatch(tmp_path, monkeypatch):
    argv(monkeypatch, "configure_build", "cpu")
    with pytest.raises(SystemExit):
        configure_build.main()
    common.write_json(tmp_path / "manifest.json", plan())
    argv(
        monkeypatch,
        "configure_build",
        "cpu",
        "--manifest",
        tmp_path / "manifest.json",
        "--channel",
        "cpu",
        "--version",
        "0.3.50",
    )
    with pytest.raises(ValueError, match="does not match"):
        configure_build.main()


def test_gate_cli_and_publisher_dry_run(tmp_path, monkeypatch, capsys):
    p, prepared, artifacts = prepared_build(tmp_path)
    common.write_json(tmp_path / "plan.json", p)
    receipts_for(p, prepared, artifacts, tmp_path / "receipts")
    argv(
        monkeypatch,
        "validate_receipts",
        "--plan",
        tmp_path / "plan.json",
        "--prepared",
        prepared,
        "--receipts",
        tmp_path / "receipts",
        "--output",
        tmp_path / "gate.json",
    )
    validate_receipts.main()
    assert len(json.loads((tmp_path / "gate.json").read_text())["channels"]) == 2
    argv(
        monkeypatch,
        "publish_release",
        "--plan",
        tmp_path / "plan.json",
        "--prepared",
        prepared,
        "--artifacts",
        artifacts,
        "--output",
        tmp_path / "staged",
        "--gate",
        tmp_path / "gate.json",
    )
    publish_release.main()
    assert "DRY RUN" in capsys.readouterr().out


def test_publisher_preflight_cli_is_read_only(tmp_path, monkeypatch):
    common.write_json(tmp_path / "plan.json", plan(["cpu"]))
    api = PublishingAPI()
    monkeypatch.setattr(publish_release, "GitHub", lambda: api)
    argv(monkeypatch, "publish_release", "--plan", tmp_path / "plan.json", "--preflight")
    publish_release.main()
    assert not api.calls
    argv(
        monkeypatch, "publish_release", "--plan", tmp_path / "plan.json", "--preflight", "--publish"
    )
    with pytest.raises(SystemExit):
        publish_release.main()
    argv(monkeypatch, "publish_release", "--plan", tmp_path / "plan.json")
    with pytest.raises(SystemExit):
        publish_release.main()


def test_publisher_cli_requires_gate_and_repository_binding(tmp_path, monkeypatch):
    p, prepared, artifacts = prepared_build(tmp_path)
    common.write_json(tmp_path / "plan.json", p)
    base = [
        "--plan",
        tmp_path / "plan.json",
        "--prepared",
        prepared,
        "--artifacts",
        artifacts,
        "--output",
        tmp_path / "staged",
        "--publish",
    ]
    argv(monkeypatch, "publish_release", *base, "--channel", "cpu")
    with pytest.raises(SystemExit):
        publish_release.main()
    monkeypatch.setenv("GITHUB_REPOSITORY", "someone/else")
    argv(monkeypatch, "publish_release", *base)
    with pytest.raises(ValueError, match="repository does not match"):
        publish_release.main()


def test_publisher_cli_only_calls_writable_client_after_validation(tmp_path, monkeypatch):
    p, prepared, artifacts = prepared_build(tmp_path)
    common.write_json(tmp_path / "plan.json", p)
    monkeypatch.setenv("GITHUB_REPOSITORY", p["repository"])
    monkeypatch.setenv("GITHUB_OUTPUT", str(tmp_path / "outputs"))
    api, calls = PublishingAPI(), []
    monkeypatch.setattr(publish_release, "GitHub", lambda **kwargs: (calls.append(kwargs), api)[1])
    monkeypatch.setattr(
        publish_release, "publish", lambda client, p, folders: calls.append(sorted(folders))
    )
    argv(
        monkeypatch,
        "publish_release",
        "--plan",
        tmp_path / "plan.json",
        "--prepared",
        prepared,
        "--artifacts",
        artifacts,
        "--output",
        tmp_path / "staged",
        "--publish",
    )
    publish_release.main()
    assert calls == [{"writable": True}, ["avx2", "cpu"]]
    assert "published=true" in (tmp_path / "outputs").read_text()


def test_wheel_cli_selectors_and_single_receipt(tmp_path, monkeypatch, capsys):
    manifest, runtime = manifest_for(plan())
    manifest["source_archive_sha256"] = "a" * 64
    common.write_json(tmp_path / "manifest.json", manifest)
    argv(
        monkeypatch,
        "verify_wheels",
        "--manifest",
        tmp_path / "manifest.json",
        "--platform",
        "linux",
        "--selectors",
    )
    verify_wheels.main()
    assert "cp314-manylinux_x86_64" in capsys.readouterr().out
    write_wheel(tmp_path / "wheelhouse", manifest, runtime, "cu124", "linux")
    base = [
        "--manifest",
        tmp_path / "manifest.json",
        "--platform",
        "linux",
        "--channel",
        "cu124",
        "--directory",
        tmp_path / "wheelhouse",
        "--single",
    ]
    argv(
        monkeypatch,
        "verify_wheels",
        *base,
        "--python",
        "3.13",
        "--receipt",
        tmp_path / "receipt.json",
    )
    verify_wheels.main()
    assert json.loads((tmp_path / "receipt.json").read_text())["channel"] == "cu124"
    argv(monkeypatch, "verify_wheels", *base, "--python", "3.12")
    with pytest.raises(ValueError, match="Python version"):
        verify_wheels.main()
    argv(
        monkeypatch, "verify_wheels", *base, "--unrepaired", "--receipt", tmp_path / "receipt.json"
    )
    with pytest.raises(SystemExit):
        verify_wheels.main()


def test_wheel_cli_requires_complete_matrix_and_explicit_directory(tmp_path, monkeypatch):
    manifest, runtime = manifest_for(plan())
    common.write_json(tmp_path / "manifest.json", manifest)
    write_wheel(tmp_path / "wheelhouse", manifest, runtime, "cpu", "linux")
    base = ["--manifest", tmp_path / "manifest.json", "--platform", "linux"]
    argv(monkeypatch, "verify_wheels", *base)
    with pytest.raises(SystemExit):
        verify_wheels.main()
    argv(monkeypatch, "verify_wheels", *base, "--directory", tmp_path / "wheelhouse")
    with pytest.raises(ValueError, match="Expected 6 wheels"):
        verify_wheels.main()
    argv(monkeypatch, "verify_wheels", *base, "--directory", tmp_path / "empty", "--single")
    with pytest.raises(ValueError, match="exactly one"):
        verify_wheels.main()


def fake_materialize(api, repository, commit, destination):
    # fixture_source creates its own subdirectory, so move the files to the
    # materializer's requested root without involving archives or the network.
    source = fixture_source(destination.parent)
    if source != destination:
        source.replace(destination)
    return [{"path": "vendor/llama.cpp", "repository": "ggml-org/llama.cpp", "commit": SHA_A}]


def test_prepare_and_unpack_cli_preserve_source_and_checksums(tmp_path, monkeypatch):
    common.write_json(tmp_path / "plan.json", plan())
    monkeypatch.setattr(prepare_source, "materialize", fake_materialize)
    argv(
        monkeypatch,
        "prepare_source",
        "--plan",
        tmp_path / "plan.json",
        "--output",
        tmp_path / "prepared",
    )
    prepare_source.main()
    manifest = json.loads((tmp_path / "prepared/build-manifest.json").read_text())
    assert manifest["native_commit"] == SHA_A
    assert manifest["source_archive_sha256"] == common.sha256(tmp_path / "prepared/source.tar.gz")
    argv(
        monkeypatch,
        "unpack_source",
        tmp_path / "prepared",
        tmp_path / "source",
        "--version",
        "0.3.49",
    )
    unpack_source.main()
    assert '__version__ = "0.3.49"' in (tmp_path / "source/llama_cpp/__init__.py").read_text()


def test_failed_preparation_does_not_leave_mixed_artifacts(tmp_path, monkeypatch):
    monkeypatch.setattr(prepare_source, "materialize", fake_materialize)

    def broken_tar(source, path):
        path.write_bytes(b"partial archive")
        raise OSError("simulated full disk")

    monkeypatch.setattr(prepare_source, "source_tar", broken_tar)
    with pytest.raises(OSError, match="full disk"):
        prepare_source.prepare(FakeGitHub(), plan(), tmp_path / "prepared")
    assert not (tmp_path / "prepared").exists()


def test_archive_downloader_pins_the_commit_and_rejects_branches(tmp_path, monkeypatch):
    calls = []

    def download(url, destination):
        calls.append(url)
        destination.write_bytes(b"archive")
        return "a" * 64

    monkeypatch.setattr(prepare_source, "download", download)
    assert prepare_source.download_archive("test/repo", SHA_A, tmp_path / "archive") == "a" * 64
    assert calls == [f"https://codeload.github.com/test/repo/zip/{SHA_A}"]
    with pytest.raises(ValueError, match="immutable SHA"):
        prepare_source.download_archive("test/repo", "main", tmp_path / "archive")


def test_index_cli_accepts_flat_releases_and_rejects_api_errors(tmp_path, monkeypatch):
    (tmp_path / "releases.json").write_text(json.dumps([owned("cpu")]))
    argv(monkeypatch, "index", tmp_path / "releases.json", tmp_path / "site")
    wheel_index.main()
    assert (tmp_path / "site/index.html").exists()
    for malformed in ({"message": "API error"}, [["not a release"]]):
        (tmp_path / "releases.json").write_text(json.dumps(malformed))
        with pytest.raises(ValueError):
            wheel_index.generate(tmp_path / "releases.json", tmp_path / "site")


@pytest.mark.parametrize(
    "mutation,message",
    [
        ("wheel_version", "Wheel-Version"),
        ("purelib", "pure-Python"),
        ("tag", "compatibility tags"),
        ("duplicate_name", "repeated wheel header"),
        ("self_dependency", "upstream distribution"),
        ("record_hash", "RECORD hash"),
        ("record_missing", "RECORD does not describe"),
        ("record_duplicate", "duplicate RECORD"),
        ("metadata_location", "renamed without rebuilding"),
    ],
)
def test_wheel_metadata_and_record_failures(tmp_path, mutation, message):
    manifest, runtime = manifest_for(plan())
    wheel = write_wheel(tmp_path, manifest, runtime, "cpu", "linux")
    data = wheel_data(wheel)
    info = "guanaco_py-0.3.49.dist-info/"
    record = True
    if mutation == "wheel_version":
        data[info + "WHEEL"] = data[info + "WHEEL"].replace(b"1.0", b"2.0")
    elif mutation == "purelib":
        data[info + "WHEEL"] = data[info + "WHEEL"].replace(b"false", b"true")
    elif mutation == "tag":
        data[info + "WHEEL"] = data[info + "WHEEL"].replace(b"cp313-cp313", b"cp312-cp312")
    elif mutation == "duplicate_name":
        data[info + "METADATA"] += b"Name: guanaco-py\n"
    elif mutation == "self_dependency":
        data[info + "METADATA"] += b"Requires-Dist: llama__cpp-python[server]\n"
    elif mutation == "record_hash":
        data[info + "RECORD"] = data[info + "RECORD"].replace(b"sha256=", b"md5=")
        record = False
    elif mutation == "record_missing":
        data[info + "RECORD"] = b""
        record = False
    elif mutation == "record_duplicate":
        data[info + "RECORD"] *= 2
        record = False
    elif mutation == "metadata_location":
        data["other.dist-info/METADATA"] = data.pop(info + "METADATA")
    zip_contents(wheel, data, record=record)
    with pytest.raises(ValueError, match=message):
        verify_wheels.verify(wheel, manifest, "cpu", "linux")


def test_upload_digest_or_inventory_failure_never_publishes(tmp_path):
    p, prepared, artifacts = prepared_build(tmp_path)
    _, folders = publish_release.stage(p, prepared, artifacts, tmp_path / "staged")
    api = PublishingAPI()

    def bad_upload(tag, files):
        api.upload(tag, files)
        api.release("", tag)["assets"][0]["digest"] = "sha256:" + "0" * 64

    with pytest.raises(ValueError, match="digest mismatch"):
        publish_release.publish(api, p, folders, uploader=bad_upload)
    assert all(release["draft"] for release in api.items.values())
    assert not any(
        method == "PATCH" and data.get("draft") is False for method, _, data in api.calls
    )


def test_bad_upload_sizes_and_extra_assets_are_rejected(tmp_path):
    path = tmp_path / "wheel"
    path.write_bytes(b"wheel")
    with pytest.raises(ValueError, match="Unexpected or duplicate"):
        publish_release.check_uploaded({"assets": []}, [path])
    with pytest.raises(ValueError, match="Incomplete upload"):
        publish_release.check_uploaded(
            {"assets": [{"name": "wheel", "size": 1, "state": "uploaded"}]}, [path]
        )


@pytest.mark.parametrize("value", ["0.03.49", "٠.3.49", "0.3.49\n", None])
def test_stable_versions_are_ascii_and_canonical(value):
    with pytest.raises(ValueError):
        common.version_key(value)


@pytest.mark.parametrize("versions", [[], ["3.8"], ["3.13", "3.13"], [3.13], "3.13"])
def test_python_matrix_validation(versions):
    with pytest.raises(ValueError):
        common.validate_python_versions(versions)


@pytest.mark.parametrize("change", ["channels", "toolkit", "architectures", "legacy", "cuda_type"])
def test_invalid_frozen_matrices_fail_closed(change):
    p = plan()
    if change == "channels":
        p["channels"].pop()
    elif change == "toolkit":
        p["cuda"]["cu124"]["toolkit"] = "12.8.1"
    elif change == "architectures":
        p["cuda"]["cu124"]["architectures"] = "90;$(echo injected)"
    elif change == "legacy":
        p["cuda"]["cu124"]["legacy_msvc"] = "false"
    else:
        p["cuda"] = []
    with pytest.raises(ValueError):
        common.validate_build_matrix(p)


def test_missing_native_submodule_prevents_source_publication(tmp_path, monkeypatch):
    monkeypatch.setattr(prepare_source, "materialize", lambda *_: [])
    with pytest.raises(ValueError, match="expected llama.cpp"):
        prepare_source.prepare(FakeGitHub(), plan(), tmp_path / "prepared")
    assert not (tmp_path / "prepared").exists()


def test_zip_limit_and_multi_root_checks_are_enforced(tmp_path, monkeypatch):
    archive = tmp_path / "source.zip"
    with zipfile.ZipFile(archive, "w") as zipped:
        zipped.writestr("one/file", b"x")
        zipped.writestr("two/file", b"x")
    with pytest.raises(ValueError, match="single GitHub archive root"):
        archive_utils.extract_zip(archive, tmp_path / "source")
    monkeypatch.setattr(archive_utils, "MAX_UNPACKED_BYTES", 1)
    with pytest.raises(ValueError, match="extraction limits"):
        archive_utils.extract_zip(archive, tmp_path / "source")


@pytest.mark.parametrize("filename", ["model.gguf", "obsolete.bin"])
def test_optional_model_downloader_requires_explicit_gguf(tmp_path, monkeypatch, filename):
    import importlib.util
    from types import SimpleNamespace

    path = Path(__file__).resolve().parents[1] / "docker/open_llama/hug_model.py"
    spec = importlib.util.spec_from_file_location("model_helper", path)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    calls = []
    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(hf_hub_download=lambda **kwargs: calls.append(kwargs) or "model.gguf"),
    )
    argv(
        monkeypatch,
        "hug_model",
        "example/models",
        filename,
        "--revision",
        SHA_A,
        "--directory",
        tmp_path,
    )
    if filename.endswith(".gguf"):
        helper.main()
        assert calls == [
            {
                "repo_id": "example/models",
                "filename": filename,
                "revision": SHA_A,
                "local_dir": str(tmp_path),
            }
        ]
    else:
        with pytest.raises(SystemExit):
            helper.main()
        assert not calls
