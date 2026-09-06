"""Frozen build options, bounded publication, and Docker's pinned/checksummed installer."""

import importlib.util
import json
import shlex
import shutil
import sys
from pathlib import Path

import configure_build
import publish_release
import pytest
import validate_receipts as gate
from helpers import manifest_for, plan, prepared_build, receipts_for, write_wheel
from release_common import outputs, sha256, write_json

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("docker_fetch", ROOT / "docker/fetch_release.py")
docker_fetch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(docker_fetch)


@pytest.mark.parametrize("channel", ["cpu", "avx2"])
@pytest.mark.parametrize("platform", ["linux", "windows"])
def test_cpu_pipeline_only_changes_the_intended_simd_flags(channel, platform):
    options = configure_build.cpu_options(plan(), channel, platform)
    cmake = shlex.split(options["cibw_environment"])[-1].removeprefix("CMAKE_ARGS=")
    flags = dict(flag.removeprefix("-DGGML_").split("=") for flag in shlex.split(cmake))
    assert all(flags[key] == "OFF" for key in configure_build.CPU_OFF)
    assert all(
        flags[key] == ("ON" if channel == "avx2" else "OFF") for key in configure_build.CPU_SIMD
    )
    assert len(options["build"].split()) == 6
    assert options["artifact"] == f"guanaco-py-{channel}-{platform}-x64"
    assert ("CC=/usr/bin/gcc" in options["cibw_environment"]) == (platform == "linux")


def test_cuda_options_come_from_manifest_not_current_config():
    manifest = plan()
    manifest["python_versions"] = ["3.12"]
    manifest["cuda"]["cu124"]["architectures"] = "80;90"
    options = configure_build.cuda_options(manifest, "cu124")
    assert options["python"] == ["3.12"]
    assert options["version"] == "12.4.1"
    assert "-DCMAKE_CUDA_ARCHITECTURES=80;90" in shlex.split(options["cmake_windows"])
    assert "-DCMAKE_EXE_LINKER_FLAGS=-L/usr/local/cuda/lib64/stubs -lcuda" in shlex.split(
        options["cmake_linux"]
    )
    assert options["cuda_flags"] == ""
    assert (
        "--allow-unsupported-compiler"
        in configure_build.cuda_options(manifest, "cu121")["cuda_flags"]
    )
    with pytest.raises(ValueError, match="prepared manifest"):
        configure_build.cuda_options(manifest, "cu999")
    with pytest.raises(ValueError, match="CPU/AVX2"):
        configure_build.cpu_options(manifest, "cu124", "linux")


def test_receipts_cover_108_wheels_in_88_build_jobs():
    specs = gate.artifact_specs(plan())
    assert len(specs) == 88
    assert sum(len(job["python_versions"]) for job in specs.values()) == 108
    assert specs["guanaco-py-cuda-windows-x64-cu131-py3.14"]["python_versions"] == ["3.14"]


def test_full_matrix_gate_and_bounded_single_channel_staging(tmp_path):
    p, prepared, artifacts = prepared_build(tmp_path)
    manifest = receipts_for(p, prepared, artifacts, tmp_path / "receipts")
    validation = gate.validate_receipts(p, manifest, tmp_path / "receipts")
    assert {key: len(value) for key, value in validation["channels"].items()} == {
        "cpu": 4,
        "avx2": 4,
    }
    subset = tmp_path / "cpu-artifacts"
    for directory in artifacts.glob("guanaco-py-cpu-*"):
        shutil.copytree(directory, subset / directory.name)
    _, folders = publish_release.stage(
        p, prepared, subset, tmp_path / "staged", channel="cpu", gate=validation
    )
    assert set(folders) == {"cpu"}
    original = next(subset.rglob("*.whl"))
    assert sha256(folders["cpu"] / original.name) == sha256(original)


@pytest.mark.parametrize(
    "change,message",
    [
        ("missing", "Missing validation receipt"),
        ("source", "identity mismatch"),
        ("duplicate", "Incomplete wheel matrix"),
        ("hash", "Invalid wheel checksum"),
    ],
)
def test_missing_mixed_or_malformed_receipts_block_the_global_gate(tmp_path, change, message):
    p, prepared, artifacts = prepared_build(tmp_path)
    manifest = receipts_for(p, prepared, artifacts, tmp_path / "receipts")
    path = next((tmp_path / "receipts").glob("*.json"))
    data = json.loads(path.read_text())
    if change == "missing":
        path.unlink()
    else:
        if change == "source":
            data["source_archive_sha256"] = "c" * 64
        elif change == "duplicate":
            data["wheels"][1] = data["wheels"][0]
        else:
            data["wheels"][0]["sha256"] = "not a hash"
        path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match=message):
        gate.validate_receipts(p, manifest, tmp_path / "receipts")


@pytest.mark.parametrize("change", ["plan", "channel", "matrix"])
def test_publisher_rejects_incomplete_or_foreign_global_gate(tmp_path, change):
    p, prepared, artifacts = prepared_build(tmp_path)
    manifest = receipts_for(p, prepared, artifacts, tmp_path / "receipts")
    validation = gate.validate_receipts(p, manifest, tmp_path / "receipts")
    if change == "plan":
        validation["plan"] = {**p, "recipe_commit": "d" * 40}
    elif change == "channel":
        validation["channels"].pop("avx2")
    else:
        validation["channels"]["cpu"].popitem()
    with pytest.raises(ValueError, match="Global validation gate"):
        publish_release.stage(p, prepared, artifacts, tmp_path / "staged", gate=validation)
    assert not (tmp_path / "staged").exists()


def test_receipt_hash_is_rechecked_before_publication(tmp_path):
    p, prepared, artifacts = prepared_build(tmp_path)
    manifest = receipts_for(p, prepared, artifacts, tmp_path / "receipts")
    validation = gate.validate_receipts(p, manifest, tmp_path / "receipts")
    next(iter(validation["channels"]["cpu"].values()))["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="globally validated receipt"):
        publish_release.stage(p, prepared, artifacts, tmp_path / "staged", gate=validation)
    assert not (tmp_path / "staged").exists()


def test_cuda_artifact_cannot_claim_another_python_version(tmp_path):
    p, prepared, artifacts = prepared_build(tmp_path)
    p["missing_channels"] = ["cu124"]
    manifest = json.loads((prepared / "build-manifest.json").read_text())
    manifest["missing_channels"] = ["cu124"]
    write_json(prepared / "build-manifest.json", manifest)
    shutil.rmtree(artifacts)
    _, runtime = manifest_for(p)
    write_wheel(
        artifacts / "guanaco-py-cuda-linux-x64-cu124-py3.12",
        manifest,
        runtime,
        "cu124",
        "linux",
        cp="cp313",
    )
    with pytest.raises(ValueError, match="Python mismatch"):
        publish_release.stage(p, prepared, artifacts, tmp_path / "staged")


def test_prepared_checksums_and_channel_selection_are_enforced(tmp_path):
    p, prepared, artifacts = prepared_build(tmp_path)
    with pytest.raises(ValueError, match="absent from"):
        publish_release.stage(p, prepared, artifacts, tmp_path / "staged", channel="cu124")
    (prepared / "packaging.patch").write_text("changed")
    with pytest.raises(ValueError, match="checksum mismatch"):
        publish_release.stage(p, prepared, artifacts, tmp_path / "staged")


def test_staging_falls_back_to_copy_across_filesystems(tmp_path, monkeypatch):
    source, destination = tmp_path / "source", tmp_path / "destination"
    source.write_bytes(b"wheel")
    monkeypatch.setattr(
        publish_release.os, "link", lambda *_: (_ for _ in ()).throw(OSError("cross-device"))
    )
    publish_release.link_or_copy(source, destination)
    assert destination.read_bytes() == b"wheel"


def test_image_tags_validate_input_and_never_promote_a_backfill_implicitly():
    assert (
        configure_build.image_tags("TheBigEye/guanaco-py", "0.3.49", False)
        == "ghcr.io/thebigeye/guanaco-py:v0.3.49"
    )
    assert configure_build.image_tags("TheBigEye/guanaco-py", "0.3.49", True).endswith(
        ",ghcr.io/thebigeye/guanaco-py:latest"
    )
    with pytest.raises(ValueError):
        configure_build.image_tags("../repo", "1.2.3", False)
    with pytest.raises(ValueError):
        configure_build.image_tags("test/repo", "01.2.3", False)


def test_workflow_outputs_are_atomic_against_newline_injection(tmp_path):
    path = tmp_path / "github-output"
    with pytest.raises(ValueError, match="single line"):
        outputs(path, first="safe", evil="bad\nsecret=oops")
    assert not path.exists()
    outputs(path, enabled=True, disabled=False, matrix=["a", "b"])
    assert path.read_text() == 'enabled=true\ndisabled=false\nmatrix=["a","b"]\n'
    with pytest.raises(ValueError, match="output key"):
        outputs(path, **{"bad key": "value"})


@pytest.mark.parametrize(
    "repository,version,channel",
    [
        ("../repo", "1.2.3", "cpu"),
        ("test/repo", "01.2.3", "cpu"),
        ("test/repo", "１.2.3", "cpu"),
        ("test/repo", "1.2.3", "cu124;echo"),
    ],
)
def test_docker_url_rejects_unsafe_repository_version_or_channel(repository, version, channel):
    with pytest.raises(ValueError):
        docker_fetch.release_base(repository, version, channel)


def test_docker_checksum_inventory_parser(tmp_path):
    path = tmp_path / "SHA256SUMS"
    path.write_text("a" * 64 + "  wheel.whl\n")
    assert docker_fetch.read_checksums(path) == {"wheel.whl": "a" * 64}
    for data in ("", "bad  wheel.whl\n", "a" * 64 + "  ../escape\n", path.read_text() * 2):
        path.write_text(data)
        with pytest.raises(ValueError):
            docker_fetch.read_checksums(path)


def test_docker_wheel_uses_exact_python_and_platform(monkeypatch):
    monkeypatch.setattr(docker_fetch.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(docker_fetch.sys, "platform", "linux")
    cp = f"cp{sys.version_info.major}{sys.version_info.minor}"
    assert (
        docker_fetch.wheel_name("0.3.49", "cpu")
        == f"guanaco_py-0.3.49-{cp}-{cp}-manylinux_2_34_x86_64.whl"
    )
    assert docker_fetch.wheel_name("0.3.49", "cu124").endswith("linux_x86_64.whl")
    monkeypatch.setattr(docker_fetch.platform, "machine", lambda: "aarch64")
    with pytest.raises(ValueError, match="linux/amd64"):
        docker_fetch.wheel_name("0.3.49", "cpu")


def test_docker_checked_asset_passes_checksum_before_replacing_file(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(
        docker_fetch, "download", lambda *args, **kwargs: calls.append((args, kwargs))
    )
    docker_fetch.checked_asset(
        "https://example.com", "wheel.whl", {"wheel.whl": "a" * 64}, tmp_path / "wheel"
    )
    assert calls[0][1]["expected_sha256"] == "a" * 64
    with pytest.raises(ValueError, match="No release checksum"):
        docker_fetch.checked_asset("https://example.com", "wheel.whl", {}, tmp_path / "wheel")


@pytest.mark.parametrize("mode", ["source", "wheel"])
def test_docker_cli_downloads_from_release_and_installs_local_wheel_only(
    tmp_path, monkeypatch, mode
):
    monkeypatch.setattr(docker_fetch.platform, "machine", lambda: "amd64")
    monkeypatch.setattr(docker_fetch.sys, "platform", "linux")
    wheel = docker_fetch.wheel_name("0.3.49", "cu124")
    hashes = dict.fromkeys((wheel, "guanaco-source-0.3.49.tar.gz", "guanaco-build.json"), "a" * 64)
    downloads, assets, commands = [], [], []

    def download(url, destination, **kwargs):
        downloads.append(url)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text("".join(f"{digest}  {name}\n" for name, digest in hashes.items()))

    monkeypatch.setattr(docker_fetch, "download", download)
    monkeypatch.setattr(
        docker_fetch,
        "checked_asset",
        lambda base, name, hashes, destination: assets.append((base, name)),
    )
    monkeypatch.setattr(
        docker_fetch.subprocess, "run", lambda args, **kwargs: commands.append(args)
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "fetch_release.py",
            mode,
            "--version",
            "0.3.49",
            "--channel",
            "cu124",
            "--directory",
            str(tmp_path / "release"),
        ],
    )
    docker_fetch.main()
    tag = "v0.3.49" if mode == "source" else "v0.3.49-cu124"
    assert f"/releases/download/{tag}/SHA256SUMS" in downloads[0]
    if mode == "source":
        assert [name for _, name in assets] == [
            "guanaco-source-0.3.49.tar.gz",
            "guanaco-build.json",
        ]
        assert not commands
    else:
        assert assets[0][1] == wheel
        assert commands[0][-1] == str(tmp_path / "release" / wheel) + "[server]"
        assert "llama-cpp-python" not in " ".join(commands[0])


@pytest.mark.parametrize("empty_gate", [{}, [], False, 0])
def test_an_empty_or_false_gate_cannot_bypass_global_validation(tmp_path, empty_gate):
    p, prepared, artifacts = prepared_build(tmp_path)
    with pytest.raises(ValueError, match="Global validation gate"):
        publish_release.stage(p, prepared, artifacts, tmp_path / "staged", gate=empty_gate)
    assert not (tmp_path / "staged").exists()
