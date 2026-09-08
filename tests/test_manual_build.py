"""Manual build selection, no-publication contract and diagnostic artifacts."""

import argparse
import difflib
import itertools
import json
import sys
import tarfile
from pathlib import Path
from unittest.mock import patch

import configure_build
import inspect_test_build as inspect_build
import plan_test_build as planner
import prepare_source
import publish_release
import pytest
import release_common as common
import validate_receipts
from helpers import SHA_A, SHA_B, PublishingAPI, plan, receipts_for, upstream, write_wheel
from source_helpers import fixture_source
from test_index_and_workflows import workflow

ROOT = Path(__file__).resolve().parents[1]


class UpstreamOnly:
    def __init__(self):
        self.reads = []

    def releases(self, repository):
        assert repository == common.UPSTREAM, "Test builds must not inspect Guanaco releases"
        self.reads.append(repository)
        return [upstream(), upstream("v0.3.48-cu124-linux-20260820", id=48)]

    def commit(self, repository, tag):
        assert repository == common.UPSTREAM
        return SHA_A


def test_plan(**options):
    return planner.make_test_plan(UpstreamOnly(), "TheBigEye/guanaco-py", **options)


test_plan.__test__ = False


def prepared_test(tmp_path, *, platforms="both", avx2=False, cuda=True):
    p = test_plan(systems=platforms, avx2=avx2, cuda=cuda)
    p["recipe_commit"] = SHA_B
    patch_dir = tmp_path / "patches"
    patch_dir.mkdir()
    before, after = "VALUE = 1\n", "VALUE = 2\n"
    text = "".join(
        difflib.unified_diff(
            before.splitlines(True),
            after.splitlines(True),
            fromfile="a/llama_cpp/example.py",
            tofile="b/llama_cpp/example.py",
        )
    )
    (patch_dir / "0001-example.patch").write_bytes(text.encode("utf-8"))

    def materialize(api, repository, commit, destination):
        source = fixture_source(destination.parent)
        assert source == destination
        (source / "llama_cpp/example.py").write_bytes(before.encode())
        return [{"path": "vendor/llama.cpp", "repository": "ggml-org/llama.cpp", "commit": SHA_A}]

    prepared = tmp_path / "prepared"
    with (
        patch.object(prepare_source, "materialize", materialize),
        patch.object(prepare_source, "PATCHES_DIR", patch_dir),
    ):
        manifest = prepare_source.prepare(UpstreamOnly(), p, prepared)
    with tarfile.open(prepared / "source.tar.gz") as archive:
        runtime = {name: archive.extractfile(name).read() for name in manifest["runtime_sha256"]}
    artifacts = tmp_path / "artifacts"
    for name, spec in validate_receipts.artifact_specs(p).items():
        for version in spec["python_versions"]:
            write_wheel(
                artifacts / name,
                manifest,
                runtime,
                spec["channel"],
                spec["platform"],
                "cp" + version.replace(".", ""),
            )
    receipts_for(p, prepared, artifacts, tmp_path / "receipts")
    return p, manifest, prepared, artifacts, patch_dir


@pytest.mark.parametrize(
    "cpu,avx2,cuda",
    [values for values in itertools.product((False, True), repeat=3) if any(values)],
)
@pytest.mark.parametrize("systems", ["linux", "windows", "both"])
def test_channel_and_system_combinations_build_exactly_the_requested_matrix(
    cpu, avx2, cuda, systems
):
    p = test_plan(
        cpu=cpu,
        avx2=avx2,
        cuda=cuda,
        cuda_channels="cu124,cu128",
        python_versions="3.14,3.9",
        systems=systems,
    )
    selected = (
        (["cpu"] if cpu else []) + (["avx2"] if avx2 else []) + (["cu124", "cu128"] if cuda else [])
    )
    platforms = ["linux", "windows"] if systems == "both" else [systems]
    assert p["missing_channels"] == selected and p["platforms"] == platforms
    assert p["python_versions"] == ["3.9", "3.14"]
    assert p["test_only"] is True and p["promote_latest"] is False
    specs = validate_receipts.artifact_specs(p)
    assert (
        sum(len(spec["python_versions"]) for spec in specs.values())
        == len(selected) * len(platforms) * 2
    )
    assert {spec["platform"] for spec in specs.values()} == set(platforms)
    assert all(name.startswith("test-guanaco-py-") for name in specs)
    common.validate_build_matrix(p)


def test_defaults_are_two_cpu_wheels_and_all_means_the_full_release_matrix():
    specs = validate_receipts.artifact_specs(test_plan())
    assert len(specs) == 2
    assert sum(len(spec["python_versions"]) for spec in specs.values()) == 2
    p = test_plan(cpu=True, avx2=True, cuda=True, cuda_channels="all", python_versions="all")
    specs = validate_receipts.artifact_specs(p)
    assert len(specs) == 88
    assert sum(len(spec["python_versions"]) for spec in specs.values()) == 108


def test_already_published_versions_can_be_tested_without_querying_target_releases():
    api = UpstreamOnly()
    p = planner.make_test_plan(api, "TheBigEye/guanaco-py", "0.3.49")
    assert p["build"] is True and p["upstream"]["commit"] == SHA_A
    assert api.reads == [common.UPSTREAM]
    assert test_plan(version="0.3.48")["version"] == "0.3.48"


@pytest.mark.parametrize(
    "options",
    [
        {"cpu": False, "avx2": False, "cuda": False},
        {"cpu": "false"},
        {"systems": "macos"},
        {"python_versions": ""},
        {"python_versions": "3.8"},
        {"python_versions": "3.13,3.13"},
        {"python_versions": "all,3.13"},
        {"python_versions": "3.13,"},
        {"cuda": True, "cuda_channels": ""},
        {"cuda": True, "cuda_channels": "cu999"},
        {"cuda": True, "cuda_channels": "cu124,cu124"},
        {"version": "main"},
        {"version": "0.3.49; echo bad"},
    ],
)
def test_invalid_inputs_fail_before_network_or_builds(options):
    api = UpstreamOnly()
    with pytest.raises(ValueError):
        planner.make_test_plan(api, "TheBigEye/guanaco-py", **options)
    assert not api.reads


def test_unused_cuda_selection_is_ignored_and_values_are_trimmed():
    p = test_plan(cuda=False, cuda_channels="not used", python_versions=" 3.14 , 3.9 ")
    assert p["missing_channels"] == ["cpu"] and p["python_versions"] == ["3.9", "3.14"]
    assert planner.boolean("TRUE") is True and planner.boolean("false") is False
    with pytest.raises(argparse.ArgumentTypeError, match="true or false"):
        planner.boolean("yes")


def test_planner_cli_outputs_real_boolean_flags_and_a_readable_summary(tmp_path, monkeypatch):
    monkeypatch.setattr(planner, "GitHub", UpstreamOnly)
    monkeypatch.setenv("GITHUB_OUTPUT", str(tmp_path / "outputs"))
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(tmp_path / "summary"))
    monkeypatch.setenv("GITHUB_SHA", SHA_B)
    monkeypatch.setenv("GITHUB_RUN_ID", "1234")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "plan_test_build",
            "--cpu",
            "false",
            "--cuda",
            "true",
            "--cuda-channels",
            "cu128",
            "--systems",
            "windows",
            "--output",
            str(tmp_path / "plan.json"),
        ],
    )
    planner.main()
    p = json.loads((tmp_path / "plan.json").read_text())
    assert p["missing_channels"] == ["cu128"]
    assert p["recipe_commit"] == SHA_B and p["run_url"].endswith("/1234")
    output = (tmp_path / "outputs").read_text()
    assert "cpu=false\n" in output and 'cuda=["cu128"]' in output
    assert "1 wheels" in (tmp_path / "summary").read_text()


@pytest.mark.parametrize("platform", ["linux", "windows"])
def test_shared_builders_filter_systems_but_keep_the_release_flags(platform):
    p = test_plan(systems=platform, cuda=True)
    matrix = configure_build.cpu_matrix(p)["matrix"]["include"]
    assert len(matrix) == 1 and matrix[0]["platform"] == platform
    cpu = configure_build.cpu_options(p, "cpu", platform)
    original = configure_build.cpu_options(plan(), "cpu", platform)
    assert cpu["cibw_environment"] == original["cibw_environment"]
    assert cpu["artifact"] == "test-" + original["artifact"]
    assert len(cpu["build"].split()) == 1
    cuda = configure_build.cuda_options(p, "cu124")
    assert cuda[platform] is True
    assert cuda["windows" if platform == "linux" else "linux"] is False
    assert cuda["cmake_linux"] == configure_build.cuda_options(plan(), "cu124")["cmake_linux"]
    assert cuda["cmake_windows"] == configure_build.cuda_options(plan(), "cu124")["cmake_windows"]
    with pytest.raises(ValueError, match="not selected"):
        configure_build.cpu_options(p, "cpu", "windows" if platform == "linux" else "linux")


def test_release_matrix_and_artifact_names_are_unchanged():
    original = plan()
    original["platforms"] = ["linux"]  # Only test plans can narrow the platform selection.
    assert common.build_platforms(original) == ["linux", "windows"]
    assert len(configure_build.cpu_matrix(original)["matrix"]["include"]) == 2
    assert configure_build.cuda_options(original, "cu124")["windows"] is True
    assert (
        configure_build.cpu_options(original, "cpu", "linux")["artifact"]
        == "guanaco-py-cpu-linux-x64"
    )
    assert len(validate_receipts.artifact_specs(original)) == 88


@pytest.mark.parametrize("value", [[], ["macos"], ["linux", "linux"], "linux", None])
def test_empty_or_malformed_test_platforms_fail_closed(value):
    p = test_plan()
    p["platforms"] = value
    with pytest.raises(ValueError, match="platforms"):
        configure_build.cpu_matrix(p)


def test_malformed_test_marker_is_not_treated_as_a_release():
    p = plan()
    p["test_only"] = "false"
    with pytest.raises(ValueError, match="boolean"):
        common.build_platforms(p)
    with pytest.raises(ValueError, match="boolean"):
        publish_release.publish(PublishingAPI(), p, {})


def test_test_plans_cannot_enter_any_release_path(tmp_path):
    p = test_plan()
    p["recipe_commit"] = SHA_B
    api = PublishingAPI()
    for action in (
        lambda: publish_release.preflight(api, p, ["cpu"]),
        lambda: publish_release.publish(api, p, {}),
        lambda: publish_release.stage(p, tmp_path, tmp_path, tmp_path / "release"),
    ):
        with pytest.raises(ValueError, match="Test-build plans"):
            action()
    assert api.calls == [] and not (tmp_path / "release").exists()


def test_cpu_matrix_cli_reads_the_manifest(tmp_path, monkeypatch):
    common.write_json(tmp_path / "manifest.json", test_plan(systems="windows"))
    monkeypatch.setenv("GITHUB_OUTPUT", str(tmp_path / "outputs"))
    monkeypatch.setattr(
        sys, "argv", ["configure_build", "matrix", "--manifest", str(tmp_path / "manifest.json")]
    )
    configure_build.main()
    value = (tmp_path / "outputs").read_text().split("=", 1)[1]
    assert json.loads(value)["include"][0]["platform"] == "windows"


@pytest.mark.parametrize("autocrlf", ["false", "true"], ids=["git-lf", "git-crlf"])
def test_source_diagnostics_preserve_patch_inputs_final_files_and_logs(
    tmp_path, monkeypatch, autocrlf
):
    # Exercise Git's checkout/write conversion, not just a CRLF copy of this repo.
    # Overrides apply only to this test and its git apply subprocesses.
    monkeypatch.setenv("GIT_CONFIG_COUNT", "1")
    monkeypatch.setenv("GIT_CONFIG_KEY_0", "core.autocrlf")
    monkeypatch.setenv("GIT_CONFIG_VALUE_0", autocrlf)
    p, manifest, prepared, _, patch_dir = prepared_test(tmp_path)
    log = tmp_path / "preparation.log"
    log.write_text("Build preparation trace\n", encoding="utf-8")
    output = tmp_path / "inspection"
    inspect_build.source_report(p, prepared, output, log, patches=patch_dir)
    exported_path = output / "patched-source/llama_cpp/example.py"
    exported = exported_path.read_bytes()
    with tarfile.open(prepared / "source.tar.gz", "r:gz") as archive:
        archived = archive.extractfile("llama_cpp/example.py").read()
    # Diagnostics must preserve the actual prepared bytes, including Git's EOLs.
    assert exported == archived
    assert common.sha256(exported_path) == manifest["runtime_sha256"]["llama_cpp/example.py"]
    assert exported.decode("utf-8").splitlines() == ["VALUE = 2"]
    assert (output / "patches/0001-example.patch").read_bytes() == (
        patch_dir / "0001-example.patch"
    ).read_bytes()
    assert json.loads((output / "build-manifest.json").read_text()) == manifest
    assert (
        json.loads((output / "build-options.json").read_text())["cpu"]["windows"]["test_only"]
        is True
    )
    assert (output / "preparation.log").read_text() == log.read_text()
    assert "NOT a release" in (output / "README.md").read_text()


def test_failed_preparation_still_has_diagnostics(tmp_path):
    log = tmp_path / "prepare.log"
    log.write_text("patch no longer applies\n")
    inspect_build.source_report(
        test_plan(), tmp_path / "missing", tmp_path / "report", log, patches=tmp_path / "no-patches"
    )
    assert "did not produce a manifest" in (tmp_path / "report/README.md").read_text()
    assert (tmp_path / "report/preparation.log").read_text() == log.read_text()


def test_tampered_prepared_archive_is_not_inspected_as_valid(tmp_path):
    p, _, prepared, _, patch_dir = prepared_test(tmp_path)
    (prepared / "source.tar.gz").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="checksum mismatch"):
        inspect_build.source_report(p, prepared, tmp_path / "report", patches=patch_dir)


def test_wheel_inventory_is_downloadable_without_installing_the_package(tmp_path):
    _, m, _, artifacts, _ = prepared_test(tmp_path)
    directory = artifacts / "test-guanaco-py-cpu-linux-x64"
    inspect_build.wheel_report(m, directory, tmp_path / "report", "cpu", "linux", "success")
    report = json.loads((tmp_path / "report/wheels.json").read_text())
    wheel = report["wheels"][0]
    assert report["verification_step"] == "success"
    assert "Name: guanaco-py" in wheel["metadata"]["METADATA"]
    assert "Tag: cp313-cp313-manylinux_2_34_x86_64" in wheel["metadata"]["WHEEL"]
    assert any(item["path"] == "llama_cpp/example.py" for item in wheel["files"])
    assert "sha256" in wheel and "size" in wheel


def test_corrupt_or_absent_wheels_are_labelled_as_diagnostics(tmp_path):
    wheelhouse = tmp_path / "wheelhouse"
    wheelhouse.mkdir()
    (wheelhouse / "bad.whl").write_bytes(b"corrupt zip")
    inspect_build.wheel_report(
        test_plan(), wheelhouse, tmp_path / "report", "cpu", "linux", "failure"
    )
    result = json.loads((tmp_path / "report/wheels.json").read_text())
    assert result["verification_step"] == "failure" and "inspection_error" in result["wheels"][0]
    inspect_build.wheel_report(
        test_plan(), tmp_path / "absent", tmp_path / "empty", "cpu", "linux", "skipped"
    )
    assert json.loads((tmp_path / "empty/wheels.json").read_text())["wheels"] == []


@pytest.mark.parametrize("systems", ["linux", "windows", "both"])
def test_final_gate_checks_only_the_selected_os_and_channels(tmp_path, systems, monkeypatch):
    p, m, prepared, _, _ = prepared_test(tmp_path, platforms=systems)
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(tmp_path / "summary.md"))
    jobs = {name: {"result": "success"} for name in ("source", "cpu", "cuda")}
    jobs["avx2"] = {"result": "skipped"}
    assert inspect_build.result_report(
        p, prepared, tmp_path / "receipts", tmp_path / "report", jobs
    )
    result = json.loads((tmp_path / "report/result.json").read_text())
    assert result["success"] and result["expected_wheels"] == 2 * len(common.build_platforms(p))
    gate = json.loads((tmp_path / "report/validated-test-build.json").read_text())
    assert (
        gate["plan"]["test_only"] is True
        and gate["source_archive_sha256"] == m["source_archive_sha256"]
    )
    assert "no release created" in (tmp_path / "summary.md").read_text()


def test_failed_job_cannot_pass_even_if_receipts_exist(tmp_path):
    p, _, prepared, _, _ = prepared_test(tmp_path)
    jobs = {name: {"result": "success"} for name in ("source", "cpu", "cuda")}
    jobs["cuda"]["result"] = "failure"
    assert not inspect_build.result_report(
        p, prepared, tmp_path / "receipts", tmp_path / "report", jobs
    )
    assert json.loads((tmp_path / "report/result.json").read_text())["success"] is False


def test_missing_receipts_or_source_make_a_failed_not_empty_green_report(tmp_path):
    p, _, prepared, _, _ = prepared_test(tmp_path)
    next((tmp_path / "receipts").glob("*.json")).unlink()
    jobs = {name: {"result": "success"} for name in ("source", "cpu", "cuda")}
    assert not inspect_build.result_report(
        p, prepared, tmp_path / "receipts", tmp_path / "report", jobs
    )
    report = json.loads((tmp_path / "report/result.json").read_text())
    assert "Missing validation receipt" in report["validation_error"]
    assert not inspect_build.result_report(
        p,
        tmp_path / "missing",
        tmp_path / "receipts",
        tmp_path / "failure",
        {"source": {"result": "failure"}},
    )
    assert (tmp_path / "failure/README.md").is_file()


def test_report_cli_can_fail_after_writing_downloadable_results(tmp_path, monkeypatch):
    common.write_json(tmp_path / "plan.json", test_plan())
    monkeypatch.setenv("BUILD_JOBS", '{"source":{"result":"failure"}}')
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "report",
            "result",
            "--plan",
            str(tmp_path / "plan.json"),
            "--prepared",
            str(tmp_path / "missing"),
            "--receipts",
            str(tmp_path / "receipts"),
            "--output",
            str(tmp_path / "report"),
        ],
    )
    with pytest.raises(SystemExit) as failure:
        inspect_build.main()
    assert failure.value.code == 1
    assert (tmp_path / "report/result.json").is_file()
    common.write_json(tmp_path / "plan.json", plan())
    with pytest.raises(ValueError, match="only for test"):
        inspect_build.test_input(tmp_path / "plan.json")


def test_manual_workflow_is_only_dispatch_and_has_no_publication_path():
    data = workflow("build-test.yaml")
    assert set(data["on"]) == {"workflow_dispatch"}
    assert data["permissions"] == {"contents": "read"}
    assert data["concurrency"]["group"] != workflow("build-release.yaml")["concurrency"]["group"]
    fields = data["on"]["workflow_dispatch"]["inputs"]
    assert set(fields) == {
        "version",
        "cpu",
        "avx2",
        "cuda",
        "cuda_channels",
        "python_versions",
        "systems",
    }
    for flag in ("cpu", "avx2", "cuda"):
        assert fields[flag]["type"] == "boolean"
    for job in data["jobs"].values():
        assert "permissions" not in job or job["permissions"] == {"contents": "read"}
        assert not any(
            word in job.get("uses", "")
            for word in ("build-release", "deploy-pages", "build-docker")
        )
        for step in job.get("steps", []):
            assert "publish_release.py" not in step.get("run", "")
            assert "gh release" not in step.get("run", "")
    assert data["jobs"]["cpu"]["uses"] == workflow("build-release.yaml")["jobs"]["cpu"]["uses"]
    assert data["jobs"]["avx2"]["uses"] == workflow("build-release.yaml")["jobs"]["avx2"]["uses"]
    assert data["jobs"]["cuda"]["uses"] == workflow("build-release.yaml")["jobs"]["cuda"]["uses"]


def test_reusable_builders_read_selected_platforms_and_retain_failed_test_outputs():
    cpu = workflow("build-wheel-cpu.yml")["jobs"]
    assert "needs.config.outputs.matrix" in cpu["build"]["strategy"]["matrix"]
    cuda = workflow("build-wheel-cuda.yaml")["jobs"]
    for platform in ("linux", "windows"):
        assert f"needs.config.outputs.{platform}" in cuda[platform]["if"]
    for job in (cpu["build"], cuda["linux"], cuda["windows"]):
        uploads = [
            step
            for step in job["steps"]
            if step.get("uses", "").startswith("actions/upload-artifact@")
        ]
        wheel_upload = next(step for step in uploads if step["with"]["path"] == "wheelhouse/*.whl")
        assert "success()" in wheel_upload["if"] and "test_only" in wheel_upload["if"]
        assert "'warn'" in wheel_upload["with"]["if-no-files-found"]
        receipt = next(step for step in uploads if step["with"]["path"] == "work/receipts/*.json")
        assert "if" not in receipt  # Failed verification must not upload a valid receipt.
    report = workflow("build-test.yaml")["jobs"]["report"]
    assert "always()" in report["if"]
    downloads = [
        step["with"]
        for step in report["steps"]
        if step.get("uses", "").startswith("actions/download-artifact@")
    ]
    assert [step["pattern"] for step in downloads if "pattern" in step] == [
        "receipt-test-guanaco-py-*"
    ]
    assert "!cancelled()" in report["steps"][-1]["if"]


def test_a_failed_artifact_download_is_not_hidden_by_continue_on_error(tmp_path):
    p, _, prepared, _, _ = prepared_test(tmp_path)
    jobs = {name: {"result": "success"} for name in ("source", "cpu", "cuda")}
    downloads = {"source": {"outcome": "success"}, "receipts": {"outcome": "failure"}}
    assert not inspect_build.result_report(
        p, prepared, tmp_path / "receipts", tmp_path / "report", jobs, downloads
    )
    report = json.loads((tmp_path / "report/result.json").read_text())
    assert "download failed" in report["validation_error"]


def test_source_wheel_and_successful_result_cli_commands(tmp_path, monkeypatch):
    p, m, prepared, artifacts, _ = prepared_test(tmp_path)
    common.write_json(tmp_path / "plan.json", p)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "inspect",
            "source",
            "--plan",
            str(tmp_path / "plan.json"),
            "--prepared",
            str(prepared),
            "--output",
            str(tmp_path / "source-report"),
        ],
    )
    inspect_build.main()
    assert (tmp_path / "source-report/build-options.json").exists()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "inspect",
            "wheels",
            "--manifest",
            str(prepared / "build-manifest.json"),
            "--directory",
            str(artifacts / "test-guanaco-py-cpu-linux-x64"),
            "--output",
            str(tmp_path / "wheel-report"),
            "--channel",
            "cpu",
            "--platform",
            "linux",
            "--verification",
            "success",
        ],
    )
    inspect_build.main()
    assert (
        json.loads((tmp_path / "wheel-report/wheels.json").read_text())["version"] == m["version"]
    )
    monkeypatch.setenv(
        "BUILD_JOBS",
        json.dumps({name: {"result": "success"} for name in ("source", "cpu", "cuda")}),
    )
    monkeypatch.setenv(
        "ARTIFACT_DOWNLOADS",
        json.dumps({name: {"outcome": "success"} for name in ("source", "receipts")}),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "inspect",
            "result",
            "--plan",
            str(tmp_path / "plan.json"),
            "--prepared",
            str(prepared),
            "--receipts",
            str(tmp_path / "receipts"),
            "--output",
            str(tmp_path / "result"),
        ],
    )
    inspect_build.main()
    assert json.loads((tmp_path / "result/result.json").read_text())["success"] is True
