"""The three diagnostics a rehearsal produces: source, wheels and result."""

from __future__ import annotations

import json
from dataclasses import replace

import pytest
from helpers import (
    make_plan,
    manifest_for,
    prepared_build,
    receipts_for,
    write_source_tarball,
    write_wheel,
)

from guanaco.models import Platform, read_json, write_json
from guanaco.reports import ReportError, ReportWriter, require_test_plan
from guanaco.transfer import sha256


@pytest.fixture
def writer(settings):
    """A report writer wired to the synthetic configuration."""
    return ReportWriter(settings)


def rehearsal(settings, **options):
    """A complete, rehearsed build laid out under a temporary directory."""
    root = options.pop("root")
    plan, prepared, artifacts = prepared_build(
        root, settings, test_only=True, platforms=("linux",), **options
    )
    receipts = root / "receipts"
    receipts_for(settings, plan, prepared, artifacts, receipts, platforms=("linux",))
    return plan, prepared, receipts


class TestRequireTestPlan:
    def test_accepts_a_rehearsal(self, tmp_path, settings):
        path = tmp_path / "plan.json"
        write_json(path, make_plan(settings, ["cpu"], test_only=True).to_mapping())
        assert require_test_plan(path).test_only

    def test_refuses_a_release_plan(self, tmp_path, settings):
        path = tmp_path / "plan.json"
        write_json(path, make_plan(settings).to_mapping())
        with pytest.raises(ReportError, match="only for test-build plans"):
            require_test_plan(path)


class TestSourceReport:
    def test_describes_a_prepared_snapshot(self, tmp_path, settings, writer):
        plan, prepared, _ = prepared_build(tmp_path, settings, test_only=True, platforms=("linux",))
        output = tmp_path / "report"
        writer.source(plan, prepared, output)
        document = read_json(output / "build-options.json")
        assert "cpu" in document
        text = (output / "README.md").read_text(encoding="utf-8")
        assert "Source archive SHA256" in text
        assert (output / "build-manifest.json").is_file()
        assert (output / "patch-inputs.json").is_file()
        assert (output / "applied-patches.json").is_file()

    def test_copies_the_preparation_log(self, tmp_path, settings, writer):
        plan, prepared, _ = prepared_build(tmp_path, settings, test_only=True, platforms=("linux",))
        log = tmp_path / "build.log"
        log.write_text("preparation output", encoding="utf-8")
        output = tmp_path / "report"
        writer.source(plan, prepared, output, log=log)
        assert (output / "preparation.log").read_text(encoding="utf-8") == "preparation output"

    def test_reports_a_failed_preparation(self, tmp_path, settings, writer):
        plan = make_plan(settings, ["cpu"], test_only=True)
        output = tmp_path / "report"
        writer.source(plan, tmp_path / "empty", output)
        assert "did not produce a manifest" in (output / "README.md").read_text(encoding="utf-8")

    def test_refuses_a_tampered_snapshot(self, tmp_path, settings, writer):
        plan, prepared, _ = prepared_build(tmp_path, settings, test_only=True, platforms=("linux",))
        (prepared / "source.tar.gz").write_bytes(b"tampered")
        with pytest.raises(ReportError, match="checksum mismatch"):
            writer.source(plan, prepared, tmp_path / "report")

    def test_reports_the_compiler_flags_of_a_cuda_channel(self, tmp_path, settings, writer):
        plan = make_plan(settings, ["cu124"], test_only=True, platforms=(Platform.LINUX,))
        manifest, _ = manifest_for(plan)
        prepared = tmp_path / "prepared"
        prepared.mkdir()
        write_source_tarball(prepared / "source.tar.gz", tmp_path / "raw")
        (prepared / "packaging.patch").write_text("", encoding="utf-8")
        manifest = replace(
            manifest,
            source_archive_sha256=sha256(prepared / "source.tar.gz"),
            packaging_patch_sha256=sha256(prepared / "packaging.patch"),
        )
        write_json(prepared / "build-manifest.json", manifest.to_mapping())
        output = tmp_path / "report"
        writer.source(plan, prepared, output)
        document = read_json(output / "build-options.json")
        assert "-DGGML_CUDA=ON" in document["cu124"]["cmake_linux"]


class TestWheelReport:
    def _manifest(self, settings):
        plan = make_plan(settings, ["cpu"], test_only=True)
        return manifest_for(plan)

    def test_describes_every_wheel(self, tmp_path, settings, writer):
        manifest, runtime = self._manifest(settings)
        directory = tmp_path / "wheels"
        write_wheel(directory, manifest, runtime, "cpu", "linux", "cp313")
        output = tmp_path / "report"
        writer.wheels(manifest, directory, output, "cpu", "linux", "verified")
        document = read_json(output / "wheels.json")
        assert document["test_only"] and len(document["wheels"]) == 1
        assert "METADATA" in document["wheels"][0]["metadata"]

    def test_records_an_unreadable_wheel(self, tmp_path, settings, writer):
        manifest, _ = self._manifest(settings)
        directory = tmp_path / "wheels"
        directory.mkdir()
        (directory / "broken.whl").write_bytes(b"not a zip")
        output = tmp_path / "report"
        writer.wheels(manifest, directory, output, "cpu", "linux", "skipped")
        document = read_json(output / "wheels.json")
        assert "inspection_error" in document["wheels"][0]

    def test_records_an_ambiguous_metadata(self, tmp_path, settings, writer):
        manifest, runtime = self._manifest(settings)
        directory = tmp_path / "wheels"
        wheel = write_wheel(directory, manifest, runtime, "cpu", "linux", "cp313")
        wheel.write_bytes(b"PK\x03\x04" + b"\x00" * 100)
        output = tmp_path / "report"
        writer.wheels(manifest, directory, output, "cpu", "linux", "skipped")
        document = read_json(output / "wheels.json")
        assert "inspection_error" in document["wheels"][0]

    def test_reports_an_empty_directory(self, tmp_path, settings, writer):
        manifest, _ = self._manifest(settings)
        output = tmp_path / "report"
        writer.wheels(manifest, tmp_path / "absent", output, "cpu", "linux", "skipped")
        assert read_json(output / "wheels.json")["wheels"] == []


class TestResultReport:
    def test_a_complete_rehearsal_passes(self, tmp_path, settings, writer):
        plan, prepared, receipts = rehearsal(settings, root=tmp_path)
        output = tmp_path / "report"
        jobs = {
            "source": {"result": "success"},
            "cpu": {"result": "success"},
            "avx2": {"result": "success"},
        }
        assert writer.result(plan, prepared, receipts, output, jobs) is True
        assert read_json(output / "result.json")["success"]

    def test_a_failed_job_fails_the_rehearsal(self, tmp_path, settings, writer):
        plan, prepared, receipts = rehearsal(settings, root=tmp_path)
        output = tmp_path / "report"
        jobs = {
            "source": {"result": "success"},
            "cpu": {"result": "failure"},
            "avx2": {"result": "success"},
        }
        assert writer.result(plan, prepared, receipts, output, jobs) is False

    def test_a_missing_job_is_reported_as_missing(self, tmp_path, settings, writer):
        plan, prepared, receipts = rehearsal(settings, root=tmp_path)
        output = tmp_path / "report"
        jobs = {"source": {"result": "success"}, "cpu": {"result": "success"}}
        assert writer.result(plan, prepared, receipts, output, jobs) is False
        document = read_json(output / "result.json")
        assert document["jobs"]["avx2"] == "missing"

    def test_a_missing_receipt_fails_the_rehearsal(self, tmp_path, settings, writer):
        plan, prepared, receipts = rehearsal(settings, root=tmp_path)
        for path in receipts.glob("*.json"):
            path.unlink()
        output = tmp_path / "report"
        jobs = {
            "source": {"result": "success"},
            "cpu": {"result": "success"},
            "avx2": {"result": "success"},
        }
        assert writer.result(plan, prepared, receipts, output, jobs) is False
        assert read_json(output / "result.json")["validation_error"]

    def test_a_failed_download_fails_the_rehearsal(self, tmp_path, settings, writer):
        plan, prepared, receipts = rehearsal(settings, root=tmp_path)
        output = tmp_path / "report"
        jobs = {
            "source": {"result": "success"},
            "cpu": {"result": "success"},
            "avx2": {"result": "success"},
        }
        downloads = {"source": {"outcome": "failure"}, "receipts": {"outcome": "success"}}
        assert writer.result(plan, prepared, receipts, output, jobs, downloads) is False

    def test_the_summary_is_appended_to_the_step_summary(
        self, tmp_path, settings, writer, monkeypatch
    ):
        summary = tmp_path / "summary.md"
        monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary))
        plan, prepared, receipts = rehearsal(settings, root=tmp_path)
        jobs = {
            "source": {"result": "success"},
            "cpu": {"result": "success"},
            "avx2": {"result": "success"},
        }
        writer.result(plan, prepared, receipts, tmp_path / "report", jobs)
        assert "Test build: PASS" in summary.read_text(encoding="utf-8")

    def test_an_unreadable_prepared_snapshot_is_reported(self, tmp_path, settings, writer):
        plan = make_plan(settings, ["cpu"], test_only=True)
        output = tmp_path / "report"
        jobs = {
            "source": {"result": "success"},
            "cpu": {"result": "success"},
            "avx2": {"result": "success"},
        }
        assert (
            writer.result(plan, tmp_path / "absent", tmp_path / "receipts", output, jobs) is False
        )

    def test_a_release_plan_is_refused(self, tmp_path, settings, writer):
        plan, prepared, _ = prepared_build(tmp_path, settings)
        jobs = {"source": {"result": "success"}}
        assert (
            writer.result(plan, prepared, tmp_path / "receipts", tmp_path / "report", jobs) is False
        )

    def test_the_environment_helper_tolerates_empty_values(self, writer, monkeypatch):
        monkeypatch.delenv("GUANACO_JOBS", raising=False)
        assert writer.read_json_env("GUANACO_JOBS") == {}
        monkeypatch.setenv("GUANACO_JOBS", json.dumps({"cpu": 1}))
        assert writer.read_json_env("GUANACO_JOBS") == {"cpu": 1}
