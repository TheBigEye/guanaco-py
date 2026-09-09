"""The command line interface: every subcommand, and how it fails."""

from __future__ import annotations

import json

import pytest
from helpers import (
    FakeGitHub,
    make_plan,
    manifest_for,
    owned_release,
    prepared_build,
    receipts_for,
    upstream_payload,
    write_wheel,
)

from guanaco import cli
from guanaco.models import Plan, read_json, write_json


@pytest.fixture(autouse=True)
def synthetic_matrix(monkeypatch, settings, root):
    """Point the CLI at the synthetic matrix instead of the repository's own."""
    from helpers import matrix_path

    monkeypatch.setenv("GUANACO_MATRIX", str(matrix_path(root)))
    yield settings


@pytest.fixture
def offline(monkeypatch):
    """Replace the GitHub client with a canned, offline one."""

    def install(releases=None, existing=None):
        """Point the CLI at a fake client serving these releases."""
        monkeypatch.setattr(cli, "GithubClient", lambda **kwargs: FakeGitHub(releases, existing))

    return install


def run(*arguments):
    """Run the CLI the way a workflow would, returning its exit code."""
    return cli.main(list(arguments))


class TestExplain:
    def test_prints_the_resolved_configuration(self, settings, capsys):
        assert run("explain") == 0
        text = capsys.readouterr().out
        assert "package:" in text and "channels:" in text


class TestPlan:
    def test_writes_a_plan_and_reports_the_channels(self, tmp_path, offline, capsys):
        offline()
        output = tmp_path / "plan.json"
        assert run("plan", "--output", str(output)) == 0
        plan = Plan.from_mapping(read_json(output))
        assert plan.version == "0.3.49"
        assert "Missing channels" in capsys.readouterr().out

    def test_writes_the_workflow_outputs(self, tmp_path, offline, monkeypatch):
        offline()
        outputs = tmp_path / "outputs.txt"
        outputs.write_text("", encoding="utf-8")
        monkeypatch.setenv("GITHUB_OUTPUT", str(outputs))
        summaries = tmp_path / "summary.md"
        monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summaries))
        assert run("plan", "--output", str(tmp_path / "plan.json")) == 0
        text = outputs.read_text(encoding="utf-8")
        assert "build=true" in text
        assert "version=0.3.49" in text
        assert "publish_matrix=" in text
        assert "Upstream release check" in summaries.read_text(encoding="utf-8")

    def test_an_explicit_version_is_honoured(self, tmp_path, offline):
        offline([upstream_payload("v0.3.48-cu124-win-20260801", id=48), upstream_payload()])
        output = tmp_path / "plan.json"
        assert run("plan", "--version", "0.3.48", "--output", str(output)) == 0
        assert Plan.from_mapping(read_json(output)).version == "0.3.48"

    def test_an_unknown_version_is_an_error(self, tmp_path, offline):
        offline()
        assert run("plan", "--version", "9.9.9", "--output", str(tmp_path / "plan.json")) == 1


class TestPlanTest:
    def test_plans_a_rehearsal(self, tmp_path, offline, capsys):
        offline()
        output = tmp_path / "test-plan.json"
        assert run("plan-test", "--output", str(output)) == 0
        assert Plan.from_mapping(read_json(output)).test_only
        assert "NOT a release" in capsys.readouterr().out

    def test_accepts_the_whole_matrix(self, tmp_path, offline):
        offline()
        output = tmp_path / "test-plan.json"
        assert (
            run(
                "plan-test",
                "--cpu",
                "true",
                "--avx2",
                "true",
                "--cuda",
                "true",
                "--cuda-channels",
                "all",
                "--python-versions",
                "all",
                "--systems",
                "linux",
                "--output",
                str(output),
            )
            == 0
        )
        plan = Plan.from_mapping(read_json(output))
        assert [c.name for c in plan.missing_channels] == ["cpu", "avx2", "cu124", "cu128"]

    def test_a_bad_system_is_rejected_by_the_parser(self, offline):
        offline()
        with pytest.raises(SystemExit) as error:
            run("plan-test", "--systems", "macos")
        assert error.value.code == 2

    def test_a_bad_selection_is_an_error(self, tmp_path, offline):
        offline()
        assert run("plan-test", "--python-versions", "2.7") == 1


class TestConfigure:
    def _manifest(self, tmp_path, settings):
        plan = make_plan(settings)
        manifest, _ = manifest_for(plan)
        path = tmp_path / "manifest.json"
        write_json(path, manifest.to_mapping())
        return path

    def test_cpu(self, tmp_path, settings, capsys):
        path = self._manifest(tmp_path, settings)
        assert (
            run(
                "configure",
                "cpu",
                "--manifest",
                str(path),
                "--channel",
                "cpu",
                "--platform",
                "linux",
            )
            == 0
        )
        assert "guanaco-py-cpu-linux-x64" in capsys.readouterr().out

    def test_cuda(self, tmp_path, settings, capsys):
        path = self._manifest(tmp_path, settings)
        assert run("configure", "cuda", "--manifest", str(path), "--channel", "cu124") == 0
        assert "12.4.1" in capsys.readouterr().out

    def test_matrix(self, tmp_path, settings, capsys):
        path = self._manifest(tmp_path, settings)
        assert run("configure", "matrix", "--manifest", str(path)) == 0
        assert "ubuntu-latest" in capsys.readouterr().out

    def test_docker(self, settings, capsys):
        assert run("configure", "docker", "--version", "0.3.49", "--promote-latest") == 0
        assert "latest" in capsys.readouterr().out

    def test_a_version_mismatch_is_an_error(self, tmp_path, settings):
        path = self._manifest(tmp_path, settings)
        assert (
            run(
                "configure",
                "cpu",
                "--manifest",
                str(path),
                "--channel",
                "cpu",
                "--platform",
                "linux",
                "--version",
                "9.9.9",
            )
            == 1
        )


class TestVerifyWheels:
    def test_prints_the_build_selector(self, tmp_path, settings, capsys):
        plan, prepared, _ = prepared_build(tmp_path, settings)
        assert (
            run(
                "verify-wheels",
                "--manifest",
                str(prepared / "build-manifest.json"),
                "--channel",
                "cpu",
                "--platform",
                "linux",
                "--selectors",
            )
            == 0
        )
        assert "build=cp312-manylinux_x86_64" in capsys.readouterr().out

    def test_validates_a_directory_and_writes_a_receipt(self, tmp_path, settings):
        plan, prepared, artifacts = prepared_build(tmp_path, settings)
        receipt = tmp_path / "receipt.json"
        assert (
            run(
                "verify-wheels",
                "--manifest",
                str(prepared / "build-manifest.json"),
                "--channel",
                "cpu",
                "--platform",
                "linux",
                "--directory",
                str(artifacts / "guanaco-py-cpu-linux-x64"),
                "--receipt",
                str(receipt),
            )
            == 0
        )
        assert read_json(receipt)["channel"] == "cpu"

    def test_validates_a_single_cuda_wheel(self, tmp_path, settings):
        plan, prepared, artifacts = prepared_build(tmp_path, settings)
        manifest = manifest_of(prepared)
        directory = tmp_path / "single"
        runtime = {"llama_cpp/__init__.py": b'__version__ = "0.3.49"\n', "llama_cpp/py.typed": b""}
        write_wheel(directory, manifest, runtime, "cu124", "linux", "cp313")
        assert (
            run(
                "verify-wheels",
                "--manifest",
                str(prepared / "build-manifest.json"),
                "--channel",
                "cu124",
                "--platform",
                "linux",
                "--directory",
                str(directory),
                "--single",
            )
            == 0
        )

    def test_a_wrong_python_version_is_rejected(self, tmp_path, settings):
        _, prepared, _ = prepared_build(tmp_path, settings)
        manifest = manifest_of(prepared)
        directory = tmp_path / "single"
        runtime = {"llama_cpp/__init__.py": b'__version__ = "0.3.49"\n', "llama_cpp/py.typed": b""}
        write_wheel(directory, manifest, runtime, "cu124", "linux", "cp313")
        assert (
            run(
                "verify-wheels",
                "--manifest",
                str(prepared / "build-manifest.json"),
                "--channel",
                "cu124",
                "--platform",
                "linux",
                "--directory",
                str(directory),
                "--single",
                "--python",
                "3.12",
            )
            == 1
        )

    def test_unrepaired_requires_a_single_wheel(self, tmp_path, settings):
        _, prepared, artifacts = prepared_build(tmp_path, settings)
        with pytest.raises(SystemExit) as error:
            run(
                "verify-wheels",
                "--manifest",
                str(prepared / "build-manifest.json"),
                "--channel",
                "cpu",
                "--platform",
                "linux",
                "--directory",
                str(artifacts / "guanaco-py-cpu-linux-x64"),
                "--unrepaired",
            )
        assert error.value.code == 2

    def test_a_directory_is_required(self, tmp_path, settings):
        _, prepared, _ = prepared_build(tmp_path, settings)
        with pytest.raises(SystemExit) as error:
            run(
                "verify-wheels",
                "--manifest",
                str(prepared / "build-manifest.json"),
                "--channel",
                "cpu",
                "--platform",
                "linux",
            )
        assert error.value.code == 2


class TestValidateReceipts:
    def test_writes_the_gate(self, tmp_path, settings, capsys):
        plan, prepared, artifacts = prepared_build(tmp_path, settings)
        receipts = tmp_path / "receipts"
        receipts_for(settings, plan, prepared, artifacts, receipts)
        plan_path = tmp_path / "plan.json"
        write_json(plan_path, plan.to_mapping())
        gate = tmp_path / "gate.json"
        assert (
            run(
                "validate-receipts",
                "--plan",
                str(plan_path),
                "--prepared",
                str(prepared),
                "--receipts",
                str(receipts),
                "--output",
                str(gate),
            )
            == 0
        )
        assert set(read_json(gate)["channels"]) == {"cpu", "avx2"}
        assert "Validated" in capsys.readouterr().out

    def test_a_missing_receipt_fails(self, tmp_path, settings):
        plan, prepared, artifacts = prepared_build(tmp_path, settings)
        receipts = tmp_path / "receipts"
        receipts.mkdir()
        plan_path = tmp_path / "plan.json"
        write_json(plan_path, plan.to_mapping())
        assert (
            run(
                "validate-receipts",
                "--plan",
                str(plan_path),
                "--prepared",
                str(prepared),
                "--receipts",
                str(receipts),
                "--output",
                str(tmp_path / "gate.json"),
            )
            == 1
        )


class TestPublish:
    def _plan(self, tmp_path, settings):
        plan, prepared, artifacts = prepared_build(tmp_path, settings)
        plan_path = tmp_path / "plan.json"
        write_json(plan_path, plan.to_mapping())
        return plan, plan_path, prepared, artifacts

    def test_a_dry_run_stages_without_writing(self, tmp_path, settings, capsys):
        plan, plan_path, prepared, artifacts = self._plan(tmp_path, settings)
        assert (
            run(
                "publish",
                "--plan",
                str(plan_path),
                "--prepared",
                str(prepared),
                "--artifacts",
                str(artifacts),
                "--output",
                str(tmp_path / "staged"),
            )
            == 0
        )
        assert "DRY RUN" in capsys.readouterr().out

    def test_staging_requires_the_inputs(self, tmp_path, settings):
        plan, plan_path, _, _ = self._plan(tmp_path, settings)
        assert run("publish", "--plan", str(plan_path)) == 1

    def test_preflight_passes_without_writing(self, tmp_path, settings, offline, capsys):
        offline(existing=[])
        plan, plan_path, _, _ = self._plan(tmp_path, settings)
        assert run("publish", "--plan", str(plan_path), "--preflight") == 0
        assert "preflight passed" in capsys.readouterr().out

    def test_preflight_refuses_a_foreign_release(self, tmp_path, settings, offline):
        from helpers import owned_release

        plan, plan_path, _, _ = self._plan(tmp_path, settings)
        foreign = owned_release(settings, plan, "cpu")
        foreign["body"] = "no provenance here"
        offline(existing=[foreign])
        assert run("publish", "--plan", str(plan_path), "--preflight") == 1

    def test_a_gate_can_be_supplied(self, tmp_path, settings):
        plan, plan_path, prepared, artifacts = self._plan(tmp_path, settings)
        receipts = tmp_path / "receipts"
        receipts_for(settings, plan, prepared, artifacts, receipts)
        gate = tmp_path / "gate.json"
        assert (
            run(
                "validate-receipts",
                "--plan",
                str(plan_path),
                "--prepared",
                str(prepared),
                "--receipts",
                str(receipts),
                "--output",
                str(gate),
            )
            == 0
        )
        assert (
            run(
                "publish",
                "--plan",
                str(plan_path),
                "--prepared",
                str(prepared),
                "--artifacts",
                str(artifacts),
                "--gate",
                str(gate),
                "--output",
                str(tmp_path / "staged"),
            )
            == 0
        )

    def test_preflight_and_publish_are_exclusive(self, tmp_path, settings):
        plan, plan_path, _, _ = self._plan(tmp_path, settings)
        with pytest.raises(SystemExit) as error:
            run("publish", "--plan", str(plan_path), "--preflight", "--publish")
        assert error.value.code == 2


class TestBuildIndex:
    def test_renders_from_a_release_payload(self, tmp_path, settings, capsys):
        plan, _, _ = prepared_build(tmp_path, settings)
        payload = tmp_path / "releases.json"
        payload.write_text(json.dumps([owned_release(settings, plan, "cpu")]), encoding="utf-8")
        assert run("build-index", str(payload), str(tmp_path / "site")) == 0
        assert (tmp_path / "site" / "index.html").is_file()

    def test_accepts_an_empty_payload(self, tmp_path, settings):
        payload = tmp_path / "releases.json"
        payload.write_text("[]", encoding="utf-8")
        assert run("build-index", str(payload), str(tmp_path / "site")) == 0

    def test_accepts_a_paginated_payload(self, tmp_path, settings):
        plan, _, _ = prepared_build(tmp_path, settings)
        payload = tmp_path / "releases.json"
        payload.write_text(
            json.dumps([[owned_release(settings, plan, "cpu")], []]),
            encoding="utf-8",
        )
        assert run("build-index", str(payload), str(tmp_path / "site")) == 0


class TestInspect:
    def test_source_report(self, tmp_path, settings):
        plan, prepared, _ = prepared_build(tmp_path, settings, test_only=True, platforms=("linux",))
        plan_path = tmp_path / "plan.json"
        write_json(plan_path, plan.to_mapping())
        assert (
            run(
                "inspect",
                "source",
                "--plan",
                str(plan_path),
                "--prepared",
                str(prepared),
                "--output",
                str(tmp_path / "report"),
            )
            == 0
        )
        assert (tmp_path / "report" / "README.md").is_file()

    def test_wheels_report(self, tmp_path, settings):
        plan, prepared, artifacts = prepared_build(
            tmp_path, settings, test_only=True, platforms=("linux",)
        )
        assert (
            run(
                "inspect",
                "wheels",
                "--manifest",
                str(prepared / "build-manifest.json"),
                "--directory",
                str(artifacts / "test-guanaco-py-cpu-linux-x64"),
                "--output",
                str(tmp_path / "report"),
                "--channel",
                "cpu",
                "--platform",
                "linux",
                "--verification",
                "success",
            )
            == 0
        )
        assert (tmp_path / "report" / "wheels.json").is_file()

    def test_result_report(self, tmp_path, settings, monkeypatch):
        plan, prepared, artifacts = prepared_build(
            tmp_path, settings, test_only=True, platforms=("linux",)
        )
        receipts = tmp_path / "receipts"
        receipts_for(settings, plan, prepared, artifacts, receipts, platforms=("linux",))
        plan_path = tmp_path / "plan.json"
        write_json(plan_path, plan.to_mapping())
        monkeypatch.setenv(
            "BUILD_JOBS",
            json.dumps(
                {
                    "source": {"result": "success"},
                    "cpu": {"result": "success"},
                    "avx2": {"result": "success"},
                }
            ),
        )
        assert (
            run(
                "inspect",
                "result",
                "--plan",
                str(plan_path),
                "--prepared",
                str(prepared),
                "--receipts",
                str(receipts),
                "--output",
                str(tmp_path / "report"),
            )
            == 0
        )

    def test_a_failing_rehearsal_exits_non_zero(self, tmp_path, settings, monkeypatch):
        plan, prepared, artifacts = prepared_build(
            tmp_path, settings, test_only=True, platforms=("linux",)
        )
        plan_path = tmp_path / "plan.json"
        write_json(plan_path, plan.to_mapping())
        monkeypatch.setenv("BUILD_JOBS", json.dumps({"source": {"result": "failure"}}))
        assert (
            run(
                "inspect",
                "result",
                "--plan",
                str(plan_path),
                "--prepared",
                str(prepared),
                "--receipts",
                str(tmp_path / "absent"),
                "--output",
                str(tmp_path / "report"),
            )
            == 1
        )

    def test_a_release_plan_is_refused(self, tmp_path, settings):
        plan, prepared, _ = prepared_build(tmp_path, settings)
        plan_path = tmp_path / "plan.json"
        write_json(plan_path, plan.to_mapping())
        assert (
            run(
                "inspect",
                "source",
                "--plan",
                str(plan_path),
                "--prepared",
                str(prepared),
                "--output",
                str(tmp_path / "report"),
            )
            == 1
        )


def manifest_of(prepared):
    """Read the manifest written by :func:`helpers.prepared_build`."""
    from guanaco.source import SourceArchive

    return SourceArchive.verify(prepared)
