"""Keep simulated build reports out of the real Actions file-command channels."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from helpers import plan
from release_common import write_json

ROOT = Path(__file__).resolve().parents[1]
FILE_COMMANDS = (
    "GITHUB_STEP_SUMMARY",
    "GITHUB_OUTPUT",
    "GITHUB_ENV",
    "GITHUB_PATH",
    "GITHUB_STATE",
)


@pytest.mark.parametrize("name", FILE_COMMANDS)
def test_github_file_commands_are_opt_in(name):
    # In CI these variables exist before pytest starts. The autouse fixture
    # removes them; a writer test can opt in with its own tmp_path afterwards.
    assert name not in os.environ


def test_nested_pytest_cannot_write_to_runner_command_files(tmp_path):
    env = os.environ.copy()
    env.pop("PYTEST_ADDOPTS", None)
    sentinels = {}
    for name in FILE_COMMANDS:
        path = tmp_path / name
        data = f"Untouched runner file: {name}\n".encode()
        path.write_bytes(data)
        env[name] = str(path)
        sentinels[path] = data
    # Run the real project conftest plus the exact negative test that used to
    # emit the fake CUDA failure. Exclude this parent test to avoid recursion.
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            str(Path(__file__)),
            str(ROOT / "tests/test_manual_build.py"),
            "-k",
            "test_github_file_commands_are_opt_in or "
            "test_failed_job_cannot_pass_even_if_receipts_exist",
        ],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "6 passed" in result.stdout, result.stdout
    for path, expected in sentinels.items():
        assert path.read_bytes() == expected, path.name


def test_real_report_cli_still_writes_a_summary_when_explicitly_configured(tmp_path):
    p = {**plan(["cpu"]), "test_only": True, "platforms": ["linux"]}
    plan_path = tmp_path / "test-plan.json"
    write_json(plan_path, p)
    summary = tmp_path / "manual-workflow-summary.md"
    env = {
        **os.environ,
        "GITHUB_STEP_SUMMARY": str(summary),
        "BUILD_JOBS": json.dumps({"source": {"result": "failure"}}),
    }
    env.pop("ARTIFACT_DOWNLOADS", None)
    # This is a CLI process, not a pytest process: its real reporting behavior
    # must remain enabled. A genuine failed build should still report FAIL.
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / ".github/scripts/inspect_test_build.py"),
            "result",
            "--plan",
            str(plan_path),
            "--prepared",
            str(tmp_path / "missing-source"),
            "--receipts",
            str(tmp_path / "receipts"),
            "--output",
            str(tmp_path / "report"),
        ],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1, result.stdout + result.stderr
    assert "Test build: FAIL" in summary.read_text(encoding="utf-8")
    assert "source=failure" in summary.read_text(encoding="utf-8")
    assert json.loads((tmp_path / "report/result.json").read_text())["success"] is False
