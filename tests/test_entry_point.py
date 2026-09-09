"""``python -m guanaco`` really runs the package."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(*arguments: str) -> subprocess.CompletedProcess:
    """Run the module as a subprocess, from the repository root."""
    return subprocess.run(
        [sys.executable, "-m", "guanaco", *arguments],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )


def test_the_module_runs_and_explains_itself():
    result = run("explain")
    assert result.returncode == 0
    assert "package:" in result.stdout


def test_a_failing_command_exits_with_one(tmp_path):
    result = run("unpack-source", str(tmp_path / "absent"), str(tmp_path / "out"))
    assert result.returncode == 1
    assert result.stderr.startswith("error:")


def test_the_configuration_is_self_contained(monkeypatch):
    """The repository's own matrix must resolve without any environment help."""
    for name in ("GITHUB_REPOSITORY", "GUANACO_REPOSITORY", "GUANACO_MATRIX"):
        monkeypatch.delenv(name, raising=False)
    from guanaco.settings import Settings

    settings = Settings.load()
    assert settings.repository == "TheBigEye/guanaco-py"
    assert settings.package == "guanaco-py"
    assert settings.upstream == "JamePeng/llama-cpp-python"
    assert settings.channels and settings.python_versions


def test_the_help_lists_every_command():
    result = run("--help")
    assert result.returncode == 0
    for name in (
        "plan",
        "plan-test",
        "prepare-source",
        "unpack-source",
        "configure",
        "verify-wheels",
        "validate-receipts",
        "publish",
        "build-index",
        "inspect",
        "explain",
    ):
        assert name in result.stdout
