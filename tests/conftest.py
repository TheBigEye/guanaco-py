"""Shared pytest fixtures for the guanaco build system tests.

Two things are guaranteed for every test:

* the repository root is importable, so ``import guanaco`` works;
* the GitHub Actions file-command environment variables are absent, so a test
  that writes a step summary or workflow output can never write into a real
  runner. Tests that exercise those writers set their own temporary paths.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture(autouse=True)
def _isolate_github_file_commands(monkeypatch):
    """Remove the Actions file-command variables for the duration of a test."""
    for name in (
        "GITHUB_STEP_SUMMARY",
        "GITHUB_OUTPUT",
        "GITHUB_ENV",
        "GITHUB_PATH",
        "GITHUB_STATE",
        "GITHUB_REPOSITORY",
        "GUANACO_REPOSITORY",
        "GUANACO_MATRIX",
        "GUANACO_ROOT",
    ):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def root(tmp_path) -> Path:
    """An empty directory that can be used as a repository root."""
    return tmp_path / "repository"


@pytest.fixture
def repository_root() -> Path:
    """The real repository this test suite ships with."""
    return ROOT


@pytest.fixture
def settings(root) -> Path:
    """A :class:`~guanaco.settings.Settings` loaded from a synthetic matrix."""
    from helpers import make_settings

    root.mkdir(parents=True, exist_ok=True)
    return make_settings(root)
