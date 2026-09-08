import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / ".github/scripts"))
spec = importlib.util.spec_from_file_location(
    "wheel_index", ROOT / ".github/scripts/generate-wheel-index.py"
)
wheel_index = importlib.util.module_from_spec(spec)
sys.modules["wheel_index"] = wheel_index
spec.loader.exec_module(wheel_index)


@pytest.fixture(autouse=True)
def _isolate_github_file_commands(monkeypatch):
    """Tests must not write fixture reports/outputs into the real Actions runner.

    Tests that exercise these writers explicitly set their own temporary paths.
    Normal workflow CLI invocations do not load this pytest-only fixture.
    """
    for name in (
        "GITHUB_STEP_SUMMARY",
        "GITHUB_OUTPUT",
        "GITHUB_ENV",
        "GITHUB_PATH",
        "GITHUB_STATE",
    ):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture(autouse=True)
def _no_committed_patches_by_default(monkeypatch, tmp_path):
    """Point prepare_source at an empty patches directory unless a test opts in.

    Keeps the fake/minimal source fixtures used elsewhere (no llama_cpp/llama.py,
    no other real files) independent of whatever real patches later land in
    .github/patches/. Tests of the patch mechanism itself (test_patches.py)
    override this explicitly.
    """
    import prepare_source

    monkeypatch.setattr(prepare_source, "PATCHES_DIR", tmp_path / "unused-patches")
