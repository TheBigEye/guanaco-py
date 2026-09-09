"""The Docker helper that installs a pinned, checksummed release wheel.

Loaded directly from its path so the tests never depend on a ``docker``
package being importable (the ``docker`` name is also used by the Docker SDK).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def helper():
    """Import ``docker/fetch_release.py`` as a module."""
    path = ROOT / "docker" / "fetch_release.py"
    spec = importlib.util.spec_from_file_location("docker_fetch_release", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(autouse=True)
def _inside_a_linux_container(monkeypatch, helper):
    """Pretend we are the only platform these images support.

    The helper refuses to run anywhere but ``linux/amd64``. These tests are
    about its naming and installation logic, so the guard is satisfied instead
    of skipped: that way the Windows runners exercise the same code.
    """
    monkeypatch.setattr(helper.sys, "platform", "linux")
    monkeypatch.setattr(helper.platform, "machine", lambda: "x86_64")


class TestReleaseBase:
    def test_cpu_keeps_the_bare_tag(self, helper):
        from guanaco.channels import Channel

        base = helper.release_base("owner/repo", "0.3.49", Channel("cpu"))
        assert base.endswith("/releases/download/v0.3.49")

    def test_other_channels_append_their_name(self, helper):
        from guanaco.channels import Channel

        base = helper.release_base("owner/repo", "0.3.49", Channel("cu128"))
        assert base.endswith("/releases/download/v0.3.49-cu128")

    @pytest.mark.parametrize("repository", ["owner", "owner/../name", "a/b/c", "own er/name"])
    def test_a_bad_repository_is_refused(self, helper, repository):
        from guanaco.channels import Channel

        with pytest.raises(ValueError):
            helper.release_base(repository, "0.3.49", Channel("cpu"))

    def test_an_unstable_version_is_refused(self, helper):
        from guanaco.channels import Channel

        with pytest.raises(ValueError):
            helper.release_base("owner/repo", "0.3.49-rc1", Channel("cpu"))


class TestChecksums:
    def test_parses_a_release_inventory(self, helper, tmp_path):
        path = tmp_path / "SHA256SUMS"
        path.write_text(f"{'a' * 64}  wheel.whl\n{'b' * 64}  SHA256SUMS\n", encoding="utf-8")
        assert helper.read_checksums(path) == {"wheel.whl": "a" * 64, "SHA256SUMS": "b" * 64}

    def test_a_malformed_line_is_refused(self, helper, tmp_path):
        path = tmp_path / "SHA256SUMS"
        path.write_text("nope\n", encoding="utf-8")
        with pytest.raises(ValueError, match="Malformed"):
            helper.read_checksums(path)

    def test_a_duplicate_entry_is_refused(self, helper, tmp_path):
        path = tmp_path / "SHA256SUMS"
        path.write_text(f"{'a' * 64}  x\n{'b' * 64}  x\n", encoding="utf-8")
        with pytest.raises(ValueError, match="duplicate"):
            helper.read_checksums(path)

    def test_an_empty_inventory_is_refused(self, helper, tmp_path):
        path = tmp_path / "SHA256SUMS"
        path.write_text("", encoding="utf-8")
        with pytest.raises(ValueError, match="Empty"):
            helper.read_checksums(path)


class TestWheelName:
    def test_uses_the_distribution_prefix(self, helper):
        from guanaco.channels import Channel

        name = helper.wheel_name("guanaco-py", "0.3.49", Channel("cpu"))
        assert name.startswith("guanaco_py-0.3.49-cp")

    def test_cuda_channels_are_not_repaired(self, helper):
        from guanaco.channels import Channel

        name = helper.wheel_name("guanaco-py", "0.3.49", Channel("cu128"))
        assert name.endswith("linux_x86_64.whl")

    def test_a_renamed_distribution_follows(self, helper):
        from guanaco.channels import Channel

        name = helper.wheel_name("my-fork", "0.3.49", Channel("cpu"))
        assert name.startswith("my_fork-0.3.49-cp")

    def test_only_linux_amd64_is_supported(self, helper, monkeypatch):
        from guanaco.channels import Channel

        monkeypatch.setattr(helper.platform, "machine", lambda: "aarch64")
        with pytest.raises(ValueError, match="linux/amd64"):
            helper.wheel_name("guanaco-py", "0.3.49", Channel("cpu"))


class TestMain:
    def _serve(self, helper, monkeypatch, tmp_path, files):
        """Answer downloads from a canned inventory written to `files`."""
        directory = tmp_path / "release"
        directory.mkdir()
        (directory / "SHA256SUMS").write_text(
            "".join(f"{'a' * 64}  {name}\n" for name in files), encoding="utf-8"
        )
        for name in files:
            (directory / name).write_text("payload", encoding="utf-8")

        def fetch(url, destination, expected_sha256=None):
            """Copy the canned asset instead of reaching the network."""
            del expected_sha256
            name = url.rsplit("/", 1)[-1]
            Path(destination).write_bytes((directory / name).read_bytes())
            return "a" * 64

        monkeypatch.setattr(
            helper, "Downloader", lambda **kwargs: type("D", (), {"fetch": staticmethod(fetch)})
        )
        return directory

    def test_source_mode_downloads_the_snapshot(self, helper, monkeypatch, tmp_path):
        self._serve(
            helper,
            monkeypatch,
            tmp_path,
            ["guanaco-source-0.3.49.tar.gz", "guanaco-build.json"],
        )
        helper.main.__globals__["sys"].argv = [
            "fetch_release.py",
            "source",
            "--version",
            "0.3.49",
            "--directory",
            str(tmp_path / "download"),
        ]
        helper.main()
        assert (tmp_path / "download" / "source.tar.gz").is_file()
        assert (tmp_path / "download" / "build-manifest.json").is_file()

    def test_wheel_mode_installs_the_wheel(self, helper, monkeypatch, tmp_path):
        from guanaco.channels import Channel

        name = helper.wheel_name("guanaco-py", "0.3.49", Channel("cpu"))
        self._serve(helper, monkeypatch, tmp_path, [name])
        installed = []
        monkeypatch.setattr(
            helper.subprocess,
            "run",
            lambda command, check=False: installed.append(command) or None,
        )
        helper.main.__globals__["sys"].argv = [
            "fetch_release.py",
            "wheel",
            "--version",
            "0.3.49",
            "--directory",
            str(tmp_path / "download"),
        ]
        helper.main()
        assert installed and installed[0][-1].endswith("[server]")

    def test_an_unknown_channel_is_rejected_by_the_parser(self, helper):
        helper.main.__globals__["sys"].argv = [
            "fetch_release.py",
            "wheel",
            "--version",
            "0.3.49",
            "--channel",
            "gpu",
        ]
        with pytest.raises(SystemExit) as error:
            helper.main()
        assert error.value.code == 2
