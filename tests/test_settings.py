"""Configuration loading, validation and the value objects built from it."""

from __future__ import annotations

import json

import pytest
from helpers import make_settings, matrix_path

from guanaco.models import Channel, Platform, Version
from guanaco.settings import (
    ConfigurationError,
    CudaChannel,
    Settings,
    repository_name,
)


class TestVersions:
    def test_compares_numerically(self):
        assert Version.parse("0.3.10") > Version.parse("0.3.9")
        assert Version.parse("1.0.0") > Version.parse("0.99.99")
        assert str(Version.parse("0.3.49")) == "0.3.49"

    def test_rejects_anything_that_is_not_stable(self):
        for text in ("", "0.3", "v0.3.49", "0.3.49.1", "0.03.49", "latest"):
            with pytest.raises(ValueError):
                Version.parse(text)

    def test_extracts_version_from_an_upstream_tag(self):
        assert str(Version.from_tag("v0.3.49-cu124-win-20260831")) == "0.3.49"
        assert Version.from_tag("v0.3.50-rc1") is None
        assert Version.from_tag("nightly") is None
        assert Version.from_tag(None) is None


class TestChannels:
    def test_cpu_channels_have_no_suffix(self):
        assert Channel("cpu").release_tag("0.3.49") == "v0.3.49"
        assert Channel("avx2").release_tag("0.3.49") == "v0.3.49-avx2"
        assert Channel("cu124").release_tag("0.3.49") == "v0.3.49-cu124"

    def test_cpu_channels_are_manylinux_and_cuda_is_not(self):
        linux, windows = Platform.LINUX, Platform.WINDOWS
        assert Channel("cpu").wheel_platform(linux) == "manylinux_2_34_x86_64"
        assert Channel("cpu").wheel_platform(windows) == "win_amd64"
        assert Channel("cu124").wheel_platform(linux) == "linux_x86_64"
        assert Channel("cpu").wheel_platform(linux, repaired=False) == "linux_x86_64"

    def test_knows_what_it_is(self):
        assert Channel("cpu").is_cpu_variant and not Channel("cpu").is_cuda
        assert Channel("avx2").is_cpu_variant
        assert Channel("cu124").is_cuda and not Channel("cu124").is_cpu_variant

    def test_rejects_unknown_names(self):
        for name in ("", "gpu", "cuda", "cu", "CPU"):
            with pytest.raises(ValueError):
                Channel(name)


class TestPlatforms:
    def test_parses_and_renders(self):
        assert Platform.parse("linux") is Platform.LINUX
        assert str(Platform.WINDOWS) == "windows"
        assert Platform.WINDOWS.is_windows

    def test_rejects_unknown(self):
        with pytest.raises(ValueError):
            Platform.parse("macos")


class TestCudaChannel:
    def test_name_must_match_toolkit(self):
        with pytest.raises(ConfigurationError):
            CudaChannel(
                name="cu124", toolkit="12.8.1", architectures="75", legacy_msvc=False
            ).validate()

    def test_rejects_bad_architectures_and_flags(self):
        with pytest.raises(ConfigurationError):
            CudaChannel.from_mapping(
                "cu124", {"toolkit": "12.4.1", "architectures": "", "legacy_msvc": False}
            )
        with pytest.raises(ConfigurationError):
            CudaChannel.from_mapping(
                "cu124", {"toolkit": "12.4.1", "architectures": "75", "legacy_msvc": "yes"}
            )

    def test_label_is_human_readable(self):
        channel = CudaChannel.from_mapping(
            "cu124", {"toolkit": "12.4.1", "architectures": "75;80", "legacy_msvc": False}
        )
        assert channel.label == "CUDA 12.4"


class TestRepositoryName:
    def test_accepts_owner_and_name(self):
        assert repository_name("TheBigEye/guanaco-py") == "TheBigEye/guanaco-py"

    @pytest.mark.parametrize("value", ["", "owner", "owner/../name", "a/b/c", 5])
    def test_rejects_anything_else(self, value):
        with pytest.raises(ConfigurationError):
            repository_name(value)


class TestSettings:
    def test_reads_the_matrix(self, root):
        settings = make_settings(root)
        assert settings.repository == "TheBigEye/guanaco-py"
        assert settings.upstream == "JamePeng/llama-cpp-python"
        assert settings.package == "guanaco-py"
        assert [c.name for c in settings.channels[:2]] == ["cpu", "avx2"]
        assert settings.cuda["cu124"].toolkit == "12.4.1"

    def test_derives_the_upstream_distribution_name(self, root):
        settings = make_settings(root)
        assert settings.upstream_package == "llama-cpp-python"

    def test_exposes_derived_directories(self, root):
        settings = make_settings(root)
        assert settings.patches_dir == root / ".github" / "patches"
        assert settings.docs_dir == root / "docs"

    def test_describe_mentions_everything_important(self, root):
        text = make_settings(root).describe()
        for expected in ("guanaco-py", "cu124", "3.14"):
            assert expected in text

    def test_repository_can_be_overridden(self, root):
        settings = make_settings(root, repository="someone/else")
        assert settings.repository == "someone/else"

    def test_environment_overrides_the_repository(self, root, monkeypatch):
        make_settings(root)
        monkeypatch.setenv("GUANACO_REPOSITORY", "env/owner")
        assert Settings.load(root=root).repository == "env/owner"

    def test_actions_repository_is_used_as_a_fallback(self, root, monkeypatch):
        make_settings(root)
        path = matrix_path(root)
        data = json.loads(path.read_text(encoding="utf-8"))
        del data["repository"]
        path.write_text(json.dumps(data), encoding="utf-8")
        monkeypatch.setenv("GITHUB_REPOSITORY", "actions/owner")
        assert Settings.load(root=root).repository == "actions/owner"

    def test_missing_repository_is_an_error(self, root):
        make_settings(root)
        path = matrix_path(root)
        data = json.loads(path.read_text(encoding="utf-8"))
        del data["repository"]
        path.write_text(json.dumps(data), encoding="utf-8")
        with pytest.raises(ConfigurationError):
            Settings.load(root=root)

    def test_missing_matrix_file_is_an_error(self, root):
        with pytest.raises(ConfigurationError):
            Settings.load(root=root / "nowhere")

    def test_invalid_json_is_an_error(self, root):
        matrix_path(root).parent.mkdir(parents=True, exist_ok=True)
        matrix_path(root).write_text("{not json", encoding="utf-8")
        with pytest.raises(ConfigurationError):
            Settings.load(root=root)

    @pytest.mark.parametrize(
        "override",
        [
            {"python_versions": []},
            {"python_versions": ["3.13", "3.13"]},
            {"python_versions": ["2.7"]},
            {"cuda": {"cu999": {"toolkit": "12.4.1", "architectures": "75", "legacy_msvc": False}}},
            {"upstream": ""},
            {"package": ""},
        ],
    )
    def test_rejects_inconsistent_matrices(self, root, override):
        with pytest.raises(ConfigurationError):
            make_settings(root, **override)

    def test_channel_list_must_start_with_the_cpu_channels(self, root):
        with pytest.raises(ConfigurationError):
            make_settings(root, channels=["cu124", "cpu", "avx2", "cu128"])

    def test_duplicate_channels_are_rejected(self, root):
        with pytest.raises(ConfigurationError):
            make_settings(root, channels=["cpu", "avx2", "cu124", "cu128", "cu124"])


class TestWorkflows:
    """The workflows consume configured values; they never restate them."""

    def test_no_workflow_hardcodes_the_package_or_repository(self, repository_root, settings):
        forbidden = (settings.package, settings.repository)
        workflows = sorted((repository_root / ".github" / "workflows").glob("*.y*ml"))
        assert workflows, "no workflow files found"
        offenders = [
            f"{path.name}: {value}"
            for path in workflows
            for value in forbidden
            if value in path.read_text(encoding="utf-8")
        ]
        assert offenders == []
