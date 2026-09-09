"""Turning a plan into compiler flags, job settings and container tags."""

from __future__ import annotations

import json

import pytest
from helpers import make_plan, make_settings

from guanaco.models import Channel, Platform
from guanaco.toolchain import (
    BuildScope,
    ContainerImage,
    Toolchain,
    ToolchainError,
    cpu_artifact,
    cuda_artifact,
    parse_artifact,
)


class TestArtifactNames:
    def test_cpu_artifacts_omit_the_python_version(self):
        assert (
            cpu_artifact("guanaco-py", "", Channel("cpu"), Platform.LINUX)
            == "guanaco-py-cpu-linux-x64"
        )

    def test_rehearsals_are_prefixed(self):
        assert (
            cpu_artifact("guanaco-py", "test-", Channel("avx2"), Platform.WINDOWS)
            == "test-guanaco-py-avx2-windows-x64"
        )

    def test_cuda_artifacts_carry_the_python_version(self):
        assert (
            cuda_artifact("guanaco-py", "", Channel("cu124"), Platform.LINUX, "3.13")
            == "guanaco-py-cuda-linux-x64-cu124-py3.13"
        )

    def test_names_round_trip(self):
        for name in (
            "guanaco-py-cpu-linux-x64",
            "guanaco-py-avx2-windows-x64",
            "guanaco-py-cuda-linux-x64-cu124-py3.13",
        ):
            channel, platform = parse_artifact("guanaco-py", name)
            assert name in (
                cpu_artifact("guanaco-py", "", channel, platform),
                cuda_artifact("guanaco-py", "", channel, platform, "3.13"),
            )

    def test_foreign_names_are_rejected(self):
        for name in ("guanaco-py-cpu-linux", "something-else-cpu-linux-x64", "guanaco-py-cu124"):
            with pytest.raises(ToolchainError):
                parse_artifact("guanaco-py", name)


class TestWorkflowOutputs:
    """Jobs read their names from the configuration, never from a literal."""

    def test_cpu_outputs_name_the_package_and_artifact(self, settings):
        plan = make_plan(settings)
        outputs = Toolchain(settings).cpu(plan, Channel("cpu"), Platform.LINUX).to_mapping()
        assert outputs["package"] == settings.package
        assert outputs["artifact"] == f"{settings.package}-cpu-linux-x64"

    def test_cuda_outputs_name_one_artifact_per_platform(self, settings):
        plan = make_plan(settings)
        outputs = Toolchain(settings).cuda(plan, Channel("cu124")).to_mapping()
        assert outputs["package"] == settings.package
        assert outputs["artifact_linux"] == f"{settings.package}-cuda-linux-x64-cu124"
        assert outputs["artifact_windows"] == f"{settings.package}-cuda-windows-x64-cu124"

    def test_a_renamed_package_flows_into_the_outputs(self, root):
        renamed = make_settings(root, package="acme-llama")
        plan = make_plan(renamed)
        outputs = Toolchain(renamed).cpu(plan, Channel("cpu"), Platform.LINUX).to_mapping()
        assert outputs["package"] == "acme-llama"
        assert outputs["artifact"] == "acme-llama-cpu-linux-x64"


class TestBuildScope:
    def test_a_release_covers_both_platforms(self, settings):
        scope = BuildScope.from_plan(make_plan(settings))
        assert scope.platforms == (Platform.LINUX, Platform.WINDOWS)
        assert scope.artifact_prefix == ""
        assert scope.to_mapping()["test_only"] is False

    def test_a_rehearsal_can_narrow_the_platforms(self, settings):
        plan = make_plan(settings, test_only=True, platforms=(Platform.LINUX,))
        scope = BuildScope.from_plan(plan)
        assert scope.artifact_prefix == "test-"
        assert scope.includes(Platform.LINUX) and not scope.includes(Platform.WINDOWS)


class TestToolchain:
    def test_cpu_build_asks_for_a_manylinux_selector(self, settings):
        plan = make_plan(settings)
        build = Toolchain(settings).cpu(plan, Channel("cpu"), Platform.LINUX)
        assert "cp313-manylinux_x86_64" in build.build_selector
        assert "cp39-manylinux_x86_64" in build.build_selector
        assert build.artifact == "guanaco-py-cpu-linux-x64"

    def test_cpu_build_uses_an_old_compiler_on_linux(self, settings):
        build = Toolchain(settings).cpu(make_plan(settings), Channel("cpu"), Platform.LINUX)
        assert "CC=/usr/bin/gcc" in build.environment
        windows = Toolchain(settings).cpu(make_plan(settings), Channel("cpu"), Platform.WINDOWS)
        assert "CC=/usr/bin/gcc" not in windows.environment

    def test_avx2_enables_the_simd_features(self, settings):
        environment = (
            Toolchain(settings)
            .cpu(make_plan(settings), Channel("avx2"), Platform.LINUX)
            .environment
        )
        assert "-DGGML_AVX=ON" in environment
        assert "-DGGML_NATIVE=OFF" in environment or "-DGGML_F16C=ON" in environment

    def test_plain_cpu_disables_the_simd_features(self, settings):
        environment = (
            Toolchain(settings).cpu(make_plan(settings), Channel("cpu"), Platform.LINUX).environment
        )
        assert "-DGGML_AVX2=OFF" in environment
        assert "-DGGML_BLAS=OFF" in environment

    def test_a_cuda_channel_is_refused_by_the_cpu_builder(self, settings):
        with pytest.raises(ToolchainError):
            Toolchain(settings).cpu(make_plan(settings), Channel("cu124"), Platform.LINUX)

    def test_an_unselected_platform_is_refused(self, settings):
        plan = make_plan(settings, test_only=True, platforms=(Platform.LINUX,))
        with pytest.raises(ToolchainError, match="was not selected"):
            Toolchain(settings).cpu(plan, Channel("cpu"), Platform.WINDOWS)

    def test_cuda_build_targets_the_configured_architectures(self, settings):
        build = Toolchain(settings).cuda(make_plan(settings), Channel("cu124"))
        assert "-DGGML_CUDA=ON" in build.cmake_linux
        assert "-DCMAKE_CUDA_ARCHITECTURES=75;80;86;89;90" in build.cmake_windows
        assert build.channel.toolkit == "12.4.1"
        assert build.python_versions == settings.python_versions

    def test_cuda_flags_add_the_stubs_directory_on_linux_only(self, settings):
        build = Toolchain(settings).cuda(make_plan(settings), Channel("cu124"))
        assert "-L/usr/local/cuda/lib64/stubs" in build.cmake_linux
        assert "-L/usr/local/cuda/lib64/stubs" not in build.cmake_windows

    def test_legacy_msvc_gets_a_compiler_override(self, root):
        settings = make_settings(
            root,
            cuda={"cu121": {"toolkit": "12.1.1", "architectures": "75", "legacy_msvc": True}},
        )
        build = Toolchain(settings).cuda(make_plan(settings), Channel("cu121"))
        assert "--allow-unsupported-compiler" in build.cuda_flags

    def test_an_unknown_cuda_channel_is_refused(self, settings):
        with pytest.raises(ValueError):
            Toolchain(settings).cuda(make_plan(settings), Channel("cu999"))

    def test_platform_matrix_is_filtered(self, settings):
        plan = make_plan(settings)
        rows = Toolchain(settings).platform_matrix(plan)
        assert [row.platform for row in rows] == [Platform.LINUX, Platform.WINDOWS]
        narrow = make_plan(settings, test_only=True, platforms=(Platform.WINDOWS,))
        assert [row.platform for row in Toolchain(settings).platform_matrix(narrow)] == [
            Platform.WINDOWS
        ]

    def test_outputs_are_json_serialisable(self, settings):
        build = Toolchain(settings).cpu(make_plan(settings), Channel("cpu"), Platform.LINUX)
        document = build.to_mapping()
        assert json.loads(json.dumps(document))["artifact"] == "guanaco-py-cpu-linux-x64"
        cuda = Toolchain(settings).cuda(make_plan(settings), Channel("cu124")).to_mapping()
        assert json.loads(json.dumps(cuda))["short"] == "cu124"


class TestContainerImage:
    def test_tags_are_lowercased_and_versioned(self):
        image = ContainerImage("TheBigEye/Guanaco-Py", "0.3.49", True)
        assert image.name == "ghcr.io/thebigeye/guanaco-py"
        assert image.tags == (
            "ghcr.io/thebigeye/guanaco-py:v0.3.49",
            "ghcr.io/thebigeye/guanaco-py:latest",
        )

    def test_latest_is_optional(self):
        image = ContainerImage("owner/repo", "0.3.49", False)
        assert image.tags == ("ghcr.io/owner/repo:v0.3.49",)
        assert image.to_mapping() == {"tags": "ghcr.io/owner/repo:v0.3.49"}

    def test_the_image_follows_the_repository(self, settings):
        image = Toolchain(settings).image("0.3.49", True)
        assert image.name == "ghcr.io/thebigeye/guanaco-py"
