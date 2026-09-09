"""Compiler flags, build selectors and container tags.

This module answers "how exactly do we compile this wheel". Every value CI needs
is produced here as a typed object that can be printed, tested and turned into
GitHub Actions outputs. No configuration lives inline in a YAML file.
"""

from __future__ import annotations

import re
import shlex
from dataclasses import dataclass

from .models import Channel, Plan, Platform
from .settings import CudaChannel, Settings

# Backends that must stay off for a plain CPU wheel.
DISABLED_BACKENDS = (
    "CUDA",
    "METAL",
    "VULKAN",
    "BLAS",
    "NATIVE",
    "BACKEND_DL",
    "CPU_ALL_VARIANTS",
)

# SIMD instruction sets. The portable channel disables all of them so the wheel
# runs on any x86-64 CPU; the AVX2 channel enables them for a fixed target.
SIMD_FEATURES = ("AVX", "AVX2", "FMA", "F16C", "SSE42", "BMI2")


class ToolchainError(ValueError):
    """Raised when a build cannot be configured, e.g. an unselected platform."""


def cpu_artifact(package: str, prefix: str, channel: Channel, platform: Platform) -> str:
    """Return the artifact name of a CPU or AVX2 build job.

    Args:
        package: Distribution name, e.g. ``guanaco-py``.
        prefix: ``test-`` for a rehearsal, empty for a release.
        channel: ``cpu`` or ``avx2``.
        platform: The operating system the job runs on.
    """
    return f"{prefix}{package}-{channel}-{platform}-x64"


def cuda_artifact_base(package: str, prefix: str, channel: Channel, platform: Platform) -> str:
    """Return the artifact name of a CUDA job minus its Python suffix.

    CUDA jobs produce a single wheel each, so the workflow appends
    ``-py<version>`` per matrix cell; CPU jobs produce the whole matrix and
    leave it out.
    """
    return f"{prefix}{package}-cuda-{platform}-x64-{channel}"


def cuda_artifact(
    package: str, prefix: str, channel: Channel, platform: Platform, python: str
) -> str:
    """Return the artifact name of one CUDA build job."""
    return f"{cuda_artifact_base(package, prefix, channel, platform)}-py{python}"


def parse_artifact(package: str, name: str) -> tuple[Channel, Platform]:
    """Split a build artifact name back into its channel and platform.

    Raises:
        ToolchainError: If the name is not one of ours.
    """
    escaped = re.escape(package)
    match = re.fullmatch(rf"{escaped}-(cpu|avx2)-(linux|windows)-x64", name)
    if match:
        return Channel(match[1]), Platform(match[2])
    match = re.fullmatch(rf"{escaped}-cuda-(linux|windows)-x64-(cu[0-9]+)-py(3[.][0-9]+)", name)
    if match:
        return Channel(match[2]), Platform(match[1])
    raise ToolchainError(f"Unexpected build artifact: {name}")


@dataclass(frozen=True)
class BuildScope:
    """How far a build reaches: which platforms, and is it a real release.

    A release always builds Linux and Windows. A manual test build may narrow
    that down, and marks its artifacts with a ``test-`` prefix so a rehearsal can
    never be mistaken for a release.
    """

    test_only: bool
    platforms: tuple[Platform, ...]

    @classmethod
    def from_plan(cls, plan: Plan) -> BuildScope:
        """Derive the scope from a frozen plan."""
        return cls(test_only=plan.test_only, platforms=plan.build_platforms())

    @property
    def artifact_prefix(self) -> str:
        """``test-`` for a rehearsal, empty for a release."""
        return "test-" if self.test_only else ""

    def includes(self, platform: Platform) -> bool:
        """Whether `platform` is part of this build."""
        return platform in self.platforms

    def to_mapping(self) -> dict:
        """Return the flat workflow outputs shared by every build job."""
        return {
            "test_only": self.test_only,
            "artifact_prefix": self.artifact_prefix,
            "linux": self.includes(Platform.LINUX),
            "windows": self.includes(Platform.WINDOWS),
        }


@dataclass(frozen=True)
class PlatformRow:
    """One cell of the CPU/AVX2 job matrix."""

    platform: Platform
    runner: str
    label: str
    architectures: str

    def to_mapping(self) -> dict:
        """Return the workflow matrix entry for this row."""
        return {
            "os": self.runner,
            "platform": self.platform.value,
            "label": self.label,
            "cibw_archs": self.architectures,
        }


@dataclass(frozen=True)
class CpuBuild:
    """Everything needed to build CPU or AVX2 wheels for one platform."""

    package: str
    scope: BuildScope
    channel: Channel
    platform: Platform
    build_selector: str
    environment: str
    artifact: str

    def to_mapping(self) -> dict:
        """Return the workflow outputs for a CPU/AVX2 build job."""
        return {
            **self.scope.to_mapping(),
            "package": self.package,
            "build": self.build_selector,
            "cibw_environment": self.environment,
            "artifact": self.artifact,
        }


@dataclass(frozen=True)
class CudaBuild:
    """Everything needed to build one CUDA channel on either platform."""

    package: str
    scope: BuildScope
    channel: CudaChannel
    python_versions: tuple[str, ...]
    cmake_linux: str
    cmake_windows: str
    cuda_flags: str

    def artifact(self, platform: Platform) -> str:
        """Return the artifact name of one CUDA job, without its Python suffix."""
        return cuda_artifact_base(
            self.package, self.scope.artifact_prefix, self.channel.name, platform
        )

    def to_mapping(self) -> dict:
        """Return the workflow outputs for a CUDA build job."""
        return {
            **self.scope.to_mapping(),
            "package": self.package,
            "short": self.channel.name,
            "artifact_linux": self.artifact(Platform.LINUX),
            "artifact_windows": self.artifact(Platform.WINDOWS),
            "version": self.channel.toolkit,
            "python": list(self.python_versions),
            "legacy_msvc": self.channel.legacy_msvc,
            "cmake_linux": self.cmake_linux,
            "cmake_windows": self.cmake_windows,
            "cuda_flags": self.cuda_flags,
        }


@dataclass(frozen=True)
class ContainerImage:
    """The image tags a Docker build should push."""

    repository: str
    version: str
    promote_latest: bool

    @property
    def name(self) -> str:
        """Lowercase fully qualified image name, e.g. ``ghcr.io/owner/repo``."""
        return "ghcr.io/" + self.repository.lower()

    @property
    def tags(self) -> tuple[str, ...]:
        """The tags to push: the version always, ``latest`` only when asked."""
        result = [f"{self.name}:v{self.version}"]
        if self.promote_latest:
            result.append(f"{self.name}:latest")
        return tuple(result)

    def to_mapping(self) -> dict:
        """Return the workflow output carrying the comma separated tags."""
        return {"tags": ",".join(self.tags)}


class Toolchain:
    """Turns a frozen plan into concrete compiler flags and job settings."""

    def __init__(self, settings: Settings) -> None:
        """Remember the distribution name and the CUDA configuration."""
        self.settings = settings

    def scope(self, plan: Plan) -> BuildScope:
        """Return the platform scope of `plan`."""
        return BuildScope.from_plan(plan)

    def platform_matrix(self, plan: Plan) -> list[PlatformRow]:
        """Return the CPU/AVX2 job matrix rows selected by `plan`."""
        rows = (
            PlatformRow(
                platform=Platform.LINUX,
                runner="ubuntu-latest",
                label="Linux x64 · manylinux_2_34",
                architectures="x86_64",
            ),
            PlatformRow(
                platform=Platform.WINDOWS,
                runner="windows-latest",
                label="Windows x64 · AMD64",
                architectures="AMD64",
            ),
        )
        selected = self.scope(plan).platforms
        return [row for row in rows if row.platform in selected]

    def build_selector(self, plan: Plan, platform: Platform) -> str:
        """Return the ``CIBW_BUILD`` selector for `platform`.

        CPU jobs ask cibuildwheel for the manylinux selector and let
        ``auditwheel`` retag the result; that is what makes the published wheel
        ``manylinux_2_34_x86_64`` rather than a raw ``linux_x86_64`` build.
        """
        suffix = "manylinux_x86_64" if platform is Platform.LINUX else "win_amd64"
        return " ".join(f"cp{v.replace('.', '')}-{suffix}" for v in plan.python_versions)

    def cpu(self, plan: Plan, channel: Channel, platform: Platform) -> CpuBuild:
        """Return the CPU/AVX2 build settings for one channel and platform.

        Args:
            plan: The frozen build plan.
            channel: ``cpu`` for a portable wheel, ``avx2`` for a fixed target.
            platform: The operating system to build on.

        Raises:
            ToolchainError: If the channel is a CUDA one or the platform was not
                selected in this build.
        """
        if not channel.is_cpu_variant:
            raise ToolchainError(f"{channel.name} is not a CPU channel")
        scope = self.scope(plan)
        if not scope.includes(platform):
            raise ToolchainError(f"Platform {platform} was not selected in this build")
        flags = [f"-DGGML_{name}=OFF" for name in DISABLED_BACKENDS]
        enabled = "ON" if channel.name == "avx2" else "OFF"
        flags += [f"-DGGML_{name}={enabled}" for name in SIMD_FEATURES]
        # AlmaLinux 9's GCC 11 matches the manylinux_2_34 GLIBCXX policy; the
        # container's newer default compiler can exceed it.
        compiler = "CC=/usr/bin/gcc CXX=/usr/bin/g++ " if platform is Platform.LINUX else ""
        return CpuBuild(
            scope=scope,
            channel=channel,
            platform=platform,
            package=self.settings.package,
            build_selector=self.build_selector(plan, platform),
            environment=compiler + 'CMAKE_ARGS="' + " ".join(flags) + '"',
            artifact=cpu_artifact(self.settings.package, scope.artifact_prefix, channel, platform),
        )

    def cuda(self, plan: Plan, channel: Channel) -> CudaBuild:
        """Return the CUDA build settings for one channel.

        Raises:
            ToolchainError: If the channel is not part of the frozen matrix.
        """
        settings = self.settings.cuda.get(channel.name)
        if settings is None:
            raise ToolchainError(f"CUDA channel is not in the prepared manifest: {channel.name}")
        flags = [
            "-DGGML_CUDA=ON",
            f"-DCMAKE_CUDA_ARCHITECTURES={settings.architectures}",
            "-DGGML_CUDA_FORCE_MMQ=OFF",
            "-DGGML_NATIVE=OFF",
            "-DLLAMA_BUILD_EXAMPLES=OFF",
            "-DLLAMA_BUILD_TESTS=OFF",
            "-DLLAMA_BUILD_SERVER=OFF",
        ]
        linux = [*flags, "-DCMAKE_EXE_LINKER_FLAGS=-L/usr/local/cuda/lib64/stubs -lcuda"]
        return CudaBuild(
            package=self.settings.package,
            scope=self.scope(plan),
            channel=settings,
            python_versions=plan.python_versions,
            cmake_linux=shlex.join(linux),
            cmake_windows=shlex.join(flags),
            cuda_flags=(
                "--allow-unsupported-compiler -D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH"
                if settings.legacy_msvc
                else ""
            ),
        )

    def image(self, version: str, promote_latest: bool) -> ContainerImage:
        """Return the container image tags for a published version."""
        return ContainerImage(
            repository=self.settings.repository,
            version=version,
            promote_latest=promote_latest,
        )


__all__ = [
    "BuildScope",
    "ContainerImage",
    "CpuBuild",
    "CudaBuild",
    "PlatformRow",
    "Toolchain",
    "ToolchainError",
    "cpu_artifact",
    "cuda_artifact",
    "cuda_artifact_base",
    "parse_artifact",
]
