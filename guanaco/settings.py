"""Configuration for every part of the build system.

Everything the build needs to know -- which repository we publish to, which
upstream we follow, which Python versions and CUDA toolkits we build for --
lives in ``.github/build-matrix.json``. This module turns that file into
typed, self-validating objects and hands them to the rest of the package.

Nothing here is hardcoded on purpose. Renaming the repository, renaming the
distribution, adding a CUDA toolkit or dropping a Python version are all
configuration edits; no Python file has to change.
"""

from __future__ import annotations

import copy
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

from .channels import CPU_CHANNELS, Channel

# Environment variables that override the configuration file. They exist so the
# Docker helpers (which copy this package into an image) and local experiments
# can point the build at a different checkout without editing any file.
ROOT_VARIABLE = "GUANACO_ROOT"
MATRIX_VARIABLE = "GUANACO_MATRIX"
REPOSITORY_VARIABLE = "GUANACO_REPOSITORY"

# Where the configuration file lives, relative to the repository root.
MATRIX_LOCATION = ".github/build-matrix.json"

# Where the reviewed local patches and the index assets live.
PATCHES_LOCATION = ".github/patches"
DOCS_LOCATION = "docs"

CUDA_CHANNEL_PATTERN = re.compile(r"cu[0-9]+")
PYTHON_VERSION_PATTERN = re.compile(r"3\.(?:9|[1-9][0-9]+)")
ARCHITECTURE_PATTERN = re.compile(r"[0-9]+[af]?(?:;[0-9]+[af]?)*")


class ConfigurationError(ValueError):
    """Raised when the build configuration is missing, malformed or inconsistent."""


def _require_mapping(value: object, what: str) -> dict:
    """Return `value` as a mapping or complain with a helpful message."""
    if not isinstance(value, dict):
        raise ConfigurationError(f"{what} must be a JSON object")
    return value


@dataclass(frozen=True)
class CudaChannel:
    """One CUDA toolkit flavour, for example ``cu124`` built with toolkit 12.4.1.

    Attributes:
        name: Channel identifier, always ``cu`` followed by the toolkit's major
            and minor version, e.g. ``cu124``.
        toolkit: Exact toolkit version to install, e.g. ``12.4.1``.
        architectures: Semicolon-separated ``CMAKE_CUDA_ARCHITECTURES`` list.
        legacy_msvc: Whether this toolkit needs MSVC compatibility flags on
            Windows. Older toolkits (12.1-12.3) reject newer compilers.
    """

    name: str
    toolkit: str
    architectures: str
    legacy_msvc: bool

    @classmethod
    def from_mapping(cls, name: str, values: dict) -> CudaChannel:
        """Build a channel from its ``build-matrix.json`` entry."""
        values = _require_mapping(values, f"CUDA channel {name!r}")
        toolkit = values.get("toolkit")
        architectures = values.get("architectures")
        legacy_msvc = values.get("legacy_msvc")
        if not isinstance(toolkit, str) or not re.fullmatch(r"[0-9]+\.[0-9]+\.[0-9]+", toolkit):
            raise ConfigurationError(f"CUDA channel {name!r} needs an X.Y.Z toolkit version")
        if not isinstance(architectures, str) or not ARCHITECTURE_PATTERN.fullmatch(architectures):
            raise ConfigurationError(f"CUDA channel {name!r} has an invalid architecture list")
        if type(legacy_msvc) is not bool:
            raise ConfigurationError(f"CUDA channel {name!r}: legacy_msvc must be a boolean")
        channel = cls(name, toolkit, architectures, legacy_msvc)
        channel.validate()
        return channel

    def to_mapping(self) -> dict:
        """Return the JSON shape used by ``build-matrix.json`` and provenance."""
        return {
            "toolkit": self.toolkit,
            "architectures": self.architectures,
            "legacy_msvc": self.legacy_msvc,
        }

    def validate(self) -> None:
        """Check that the channel name agrees with its toolkit version."""
        if not CUDA_CHANNEL_PATTERN.fullmatch(self.name):
            raise ConfigurationError(f"Invalid CUDA channel name: {self.name!r}")
        major, minor, _ = self.toolkit.split(".")
        if self.name != f"cu{major}{minor}":
            raise ConfigurationError(
                f"CUDA channel {self.name!r} does not match toolkit {self.toolkit!r}"
            )

    @property
    def label(self) -> str:
        """Human readable name, e.g. ``cu124`` -> ``CUDA 12.4``."""
        return f"CUDA {self.name[2:-1]}.{self.name[-1]}"


@dataclass(frozen=True)
class BuildMatrix:
    """The part of the configuration that gets frozen into a release.

    Once a version family starts publishing, its matrix must not drift: a retry
    has to reuse the same upstream, the same Python versions and the same CUDA
    settings. This object is exactly that frozen set of values, and it is what
    gets embedded in each release's provenance marker.
    """

    upstream: str
    package: str
    python_versions: tuple[str, ...]
    channels: tuple[Channel, ...]
    cuda: dict[str, CudaChannel]

    def __post_init__(self) -> None:
        """Accept plain channel names as well as :class:`Channel` objects."""
        object.__setattr__(
            self,
            "channels",
            tuple(
                item if isinstance(item, Channel) else Channel(str(item)) for item in self.channels
            ),
        )

    @classmethod
    def from_mapping(cls, values: dict, *, default_package: str | None = None) -> BuildMatrix:
        """Build a matrix from the top level of ``build-matrix.json``.

        Args:
            values: The mapping to read.
            default_package: Fallback distribution name for documents written
                before the matrix carried one, such as the provenance snapshot
                of an already published release.
        """
        values = _require_mapping(values, "Build matrix")
        python_versions = values.get("python_versions")
        if not isinstance(python_versions, list) or not python_versions:
            raise ConfigurationError("python_versions must be a nonempty list")
        for version in python_versions:
            if not isinstance(version, str) or not PYTHON_VERSION_PATTERN.fullmatch(version):
                raise ConfigurationError(f"Unsupported Python version in matrix: {version!r}")
        if len(set(python_versions)) != len(python_versions):
            raise ConfigurationError("python_versions contains duplicates")

        raw_cuda = _require_mapping(values.get("cuda"), "CUDA matrix")
        cuda = {
            str(name): CudaChannel.from_mapping(str(name), settings)
            for name, settings in raw_cuda.items()
        }
        channels = tuple(values.get("channels", (*CPU_CHANNELS, *cuda)))
        upstream = values.get("upstream")
        package = values.get("package", default_package)
        if not isinstance(upstream, str) or not upstream:
            raise ConfigurationError("Build matrix is missing the upstream repository")
        if not isinstance(package, str) or not package:
            raise ConfigurationError("Build matrix is missing the package name")
        matrix = cls(
            upstream=upstream,
            package=package,
            python_versions=tuple(python_versions),
            channels=tuple(_channel(name) for name in channels),
            cuda=cuda,
        )
        matrix.validate()
        return matrix

    def to_mapping(self) -> dict:
        """Return the JSON shape stored in plans, manifests and provenance."""
        return {
            "upstream": self.upstream,
            "package": self.package,
            "python_versions": list(self.python_versions),
            "channels": [channel.name for channel in self.channels],
            "cuda": {name: channel.to_mapping() for name, channel in self.cuda.items()},
        }

    def validate(self) -> None:
        """Check that the channel list and the CUDA settings agree."""
        if not self.python_versions:
            raise ConfigurationError("Python matrix must not be empty")
        if len(set(self.python_versions)) != len(self.python_versions):
            raise ConfigurationError("Python matrix contains duplicate versions")
        expected = {*CPU_CHANNELS, *self.cuda}
        names = tuple(channel.name for channel in self.channels)
        if names[: len(CPU_CHANNELS)] != CPU_CHANNELS:
            raise ConfigurationError(f"Channel list must start with {list(CPU_CHANNELS)}")
        if len(set(names)) != len(names):
            raise ConfigurationError("Channel list contains duplicates")
        if set(names) != expected:
            raise ConfigurationError("Channel list does not match the CUDA matrix")
        for channel in self.cuda.values():
            channel.validate()


@dataclass(frozen=True)
class Settings:
    """Everything the build system needs, resolved once and passed around.

    Attributes:
        root: Repository root, used to find the patches and the index assets.
        repository: The GitHub repository we publish to, ``owner/name``.
        matrix: The frozen :class:`BuildMatrix`.
    """

    root: Path
    repository: str
    matrix: BuildMatrix

    @classmethod
    def load(
        cls,
        *,
        root: Path | str | None = None,
        matrix: Path | str | None = None,
        repository: str | None = None,
    ) -> Settings:
        """Resolve the configuration from arguments, environment and the file.

        Precedence is: explicit argument, then environment variable, then
        ``build-matrix.json``. The repository additionally falls back to
        ``GITHUB_REPOSITORY`` so that Actions runs need no extra wiring.
        """
        resolved_root = Path(
            root or os.environ.get(ROOT_VARIABLE) or Path(__file__).resolve().parents[1]
        ).resolve()
        matrix_path = Path(
            matrix or os.environ.get(MATRIX_VARIABLE) or resolved_root / MATRIX_LOCATION
        )
        if not matrix_path.is_file():
            raise ConfigurationError(f"Build matrix not found: {matrix_path}")
        try:
            data = json.loads(matrix_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise ConfigurationError(f"{matrix_path} is not valid JSON: {error}") from error
        data = _require_mapping(data, f"{matrix_path}")

        resolved_repository = (
            repository
            or os.environ.get(REPOSITORY_VARIABLE)
            or os.environ.get("GITHUB_REPOSITORY")
            or data.get("repository")
        )
        if not isinstance(resolved_repository, str) or not resolved_repository:
            raise ConfigurationError(
                f"No target repository: set 'repository' in {matrix_path} "
                f"or the {REPOSITORY_VARIABLE} environment variable"
            )
        return cls(
            root=resolved_root,
            repository=resolved_repository,
            matrix=BuildMatrix.from_mapping(data),
        )

    # -- Convenience accessors -------------------------------------------------

    @property
    def upstream(self) -> str:
        """Repository we rebuild, e.g. ``JamePeng/llama-cpp-python``."""
        return self.matrix.upstream

    @property
    def package(self) -> str:
        """Distribution name we publish, e.g. ``guanaco-py``."""
        return self.matrix.package

    @property
    def python_versions(self) -> tuple[str, ...]:
        """Python versions to build, e.g. ``("3.9", ..., "3.14")``."""
        return self.matrix.python_versions

    @property
    def upstream_package(self) -> str:
        """Distribution name upstream publishes, derived from its repository name.

        Deriving it means pointing the build at a fork of the bindings needs no
        code change: the name follows the repository.
        """
        return self.matrix.upstream.rsplit("/", 1)[-1].casefold()

    @property
    def channels(self) -> tuple[Channel, ...]:
        """Every channel we can build, CPU variants first."""
        return self.matrix.channels

    @property
    def cuda(self) -> dict[str, CudaChannel]:
        """CUDA channels keyed by name."""
        return self.matrix.cuda

    @property
    def patches_dir(self) -> Path:
        """Directory holding the reviewed local patches."""
        return self.root / PATCHES_LOCATION

    @property
    def docs_dir(self) -> Path:
        """Directory holding the icon and stylesheet used by the index."""
        return self.root / DOCS_LOCATION

    def describe(self) -> str:
        """Return a short human readable summary, used by ``guanaco explain``."""
        cuda = ", ".join(sorted(self.cuda)) or "none"
        return (
            f"repository:      {self.repository}\n"
            f"upstream:        {self.upstream}\n"
            f"package:         {self.package}\n"
            f"python versions: {', '.join(self.python_versions)}\n"
            f"channels:        {', '.join(channel.name for channel in self.channels)}\n"
            f"cuda toolkits:   {cuda}\n"
            f"root:            {self.root}\n"
            f"patches:         {self.patches_dir}"
        )


def _channel(name) -> Channel:
    """Build a channel, reporting a bad or unknown name as a configuration error."""
    try:
        return Channel(name)
    except ValueError as error:
        raise ConfigurationError(str(error)) from error


def repository_name(value: str) -> str:
    """Validate an ``owner/name`` GitHub repository and return it unchanged."""
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", value):
        raise ConfigurationError(f"Invalid GitHub repository: {value!r}")
    if any(part in (".", "..") for part in value.split("/")):
        raise ConfigurationError("Repository paths cannot contain dot segments")
    return value


def freeze_matrix(matrix: BuildMatrix) -> dict:
    """Return an independent copy of a matrix, safe to embed in a document."""
    return copy.deepcopy(matrix.to_mapping())
