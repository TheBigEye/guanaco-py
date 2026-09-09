"""Build channels and target platforms.

This module is a leaf: it depends on nothing else inside :mod:`guanaco`. It
exists so that both the configuration layer (:mod:`guanaco.settings`) and the
document layer (:mod:`guanaco.models`) can talk about channels without
importing each other.

A *channel* is one way of building the same upstream version: a portable CPU
wheel, an AVX2 CPU wheel, or a wheel linked against one specific CUDA toolkit.
The channel appears in the release tag, the wheel platform tag and the index
URL; it is never part of the package version, so ``guanaco-py==0.3.49`` exists
exactly once per channel.
"""

from __future__ import annotations

import enum
import re
from dataclasses import dataclass

CHANNEL_PATTERN = re.compile(r"cpu|avx2|cu[0-9]+")
CUDA_CHANNEL_PATTERN = re.compile(r"cu[0-9]+")

#: Channels built without CUDA, in the order they must appear in the matrix.
CPU_CHANNELS = ("cpu", "avx2")

#: What ``auditwheel`` retags a repaired CPU wheel to (AlmaLinux 9 / glibc 2.34).
MANYLINUX_PLATFORM = "manylinux_2_34_x86_64"

#: The tag cibuildwheel produces before ``auditwheel`` has repaired the wheel.
UNREPAIRED_LINUX_PLATFORM = "linux_x86_64"

#: Windows wheels need no repair step and keep their native tag.
WINDOWS_PLATFORM = "win_amd64"


class Platform(enum.Enum):
    """An operating system we build wheels for."""

    LINUX = "linux"
    WINDOWS = "windows"

    @classmethod
    def parse(cls, value: str) -> Platform:
        """Parse a platform name, raising :class:`ValueError` if unknown."""
        for member in cls:
            if member.value == value:
                return member
        raise ValueError(f"Unknown platform: {value!r} (expected linux or windows)")

    @property
    def is_windows(self) -> bool:
        """Whether this platform produces ``win_amd64`` wheels."""
        return self is Platform.WINDOWS

    def __str__(self) -> str:
        """Return the plain platform name, so f-strings read naturally."""
        return self.value


@dataclass(frozen=True)
class Channel:
    """One distribution channel: portable CPU, AVX2 CPU, or a CUDA toolkit.

    The channel decides the release tag, the wheel platform tag and the compiler
    flags.
    """

    name: str

    def __post_init__(self) -> None:
        """Reject a malformed channel name as early as possible."""
        if not isinstance(self.name, str) or not CHANNEL_PATTERN.fullmatch(self.name):
            raise ValueError(f"Unknown channel: {self.name!r}")

    @property
    def is_cpu_variant(self) -> bool:
        """Whether this channel is built without CUDA (``cpu`` or ``avx2``)."""
        return self.name in CPU_CHANNELS

    @property
    def is_cuda(self) -> bool:
        """Whether this channel requires a CUDA toolkit."""
        return not self.is_cpu_variant

    def release_tag(self, version) -> str:
        """Return the release tag, e.g. ``v0.3.49-cu124``.

        The portable CPU channel keeps the bare ``vX.Y.Z`` tag; every other
        channel appends its name.
        """
        text = str(version)
        return f"v{text}" if self.name == CPU_CHANNELS[0] else f"v{text}-{self.name}"

    def wheel_platform(self, platform: Platform, *, repaired: bool = True) -> str:
        """Return the wheel platform tag for this channel and platform.

        Args:
            platform: The operating system the wheel was built on.
            repaired: Whether CPU wheels went through ``auditwheel``. A local
                build without Docker produces ``linux_x86_64``, which the
                release verifier deliberately rejects.
        """
        if platform.is_windows:
            return WINDOWS_PLATFORM
        if self.is_cpu_variant and repaired:
            return MANYLINUX_PLATFORM
        return UNREPAIRED_LINUX_PLATFORM

    def __str__(self) -> str:
        """Return the channel name."""
        return self.name


__all__ = [
    "CHANNEL_PATTERN",
    "CPU_CHANNELS",
    "CUDA_CHANNEL_PATTERN",
    "MANYLINUX_PLATFORM",
    "Channel",
    "Platform",
    "UNREPAIRED_LINUX_PLATFORM",
    "WINDOWS_PLATFORM",
]
