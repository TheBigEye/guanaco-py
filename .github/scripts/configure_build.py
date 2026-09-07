"""Resolve CI build options from the frozen source manifest, never live upstream state."""

from __future__ import annotations

import argparse
import json
import os
import shlex
from pathlib import Path

from release_common import outputs, repository_name, validate_build_matrix, version_key
from verify_wheels import selector

CPU_OFF = ("CUDA", "METAL", "VULKAN", "BLAS", "NATIVE", "BACKEND_DL", "CPU_ALL_VARIANTS")
CPU_SIMD = ("AVX", "AVX2", "FMA", "F16C", "SSE42", "BMI2")


def cpu_options(manifest: dict, channel: str, platform: str) -> dict:
    validate_build_matrix(manifest)
    if channel not in ("cpu", "avx2") or platform not in ("linux", "windows"):
        raise ValueError("Expected a CPU/AVX2 channel and Linux/Windows platform")
    flags = [f"-DGGML_{name}=OFF" for name in CPU_OFF]
    flags += [f"-DGGML_{name}={'ON' if channel == 'avx2' else 'OFF'}" for name in CPU_SIMD]
    cmake = " ".join(flags)
    compiler = "CC=/usr/bin/gcc CXX=/usr/bin/g++ " if platform == "linux" else ""
    return {
        "build": selector(manifest["python_versions"], platform),
        "cibw_environment": compiler + f'CMAKE_ARGS="{cmake}"',
        "artifact": f"guanaco-py-{channel}-{platform}-x64",
    }


def cuda_options(manifest: dict, channel: str) -> dict:
    validate_build_matrix(manifest)
    if channel not in manifest["cuda"]:
        raise ValueError(f"CUDA channel is not in the prepared manifest: {channel}")
    settings = manifest["cuda"][channel]
    flags = [
        "-DGGML_CUDA=ON",
        f"-DCMAKE_CUDA_ARCHITECTURES={settings['architectures']}",
        "-DGGML_CUDA_FORCE_MMQ=OFF",
        "-DGGML_NATIVE=OFF",
        "-DLLAMA_BUILD_EXAMPLES=OFF",
        "-DLLAMA_BUILD_TESTS=OFF",
        "-DLLAMA_BUILD_SERVER=OFF",
    ]
    linux = [*flags, "-DCMAKE_EXE_LINKER_FLAGS=-L/usr/local/cuda/lib64/stubs -lcuda"]
    return {
        "short": channel,
        "version": settings["toolkit"],
        "python": manifest["python_versions"],
        "legacy_msvc": settings["legacy_msvc"],
        "cmake_linux": shlex.join(linux),
        "cmake_windows": shlex.join(flags),
        "cuda_flags": "--allow-unsupported-compiler -D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH"
        if settings["legacy_msvc"]
        else "",
    }


def image_tags(repository: str, version: str, promote_latest: bool) -> str:
    repository_name(repository)
    version_key(version)
    image = "ghcr.io/" + repository.lower()
    tags = [f"{image}:v{version}"]
    if promote_latest:
        tags.append(f"{image}:latest")
    return ",".join(tags)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=["cpu", "cuda", "docker"])
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--channel")
    parser.add_argument("--platform", choices=["linux", "windows"])
    parser.add_argument("--version", default=os.getenv("VERSION", ""))
    parser.add_argument("--repository", default=os.getenv("GITHUB_REPOSITORY", ""))
    parser.add_argument(
        "--promote-latest", action="store_true", default=os.getenv("PROMOTE_LATEST") == "true"
    )
    args = parser.parse_args()
    if args.kind == "docker":
        values = {"tags": image_tags(args.repository, args.version, args.promote_latest)}
    else:
        if args.manifest is None or not args.channel:
            parser.error("CPU/CUDA configuration requires --manifest and --channel")
        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
        if args.version and args.version != manifest["version"]:
            raise ValueError("Prepared manifest version does not match the workflow input")
        values = (
            cpu_options(manifest, args.channel, args.platform)
            if args.kind == "cpu"
            else cuda_options(manifest, args.channel)
        )
    outputs(os.getenv("GITHUB_OUTPUT"), **values)
    print(json.dumps(values, indent=2))


if __name__ == "__main__":
    main()
