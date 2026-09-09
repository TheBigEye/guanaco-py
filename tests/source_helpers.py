"""A minimal upstream source layout, used by the offline preparation tests."""

from __future__ import annotations

import zipfile

METADATA = """[build-system]
requires = ["scikit-build-core[pyproject]>=0.9.2"]
build-backend = "scikit_build_core.build"
[project]
name = "llama_cpp_python"
dynamic = ["version"]
description = "Original description"
authors = [{name = "Original author"}]
maintainers = [{name = "Upstream"}]
license = {text = "MIT"}
dependencies = ["numpy>=1.21.6"]
requires-python = ">=3.9"
[project.optional-dependencies]
server = ["fastapi>=0.100"]
all = ["llama_cpp_python[server]"]
[tool.scikit-build]
wheel.packages = ["llama_cpp"]
cmake.verbose = true
cmake.minimum-version = "3.21"
minimum-version = "0.5.1"
sdist.include = [".git", "vendor/llama.cpp/*"]
[tool.scikit-build.metadata.version]
provider = "scikit_build_core.metadata.regex"
input = "llama_cpp/__init__.py"
[project.urls]
Homepage = "https://github.com/upstream/project"
"""

GITMODULES = """[submodule "vendor/llama.cpp"]
\tpath = vendor/llama.cpp
\turl = https://github.com/ggml-org/llama.cpp
"""


def fixture_source(tmp_path, version: str = "0.3.49"):
    """Create the smallest source tree that looks like the upstream bindings."""
    source = tmp_path / "source"
    (source / "llama_cpp").mkdir(parents=True)
    (source / "llama_cpp" / "__init__.py").write_text(f'__version__ = "{version}"\n')
    (source / "pyproject.toml").write_text(METADATA)
    (source / "LICENSE.md").write_text("MIT - original attribution")
    return source


def zip_source(path, files: dict) -> None:
    """Write a GitHub-style ZIP: one root directory containing `files`."""
    with zipfile.ZipFile(path, "w") as archive:
        for name, data in files.items():
            archive.writestr("repo-sha/" + name, data)


def raw_zip_member(name: str) -> zipfile.ZipInfo:
    """Keep an adversarial ZIP name identical on every test host.

    ``ZipInfo(name)`` would replace backslashes on Windows and truncate NUL
    bytes, so both fields are set afterwards and the archive really contains
    the requested name.
    """
    member = zipfile.ZipInfo()
    member.filename = member.orig_filename = name
    return member
