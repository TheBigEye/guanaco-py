"""Download immutable source ZIPs; adapt packaging metadata, never binding code."""

from __future__ import annotations

import argparse
import configparser
import difflib
import gzip
import json
import os
import re
import subprocess
import tarfile
import tempfile
from pathlib import Path, PurePosixPath

import tomlkit
from archive_utils import empty_destination
from archive_utils import extract_zip as extract_zip
from archive_utils import portable_path as safe_path
from download_utils import download
from release_common import (
    PACKAGE,
    SHA,
    UPSTREAM,
    GitHub,
    repository_name,
    sha256,
    version_key,
    write_json,
)


def download_archive(repository: str, commit: str, destination: Path) -> str:
    repository_name(repository)
    if not SHA.fullmatch(commit):
        raise ValueError("Refusing an archive without an immutable SHA")
    url = f"https://codeload.github.com/{repository}/zip/{commit}"
    digest = download(url, destination)
    print(f"Downloaded {repository}@{commit}: {destination.stat().st_size} bytes")
    return digest


def submodule_repositories(source: Path) -> dict[str, str]:
    path = source / ".gitmodules"
    if not path.exists():
        return {}
    parser = configparser.ConfigParser(interpolation=None)
    parser.read(path, encoding="utf-8")
    modules = {}
    for section in parser.sections():
        location = str(safe_path(parser[section]["path"]))
        url = parser[section]["url"]
        match = re.fullmatch(
            r"https://github\.com/([A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+?)(?:\.git)?/?", url
        )
        if not match:
            raise ValueError(
                f"Unsupported submodule URL: {url!r}; only public HTTPS GitHub archives are supported"
            )
        if location in modules:
            raise ValueError("Duplicate submodule path")
        modules[location] = repository_name(match.group(1))
    return modules


def materialize(
    api: GitHub,
    repository: str,
    commit: str,
    destination: Path,
    *,
    prefix="",
    depth=0,
    downloader=download_archive,
) -> list[dict]:
    if depth > 8:
        raise ValueError("Excessive nested submodule depth")
    with tempfile.TemporaryDirectory() as temporary:
        archive = Path(temporary) / "source.zip"
        digest = downloader(repository, commit, archive)
        extract_zip(archive, destination)
    entries = [
        {
            "path": prefix or ".",
            "repository": repository,
            "commit": commit,
            "zip_sha256": digest,
            "zip_url": f"https://codeload.github.com/{repository}/zip/{commit}",
        }
    ]
    tree = api.tree(repository, commit)
    gitlinks = {e["path"]: e["sha"] for e in tree if e["mode"] == "160000"}
    modules = submodule_repositories(destination)
    if set(gitlinks) != set(modules):
        raise ValueError("Submodule declarations and pinned Git tree do not agree")
    for location, sha in sorted(gitlinks.items()):
        safe_path(location)
        subpath = destination.joinpath(*PurePosixPath(location).parts)
        if subpath.exists() and (not subpath.is_dir() or any(subpath.iterdir())):
            raise ValueError(f"Archive unexpectedly contains submodule contents: {location}")
        entries += materialize(
            api,
            modules[location],
            sha,
            subpath,
            prefix=f"{prefix}/{location}".lstrip("/"),
            depth=depth + 1,
            downloader=downloader,
        )
    return entries


def runtime_hashes(source: Path) -> dict[str, str]:
    return {
        p.relative_to(source).as_posix(): sha256(p)
        for p in sorted((source / "llama_cpp").rglob("*"))
        if p.is_file()
    }


PATCHES_DIR = Path(__file__).resolve().parent.parent / "patches"


def patch_targets(patch_path: Path) -> list[str]:
    """Read the 'b/<path>' target(s) declared by a unified diff, for provenance only."""
    targets = []
    for line in patch_path.read_text(encoding="utf-8").splitlines():
        if line.startswith("+++ "):
            target = line[4:].split("\t", 1)[0].strip()
            target = target.removeprefix("b/")
            if target in targets:
                raise ValueError(f"{patch_path.name} touches {target} more than once")
            targets.append(target)
    if not targets:
        raise ValueError(f"{patch_path.name} declares no target file")
    return targets


def apply_patches(source: Path, patches_dir: Path) -> list[dict]:
    """Apply reviewed local patches to specific runtime files, in strict order.

    Each patch is a small, human-reviewed unified diff kept in .github/patches/,
    applied with `git apply`, which requires an exact context match. If upstream
    changes the surrounding code, the patch stops applying cleanly and this
    raises instead of silently dropping the fix, guessing a new offset, or
    fuzzy-matching it onto code a human has not reviewed. See docs/automation.md.
    """
    applied = []
    if not patches_dir.is_dir():
        return applied
    for patch_path in sorted(patches_dir.glob("*.patch")):
        targets = patch_targets(patch_path)
        missing = [name for name in targets if not (source / name).is_file()]
        if missing:
            raise ValueError(f"{patch_path.name} targets a file upstream no longer has: {missing}")
        pre_sha256 = {name: sha256(source / name) for name in targets}
        check = subprocess.run(
            ["git", "apply", "--check", "--whitespace=nowarn", str(patch_path.resolve())],
            cwd=source,
            capture_output=True,
            text=True,
        )
        if check.returncode != 0:
            raise ValueError(
                f"{patch_path.name} no longer applies cleanly to this upstream version; "
                f"review and refresh it before releasing this version.\n{check.stderr.strip()}"
            )
        subprocess.run(
            ["git", "apply", "--whitespace=nowarn", str(patch_path.resolve())],
            cwd=source,
            check=True,
        )
        applied.append(
            {
                "patch": patch_path.name,
                "patch_sha256": sha256(patch_path),
                "files": targets,
                "pre_sha256": pre_sha256,
                "post_sha256": {name: sha256(source / name) for name in targets},
            }
        )
        print(f"Applied local patch {patch_path.name} to {', '.join(targets)}")
    return applied


def adapt_metadata(source: Path, plan: dict, native_commit: str) -> str:
    """Preserve version/authors/license/dependencies; rename distribution self-references."""
    version_key(plan["version"])
    before = (source / "pyproject.toml").read_text(encoding="utf-8")
    document = tomlkit.parse(before)
    project = document["project"]
    if re.sub(r"[-_.]+", "-", project["name"]).lower() != "llama-cpp-python":
        raise ValueError("Unexpected upstream distribution name")
    version_source = (source / "llama_cpp/__init__.py").read_text(encoding="utf-8")
    matches = re.findall(r'^__version__\s*=\s*[\'"]([^\'"]+)[\'"]\s*$', version_source, re.M)
    if matches != [plan["version"]]:
        raise ValueError(f"Tag version {plan['version']} does not match __version__: {matches}")
    project["name"] = PACKAGE
    # Only metadata is changed; logger names and every Python binding remain upstream's.
    project["description"] = (
        "CPU and CUDA builds of JamePeng's llama-cpp-python, distributed as guanaco-py"
    )
    project["maintainers"] = [{"name": plan["repository"].split("/")[0]}]
    extras = project.get("optional-dependencies", {})
    for dependencies in [project.get("dependencies", []), *extras.values()]:
        for index, value in enumerate(dependencies):
            dependencies[index] = re.sub(
                r"^llama[-_.]+cpp[-_.]+python(?=\[|\s|[<>=!~;@]|$)", PACKAGE, value, flags=re.I
            )
    urls = project.setdefault("urls", tomlkit.table())
    urls["Homepage"] = f"https://github.com/{plan['repository']}"
    urls["Issues"] = f"https://github.com/{plan['repository']}/issues"
    urls["Documentation"] = f"https://github.com/{UPSTREAM}/blob/main/docs/wiki/index.md"
    urls["Changelog"] = plan["upstream"]["release_url"]
    urls["Upstream"] = f"https://github.com/{UPSTREAM}"
    # Source ZIPs have no .git. Do not accidentally embed this automation repo's
    # commit/build count when CMake searches parent directories for Git metadata.
    build = document["tool"]["scikit-build"]
    for key, value in {"LLAMA_BUILD_COMMIT": native_commit, "LLAMA_BUILD_NUMBER": "0"}.items():
        existing = build.get("cmake", {}).get("define", {})
        if key in existing:
            existing[key] = value
        else:
            # Dotted keys preserve scikit-build's existing dotted-table layout.
            build.add(tomlkit.key(["cmake", "define", key]), value)
    wheel = build.setdefault("wheel", tomlkit.table())
    licenses = list(wheel.get("license-files", []))
    for pattern in ["LICENSE*", "vendor/llama.cpp/LICENSE*", "vendor/llama.cpp/ggml/LICENSE*"]:
        if pattern not in licenses:
            licenses.append(pattern)
    wheel["license-files"] = licenses
    after = tomlkit.dumps(document)
    if tomlkit.parse(after).unwrap() != document.unwrap():
        raise ValueError("Metadata serialization changed the TOML table structure")
    (source / "pyproject.toml").write_text(after, encoding="utf-8")
    return "".join(
        difflib.unified_diff(
            before.splitlines(keepends=True),
            after.splitlines(keepends=True),
            fromfile="a/pyproject.toml",
            tofile="b/pyproject.toml",
        )
    )


def source_tar(source: Path, destination: Path) -> None:
    """Deterministic tar/gzip, without Git state or credentials."""
    with (
        destination.open("wb") as raw,
        gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as zipped,
    ):
        with tarfile.open(fileobj=zipped, mode="w", format=tarfile.PAX_FORMAT) as archive:
            for path in sorted(source.rglob("*")):
                relative = path.relative_to(source).as_posix()
                safe_path(relative)
                if path.is_symlink():
                    raise ValueError("Refusing a symlink in prepared source")
                info = archive.gettarinfo(str(path), arcname=relative)
                info.uid = info.gid = info.mtime = 0
                info.uname = info.gname = ""
                info.mode = 0o755 if path.is_dir() or os.access(path, os.X_OK) else 0o644
                if path.is_file():
                    with path.open("rb") as content:
                        archive.addfile(info, content)
                else:
                    archive.addfile(info)


def prepare(api: GitHub, plan: dict, output: Path) -> dict:
    if plan["upstream"]["repository"] != UPSTREAM:
        raise ValueError("Unexpected upstream repository")
    repository_name(plan["repository"])
    with empty_destination(output) as prepared:
        with tempfile.TemporaryDirectory(prefix="guanaco-source-") as temporary:
            source = Path(temporary) / "source"
            snapshots = materialize(api, UPSTREAM, plan["upstream"]["commit"], source)
            native = next((s for s in snapshots if s["path"] == "vendor/llama.cpp"), None)
            if native is None:
                raise ValueError("Upstream no longer contains the expected llama.cpp submodule")
            upstream_runtime = runtime_hashes(source)
            if not upstream_runtime or not (source / "LICENSE.md").is_file():
                raise ValueError("Missing upstream runtime or license")
            applied_patches = apply_patches(source, PATCHES_DIR)
            patched_runtime = runtime_hashes(source)
            patch = adapt_metadata(source, plan, native["commit"])
            if runtime_hashes(source) != patched_runtime:
                raise ValueError("Binding code changed while preparing distribution metadata")
            (prepared / "packaging.patch").write_text(patch, encoding="utf-8")
            archive = prepared / "source.tar.gz"
            source_tar(source, archive)
        manifest = {
            **plan,
            "package": PACKAGE,
            "snapshots": snapshots,
            # What the wheel actually ships (post-patch); this is what
            # verify_wheels.py's check_runtime() validates against.
            "runtime_sha256": patched_runtime,
            # The unmodified upstream hashes, kept only so a patched file is
            # easy to diff/audit against the pristine release it came from.
            "upstream_runtime_sha256": upstream_runtime,
            "applied_patches": applied_patches,
            "source_archive_sha256": sha256(archive),
            "packaging_patch_sha256": sha256(prepared / "packaging.patch"),
            "native_commit": native["commit"],
        }
        write_json(prepared / "build-manifest.json", manifest)
    note = (
        f"{len(applied_patches)} local patch(es) applied"
        if applied_patches
        else "bindings unchanged"
    )
    print(f"Prepared guanaco-py {plan['version']}; {note}; llama.cpp @ {native['commit']}")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("work/prepared"))
    args = parser.parse_args()
    prepare(GitHub(), json.loads(args.plan.read_text(encoding="utf-8")), args.output)


if __name__ == "__main__":
    main()
