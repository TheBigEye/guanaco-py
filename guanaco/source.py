"""Build the immutable source snapshot every wheel is compiled from.

The snapshot is the single most important artifact of the whole pipeline: it is
the one place where "which upstream code did we actually ship" is decided and
recorded. It contains

* the upstream bindings and their submodules, each at an exact commit,
* any reviewed local patch, applied strictly and recorded with before/after
  hashes, and
* an adapted ``pyproject.toml`` that renames the distribution without touching
  a single byte of binding code.

Nothing here ever falls back to a branch, a ``main`` tip or a guessed revision:
if an exact commit cannot be resolved, preparation fails.
"""

from __future__ import annotations

import configparser
import difflib
import gzip
import os
import re
import subprocess
import tarfile
import tempfile
from pathlib import Path, PurePosixPath

from .models import (
    DocumentError,
    PatchRecord,
    Plan,
    Snapshot,
    SourceManifest,
    read_json,
    write_json,
)
from .settings import Settings, repository_name
from .transfer import Archive, Downloader, empty_destination, portable_path, sha256

# Depth limit for nested submodules. Two is what upstream needs; eight is a bug.
MAX_SUBMODULE_DEPTH = 8

SUBMODULE_URL_PATTERN = re.compile(
    r"https://github\.com/([A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+?)(?:\.git)?/?"
)
VERSION_ASSIGNMENT_PATTERN = re.compile(r"^__version__\s*=\s*['\"]([^'\"]+)['\"]\s*$", re.MULTILINE)
SELF_REFERENCE_PATTERN = r"^llama[-_.]+cpp[-_.]+python(?=\[|\s|[<>=!~;@]|$)"


MANIFEST_FILE = "build-manifest.json"


def _tomlkit():
    """Import ``tomlkit`` on demand.

    Only source preparation rewrites ``pyproject.toml``; every other command --
    planning, validation, publication, indexing -- runs on the standard library
    alone. Importing on demand keeps that promise true in CI, where most jobs
    never install the build requirements.
    """
    try:
        import tomlkit
    except ImportError as error:  # pragma: no cover - depends on the environment
        raise SourceError(
            "Preparing source requires tomlkit; install it with requirements-ci.txt"
        ) from error
    return tomlkit


class SourceError(ValueError):
    """Raised when the upstream source cannot be pinned, patched or packaged."""


class PatchSet:
    """The reviewed local patches kept in ``.github/patches``.

    Each patch is a small unified diff applied with ``git apply``, which requires
    an exact context match. If upstream rewrites the surrounding code the patch
    stops applying and preparation fails loudly, instead of silently dropping
    the fix or fuzzy-matching it onto code nobody reviewed.
    """

    def __init__(self, directory: Path) -> None:
        """Point the patch set at a directory; it may not exist."""
        self.directory = Path(directory)

    def apply(self, source: Path) -> tuple[PatchRecord, ...]:
        """Apply every patch, in filename order, and return what changed.

        Args:
            source: Root of the extracted upstream source.

        Raises:
            SourceError: If a patch is missing its target file, targets a file
                upstream no longer ships, or no longer applies cleanly.
        """
        if not self.directory.is_dir():
            return ()
        records = []
        for path in sorted(self.directory.glob("*.patch")):
            targets = self.targets(path)
            missing = [name for name in targets if not (source / name).is_file()]
            if missing:
                raise SourceError(f"{path.name} targets a file upstream no longer has: {missing}")
            before = {name: sha256(source / name) for name in targets}
            self._check(path, source)
            subprocess.run(
                ["git", "apply", "--whitespace=nowarn", str(path.resolve())],
                cwd=source,
                check=True,
            )
            records.append(
                PatchRecord(
                    patch=path.name,
                    patch_sha256=sha256(path),
                    files=tuple(targets),
                    pre_sha256=before,
                    post_sha256={name: sha256(source / name) for name in targets},
                )
            )
            print(f"Applied local patch {path.name} to {', '.join(targets)}")
        return tuple(records)

    @staticmethod
    def targets(patch: Path) -> list[str]:
        """Read the ``+++ b/<path>`` targets a unified diff declares."""
        targets: list[str] = []
        for line in patch.read_text(encoding="utf-8").splitlines():
            if line.startswith("+++ "):
                target = line[4:].split("\t", 1)[0].strip().removeprefix("b/")
                if target in targets:
                    raise SourceError(f"{patch.name} touches {target} more than once")
                targets.append(target)
        if not targets:
            raise SourceError(f"{patch.name} declares no target file")
        return targets

    @staticmethod
    def _check(patch: Path, source: Path) -> None:
        """Verify a patch applies cleanly, without changing anything."""
        result = subprocess.run(
            ["git", "apply", "--check", "--whitespace=nowarn", str(patch.resolve())],
            cwd=source,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise SourceError(
                f"{patch.name} no longer applies cleanly to this upstream version; "
                f"review and refresh it before releasing this version.\n{result.stderr.strip()}"
            )


class MetadataAdapter:
    """Renames the distribution inside ``pyproject.toml`` and nothing else.

    The upstream bindings are shipped byte-for-byte. Only packaging metadata
    changes: the distribution name, its description, the maintainer, project
    URLs, license inclusion, and the build identity CMake reports.
    """

    def __init__(self, package: str, upstream_package: str) -> None:
        """Remember both distribution names.

        Args:
            package: The name we publish, e.g. ``guanaco-py``.
            upstream_package: The name upstream ships, used to validate the
                downloaded project and to rewrite self-referencing extras.
        """
        self.package = package
        self.upstream_package = upstream_package

    def adapt(self, source: Path, plan: Plan, native_commit: str) -> str:
        """Rewrite the project metadata and return the unified diff.

        Args:
            source: Root of the extracted upstream source.
            plan: The build plan, used for the version and the project URLs.
            native_commit: llama.cpp commit CMake should report.

        Raises:
            SourceError: If the project is not the expected upstream one, if
                ``__version__`` disagrees with the plan, or if rewriting the
                TOML would change its structure.
        """
        tomlkit = _tomlkit()
        path = source / "pyproject.toml"
        before = path.read_text(encoding="utf-8")
        document = tomlkit.parse(before)
        project = document["project"]
        if self._normalize(project["name"]) != self._normalize(self.upstream_package):
            raise SourceError(f"Unexpected upstream distribution name: {project['name']!r}")

        declared = VERSION_ASSIGNMENT_PATTERN.findall(
            (source / "llama_cpp/__init__.py").read_text(encoding="utf-8")
        )
        if declared != [plan.version]:
            raise SourceError(f"Tag version {plan.version} does not match __version__: {declared}")

        project["name"] = self.package
        owner = plan.origin.repository.split("/")[0]
        project["description"] = (
            f"CPU and CUDA builds of {owner}'s {self.upstream_package}, "
            f"distributed as {self.package}"
        )
        maintainers = tomlkit.array()
        maintainers.append({"name": plan.repository.split("/")[0]})
        project["maintainers"] = maintainers

        for dependencies in [project.get("dependencies", []), *self._extras(project).values()]:
            for index, value in enumerate(dependencies):
                dependencies[index] = re.sub(
                    SELF_REFERENCE_PATTERN, self.package, value, flags=re.IGNORECASE
                )

        urls = project.setdefault("urls", tomlkit.table())
        urls["Homepage"] = f"https://github.com/{plan.repository}"
        urls["Issues"] = f"https://github.com/{plan.repository}/issues"
        urls["Documentation"] = f"https://github.com/{plan.upstream}/blob/main/docs/wiki/index.md"
        urls["Changelog"] = plan.origin.release_url
        urls["Upstream"] = f"https://github.com/{plan.upstream}"

        self._set_build_identity(document, native_commit)
        self._include_licenses(document)

        after = tomlkit.dumps(document)
        if tomlkit.parse(after).unwrap() != document.unwrap():
            raise SourceError("Metadata serialization changed the TOML table structure")
        with path.open("w", encoding="utf-8", newline="\n") as stream:
            stream.write(after)
        return "".join(
            difflib.unified_diff(
                before.splitlines(keepends=True),
                after.splitlines(keepends=True),
                fromfile="a/pyproject.toml",
                tofile="b/pyproject.toml",
            )
        )

    @staticmethod
    def _normalize(name: str) -> str:
        """Normalize a distribution name the way PEP 503 does."""
        return re.sub(r"[-_.]+", "-", str(name)).lower()

    @staticmethod
    def _extras(project) -> dict:
        """Return the optional-dependency tables, possibly empty."""
        return project.get("optional-dependencies", {})

    @staticmethod
    def _set_build_identity(document, native_commit: str) -> None:
        """Pin what CMake reports as the build commit and number.


        The source archive has no ``.git``, so without this CMake would walk up
        the directory tree and happily report this automation repository's own
        commit and build count.
        """
        tomlkit = _tomlkit()
        build = document["tool"]["scikit-build"]
        existing = build.get("cmake", {}).get("define", {})
        for key, value in {"LLAMA_BUILD_COMMIT": native_commit, "LLAMA_BUILD_NUMBER": "0"}.items():
            if key in existing:
                existing[key] = value
            else:
                # Dotted keys preserve scikit-build's existing dotted-table layout.
                build.add(tomlkit.key(["cmake", "define", key]), value)

    @staticmethod
    def _include_licenses(document) -> None:
        """Make sure the wheel carries upstream's and llama.cpp's licenses."""
        tomlkit = _tomlkit()
        build = document["tool"]["scikit-build"]
        wheel = build.setdefault("wheel", tomlkit.table())
        licenses = tomlkit.array()
        for pattern in list(wheel.get("license-files", [])) + [
            "LICENSE*",
            "vendor/llama.cpp/LICENSE*",
            "vendor/llama.cpp/ggml/LICENSE*",
        ]:
            if pattern not in licenses:
                licenses.append(pattern)
        wheel["license-files"] = licenses

    def check_requirement(self, requirement: str) -> bool:
        """Whether a wheel requirement still points at the upstream distribution."""
        return (
            re.match(SELF_REFERENCE_PATTERN.replace("^", ""), requirement, re.IGNORECASE)
            is not None
        )


class SourcePreparer:
    """Downloads, patches and packages one exact upstream version."""

    def __init__(self, settings: Settings, client, downloader: Downloader | None = None) -> None:
        """Wire the preparer to its collaborators.

        Args:
            settings: Configuration, used for the package names and patches.
            client: A :class:`~guanaco.github_api.GithubClient` for the tree API.
            downloader: Optional custom :class:`Downloader`.
        """
        self.settings = settings
        self.client = client
        self.downloader = downloader or Downloader()
        self.patches = PatchSet(settings.patches_dir)
        self.metadata = MetadataAdapter(settings.package, settings.upstream_package)

    def prepare(self, plan: Plan, output: Path) -> SourceManifest:
        """Build the source snapshot for `plan` and write it to `output`.

        Args:
            plan: The frozen build plan.
            output: Directory to create; must not exist or be empty.

        Returns:
            The manifest describing everything that went into the archive.
        """
        if plan.origin.repository != self.settings.upstream:
            raise SourceError(f"Unexpected upstream repository: {plan.origin.repository!r}")
        repository_name(plan.repository)
        with empty_destination(output) as prepared:
            with tempfile.TemporaryDirectory(prefix="guanaco-source-") as temporary:
                source = Path(temporary) / "source"
                snapshots = self._materialize(plan, source)
                native = next((s for s in snapshots if s.path == "vendor/llama.cpp"), None)
                if native is None:
                    raise SourceError(
                        "Upstream no longer contains the expected llama.cpp submodule"
                    )

                pristine = self._runtime_hashes(source)
                if not pristine or not (source / "LICENSE.md").is_file():
                    raise SourceError("Missing upstream runtime or license")

                patches = self.patches.apply(source)
                patched = self._runtime_hashes(source)
                diff = self.metadata.adapt(source, plan, native.commit)
                if self._runtime_hashes(source) != patched:
                    raise SourceError("Binding code changed while preparing distribution metadata")

                (prepared / "packaging.patch").write_text(diff, encoding="utf-8")
                archive = prepared / "source.tar.gz"
                self._write_archive(source, archive)

            manifest = SourceManifest(
                plan=plan,
                package=self.settings.package,
                snapshots=snapshots,
                runtime_sha256=patched,
                upstream_runtime_sha256=pristine,
                applied_patches=patches,
                source_archive_sha256=sha256(archive),
                packaging_patch_sha256=sha256(prepared / "packaging.patch"),
                native_commit=native.commit,
            )
            write_json(prepared / MANIFEST_FILE, manifest.to_mapping())

        note = f"{len(patches)} local patch(es) applied" if patches else "bindings unchanged"
        print(
            f"Prepared {self.settings.package} {plan.version}; {note}; "
            f"llama.cpp @ {manifest.native_commit}"
        )
        return manifest

    # -- Downloading --------------------------------------------------------

    def _materialize(self, plan: Plan, source: Path) -> tuple[Snapshot, ...]:
        """Download the upstream archive and every pinned submodule into `source`."""
        return self._download_tree(self.settings.upstream, plan.origin.commit, source)

    def _download_tree(
        self,
        repository: str,
        commit: str,
        destination: Path,
        *,
        prefix: str = "",
        depth: int = 0,
    ) -> tuple[Snapshot, ...]:
        """Download one repository at `commit`, then recurse into its submodules.

        GitHub source archives do not contain submodule contents, so the pinned
        gitlinks are read from the Git tree API and each one is downloaded at its
        own exact commit, recursively.

        Raises:
            SourceError: If ``.gitmodules`` and the Git tree disagree, or if an
                archive unexpectedly contains submodule contents.
        """
        if depth > MAX_SUBMODULE_DEPTH:
            raise SourceError("Excessive nested submodule depth")
        digest = self._download_archive(repository, commit, destination)
        entries = [
            Snapshot(
                path=prefix or ".",
                repository=repository,
                commit=commit,
                zip_sha256=digest,
                zip_url=f"https://codeload.github.com/{repository}/zip/{commit}",
            )
        ]
        gitlinks = {
            entry["path"]: entry["sha"]
            for entry in self.client.tree(repository, commit)
            if entry.get("mode") == "160000"
        }
        modules = submodule_repositories(destination)
        if set(gitlinks) != set(modules):
            raise SourceError("Submodule declarations and pinned Git tree do not agree")
        for location, sha in sorted(gitlinks.items()):
            portable_path(location)
            subpath = destination.joinpath(*PurePosixPath(location).parts)
            if subpath.exists() and (not subpath.is_dir() or any(subpath.iterdir())):
                raise SourceError(f"Archive unexpectedly contains submodule contents: {location}")
            entries += self._download_tree(
                modules[location],
                sha,
                subpath,
                prefix=f"{prefix}/{location}".lstrip("/"),
                depth=depth + 1,
            )
        return tuple(entries)

    def _download_archive(self, repository: str, commit: str, destination: Path) -> str:
        """Download one repository archive, extract it and return its SHA-256."""
        repository_name(repository)
        if not re.fullmatch(r"[0-9a-f]{40}", commit):
            raise SourceError("Refusing an archive without an immutable SHA")
        url = f"https://codeload.github.com/{repository}/zip/{commit}"
        with tempfile.TemporaryDirectory() as temporary:
            archive = Path(temporary) / "source.zip"
            digest = self.downloader.fetch(url, archive)
            size = archive.stat().st_size
            Archive.extract_zip(archive, destination)
        print(f"Downloaded {repository}@{commit}: {size} bytes")
        return digest

    # -- Helpers ------------------------------------------------------------

    @staticmethod
    def _runtime_hashes(source: Path) -> dict[str, str]:
        """Hash every file of the Python bindings, keyed by relative POSIX path."""
        root = source / "llama_cpp"
        return {
            path.relative_to(source).as_posix(): sha256(path)
            for path in sorted(root.rglob("*"))
            if path.is_file()
        }

    @staticmethod
    def _write_archive(source: Path, destination: Path) -> None:
        """Write a deterministic ``tar.gz``: no Git state, no credentials, no owner."""
        with (
            destination.open("wb") as raw,
            gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as zipped,
        ):
            with tarfile.open(fileobj=zipped, mode="w", format=tarfile.PAX_FORMAT) as archive:
                for path in sorted(source.rglob("*")):
                    relative = path.relative_to(source).as_posix()
                    if path.is_symlink():
                        raise SourceError("Refusing a symlink in prepared source")
                    info = archive.gettarinfo(str(path), arcname=relative)
                    info.uid = info.gid = info.mtime = 0
                    info.uname = info.gname = ""
                    info.mode = 0o755 if path.is_dir() or os.access(path, os.X_OK) else 0o644
                    if path.is_file():
                        with path.open("rb") as content:
                            archive.addfile(info, content)
                    else:
                        archive.addfile(info)


class SourceArchive:
    """The prepared snapshot as it travels between CI jobs."""

    @staticmethod
    def _manifest(artifact: Path) -> SourceManifest:
        """Read ``build-manifest.json``, reporting any problem as a source error."""
        try:
            return SourceManifest.from_mapping(read_json(artifact / MANIFEST_FILE))
        except DocumentError as error:
            raise SourceError(f"Unreadable prepared artifact: {error}") from error

    @staticmethod
    def extract(artifact: Path, destination: Path, version: str | None = None) -> SourceManifest:
        """Verify and unpack a prepared snapshot.

        Args:
            artifact: Directory holding ``source.tar.gz`` and the manifest.
            destination: Where to unpack. Must not exist or be empty.
            version: If given, the manifest must describe this version.

        Raises:
            SourceError: If the checksum or the version does not match.
        """
        artifact = Path(artifact)
        manifest = SourceArchive._manifest(artifact)
        if version and manifest.version != version:
            raise SourceError("Source artifact version does not match the requested build")
        archive = artifact / "source.tar.gz"
        if sha256(archive) != manifest.source_archive_sha256:
            raise SourceError("Prepared source SHA256 mismatch")
        Archive.extract_tar(archive, destination)
        print(f"Unpacked {manifest.package} {manifest.version} from verified source archive")
        return manifest

    @staticmethod
    def verify(artifact: Path) -> SourceManifest:
        """Check the archive and patch checksums without extracting anything."""
        artifact = Path(artifact)
        manifest = SourceArchive._manifest(artifact)
        for name, expected in (
            ("source.tar.gz", manifest.source_archive_sha256),
            ("packaging.patch", manifest.packaging_patch_sha256),
        ):
            path = artifact / name
            if not path.is_file():
                raise SourceError(f"Prepared artifact is missing: {name}")
            if sha256(path) != expected:
                raise SourceError(f"Prepared artifact checksum mismatch: {name}")
        return manifest


def submodule_repositories(source: Path) -> dict[str, str]:
    """Read ``.gitmodules`` and return ``{path: owner/name}``.

    Only public HTTPS GitHub submodules are supported: anything else cannot be
    downloaded as a pinned archive.
    """
    path = Path(source) / ".gitmodules"
    if not path.exists():
        return {}
    parser = configparser.ConfigParser(interpolation=None)
    parser.read(path, encoding="utf-8")
    modules: dict[str, str] = {}
    for section in parser.sections():
        location = str(PurePosixPath(parser[section]["path"]))
        url = parser[section]["url"]
        match = SUBMODULE_URL_PATTERN.fullmatch(url)
        if not match:
            raise SourceError(
                f"Unsupported submodule URL: {url!r}; only public HTTPS GitHub archives are supported"
            )
        if location in modules:
            raise SourceError("Duplicate submodule path")
        modules[location] = repository_name(match.group(1))
    return modules


__all__ = [
    "MetadataAdapter",
    "PatchSet",
    "SourceArchive",
    "SourceError",
    "SourcePreparer",
    "submodule_repositories",
]
