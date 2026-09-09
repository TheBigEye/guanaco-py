"""Small synthetic settings, plans and wheels for the offline test suite.

Nothing here touches the network, compiles C++ or loads a native library. The
wheel fixtures are real ZIP files with hand-written ELF/PE headers, which is
enough to exercise every validation rule without shipping a real binary.
"""

from __future__ import annotations

import base64
import copy
import csv
import hashlib
import io
import json
import struct
import tarfile
import zipfile
from dataclasses import replace
from pathlib import Path

from guanaco.models import (
    Channel,
    Plan,
    Platform,
    Provenance,
    Release,
    SourceManifest,
    UpstreamOrigin,
    Version,
    expected_asset_names,
    write_json,
)
from guanaco.releases import ReleasePublisher
from guanaco.settings import BuildMatrix, Settings
from guanaco.transfer import sha256
from guanaco.wheels import WheelValidator

SHA_A, SHA_B = "a" * 40, "b" * 40
UPSTREAM = "JamePeng/llama-cpp-python"
REPOSITORY = "TheBigEye/guanaco-py"

MATRIX = {
    "repository": REPOSITORY,
    "upstream": UPSTREAM,
    "package": "guanaco-py",
    "python_versions": ["3.9", "3.10", "3.11", "3.12", "3.13", "3.14"],
    "cuda": {
        "cu124": {"toolkit": "12.4.1", "architectures": "75;80;86;89;90", "legacy_msvc": False},
        "cu128": {"toolkit": "12.8.1", "architectures": "75;80;86;89;90", "legacy_msvc": False},
    },
}


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------


def matrix_path(root: Path) -> Path:
    """Return where ``build-matrix.json`` lives inside a repository root."""
    return root / ".github" / "build-matrix.json"


def make_settings(root: Path, **overrides) -> Settings:
    """Write ``build-matrix.json`` under `root` and load it."""
    data = copy.deepcopy(MATRIX)
    data.update(overrides)
    root.mkdir(parents=True, exist_ok=True)
    matrix = matrix_path(root)
    matrix.parent.mkdir(parents=True, exist_ok=True)
    matrix.write_text(json.dumps(data), encoding="utf-8")
    for folder in (".github/patches", "docs"):
        (root / folder).mkdir(parents=True, exist_ok=True)
    (root / "docs" / "icon.svg").write_text("<svg/>", encoding="utf-8")
    (root / "docs" / "wheel-index.css").write_text("body { color: #333; }", encoding="utf-8")
    return Settings.load(root=root, matrix=matrix)


# ---------------------------------------------------------------------------
# Plans and manifests
# ---------------------------------------------------------------------------


def upstream_payload(tag: str = "v0.3.49-cu124-win-20260831", **overrides) -> dict:
    """Return an upstream GitHub release payload."""
    payload = {
        "id": 49,
        "tag_name": tag,
        "published_at": "2026-08-31T10:00:00Z",
        "draft": False,
        "prerelease": False,
        "body": "Original upstream notes.\r\nKeep the changelog.",
        "html_url": f"https://github.com/{UPSTREAM}/releases/tag/{tag}",
        "name": tag,
    }
    payload.update(overrides)
    return payload


def make_plan(settings: Settings, channels=None, **overrides) -> Plan:
    """Return a release plan for the synthetic matrix."""
    release = Release.from_mapping(upstream_payload())
    origin = UpstreamOrigin(
        repository=settings.upstream,
        release_id=release.identifier,
        tag=release.tag,
        commit=SHA_A,
        release_url=release.url,
        release_name=release.name,
        published_at=release.published_at,
        body=release.body,
        zip_url=f"https://codeload.github.com/{settings.upstream}/zip/{SHA_A}",
    )
    options = {
        "repository": settings.repository,
        "version": "0.3.49",
        "origin": origin,
        "matrix": settings.matrix,
        "missing_channels": tuple(
            name if isinstance(name, Channel) else Channel(name)
            for name in (channels if channels is not None else settings.channels)
        ),
        "promote_latest": True,
        "recipe_commit": SHA_B,
        "run_url": "https://github.com/TheBigEye/guanaco-py/actions/runs/1",
    }
    options.update(overrides)
    return Plan(**options)


def manifest_for(plan: Plan, package: str = "guanaco-py") -> tuple[SourceManifest, dict]:
    """Return a manifest plus the runtime files it describes."""
    runtime = {
        "llama_cpp/__init__.py": f'__version__ = "{plan.version}"\n'.encode(),
        "llama_cpp/py.typed": b"",
    }
    manifest = SourceManifest(
        plan=plan,
        package=package,
        snapshots=(),
        runtime_sha256={name: hashlib.sha256(data).hexdigest() for name, data in runtime.items()},
        upstream_runtime_sha256={},
        applied_patches=(),
        source_archive_sha256="0" * 64,
        packaging_patch_sha256="1" * 64,
        native_commit=SHA_A,
    )
    return manifest, runtime


# ---------------------------------------------------------------------------
# Releases
# ---------------------------------------------------------------------------


def owned_release(
    settings: Settings,
    plan: Plan,
    channel: str,
    *,
    finished: bool = True,
    draft: bool = False,
    uploader=None,
) -> dict:
    """Return a GitHub release payload that looks fully published by us."""
    publisher = ReleasePublisher(settings, client=None)
    tag = Channel(channel).release_tag(plan.version)
    names = sorted(
        expected_asset_names(settings.package, plan.version, Channel(channel), plan.python_versions)
    )
    assets = [
        {
            "name": name,
            "state": "uploaded",
            "size": 100,
            "browser_download_url": (
                f"https://github.com/{plan.repository}/releases/download/{tag}/{name}"
            ),
            "digest": "sha256:" + "d" * 64,
        }
        for name in names
    ]
    payload = {
        "id": 100 + [item.name for item in settings.channels].index(channel),
        "tag_name": tag,
        "name": tag,
        "body": publisher.release_body(plan, Channel(channel), finished),
        "draft": draft,
        "prerelease": False,
        "published_at": "2026-08-31T11:00:00Z",
        "html_url": f"https://github.com/{plan.repository}/releases/tag/{tag}",
        "assets": assets,
    }
    del uploader
    return payload


def provenance_of(settings: Settings, plan: Plan, channel: str, finished: bool) -> Provenance:
    """Parse back the marker our own publisher would write."""
    publisher = ReleasePublisher(settings, client=None)
    body = publisher.release_body(plan, Channel(channel), finished)
    return Provenance.from_body(body, Channel(channel).release_tag(plan.version))


class FakeGitHub:
    """A GitHub client that answers from fixed lists and records its calls."""

    def __init__(self, upstream_releases=None, existing=None) -> None:
        """Store the canned upstream releases and our own existing releases."""
        self.upstream_releases = (
            upstream_releases if upstream_releases is not None else [upstream_payload()]
        )
        self.existing = existing or []
        self.commit_calls: list[tuple[str, str]] = []

    def releases(self, repository: str) -> list[Release]:
        """Return the upstream list, or our own releases."""
        source = self.upstream_releases if repository == UPSTREAM else self.existing
        return [
            item if isinstance(item, Release) else Release.from_mapping(item) for item in source
        ]

    def release(self, repository: str, tag: str) -> Release | None:
        """Find a release by tag in whichever list belongs to `repository`."""
        for item in self.releases(repository):
            if item.tag == tag:
                return item
        return None

    def commit(self, repository: str, ref: str) -> str:
        """Record the call and return the pinned commit."""
        self.commit_calls.append((repository, ref))
        return SHA_A

    def tag_commit(self, repository: str, tag: str) -> str | None:
        """Pretend no Git tags exist unless a test overrides this."""
        return None

    def request(self, path: str, **kwargs) -> None:  # pragma: no cover - safety net
        """Fail loudly: tests should not reach the network."""
        raise AssertionError(f"Unexpected API call: {path}")


class PublishingAPI:
    """A writable GitHub client stand-in that stores what it was told."""

    def __init__(self) -> None:
        """Start with no releases and no calls."""
        self.items: dict[int, dict] = {}
        self.calls: list[tuple[str, str, dict | None]] = []
        self.tag_commits: dict[str, str] = {}

    def release(self, repository: str, tag: str) -> Release | None:
        """Return the stored release with this tag, if any."""
        for item in self.items.values():
            if item["tag_name"] == tag:
                return Release.from_mapping(item)
        return None

    def releases(self, repository: str) -> list[Release]:
        """Return every stored release."""
        return [Release.from_mapping(item) for item in self.items.values()]

    def tag_commit(self, repository: str, tag: str) -> str | None:
        """Return a recorded tag target."""
        return self.tag_commits.get(tag)

    def commit(self, repository: str, ref: str) -> str:
        """Return the pinned commit."""
        return SHA_A

    def request(self, path: str, method: str = "GET", data=None):
        """Mimic the release create/patch endpoints."""
        self.calls.append((method, path, copy.deepcopy(data)))
        if method == "POST":
            number = len(self.items) + 1
            self.items[number] = {"id": number, "assets": [], **data}
            return copy.deepcopy(self.items[number])
        number = int(path.split("/")[-1])
        if method == "PATCH":
            self.items[number].update(data)
        return copy.deepcopy(self.items[number])

    def create_release(self, repository, *, tag, commit, name, body, draft, latest) -> Release:
        """Create a release through :meth:`request`."""
        payload = self.request(
            f"/repos/{repository}/releases",
            method="POST",
            data={
                "tag_name": tag,
                "target_commitish": commit,
                "name": name,
                "body": body,
                "draft": draft,
                "prerelease": False,
                "make_latest": "true" if latest else "false",
            },
        )
        return Release.from_mapping(payload)

    def update_release(self, repository, identifier, *, body=None, draft=None, latest=None):
        """Patch a release through :meth:`request`."""
        data = {"prerelease": False}
        if body is not None:
            data["body"] = body
        if draft is not None:
            data["draft"] = draft
        if latest is not None:
            data["make_latest"] = "true" if latest else "false"
        return Release.from_mapping(
            self.request(f"/repos/{repository}/releases/{identifier}", method="PATCH", data=data)
        )

    def upload_assets(self, plan: Plan, tag: str, files) -> None:
        """Record uploaded assets with their real sizes and digests."""
        release = next(item for item in self.items.values() if item["tag_name"] == tag)
        release["assets"] = [
            {
                "name": path.name,
                "state": "uploaded",
                "size": path.stat().st_size,
                "digest": "sha256:" + sha256(path),
                "browser_download_url": f"https://example.test/{tag}/{path.name}",
            }
            for path in files
        ]


# ---------------------------------------------------------------------------
# Wheels
# ---------------------------------------------------------------------------


def native_header(platform: str) -> bytes:
    """Return a minimal, non-loadable ELF or PE header fixture."""
    header = bytearray(256)
    if platform == "windows":
        header[:2] = b"MZ"
        struct.pack_into("<I", header, 60, 128)
        header[128:134] = b"PE\x00\x00\x64\x86"
    else:
        header[:7] = b"\x7fELF\x02\x01\x01"
        struct.pack_into("<HH", header, 16, 3, 62)
    return bytes(header)


ARCHIVE_TIMESTAMP = (1980, 1, 1, 0, 0, 0)


def archive_member(name: str) -> zipfile.ZipInfo:
    """Return a ZIP entry carrying a fixed timestamp and the name as written.

    ``ZipFile.writestr(name, data)`` stamps the entry with the current time, so
    two archives with identical contents differ byte for byte when they are
    built either side of a second boundary. The source manifest hashes the
    downloaded archives, and a test compares two preparations, so that
    timestamp would make the result depend on how fast the machine is.

    ``ZipInfo(name)`` also rewrites separators on Windows and truncates the
    name at a NUL byte, so both fields are restored afterwards to keep
    deliberately hostile names intact.
    """
    member = zipfile.ZipInfo(name, date_time=ARCHIVE_TIMESTAMP)
    member.filename = member.orig_filename = name
    return member


def zip_contents(path: Path, contents: dict, *, record: bool = True) -> None:
    """Write a ZIP, generating a matching ``RECORD`` unless asked not to."""
    if record:
        record_path = (
            next(name for name in contents if name.endswith(".dist-info/METADATA")).removesuffix(
                "METADATA"
            )
            + "RECORD"
        )
        rows = []
        for name, data in contents.items():
            if name == record_path:
                continue
            digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).decode().rstrip("=")
            rows.append([name, "sha256=" + digest, str(len(data))])
        rows.append([record_path, "", ""])
        output = io.StringIO(newline="")
        csv.writer(output).writerows(rows)
        contents[record_path] = output.getvalue().encode()
    with zipfile.ZipFile(path, "w") as archive:
        for name, data in contents.items():
            archive.writestr(archive_member(name), data)


def write_wheel(
    directory: Path,
    manifest: SourceManifest,
    runtime: dict,
    channel: str,
    platform: str,
    python: str = "cp313",
    *,
    metadata_name: str = "guanaco-py",
    native: bool = True,
    altered: bool = False,
    extra: bool = False,
    raw: bool = False,
) -> Path:
    """Write one synthetic wheel matching the manifest's expectations."""
    directory.mkdir(parents=True, exist_ok=True)
    policy = Channel(channel).wheel_platform(
        Platform.parse(platform), repaired=not raw and not channel.startswith("cu")
    )
    version = manifest.version
    path = directory / f"guanaco_py-{version}-{python}-{python}-{policy}.whl"
    contents = {
        name: (data + b"# altered" if altered and name.endswith(".py") else data)
        for name, data in runtime.items()
    }
    if extra:
        contents["llama_cpp/additional.py"] = b"pass"
    info = f"guanaco_py-{version}.dist-info/"
    contents[info + "METADATA"] = (
        f"Metadata-Version: 2.3\nName: {metadata_name}\nVersion: {version}\n".encode()
    )
    contents[info + "WHEEL"] = (
        f"Wheel-Version: 1.0\nRoot-Is-Purelib: false\nTag: {python}-{python}-{policy}\n".encode()
    )
    contents[info + "licenses/LICENSE.md"] = b"MIT notice"
    if native:
        name = "llama.dll" if platform == "windows" else "libllama.so"
        contents["llama_cpp/lib/" + name] = native_header(platform)
    zip_contents(path, contents)
    return path


def write_source_tarball(destination: Path, staging: Path) -> None:
    """Write a small but real ``tar.gz`` holding the Python bindings."""
    (staging / "llama_cpp").mkdir(parents=True, exist_ok=True)
    (staging / "llama_cpp" / "__init__.py").write_text('__version__ = "0.3.49"\n', encoding="utf-8")
    with tarfile.open(destination, "w:gz") as archive:
        archive.add(staging / "llama_cpp", arcname="llama_cpp")


def prepared_build(
    tmp_path: Path,
    settings: Settings,
    *,
    missing: bool = False,
    test_only: bool = False,
    platforms=("linux", "windows"),
):
    """Write a prepared snapshot plus a complete set of wheel artifacts."""
    plan = make_plan(
        settings,
        ["cpu", "avx2"],
        test_only=test_only,
        platforms=tuple(Platform(name) for name in platforms),
    )
    plan = replace(
        plan,
        matrix=BuildMatrix(
            upstream=settings.upstream,
            package=settings.package,
            python_versions=("3.12", "3.13"),
            channels=settings.channels,
            cuda=settings.cuda,
        ),
    )
    manifest, runtime = manifest_for(plan)
    prepared = tmp_path / "prepared"
    prepared.mkdir(parents=True, exist_ok=True)
    write_source_tarball(prepared / "source.tar.gz", tmp_path / "raw-source")
    (prepared / "packaging.patch").write_text("metadata patch fixture", encoding="utf-8")
    manifest = SourceManifest(
        plan=plan,
        package=manifest.package,
        snapshots=manifest.snapshots,
        runtime_sha256=manifest.runtime_sha256,
        upstream_runtime_sha256=manifest.upstream_runtime_sha256,
        applied_patches=manifest.applied_patches,
        source_archive_sha256=sha256(prepared / "source.tar.gz"),
        packaging_patch_sha256=sha256(prepared / "packaging.patch"),
        native_commit=manifest.native_commit,
    )
    write_json(prepared / "build-manifest.json", manifest.to_mapping())

    artifacts = tmp_path / "artifacts"
    prefix = "test-" if test_only else ""
    for channel in plan.missing_channels:
        for platform in platforms:
            for python in ("3.12", "3.13"):
                if (
                    missing
                    and channel.name == "avx2"
                    and platform == "windows"
                    and python == "3.13"
                ):
                    continue
                write_wheel(
                    artifacts / f"{prefix}{settings.package}-{channel}-{platform}-x64",
                    manifest,
                    runtime,
                    channel.name,
                    platform,
                    "cp" + python.replace(".", ""),
                )
    return plan, prepared, artifacts


def receipts_for(
    settings: Settings,
    plan: Plan,
    prepared: Path,
    artifacts: Path,
    directory: Path,
    platforms=("linux", "windows"),
):
    """Validate every artifact folder and write the receipts for it."""
    from guanaco.releases import artifact_specs
    from guanaco.source import SourceArchive

    directory.mkdir(parents=True, exist_ok=True)
    manifest = SourceArchive.verify(prepared)
    validator = WheelValidator(settings)
    for name, spec in artifact_specs(plan, settings.package).items():
        if spec.platform.value not in platforms:
            continue
        wheels = validator.verify_directory(artifacts / name, manifest, spec.channel, spec.platform)
        write_json(
            directory / f"{name}.json",
            validator.receipt(wheels, manifest, spec.channel, spec.platform).to_mapping(),
        )
    return manifest


def wheel_data(wheel: Path) -> dict:
    """Read a wheel's members into memory."""
    with zipfile.ZipFile(wheel) as archive:
        return {name: archive.read(name) for name in archive.namelist()}


def version_of(text: str) -> Version:
    """Parse a version, for readable test assertions."""
    return Version.parse(text)


__all__ = [
    "FakeGitHub",
    "matrix_path",
    "PublishingAPI",
    "make_plan",
    "make_settings",
    "manifest_for",
    "native_header",
    "owned_release",
    "prepared_build",
    "provenance_of",
    "receipts_for",
    "upstream_payload",
    "version_of",
    "wheel_data",
    "write_source_tarball",
    "write_wheel",
    "zip_contents",
]
