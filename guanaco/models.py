"""Value objects and the JSON documents the build system exchanges.

Every document that travels between the CI jobs -- the build plan, the source
manifest, a validation receipt, the publication gate, the provenance marker
hidden inside a release body -- is represented here by a frozen dataclass with
``from_mapping`` / ``to_mapping``. That gives the rest of the package typed
attributes instead of stringly-keyed dictionaries, while the on-disk JSON stays
exactly the same.
"""

from __future__ import annotations

import copy
import enum
import hashlib
import json
import os
import re
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from .channels import CHANNEL_PATTERN, CPU_CHANNELS, Channel, Platform
from .settings import BuildMatrix, ConfigurationError

PLAN_SCHEMA = 1
RECEIPT_SCHEMA = 1
GATE_SCHEMA = 1
COMMIT_PATTERN = re.compile(r"[0-9a-f]{40}")
SHA256_PATTERN = re.compile(r"[a-f0-9]{64}")

# Names of the non-wheel assets of a release. These are part of the published
# format of already released versions, so they follow the distribution name
# rather than the repository name, and renaming them would make every existing
# release look incomplete.
MANIFEST_ASSET = "guanaco-build.json"
CHECKSUM_ASSET = "SHA256SUMS"
PACKAGING_PATCH_ASSET = "packaging.patch"


def source_asset(package: str, version: Version | str) -> str:
    """Return the name of the reconstructed source archive, e.g. ``guanaco-source-0.3.49.tar.gz``."""
    return f"{package.rsplit('-', 1)[0]}-source-{version}.tar.gz"


# Marks the embedded provenance block inside a published release body. The name
# is part of the on-disk format of already published releases: never rename it.
PROVENANCE_MARKER = "guanaco-upstream-build-v1"
PROVENANCE_SEPARATOR = "\n\n---\n\n### Guanaco build provenance\n\n"
PROVENANCE_PATTERN = re.compile(r"<!-- " + PROVENANCE_MARKER + r"\r?\n(.*?)\r?\n-->\s*$", re.DOTALL)
PRERELEASE_PATTERN = re.compile(
    r"(?:^|[-+.])(?:alpha|beta|rc|dev|pre|preview|nightly|snapshot)(?:[0-9]*|[.-][0-9]+)?(?:$|[-+.])",
    re.IGNORECASE,
)
UPSTREAM_TAG_PATTERN = re.compile(
    r"v?((?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*))(?:[-+][A-Za-z0-9_.+\-]+)?"
)


class DocumentError(ValueError):
    """Raised when a plan, manifest, receipt or provenance marker is invalid."""


# ---------------------------------------------------------------------------
# Small value objects
# ---------------------------------------------------------------------------


@dataclass(frozen=True, order=True)
class Version:
    """A stable ``X.Y.Z`` version, compared numerically.

    Upstream tags carry a backend and a date (``v0.3.49-cu124-win-20260831``),
    but what identifies a package version is only the numeric part, so
    ``0.3.10`` sorts above ``0.3.9``.
    """

    major: int
    minor: int
    patch: int

    @classmethod
    def parse(cls, text: str | None) -> Version:
        """Parse ``X.Y.Z``, rejecting anything else (no leading zeros, no suffix)."""
        if not isinstance(text, str):
            raise DocumentError(f"Expected a version string, got {text!r}")
        parts = text.split(".")
        if len(parts) != 3 or not all(part.isdigit() for part in parts):
            raise DocumentError(f"Expected a stable X.Y.Z version, got {text!r}")
        if any(len(part) > 1 and part[0] == "0" for part in parts):
            raise DocumentError(f"Version components must not have leading zeros: {text!r}")
        return cls(int(parts[0]), int(parts[1]), int(parts[2]))

    @classmethod
    def from_tag(cls, tag: str | None) -> Version | None:
        """Extract the version of an upstream tag, or ``None`` if it has none.

        Preview, release candidate and nightly tags are rejected as well: those
        are never rebuilt into a stable wheel channel.
        """
        if not isinstance(tag, str):
            return None
        match = UPSTREAM_TAG_PATTERN.fullmatch(tag)
        if not match or PRERELEASE_PATTERN.search(tag):
            return None
        return cls.parse(match.group(1))

    def __str__(self) -> str:
        """Render the version the way it appears in filenames and tags."""
        return f"{self.major}.{self.minor}.{self.patch}"


@dataclass(frozen=True)
class UpstreamOrigin:
    """The exact upstream release a build was made from.

    Attributes:
        repository: Upstream ``owner/name`` we rebuild.
        release_id: Numeric GitHub release id, kept to detect a moved release.
        tag: Upstream tag, e.g. ``v0.3.49-cu124-win-20260831``.
        commit: Full commit SHA the source archive was downloaded at.
        release_url: Human readable link to the upstream release.
        release_name: Upstream release title.
        published_at: Upstream publication timestamp, or ``None``.
        body: Upstream release notes, preserved in our release body.
        zip_url: Archive URL, always pinned to `commit`.
    """

    repository: str
    release_id: int
    tag: str
    commit: str
    release_url: str
    release_name: str
    published_at: str | None
    body: str
    zip_url: str

    @classmethod
    def from_mapping(cls, values: dict) -> UpstreamOrigin:
        """Rebuild an origin from a plan/manifest/provenance mapping."""
        try:
            origin = cls(
                repository=values["repository"],
                release_id=values["release_id"],
                tag=values["tag"],
                commit=values["commit"],
                release_url=values["release_url"],
                release_name=values["release_name"],
                published_at=values["published_at"],
                body=values["body"],
                zip_url=values["zip_url"],
            )
        except KeyError as error:
            raise DocumentError(f"Upstream origin is missing {error}") from error
        origin.validate()
        return origin

    def to_mapping(self) -> dict:
        """Return the JSON shape stored in plans, manifests and provenance."""
        return {
            "repository": self.repository,
            "release_id": self.release_id,
            "tag": self.tag,
            "commit": self.commit,
            "release_url": self.release_url,
            "release_name": self.release_name,
            "published_at": self.published_at,
            "body": self.body,
            "zip_url": self.zip_url,
        }

    def with_body_hash(self) -> dict:
        """Return a mapping where the notes are replaced by their SHA-256.

        Release bodies already carry the upstream notes above the provenance
        separator, so the hidden marker stores only their hash. This keeps the
        marker small and makes accidental edits detectable.
        """
        values = self.to_mapping()
        values["body_sha256"] = hashlib.sha256(self.body.encode("utf-8")).hexdigest()
        del values["body"]
        return values

    def validate(self) -> None:
        """Check the identifiers that other modules rely on."""
        if not COMMIT_PATTERN.fullmatch(self.commit):
            raise DocumentError(f"Upstream origin has an invalid commit: {self.commit!r}")
        if type(self.release_id) is not int or self.release_id <= 0:
            raise DocumentError("Upstream origin has an invalid release id")
        if Version.from_tag(self.tag) is None:
            raise DocumentError(f"Upstream tag carries no stable version: {self.tag!r}")


# ---------------------------------------------------------------------------
# Build plan
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FrozenSource:
    """An upstream origin plus the build matrix it was first built with.

    Once a version family starts publishing, a retry has to reuse both: the
    upstream commit cannot change under a half-finished family, and neither can
    the Python or CUDA matrix that its releases were planned with.
    """

    origin: UpstreamOrigin
    matrix: BuildMatrix


@dataclass(frozen=True)
class Plan:
    """What to build: one upstream version, for a set of missing channels.

    A plan is frozen the moment it is written to ``plan.json``. Every later job
    reads it instead of asking GitHub again, so a long build can never drift
    onto a newer upstream commit halfway through.
    """

    repository: str
    version: str
    origin: UpstreamOrigin
    matrix: BuildMatrix
    missing_channels: tuple[Channel, ...]
    promote_latest: bool
    recipe_commit: str
    run_url: str | None = None
    test_only: bool = False
    platforms: tuple[Platform, ...] = (Platform.LINUX, Platform.WINDOWS)

    @property
    def package(self) -> str:
        """Distribution name this plan publishes."""
        return self.matrix.package

    @property
    def upstream(self) -> str:
        """Upstream repository this plan rebuilds."""
        return self.matrix.upstream

    @property
    def python_versions(self) -> tuple[str, ...]:
        """Python versions to build."""
        return self.matrix.python_versions

    @property
    def channels(self) -> tuple[str, ...]:
        """Every channel this plan knows about."""
        return self.matrix.channels

    @property
    def cuda(self) -> dict:
        """CUDA channel settings, keyed by channel name."""
        return self.matrix.cuda

    @property
    def needs_build(self) -> bool:
        """Whether anything is missing and a build should run."""
        return bool(self.missing_channels)

    def build_platforms(self) -> tuple[Platform, ...]:
        """Return the platforms this plan targets.

        A release always builds Linux and Windows. A manual test build may
        narrow that down to one of them.
        """
        if not self.test_only:
            return (Platform.LINUX, Platform.WINDOWS)
        if (
            not self.platforms
            or len(set(self.platforms)) != len(self.platforms)
            or any(not isinstance(item, Platform) for item in self.platforms)
        ):
            raise DocumentError("Test platforms must be a unique non-empty Linux/Windows selection")
        return self.platforms

    def to_mapping(self) -> dict:
        """Return the JSON document written to ``plan.json``.

        The matrix carries an ``upstream`` key of its own (the repository
        name); a plan keeps the *origin* under that key instead, because that is
        what every reader of a plan has always expected.
        """
        matrix = self.matrix.to_mapping()
        del matrix["upstream"]
        document = {
            "schema": PLAN_SCHEMA,
            "repository": self.repository,
            "version": self.version,
            "upstream": self.origin.to_mapping(),
            **matrix,
            "missing_channels": [channel.name for channel in self.missing_channels],
            "build": self.needs_build,
            "promote_latest": self.promote_latest,
            "recipe_commit": self.recipe_commit,
            "run_url": self.run_url,
        }
        if self.test_only:
            document["test_only"] = True
            document["platforms"] = [platform.value for platform in self.platforms]
        return document

    @classmethod
    def from_mapping(cls, values: dict) -> Plan:
        """Rebuild a plan from its JSON document."""
        if not isinstance(values, dict):
            raise DocumentError("A plan must be a JSON object")
        schema = values.get("schema")
        if type(schema) is not int or schema != PLAN_SCHEMA:
            raise DocumentError(f"Unsupported plan schema: {schema!r}")
        test_only = values.get("test_only", False)
        if type(test_only) is not bool:
            raise DocumentError("test_only must be a boolean")
        try:
            origin = UpstreamOrigin.from_mapping(values["upstream"])
            plan = cls(
                repository=values["repository"],
                version=values["version"],
                origin=origin,
                # The matrix stores the upstream repository name under the same
                # key the plan uses for the origin, so pass it explicitly.
                matrix=BuildMatrix.from_mapping({**values, "upstream": origin.repository}),
                missing_channels=tuple(
                    Channel(str(name)) for name in values.get("missing_channels", [])
                ),
                promote_latest=values["promote_latest"],
                recipe_commit=values["recipe_commit"],
                run_url=values.get("run_url"),
                test_only=test_only,
                platforms=tuple(Platform.parse(str(name)) for name in values.get("platforms", []))
                or (Platform.LINUX, Platform.WINDOWS),
            )
        except KeyError as error:
            raise DocumentError(f"Plan is missing {error}") from error
        Version.parse(plan.version)
        return plan


# ---------------------------------------------------------------------------
# Prepared source
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Snapshot:
    """One repository downloaded at an exact commit.

    GitHub source archives do not contain submodules, so the preparer records
    one snapshot per gitlink: the bindings themselves and, nested inside them,
    ``vendor/llama.cpp``.
    """

    path: str
    repository: str
    commit: str
    zip_sha256: str
    zip_url: str

    def to_mapping(self) -> dict:
        """Return the JSON shape recorded in the manifest."""
        return {
            "path": self.path,
            "repository": self.repository,
            "commit": self.commit,
            "zip_sha256": self.zip_sha256,
            "zip_url": self.zip_url,
        }

    @classmethod
    def from_mapping(cls, values: dict) -> Snapshot:
        """Rebuild a snapshot from the manifest."""
        return cls(
            path=values["path"],
            repository=values["repository"],
            commit=values["commit"],
            zip_sha256=values["zip_sha256"],
            zip_url=values["zip_url"],
        )


@dataclass(frozen=True)
class PatchRecord:
    """A local patch that was applied to the downloaded upstream source.

    Keeping the hashes of every patched file before and after makes the change
    auditable without diffing the whole tree.
    """

    patch: str
    patch_sha256: str
    files: tuple[str, ...]
    pre_sha256: dict[str, str]
    post_sha256: dict[str, str]

    def to_mapping(self) -> dict:
        """Return the JSON shape recorded in the manifest."""
        return {
            "patch": self.patch,
            "patch_sha256": self.patch_sha256,
            "files": list(self.files),
            "pre_sha256": dict(self.pre_sha256),
            "post_sha256": dict(self.post_sha256),
        }

    @classmethod
    def from_mapping(cls, values: dict) -> PatchRecord:
        """Rebuild a patch record from the manifest."""
        return cls(
            patch=values["patch"],
            patch_sha256=values["patch_sha256"],
            files=tuple(values["files"]),
            pre_sha256=dict(values["pre_sha256"]),
            post_sha256=dict(values["post_sha256"]),
        )


@dataclass(frozen=True)
class SourceManifest:
    """Everything known about the immutable source snapshot.

    This is the contract between the ``source`` job and every builder: it says
    which commits went in, which files came out, and what the Python bindings
    must hash to inside a finished wheel.
    """

    plan: Plan
    package: str
    snapshots: tuple[Snapshot, ...]
    runtime_sha256: dict[str, str]
    upstream_runtime_sha256: dict[str, str]
    applied_patches: tuple[PatchRecord, ...]
    source_archive_sha256: str
    packaging_patch_sha256: str
    native_commit: str

    @property
    def version(self) -> str:
        """Upstream version this source was prepared from."""
        return self.plan.version

    @property
    def python_versions(self) -> tuple[str, ...]:
        """Python versions to build from this source."""
        return self.plan.python_versions

    def to_mapping(self) -> dict:
        """Return the JSON document written to ``build-manifest.json``."""
        return {
            **self.plan.to_mapping(),
            "package": self.package,
            "snapshots": [snapshot.to_mapping() for snapshot in self.snapshots],
            "runtime_sha256": dict(self.runtime_sha256),
            "upstream_runtime_sha256": dict(self.upstream_runtime_sha256),
            "applied_patches": [record.to_mapping() for record in self.applied_patches],
            "source_archive_sha256": self.source_archive_sha256,
            "packaging_patch_sha256": self.packaging_patch_sha256,
            "native_commit": self.native_commit,
        }

    @classmethod
    def from_mapping(cls, values: dict) -> SourceManifest:
        """Rebuild a manifest from its JSON document."""
        if not isinstance(values, dict):
            raise DocumentError("A source manifest must be a JSON object")
        try:
            return cls(
                plan=Plan.from_mapping(values),
                package=values["package"],
                snapshots=tuple(Snapshot.from_mapping(item) for item in values["snapshots"]),
                runtime_sha256=dict(values["runtime_sha256"]),
                upstream_runtime_sha256=dict(values["upstream_runtime_sha256"]),
                applied_patches=tuple(
                    PatchRecord.from_mapping(item) for item in values["applied_patches"]
                ),
                source_archive_sha256=values["source_archive_sha256"],
                packaging_patch_sha256=values["packaging_patch_sha256"],
                native_commit=values["native_commit"],
            )
        except KeyError as error:
            raise DocumentError(f"Source manifest is missing {error}") from error


# ---------------------------------------------------------------------------
# Validation receipts and the publication gate
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WheelRecord:
    """One validated wheel, as recorded in a receipt."""

    name: str
    size: int
    sha256: str

    def to_mapping(self) -> dict:
        """Return the JSON shape recorded in a receipt."""
        return {"name": self.name, "size": self.size, "sha256": self.sha256}

    @classmethod
    def from_mapping(cls, values: dict) -> WheelRecord:
        """Rebuild a wheel record from a receipt."""
        return cls(name=values["name"], size=values["size"], sha256=values["sha256"])


@dataclass(frozen=True)
class Receipt:
    """Proof that one build job produced and validated its wheels.

    Receipts are tiny (a few hundred bytes) which is the whole point: the
    validation job can require the entire matrix without downloading a
    multi-gigabyte wheel set.
    """

    version: str
    channel: Channel
    platform: Platform
    recipe_commit: str
    source_archive_sha256: str
    wheels: tuple[WheelRecord, ...]

    def to_mapping(self) -> dict:
        """Return the JSON document uploaded as a receipt artifact."""
        return {
            "schema": RECEIPT_SCHEMA,
            "version": self.version,
            "channel": self.channel.name,
            "platform": self.platform.value,
            "recipe_commit": self.recipe_commit,
            "source_archive_sha256": self.source_archive_sha256,
            "wheels": [wheel.to_mapping() for wheel in self.wheels],
        }

    @classmethod
    def from_mapping(cls, values: dict) -> Receipt:
        """Rebuild a receipt from its JSON document."""
        if not isinstance(values, dict):
            raise DocumentError("A receipt must be a JSON object")
        schema = values.get("schema")
        if type(schema) is not int or schema != RECEIPT_SCHEMA:
            raise DocumentError(f"Unsupported receipt schema: {schema!r}")
        try:
            return cls(
                version=values["version"],
                channel=Channel(str(values["channel"])),
                platform=Platform.parse(str(values["platform"])),
                recipe_commit=values["recipe_commit"],
                source_archive_sha256=values["source_archive_sha256"],
                wheels=tuple(WheelRecord.from_mapping(item) for item in values["wheels"]),
            )
        except KeyError as error:
            raise DocumentError(f"Receipt is missing {error}") from error


@dataclass(frozen=True)
class PublicationGate:
    """The whole validated matrix, required before anything is published."""

    plan: Plan
    source_archive_sha256: str
    channels: dict[str, dict[str, dict]] = field(default_factory=dict)

    def to_mapping(self) -> dict:
        """Return the JSON document written to ``validated-build.json``."""
        return {
            "schema": GATE_SCHEMA,
            "plan": self.plan.to_mapping(),
            "source_archive_sha256": self.source_archive_sha256,
            "channels": copy.deepcopy(self.channels),
        }

    @classmethod
    def from_mapping(cls, values: dict) -> PublicationGate:
        """Rebuild the gate from its JSON document."""
        if not isinstance(values, dict):
            raise DocumentError("A publication gate must be a JSON object")
        schema = values.get("schema")
        if type(schema) is not int or schema != GATE_SCHEMA:
            raise DocumentError(f"Unsupported gate schema: {schema!r}")
        return cls(
            plan=Plan.from_mapping(values["plan"]),
            source_archive_sha256=values["source_archive_sha256"],
            channels=copy.deepcopy(values["channels"]),
        )


# ---------------------------------------------------------------------------
# GitHub releases and provenance
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ReleaseAsset:
    """One file attached to a GitHub release."""

    name: str
    state: str
    size: int
    url: str
    digest: str

    @classmethod
    def from_mapping(cls, values: dict) -> ReleaseAsset:
        """Rebuild an asset from the GitHub API payload."""
        return cls(
            name=str(values.get("name", "")),
            state=str(values.get("state", "")),
            size=values.get("size") if type(values.get("size")) is int else -1,
            url=str(values.get("browser_download_url", "")),
            digest=str(values.get("digest") or ""),
        )

    @property
    def is_uploaded(self) -> bool:
        """Whether GitHub finished processing this asset."""
        return self.state == "uploaded" and self.size > 0


@dataclass(frozen=True)
class Release:
    """A GitHub release we either read (upstream) or publish to (our own)."""

    identifier: int
    tag: str
    name: str
    draft: bool
    prerelease: bool
    published_at: str | None
    body: str
    url: str
    assets: tuple[ReleaseAsset, ...]

    @classmethod
    def from_mapping(cls, values: dict) -> Release:
        """Rebuild a release from the GitHub API payload."""
        if not isinstance(values, dict):
            raise DocumentError("A release must be a JSON object")
        identifier = values.get("id")
        if type(identifier) is not int:
            raise DocumentError("Release payload has no numeric id")
        assets = values.get("assets") or []
        if not isinstance(assets, list):
            raise DocumentError("Release assets must be a list")
        return cls(
            identifier=identifier,
            tag=str(values.get("tag_name", "")),
            name=str(values.get("name") or values.get("tag_name") or ""),
            draft=bool(values.get("draft")),
            prerelease=bool(values.get("prerelease")),
            published_at=values.get("published_at"),
            body=str(values.get("body") or ""),
            url=str(values.get("html_url", "")),
            assets=tuple(ReleaseAsset.from_mapping(item) for item in assets),
        )

    @property
    def version(self) -> Version | None:
        """The stable version carried by this release's tag, if any."""
        return Version.from_tag(self.tag)

    def provenance(self, upstream: str = "", default_package: str = "") -> Provenance | None:
        """Parse the provenance marker, or return ``None`` if there is none."""
        return Provenance.from_body(self.body, self.tag, upstream, default_package)

    def is_complete(self, package: str, python_versions: tuple[str, ...]) -> bool:
        """Whether this release is fully published for `python_versions`.

        A complete release is public, not a draft, not a prerelease, marked
        complete in its provenance, and every expected asset is uploaded with a
        non-zero size.
        """
        return is_complete(self, self.provenance("", package), package, python_versions)


@dataclass(frozen=True)
class Provenance:
    """The hidden marker that makes a published release resumable.

    Every Guanaco release body ends with an HTML comment holding this document.
    It records which upstream commit and recipe commit produced the release, so
    a rerun can finish an interrupted family and will refuse to touch anything
    it did not create.
    """

    version: str
    channel: Channel
    tag: str
    upstream_repository: str
    upstream_commit: str
    upstream_tag: str
    upstream_release_id: int
    recipe_commit: str
    python_versions: tuple[str, ...]
    complete: bool
    origin: UpstreamOrigin | None = None
    matrix: BuildMatrix | None = None

    @classmethod
    def from_body(
        cls, body: str, release_tag: str, upstream: str = "", default_package: str = ""
    ) -> Provenance | None:
        """Parse the marker of a release body, or return ``None``.

        Args:
            body: The full release body.
            release_tag: Tag of the release the body belongs to.
            upstream: Upstream repository the marker must point at, if known.
            default_package: Distribution name to assume for markers written
                before the frozen matrix carried one.

        Raises:
            DocumentError: If a marker exists but is ambiguous or malformed.
        """
        text = body or ""
        boundary = max(
            text.rfind(PROVENANCE_SEPARATOR),
            text.rfind(PROVENANCE_SEPARATOR.replace("\n", "\r\n")),
        )
        matches = PROVENANCE_PATTERN.findall(text[boundary:] if boundary >= 0 else text)
        if not matches:
            return None
        if len(matches) != 1 or matches[0].count("<!-- " + PROVENANCE_MARKER):
            raise DocumentError("Ambiguous provenance marker")
        try:
            values = json.loads(matches[0])
        except json.JSONDecodeError as error:
            raise DocumentError("Provenance marker is not valid JSON") from error
        return cls.from_document(values, release_tag, upstream, body, default_package)

    @classmethod
    def from_document(
        cls,
        values: dict,
        release_tag: str,
        upstream: str,
        body: str,
        default_package: str = "",
    ) -> Provenance:
        """Rebuild provenance from its decoded JSON document.

        Args:
            values: The decoded marker.
            release_tag: Tag of the release the marker was found in.
            upstream: Upstream repository we expect the marker to point at.
            body: Full release body, used to verify the frozen note hash.
            default_package: Fallback distribution name for old markers.
        """
        try:
            snapshot = values.get("snapshot")
            provenance = cls(
                version=str(Version.parse(values["version"])),
                channel=Channel(str(values["channel"])),
                tag=values["tag"],
                upstream_repository=values["upstream_repository"],
                upstream_commit=values["upstream_commit"],
                upstream_tag=values["upstream_tag"],
                upstream_release_id=values["upstream_release_id"],
                recipe_commit=values["recipe_commit"],
                python_versions=tuple(str(item) for item in values["python_versions"]),
                complete=values["complete"],
                origin=cls._restore_origin(snapshot, body) if snapshot else None,
                matrix=BuildMatrix.from_mapping(
                    {**snapshot, "upstream": snapshot["upstream"]["repository"]},
                    default_package=default_package,
                )
                if snapshot
                else None,
            )
        except (KeyError, TypeError, AttributeError) as error:
            raise DocumentError("Malformed Guanaco release provenance") from error
        provenance.validate(release_tag, upstream)
        return provenance

    @staticmethod
    def _restore_origin(snapshot: dict, body: str) -> UpstreamOrigin:
        """Rebuild the frozen upstream origin, verifying the note hash.

        Markers written today store ``body_sha256`` instead of the notes
        themselves, because the notes already sit above the separator. Verifying
        the hash here means a hand-edited changelog cannot go unnoticed.
        """
        raw = copy.deepcopy(snapshot["upstream"])
        expected = raw.pop("body_sha256", None)
        if expected is None:
            return UpstreamOrigin.from_mapping(raw)
        notes = source_notes(body)
        if hashlib.sha256(notes.encode("utf-8")).hexdigest() != expected:
            raise DocumentError("Upstream note snapshot checksum mismatch")
        if "body" in raw:
            raise DocumentError("Ambiguous note snapshot")
        raw["body"] = notes
        return UpstreamOrigin.from_mapping(raw)

    def to_document(self, *, hash_notes: bool) -> dict:
        """Return the document embedded in a release body.

        Args:
            hash_notes: Store only the SHA-256 of the upstream notes. The notes
                are already at the top of the release body, so duplicating them
                inside the hidden marker would double its size.
        """
        document = {
            "version": self.version,
            "channel": self.channel.name,
            "tag": self.tag,
            "upstream_repository": self.upstream_repository,
            "upstream_commit": self.upstream_commit,
            "upstream_tag": self.upstream_tag,
            "upstream_release_id": self.upstream_release_id,
            "recipe_commit": self.recipe_commit,
            "python_versions": list(self.python_versions),
            "complete": self.complete,
        }
        if self.matrix is not None and self.origin is not None:
            origin = self.origin.with_body_hash() if hash_notes else self.origin.to_mapping()
            # The snapshot keeps the upstream *origin* under the "upstream" key,
            # exactly as a plan does; the matrix's own repository name is dropped
            # so the two cannot collide.
            matrix = self.matrix.to_mapping()
            del matrix["upstream"]
            document["snapshot"] = {**matrix, "upstream": origin}
        return document

    def render(self, *, hash_notes: bool = True) -> str:
        """Render the marker comment exactly as it appears in a release body."""
        document = json.dumps(
            self.to_document(hash_notes=hash_notes), sort_keys=True, separators=(",", ":")
        )
        # A literal "-->" inside the JSON would close the HTML comment early.
        escaped = document.replace("-->", "--\\u003e")
        return f"<!-- {PROVENANCE_MARKER}\n{escaped}\n-->"

    def validate(self, release_tag: str, upstream: str) -> None:
        """Check that the marker is internally consistent."""
        Version.parse(self.version)
        if not COMMIT_PATTERN.fullmatch(self.upstream_commit):
            raise DocumentError("Invalid upstream commit in provenance")
        if not COMMIT_PATTERN.fullmatch(self.recipe_commit):
            raise DocumentError("Invalid build recipe commit in provenance")
        if type(self.upstream_release_id) is not int or self.upstream_release_id <= 0:
            raise DocumentError("Invalid upstream release id in provenance")
        if type(self.complete) is not bool:
            raise DocumentError("Provenance complete must be a boolean")
        if not self.python_versions or len(set(self.python_versions)) != len(self.python_versions):
            raise DocumentError("Provenance Python versions must be a unique non-empty list")
        if self.tag != self.channel.release_tag(self.version) or self.tag != release_tag:
            raise DocumentError("Provenance tag does not match its channel and release")
        if upstream and self.upstream_repository != upstream:
            raise DocumentError("Provenance belongs to a different upstream repository")
        if Version.from_tag(self.upstream_tag) != Version.parse(self.version):
            raise DocumentError("Provenance does not match the upstream version")
        if self.matrix is None:
            return
        self.matrix.validate()
        if self.origin is None:
            return
        if (
            self.origin.repository != self.upstream_repository
            or self.origin.commit != self.upstream_commit
            or self.origin.tag != self.upstream_tag
            or self.origin.release_id != self.upstream_release_id
            or self.matrix.python_versions != self.python_versions
        ):
            raise DocumentError("Frozen snapshot does not match release provenance")


def source_notes(body: str) -> str:
    """Return the upstream notes stored above the provenance separator.

    Raises:
        DocumentError: If the body has no separator, which means it predates the
            current provenance format and needs manual recovery.
    """
    text = body or ""
    boundary = max(
        text.rfind(PROVENANCE_SEPARATOR),
        text.rfind(PROVENANCE_SEPARATOR.replace("\n", "\r\n")),
    )
    if boundary < 0:
        raise DocumentError(
            "Legacy provenance has no upstream note snapshot; manual recovery required"
        )
    return text[:boundary]


def wheel_prefix(package: str) -> str:
    """Return the normalised wheel filename prefix, e.g. ``guanaco-py`` -> ``guanaco_py``."""
    return re.sub(r"[-_.]+", "_", package).lower()


def expected_asset_names(
    package: str, version: Version | str, channel: Channel, python_versions: tuple[str, ...]
) -> set[str]:
    """Return every asset a complete release of one channel must contain.

    Args:
        package: Distribution name, used for the wheel filename prefix.
        version: Upstream version being published.
        channel: The channel the assets belong to.
        python_versions: Python versions the channel was built for.

    Returns:
        The wheel names for both platforms plus the manifest, the checksum file
        and -- on the CPU channel, which doubles as the source release -- the
        reconstructed source archive and the packaging patch.
    """
    prefix = wheel_prefix(package)
    text = str(version)
    names = {
        f"{prefix}-{text}-cp{v.replace('.', '')}-cp{v.replace('.', '')}-{platform}.whl"
        for v in python_versions
        for platform in (
            channel.wheel_platform(Platform.LINUX),
            channel.wheel_platform(Platform.WINDOWS),
        )
    }
    names.update({MANIFEST_ASSET, CHECKSUM_ASSET})
    if channel.name == CPU_CHANNELS[0]:
        names.update({source_asset(package, text), PACKAGING_PATCH_ASSET})
    return names


def is_complete(
    release: Release, state: Provenance | None, package: str, python_versions: tuple[str, ...]
) -> bool:
    """Whether a release can be considered fully and correctly published.

    Args:
        release: The GitHub release to inspect.
        state: Its parsed provenance; without one the release is not ours.
        package: Distribution name, needed to compute expected wheel names.
        python_versions: Fallback Python versions for markers without one.
    """
    if (
        release is None
        or state is None
        or release.draft
        or release.prerelease
        or not state.complete
    ):
        return False
    names = [asset.name for asset in release.assets]
    if len(names) != len(set(names)):
        return False
    uploaded = {asset.name for asset in release.assets if asset.is_uploaded}
    expected = expected_asset_names(
        package,
        state.version,
        state.channel,
        state.python_versions or python_versions,
    )
    return expected <= uploaded


# ---------------------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------------------


def read_json(path: Path) -> dict:
    """Read a JSON document, raising :class:`DocumentError` if it is unreadable."""
    try:
        values = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise DocumentError(f"Cannot read {path}: {error}") from error
    if not isinstance(values, dict):
        raise DocumentError(f"{path} must contain a JSON object")
    return values


def write_json(path: Path, value: dict) -> None:
    """Write a JSON document atomically, so a crash cannot leave a half file.

    The content is written to a temporary file in the same directory and then
    moved into place, which is atomic on both Linux and Windows.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(value, indent=2, sort_keys=True) + "\n"
    handle, temporary_name = tempfile.mkstemp(prefix=".json-", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(content)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def write_outputs(path: str | Path | None, **values: object) -> None:
    """Append GitHub Actions workflow outputs.

    Every value is validated before anything is written, because a stray
    newline in an output can corrupt the whole ``GITHUB_OUTPUT`` file.

    Args:
        path: The ``GITHUB_OUTPUT`` file, or ``None`` to only print.
        **values: Output names and values; booleans become ``true``/``false``
            and containers are serialised as compact JSON.
    """
    records = []
    for key, value in values.items():
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
            raise DocumentError(f"Invalid workflow output key: {key!r}")
        if isinstance(value, bool):
            value = "true" if value else "false"
        elif isinstance(value, (dict, list, tuple)):
            value = json.dumps(value, separators=(",", ":"))
        elif isinstance(value, enum.Enum):
            value = str(value.value)
        text = str(value)
        if "\n" in text or "\r" in text:
            raise DocumentError(f"Workflow output {key!r} must be a single line")
        records.append(f"{key}={text}\n")
    if path:
        with Path(path).open("a", encoding="utf-8", newline="\n") as stream:
            stream.writelines(records)


__all__ = [
    "CPU_CHANNELS",
    "CHANNEL_PATTERN",
    "Channel",
    "ConfigurationError",
    "DocumentError",
    "PatchRecord",
    "Plan",
    "Platform",
    "Provenance",
    "PublicationGate",
    "Receipt",
    "Release",
    "ReleaseAsset",
    "Snapshot",
    "SourceManifest",
    "UpstreamOrigin",
    "Version",
    "WheelRecord",
    "expected_asset_names",
    "is_complete",
    "read_json",
    "source_notes",
    "wheel_prefix",
    "write_json",
    "write_outputs",
]
