"""Upstream discovery, release planning and publication.

Three closely related jobs live here, each with its own class:

* :class:`UpstreamSelector` decides *which* upstream version to build.
* :class:`ReleasePlanner` turns that into a frozen :class:`~guanaco.models.Plan`,
  reusing the snapshot of a partially published version family when there is one.
* :class:`TestBuildPlanner` plans a manual rehearsal, which is always a fresh
  plan and never touches published releases.

:class:`ReceiptCollector` is the gate: it proves the whole requested matrix was
built and validated, using the small receipts instead of the wheels themselves.
:class:`ReleasePublisher` then stages and publishes, checking every destination
before it writes anything.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import urllib.parse
from dataclasses import dataclass
from pathlib import Path

from .models import (
    CHECKSUM_ASSET,
    COMMIT_PATTERN,
    MANIFEST_ASSET,
    PACKAGING_PATCH_ASSET,
    PROVENANCE_SEPARATOR,
    BuildMatrix,
    Channel,
    FrozenSource,
    Plan,
    Platform,
    Provenance,
    PublicationGate,
    Receipt,
    Release,
    SourceManifest,
    UpstreamOrigin,
    Version,
    expected_asset_names,
    is_complete,
    read_json,
    source_asset,
    source_notes,
    write_json,
)
from .settings import Settings
from .source import SourceArchive
from .toolchain import cpu_artifact, cuda_artifact
from .transfer import empty_destination, sha256
from .wheels import WheelValidator


class ReleaseError(ValueError):
    """Raised when a release cannot be planned, gated or published safely."""


@dataclass(frozen=True)
class ArtifactSpec:
    """One expected build artifact: its name and what it must contain."""

    name: str
    channel: Channel
    platform: Platform
    python_versions: tuple[str, ...]


def artifact_specs(plan: Plan, package: str) -> dict[str, ArtifactSpec]:
    """Return every artifact `plan` expects, keyed by artifact name.

    CPU and AVX2 jobs produce the whole Python matrix in one artifact; CUDA jobs
    produce a single wheel each, so their names carry the Python version.
    """
    prefix = "test-" if plan.test_only else ""
    result: dict[str, ArtifactSpec] = {}
    for channel in plan.missing_channels:
        for platform in plan.build_platforms():
            if channel.is_cpu_variant:
                name = cpu_artifact(package, prefix, channel, platform)
                result[name] = ArtifactSpec(name, channel, platform, plan.python_versions)
            else:
                for python in plan.python_versions:
                    name = cuda_artifact(package, prefix, channel, platform, python)
                    result[name] = ArtifactSpec(name, channel, platform, (python,))
    return result


def fresh_origin(client, release: Release, upstream: str) -> UpstreamOrigin:
    """Resolve an upstream release to an immutable commit and describe it."""
    commit = client.commit(upstream, release.tag)
    return UpstreamOrigin(
        repository=upstream,
        release_id=release.identifier,
        tag=release.tag,
        commit=commit,
        release_url=release.url,
        release_name=release.name,
        published_at=release.published_at,
        body=release.body,
        zip_url=f"https://codeload.github.com/{upstream}/zip/{commit}",
    )


def run_url(repository: str) -> str | None:
    """Return the URL of the current Actions run, or ``None`` when running locally."""
    identifier = os.getenv("GITHUB_RUN_ID")
    return f"https://github.com/{repository}/actions/runs/{identifier}" if identifier else None


def recipe_commit() -> str:
    """Return the recipe commit to record, or a local marker outside Actions."""
    return os.getenv("GITHUB_SHA", "local-working-tree")


class UpstreamSelector:
    """Chooses the newest stable upstream version, ignoring drafts and previews."""

    def __init__(self, settings: Settings, client) -> None:
        """Remember the upstream repository and the API client."""
        self.settings = settings
        self.client = client

    def select(self, releases: list[Release], requested: str = "") -> Release:
        """Return the release to build.

        Args:
            releases: Every release of the upstream repository.
            requested: An explicit ``X.Y.Z``, or empty for the newest one.

        Raises:
            ReleaseError: If nothing stable matches.
        """
        if requested:
            Version.parse(requested)
        eligible = [
            release
            for release in releases
            if release.version is not None
            and not release.draft
            and not release.prerelease
            and release.published_at
            and (not requested or str(release.version) == requested)
        ]
        if not eligible:
            raise ReleaseError(
                f"No published stable upstream release found for {requested or 'latest'}"
            )
        return max(eligible, key=lambda item: (item.version, item.published_at, item.tag))

    def latest_version(self, releases: list[Release]) -> Version:
        """Return the newest stable version available upstream."""
        return self.select(releases).version


class ReleasePlanner:
    """Builds the frozen plan for a release, resuming a partial family if needed.

    Upstream publishes one tag per backend and date, all belonging to a single
    ``X.Y.Z``. A version family is only complete once every channel has all of
    its wheels. If a previous run got halfway, this planner *reuses the frozen
    snapshot* of that family instead of silently switching to newer upstream
    code.
    """

    def __init__(self, settings: Settings, client) -> None:
        """Remember the configuration and the API client."""
        self.settings = settings
        self.client = client
        self.selector = UpstreamSelector(settings, client)

    def plan(self, requested_version: str = "") -> Plan:
        """Return the plan for `requested_version`, or for the newest one."""
        upstream_releases = self.client.releases(self.settings.upstream)
        selected = self.selector.select(upstream_releases, requested_version)
        version = selected.version
        latest = self.selector.latest_version(upstream_releases)
        family = self._family(version)
        frozen, states = self._frozen_family(family, upstream_releases)
        if frozen is None:
            frozen = FrozenSource(
                origin=fresh_origin(self.client, selected, self.settings.upstream),
                matrix=self.settings.matrix,
            )
        missing = tuple(
            channel
            for channel in frozen.matrix.channels
            if not is_complete(
                family.get(channel),
                states.get(channel),
                self.settings.package,
                frozen.matrix.python_versions,
            )
        )
        return Plan(
            repository=self.settings.repository,
            version=str(version),
            origin=frozen.origin,
            matrix=frozen.matrix,
            missing_channels=missing,
            promote_latest=version == latest,
            recipe_commit=recipe_commit(),
            run_url=run_url(self.settings.repository),
        )

    # -- Family inspection ---------------------------------------------------

    def _family(self, version: Version) -> dict[Channel, Release]:
        """Group our own releases that belong to one upstream version."""
        pattern = re.compile(rf"v{re.escape(str(version))}(?:-(avx2|cu[0-9]+))?")
        family: dict[Channel, Release] = {}
        for release in self.client.releases(self.settings.repository):
            match = pattern.fullmatch(release.tag)
            if not match:
                continue
            channel = Channel(match[1] or "cpu")
            if channel in family:
                raise ReleaseError(f"Duplicate releases for {release.tag}; resolve the ambiguity")
            family[channel] = release
        return family

    def _frozen_family(
        self, family: dict[Channel, Release], upstream_releases: list[Release]
    ) -> tuple[FrozenSource | None, dict[Channel, Provenance]]:
        """Read the provenance of every release in the family.

        Raises:
            ReleaseError: If a release exists without provenance, if a published
                release is incomplete, or if the family disagrees with itself.
        """
        states: dict[Channel, Provenance] = {}
        snapshots: list[FrozenSource] = []
        for channel, release in family.items():
            state = release.provenance(self.settings.upstream, self.settings.package)
            if state is None:
                raise ReleaseError(
                    f"{release.tag} already exists without Guanaco provenance; "
                    "refusing to overwrite a legacy/manual release"
                )
            if not release.draft and not is_complete(
                release, state, self.settings.package, state.python_versions
            ):
                raise ReleaseError(
                    f"Published release {release.tag} is incomplete; refusing automatic replacement"
                )
            states[channel] = state
            snapshots.append(
                FrozenSource(state.origin, state.matrix)
                if state.matrix is not None
                else self._legacy_snapshot(state, release, upstream_releases)
            )
        if len({item.origin.commit for item in snapshots}) > 1:
            raise ReleaseError(
                "Mixed upstream commits in this release family; manual investigation required"
            )
        if snapshots and any(item != snapshots[0] for item in snapshots[1:]):
            raise ReleaseError("Mixed source/notes/build matrices in the release family")
        return (snapshots[0] if snapshots else None), states

    def _legacy_snapshot(
        self, state: Provenance, release: Release, upstream_releases: list[Release]
    ) -> FrozenSource:
        """Rebuild a snapshot from a marker written before snapshots existed.

        The notes are recovered from the release body above the separator, the
        pinned Python versions come from the marker, and the CUDA matrix comes
        from today's configuration -- an old marker cannot recover a historical
        one, and that is stated in the log.
        """
        upstream = self.settings.upstream
        original = next(
            (item for item in upstream_releases if item.identifier == state.upstream_release_id),
            None,
        )
        quoted = urllib.parse.quote(state.upstream_tag, safe="")
        origin = UpstreamOrigin(
            repository=upstream,
            release_id=state.upstream_release_id,
            tag=state.upstream_tag,
            commit=state.upstream_commit,
            release_url=f"https://github.com/{upstream}/releases/tag/{quoted}",
            release_name=(original.name if original else "") or state.upstream_tag,
            published_at=original.published_at if original else None,
            body=source_notes(release.body),
            zip_url=f"https://codeload.github.com/{upstream}/zip/{state.upstream_commit}",
        )
        matrix = BuildMatrix(
            upstream=upstream,
            package=self.settings.package,
            python_versions=state.python_versions,
            channels=self.settings.channels,
            cuda=self.settings.cuda,
        )
        print("Legacy marker: using its pinned source/Python versions and the current CUDA matrix")
        return FrozenSource(origin=origin, matrix=matrix)


class TestBuildPlanner:
    """Plans a manual, artifact-only rehearsal.

    Unlike :class:`ReleasePlanner` this never looks at our own releases or
    drafts: a rehearsal always tests the *current* recipe and patches against
    the selected upstream version, even one that is already published.
    """

    # The class plans test builds; it is not a pytest test case.
    __test__ = False

    def __init__(self, settings: Settings, client) -> None:
        """Remember the configuration and the API client."""
        self.settings = settings
        self.client = client
        self.selector = UpstreamSelector(settings, client)

    def plan(
        self,
        *,
        version: str = "",
        cpu: bool = True,
        avx2: bool = False,
        cuda: bool = False,
        cuda_channels: str = "",
        python_versions: str = "3.13",
        systems: str = "both",
    ) -> Plan:
        """Return a ``test_only`` plan for the requested subset of the matrix.

        Args:
            version: Upstream ``X.Y.Z``; empty selects the newest stable release.
            cpu: Build the portable CPU channel.
            avx2: Build the AVX2 channel.
            cuda: Build CUDA channels.
            cuda_channels: Comma separated CUDA channels, or ``all``.
            python_versions: Comma separated Python versions, or ``all``.
            systems: ``linux``, ``windows`` or ``both``.
        """
        version = version.strip()
        if version:
            Version.parse(version)
        if any(type(value) is not bool for value in (cpu, avx2, cuda)):
            raise ReleaseError("Channel switches must be booleans")
        if not (cpu or avx2 or cuda):
            raise ReleaseError("Select at least one of CPU, AVX2 or CUDA")
        if systems not in ("linux", "windows", "both"):
            raise ReleaseError("Systems must be linux, windows or both")

        selected_python = tuple(
            self.selection(python_versions, list(self.settings.python_versions), "Python versions")
        )
        channels = (["cpu"] if cpu else []) + (["avx2"] if avx2 else [])
        if cuda:
            channels += self.selection(
                cuda_channels or "all", list(self.settings.cuda), "CUDA channels"
            )
        release = self.selector.select(self.client.releases(self.settings.upstream), version)
        origin = fresh_origin(self.client, release, self.settings.upstream)
        matrix = BuildMatrix(
            upstream=self.settings.upstream,
            package=self.settings.package,
            python_versions=selected_python,
            channels=self.settings.channels,
            cuda=self.settings.cuda,
        )
        platforms = (
            (Platform.LINUX, Platform.WINDOWS) if systems == "both" else (Platform.parse(systems),)
        )
        return Plan(
            repository=self.settings.repository,
            version=str(Version.from_tag(origin.tag)),
            origin=origin,
            matrix=matrix,
            missing_channels=tuple(Channel(name) for name in channels),
            promote_latest=False,
            recipe_commit=recipe_commit(),
            run_url=run_url(self.settings.repository),
            test_only=True,
            platforms=platforms,
        )

    @staticmethod
    def selection(text: str, allowed: list[str], label: str) -> list[str]:
        """Parse a comma separated selection, or ``all``.

        Returns the selection in `allowed` order, so the result is stable no
        matter how the caller typed it.
        """
        values = [value.strip().lower() for value in (text or "").split(",")]
        if values == ["all"]:
            return list(allowed)
        if not values or any(value not in allowed for value in values):
            raise ReleaseError(
                f"Invalid {label}: use a comma-separated selection of {', '.join(allowed)}, or all"
            )
        if len(values) != len(set(values)):
            raise ReleaseError(f"Duplicate {label} selection")
        return [value for value in allowed if value in values]


class ReceiptCollector:
    """Collects the small validation receipts into the publication gate.

    The gate proves the *entire* requested matrix was built and verified. It is
    deliberately built from receipts rather than wheels, so the validation job
    never has to download gigabytes of binaries.
    """

    def __init__(self, settings: Settings) -> None:
        """Remember the distribution name used to compute expected assets."""
        self.settings = settings

    def collect(self, plan: Plan, manifest: SourceManifest, receipts: Path) -> PublicationGate:
        """Require every receipt `plan` expects and return the gate.

        Raises:
            ReleaseError: If a receipt is missing, belongs to another build, or
                does not cover the complete wheel matrix.
        """
        self.check_plan(plan, manifest)
        inventory: dict[str, dict[str, dict]] = {
            channel.name: {} for channel in plan.missing_channels
        }
        for artifact, spec in artifact_specs(plan, self.settings.package).items():
            path = Path(receipts) / f"{artifact}.json"
            if not path.is_file():
                raise ReleaseError(f"Missing validation receipt: {artifact}")
            receipt = Receipt.from_mapping(read_json(path))
            self._check_identity(receipt, plan, manifest, spec, artifact)
            allowed = self._expected_wheels(plan, spec)
            if len(receipt.wheels) != len(allowed) or (
                {wheel.name for wheel in receipt.wheels} != allowed
            ):
                raise ReleaseError(f"Incomplete wheel matrix in receipt: {artifact}")
            for wheel in receipt.wheels:
                if (
                    type(wheel.size) is not int
                    or wheel.size <= 0
                    or not re.fullmatch(r"[a-f0-9]{64}", wheel.sha256)
                ):
                    raise ReleaseError(f"Invalid wheel checksum/size in receipt: {artifact}")
                inventory[spec.channel.name][wheel.name] = {
                    "size": wheel.size,
                    "sha256": wheel.sha256,
                }
        return PublicationGate(
            plan=plan,
            source_archive_sha256=manifest.source_archive_sha256,
            channels=inventory,
        )

    @staticmethod
    def check_plan(plan: Plan, manifest: SourceManifest) -> None:
        """Check that a prepared manifest was really made from this plan."""
        document = manifest.to_mapping()
        for key, value in plan.to_mapping().items():
            if key not in document or document[key] != value:
                raise ReleaseError(f"Prepared source does not match the release plan: {key}")

    def _check_identity(
        self,
        receipt: Receipt,
        plan: Plan,
        manifest: SourceManifest,
        spec: ArtifactSpec,
        artifact: str,
    ) -> None:
        """Check that a receipt belongs to this exact build."""
        expected = {
            "version": plan.version,
            "recipe_commit": plan.recipe_commit,
            "source_archive_sha256": manifest.source_archive_sha256,
            "channel": spec.channel.name,
            "platform": spec.platform.value,
        }
        actual = {
            "version": receipt.version,
            "recipe_commit": receipt.recipe_commit,
            "source_archive_sha256": receipt.source_archive_sha256,
            "channel": receipt.channel.name,
            "platform": receipt.platform.value,
        }
        if actual != expected:
            raise ReleaseError(f"Receipt identity mismatch: {artifact}")

    def _expected_wheels(self, plan: Plan, spec: ArtifactSpec) -> set[str]:
        """Return the wheel names one receipt must contain."""
        names = expected_asset_names(
            self.settings.package, plan.version, spec.channel, spec.python_versions
        )
        return {
            name
            for name in names
            if name.endswith(".whl") and ("win_amd64" in name) == spec.platform.is_windows
        }


class ReleasePublisher:
    """Stages validated wheels and publishes immutable channel releases.

    Publication is deliberately conservative:

    * every destination is inspected *before* anything is written;
    * releases are created as drafts, uploaded, verified, and only then made
      public;
    * a complete public release is never replaced, and a Git tag is never moved.
    """

    def __init__(self, settings: Settings, client) -> None:
        """Remember the configuration, the API client and the wheel validator."""
        self.settings = settings
        self.client = client
        self.validator = WheelValidator(settings)

    # -- Inspection ----------------------------------------------------------

    def preflight(self, plan: Plan) -> dict[Channel, Release | None]:
        """Inspect every destination a publication would touch.

        Raises:
            ReleaseError: If a destination belongs to another build, if a
                published release is incomplete, or if a Git tag would move.
        """
        self._require_release(plan)
        existing: dict[Channel, Release | None] = {}
        for channel in plan.missing_channels:
            tag = channel.release_tag(plan.version)
            release = self.client.release(plan.repository, tag)
            expected = self.state(plan, channel, False)
            state = (
                release.provenance(self.settings.upstream, self.settings.package)
                if release
                else None
            )
            if release:
                if state is None or any(
                    getattr(state, key) != getattr(expected, key)
                    for key in (
                        "upstream_commit",
                        "upstream_tag",
                        "upstream_release_id",
                        "python_versions",
                    )
                ):
                    raise ReleaseError(f"Refusing to replace unrelated release {tag}")
                if state.matrix is not None and state.matrix != plan.matrix:
                    raise ReleaseError(f"Frozen source/notes/matrix mismatch: {tag}")
                if not release.draft and not is_complete(
                    release, state, self.settings.package, plan.python_versions
                ):
                    raise ReleaseError(
                        f"Published release {tag} is incomplete; "
                        "refusing to make it a draft or replace it"
                    )
            target = self.client.tag_commit(plan.repository, tag)
            expected_recipe = (
                state.recipe_commit
                if state is not None and release is not None and not release.draft
                else plan.recipe_commit
            )
            if target is not None and target != expected_recipe:
                raise ReleaseError(
                    f"Existing Git tag {tag} points to another build recipe; "
                    "tags are never moved automatically"
                )
            if (
                release is not None
                and release.draft
                and state is not None
                and state.recipe_commit != plan.recipe_commit
            ):
                raise ReleaseError(
                    f"Draft {tag} belongs to another build recipe; resolve it before retrying"
                )
            existing[channel] = release
        return existing

    # -- Staging -------------------------------------------------------------

    def stage(
        self,
        plan: Plan,
        prepared: Path,
        artifacts: Path,
        output: Path,
        *,
        channel: Channel | None = None,
        gate: PublicationGate | None = None,
    ) -> tuple[SourceManifest, dict[Channel, Path]]:
        """Verify and stage the wheels of one or every channel.

        Args:
            plan: The frozen release plan.
            prepared: Directory holding the verified source snapshot.
            artifacts: Directory of downloaded build artifacts.
            output: Where to build the staged release folders.
            channel: Stage only this channel, to keep CI disk usage bounded.
            gate: The global validation gate, when publishing one channel.

        Returns:
            The source manifest and the staged folder per channel. Nothing is
            uploaded; call :meth:`publish` for that.
        """
        self._require_release(plan)
        manifest = SourceArchive.verify(prepared)
        ReceiptCollector.check_plan(plan, manifest)
        if channel is not None and channel not in plan.missing_channels:
            raise ReleaseError("Cannot stage a channel absent from the release plan")
        channels = [channel] if channel else list(plan.missing_channels)
        if gate is not None:
            self._check_gate(plan, manifest, gate)
        with empty_destination(output) as temporary:
            self._collect(plan, Path(artifacts), temporary, channels)
            staged: dict[Channel, Path] = {}
            for selected in channels:
                folder = temporary / "releases" / selected.name
                folder.mkdir(parents=True)
                for platform in (Platform.LINUX, Platform.WINDOWS):
                    wheels = self.validator.verify_directory(
                        temporary / "validation" / selected.name / platform.value,
                        manifest,
                        selected,
                        platform,
                    )
                    for wheel in wheels:
                        wheel.replace(folder / wheel.name)
                if selected.name == "cpu":
                    self._link_or_copy(
                        Path(prepared) / "source.tar.gz",
                        folder / source_asset(self.settings.package, plan.version),
                    )
                    self._link_or_copy(
                        Path(prepared) / "packaging.patch", folder / PACKAGING_PATCH_ASSET
                    )
                self._write_inventory(plan, manifest, selected, folder, gate)
                staged[selected] = folder
        return manifest, {
            selected: Path(output) / "releases" / selected.name for selected in staged
        }

    def _collect(
        self,
        plan: Plan,
        artifacts: Path,
        temporary: Path,
        channels: list[Channel],
    ) -> None:
        """Move downloaded wheels into per-channel validation folders."""
        specs = artifact_specs(plan, self.settings.package)
        for wheel in sorted(artifacts.rglob("*.whl")):
            relative = wheel.relative_to(artifacts)
            if len(relative.parts) != 2 or wheel.is_symlink() or wheel.parent.is_symlink():
                raise ReleaseError(f"Unexpected wheel artifact layout: {relative}")
            artifact = relative.parts[0]
            if artifact not in specs:
                raise ReleaseError(f"Unexpected or unrequested build artifact: {artifact}")
            spec = specs[artifact]
            if spec.channel not in channels:
                raise ReleaseError(f"Unrequested build channel: {spec.channel}")
            if spec.channel.is_cuda and wheel.name.split("-")[2] != "cp" + spec.python_versions[
                0
            ].replace(".", ""):
                raise ReleaseError(f"CUDA artifact/wheel Python mismatch: {artifact}")
            destination = (
                temporary / "validation" / spec.channel.name / spec.platform.value / wheel.name
            )
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists():
                raise ReleaseError(f"Duplicate wheel artifact: {wheel.name}")
            self._link_or_copy(wheel, destination)

    def _write_inventory(
        self,
        plan: Plan,
        manifest: SourceManifest,
        channel: Channel,
        folder: Path,
        gate: PublicationGate | None,
    ) -> None:
        """Write ``guanaco-build.json`` and ``SHA256SUMS`` for one channel."""
        hashes = {path.name: sha256(path) for path in sorted(folder.iterdir())}
        if gate is not None:
            for path in folder.glob("*.whl"):
                expected = gate.channels[channel.name][path.name]
                if expected != {"size": path.stat().st_size, "sha256": hashes[path.name]}:
                    raise ReleaseError(
                        f"Wheel differs from the globally validated receipt: {path.name}"
                    )
        write_json(
            folder / MANIFEST_ASSET,
            {
                **manifest.to_mapping(),
                "channel": channel.name,
                "release_tag": channel.release_tag(plan.version),
                "assets_sha256": hashes,
            },
        )
        hashes[MANIFEST_ASSET] = sha256(folder / MANIFEST_ASSET)
        with (folder / CHECKSUM_ASSET).open("w", encoding="utf-8", newline="\n") as stream:
            stream.write("".join(f"{digest}  {name}\n" for name, digest in sorted(hashes.items())))
        if {path.name for path in folder.iterdir()} != expected_asset_names(
            self.settings.package, plan.version, channel, plan.python_versions
        ):
            raise ReleaseError("Staged release inventory does not match the expected matrix")

    # -- Publication ---------------------------------------------------------

    def publish(self, plan: Plan, folders: dict[Channel, Path], uploader=None) -> None:
        """Publish every staged folder, or skip the ones already complete.

        Args:
            plan: The frozen release plan.
            folders: Staged folders, keyed by channel.
            uploader: Injection point for tests; defaults to ``gh release upload``.
        """
        self._require_release(plan)
        if not COMMIT_PATTERN.fullmatch(plan.recipe_commit):
            raise ReleaseError("Publishing requires an immutable automation commit (GITHUB_SHA)")
        if os.getenv("GITHUB_REPOSITORY") != plan.repository:
            raise ReleaseError("Publishing repository does not match GITHUB_REPOSITORY")
        existing = self.preflight(plan)
        upload = uploader or self.upload
        for channel, folder in folders.items():
            tag = channel.release_tag(plan.version)
            release = existing.get(channel)
            if release is not None and not release.draft:
                print(f"{tag} already complete; keeping its published binaries")
                continue
            files = sorted(folder.iterdir())
            body = self.release_body(plan, channel, False)
            if release is not None:
                release = self.client.update_release(
                    plan.repository, release.identifier, body=body, draft=True
                )
            else:
                release = self.client.create_release(
                    plan.repository,
                    tag=tag,
                    commit=plan.recipe_commit,
                    name=tag,
                    body=body,
                    draft=True,
                    latest=False,
                )
            upload(plan, tag, files)
            refreshed = self.client.release(plan.repository, tag)
            self._check_uploaded(refreshed, files)
            self.client.update_release(
                plan.repository,
                release.identifier,
                body=self.release_body(plan, channel, True),
                draft=False,
                latest=channel.name == "cpu" and plan.promote_latest,
            )
            print(f"Published {tag} with {len(files)} verified assets")

    def upload(self, plan: Plan, tag: str, files: list[Path]) -> None:
        """Upload assets with the GitHub CLI."""
        subprocess.run(
            [
                "gh",
                "release",
                "upload",
                tag,
                *[str(path) for path in files],
                "--repo",
                plan.repository,
                "--clobber",
            ],
            check=True,
        )

    def release_body(self, plan: Plan, channel: Channel, finished: bool) -> str:
        """Render the release notes for one channel.

        The upstream notes are preserved at the top, followed by our own
        provenance section and the hidden marker.
        """
        origin = plan.origin
        marker = self.state(plan, channel, finished).render(hash_notes=True)
        upstream = self.settings.upstream
        return (
            origin.body
            + PROVENANCE_SEPARATOR
            + (
                f"Rebuilt as **{self.settings.package} {plan.version} · {channel}** "
                f"from [{upstream} / {origin.tag}]({origin.release_url}). "
                "The upstream release notes above are preserved; these are "
                f"{self.settings.package}'s binaries, not upstream's wheel assets.\n\n"
                f"- Upstream source commit: [`{origin.commit}`](https://github.com/{upstream}/commit/{origin.commit})\n"
                f"- Build recipe: [`{plan.recipe_commit}`](https://github.com/{plan.repository}/commit/{plan.recipe_commit})\n"
                f"- Build run: {plan.run_url or 'local'}\n"
                f"- See `{MANIFEST_ASSET}` for pinned submodules, source/code hashes and wheel checksums; "
                f"`{CHECKSUM_ASSET}` covers the downloadable assets.\n\n"
                "Only distribution/build metadata was adapted. The Python bindings are unchanged.\n\n"
                + marker
            )
        )

    def state(self, plan: Plan, channel: Channel, finished: bool) -> Provenance:
        """Build the provenance document for one channel of `plan`."""
        return Provenance(
            version=plan.version,
            channel=channel,
            tag=channel.release_tag(plan.version),
            upstream_repository=self.settings.upstream,
            upstream_commit=plan.origin.commit,
            upstream_tag=plan.origin.tag,
            upstream_release_id=plan.origin.release_id,
            recipe_commit=plan.recipe_commit,
            python_versions=plan.python_versions,
            complete=finished,
            origin=plan.origin,
            matrix=plan.matrix,
        )

    # -- Helpers -------------------------------------------------------------

    @staticmethod
    def _require_release(plan: Plan) -> None:
        """Refuse to stage or publish a rehearsal."""
        if plan.test_only:
            raise ReleaseError("Test-build plans cannot be staged or published as releases")

    @staticmethod
    def _link_or_copy(source: Path, destination: Path) -> None:
        """Hard-link when possible, so staging does not duplicate gigabytes."""
        try:
            os.link(source, destination)
        except OSError:
            shutil.copyfile(source, destination)

    def _check_gate(self, plan: Plan, manifest: SourceManifest, gate: PublicationGate) -> None:
        """Check that the gate covers exactly the requested matrix."""
        if (
            gate.plan.to_mapping() != plan.to_mapping()
            or gate.source_archive_sha256 != manifest.source_archive_sha256
        ):
            raise ReleaseError("Global validation gate does not match the prepared build")
        if set(gate.channels) != {channel.name for channel in plan.missing_channels}:
            raise ReleaseError("Global validation gate is missing requested channels")
        for name, records in gate.channels.items():
            channel = Channel(name)
            expected = {
                item
                for item in expected_asset_names(
                    self.settings.package, plan.version, channel, plan.python_versions
                )
                if item.endswith(".whl")
            }
            if set(records) != expected:
                raise ReleaseError(f"Global validation gate has an incomplete matrix: {name}")

    @staticmethod
    def _check_uploaded(release: Release | None, files: list[Path]) -> None:
        """Verify that every asset really arrived on GitHub."""
        if release is None:
            raise ReleaseError("Release disappeared during upload")
        by_name = {asset.name: asset for asset in release.assets}
        if len(by_name) != len(release.assets) or set(by_name) != {path.name for path in files}:
            raise ReleaseError("Unexpected or duplicate release assets; release remains a draft")
        for path in files:
            asset = by_name[path.name]
            if not asset.is_uploaded or asset.size != path.stat().st_size:
                raise ReleaseError(f"Incomplete upload: {path.name}; release remains a draft")
            if asset.digest and asset.digest != "sha256:" + sha256(path):
                raise ReleaseError(f"GitHub asset digest mismatch: {path.name}")


__all__ = [
    "ArtifactSpec",
    "ReleaseError",
    "ReleasePlanner",
    "ReleasePublisher",
    "ReceiptCollector",
    "TestBuildPlanner",
    "UpstreamSelector",
    "artifact_specs",
    "fresh_origin",
]
