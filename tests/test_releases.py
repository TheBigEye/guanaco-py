"""Upstream discovery, release planning, receipts and conservative publishing."""

from __future__ import annotations

import json
import shutil
from dataclasses import replace

import pytest
from helpers import (
    SHA_A,
    SHA_B,
    FakeGitHub,
    PublishingAPI,
    make_plan,
    owned_release,
    prepared_build,
    receipts_for,
    upstream_payload,
    write_wheel,
)

from guanaco.models import (
    PROVENANCE_SEPARATOR,
    Channel,
    Platform,
    Provenance,
    SourceManifest,
    is_complete,
    read_json,
    write_json,
)
from guanaco.releases import (
    ReceiptCollector,
    ReleaseError,
    ReleasePlanner,
    ReleasePublisher,
    TestBuildPlanner,
    UpstreamSelector,
    artifact_specs,
    fresh_origin,
)
from guanaco.toolchain import cpu_artifact

# ---------------------------------------------------------------------------
# Artifacts
# ---------------------------------------------------------------------------


class TestArtifactSpecs:
    def test_cpu_artifacts_cover_the_whole_matrix(self, settings):
        plan = make_plan(settings, ["cpu"])
        specs = artifact_specs(plan, settings.package)
        assert list(specs) == [
            "guanaco-py-cpu-linux-x64",
            "guanaco-py-cpu-windows-x64",
        ]
        assert specs["guanaco-py-cpu-linux-x64"].python_versions == settings.python_versions

    def test_rehearsals_are_prefixed(self, settings):
        plan = make_plan(settings, ["cpu"], test_only=True)
        assert all(name.startswith("test-") for name in artifact_specs(plan, settings.package))

    def test_cuda_artifacts_are_one_per_python_version(self, settings):
        plan = make_plan(settings, ["cu124"])
        specs = artifact_specs(plan, settings.package)
        assert len(specs) == 2 * len(settings.python_versions)
        assert specs["guanaco-py-cuda-linux-x64-cu124-py3.13"].python_versions == ("3.13",)

    def test_a_plan_without_channels_expects_nothing(self, settings):
        assert artifact_specs(make_plan(settings, []), settings.package) == {}


# ---------------------------------------------------------------------------
# Upstream selection
# ---------------------------------------------------------------------------


class TestUpstreamSelector:
    def test_picks_the_newest_stable_release(self, settings):
        client = FakeGitHub(
            [
                upstream_payload("v0.3.49-cu124-win-20260831"),
                upstream_payload("v0.3.50-cu124-win-20260901", id=50),
            ]
        )
        assert (
            str(
                UpstreamSelector(settings, client)
                .select(client.releases(settings.upstream))
                .version
            )
            == "0.3.50"
        )

    def test_honours_an_explicit_request(self, settings):
        client = FakeGitHub(
            [
                upstream_payload("v0.3.49-cu124-win-20260831"),
                upstream_payload("v0.3.50-cu124-win-20260901", id=50),
            ]
        )
        selector = UpstreamSelector(settings, client)
        assert (
            str(selector.select(client.releases(settings.upstream), "0.3.49").version) == "0.3.49"
        )

    def test_rejects_an_invalid_request(self, settings):
        client = FakeGitHub([upstream_payload()])
        with pytest.raises(ValueError):
            UpstreamSelector(settings, client).select(client.releases(settings.upstream), "nope")

    def test_ignores_drafts_and_previews(self, settings):
        client = FakeGitHub(
            [
                upstream_payload("v0.3.49-cu124-win-20260831", draft=True),
                upstream_payload("v0.3.50-cu124-win-20260901", prerelease=True, id=50),
            ]
        )
        with pytest.raises(ReleaseError, match="No published stable upstream release"):
            UpstreamSelector(settings, client).select(client.releases(settings.upstream))

    def test_ignores_releases_without_a_stable_tag(self, settings):
        client = FakeGitHub([upstream_payload("nightly")])
        with pytest.raises(ReleaseError):
            UpstreamSelector(settings, client).select(client.releases(settings.upstream))

    def test_ignores_unpublished_releases(self, settings):
        client = FakeGitHub([upstream_payload(published_at=None)])
        with pytest.raises(ReleaseError):
            UpstreamSelector(settings, client).select(client.releases(settings.upstream))

    def test_latest_version_matches_the_selection(self, settings):
        client = FakeGitHub([upstream_payload()])
        selector = UpstreamSelector(settings, client)
        releases = client.releases(settings.upstream)
        assert selector.latest_version(releases) == selector.select(releases).version


class TestFreshOrigin:
    def test_pins_the_upstream_commit(self, settings):
        client = FakeGitHub([upstream_payload()])
        release = client.releases(settings.upstream)[0]
        origin = fresh_origin(client, release, settings.upstream)
        assert origin.commit == SHA_A
        assert origin.zip_url.endswith(SHA_A)
        assert origin.repository == settings.upstream


# ---------------------------------------------------------------------------
# Release planning
# ---------------------------------------------------------------------------


class TestReleasePlanner:
    def test_plans_everything_when_nothing_exists(self, settings):
        plan = ReleasePlanner(settings, FakeGitHub()).plan()
        assert plan.version == "0.3.49"
        assert plan.missing_channels == tuple(settings.channels)
        assert plan.promote_latest

    def test_does_not_promote_an_older_version_to_latest(self, settings):
        client = FakeGitHub(
            [
                upstream_payload("v0.3.49-cu124-win-20260831"),
                upstream_payload("v0.3.50-cu124-win-20260901", id=50),
            ]
        )
        plan = ReleasePlanner(settings, client).plan("0.3.49")
        assert not plan.promote_latest

    def test_skips_channels_that_are_already_complete(self, settings):
        plan = make_plan(settings, ["cpu"])
        client = FakeGitHub(
            [upstream_payload()],
            [owned_release(settings, plan, "cpu")],
        )
        planned = ReleasePlanner(settings, client).plan()
        assert planned.missing_channels == tuple(settings.channels)[1:]

    def test_nothing_to_build_when_the_family_is_complete(self, settings):
        plan = make_plan(settings, [])
        client = FakeGitHub(
            [upstream_payload()],
            [owned_release(settings, plan, channel.name) for channel in settings.channels],
        )
        planned = ReleasePlanner(settings, client).plan()
        assert planned.missing_channels == ()
        assert not planned.needs_build

    def test_reuses_the_frozen_snapshot_of_a_partial_family(self, settings):
        plan = make_plan(settings, ["cpu"])
        client = FakeGitHub([upstream_payload()], [owned_release(settings, plan, "cpu")])
        planned = ReleasePlanner(settings, client).plan()
        assert planned.origin.commit == planned.origin.commit
        assert planned.origin.commit == client.commit(settings.upstream, "v0.3.49")

    def test_refuses_a_release_without_provenance(self, settings):
        client = FakeGitHub([upstream_payload()], [upstream_payload("v0.3.49")])
        with pytest.raises(ReleaseError, match="without Guanaco provenance"):
            ReleasePlanner(settings, client).plan()

    def test_refuses_an_incomplete_published_release(self, settings):
        plan = make_plan(settings, ["cpu"])
        payload = owned_release(settings, plan, "cpu")
        payload["assets"] = payload["assets"][:2]
        client = FakeGitHub([upstream_payload()], [payload])
        with pytest.raises(ReleaseError, match="incomplete"):
            ReleasePlanner(settings, client).plan()

    def test_an_incomplete_draft_is_resumed(self, settings):
        plan = make_plan(settings, ["cpu"])
        payload = owned_release(settings, plan, "cpu", finished=False, draft=True)
        payload["assets"] = payload["assets"][:2]
        client = FakeGitHub([upstream_payload()], [payload])
        planned = ReleasePlanner(settings, client).plan()
        assert Channel("cpu") in planned.missing_channels

    def test_refuses_duplicate_releases_for_one_channel(self, settings):
        plan = make_plan(settings, ["cpu"])
        client = FakeGitHub(
            [upstream_payload()],
            [owned_release(settings, plan, "cpu"), owned_release(settings, plan, "cpu")],
        )
        with pytest.raises(ReleaseError, match="Duplicate releases"):
            ReleasePlanner(settings, client).plan()

    def test_recovers_a_legacy_marker(self, settings, capsys):
        plan = make_plan(settings, ["cpu"])
        payload = owned_release(settings, plan, "cpu")
        document = {
            "channel": "cpu",
            "complete": True,
            "python_versions": ["3.12"],
            "recipe_commit": plan.recipe_commit,
            "tag": "v0.3.49",
            "upstream_commit": SHA_A,
            "upstream_release_id": plan.origin.release_id,
            "upstream_repository": settings.upstream,
            "upstream_tag": plan.origin.tag,
            "version": "0.3.49",
        }
        marker = "<!-- guanaco-upstream-build-v1\n" + json.dumps(document) + "\n-->"
        payload["body"] = (
            "Legacy upstream notes.\r\nSecond line.\r\n" + PROVENANCE_SEPARATOR + marker
        )
        client = FakeGitHub([upstream_payload()], [payload])
        planned = ReleasePlanner(settings, client).plan()
        assert "Legacy upstream notes." in planned.origin.body
        assert planned.origin.tag == plan.origin.tag
        assert "Legacy marker" in capsys.readouterr().out

    def test_refuses_mixed_upstream_commits(self, settings):
        from dataclasses import replace

        plan = make_plan(settings, ["cpu"])
        other = replace(plan, origin=replace(plan.origin, commit="c" * 40))
        first = owned_release(settings, plan, "cpu")
        second = owned_release(settings, other, "avx2")
        client = FakeGitHub([upstream_payload()], [first, second])
        with pytest.raises(ReleaseError, match="Mixed upstream commits"):
            ReleasePlanner(settings, client).plan()


# ---------------------------------------------------------------------------
# Rehearsals
# ---------------------------------------------------------------------------


class RehearsalPlannerTests:
    def test_defaults_to_cpu_on_both_platforms(self, settings):
        plan = TestBuildPlanner(settings, FakeGitHub()).plan()
        assert plan.test_only
        assert plan.missing_channels == (Channel("cpu"),)
        assert plan.python_versions == ("3.13",)
        assert plan.build_platforms() == (Platform.LINUX, Platform.WINDOWS)

    def test_selects_an_explicit_matrix(self, settings):
        plan = TestBuildPlanner(settings, FakeGitHub()).plan(
            cpu=False,
            cuda=True,
            cuda_channels="cu128,cu124",
            python_versions="3.12,3.13",
            systems="linux",
        )
        assert plan.missing_channels == (Channel("cu124"), Channel("cu128"))
        assert plan.python_versions == ("3.12", "3.13")
        assert plan.build_platforms() == (Platform.LINUX,)

    def test_all_means_everything_in_matrix_order(self, settings):
        plan = TestBuildPlanner(settings, FakeGitHub()).plan(
            cpu=True, avx2=True, cuda=True, cuda_channels="all", python_versions="all"
        )
        assert plan.missing_channels == tuple(settings.channels)
        assert plan.python_versions == settings.python_versions

    def test_rejects_an_empty_channel_selection(self, settings):
        with pytest.raises(ReleaseError, match="Select at least one"):
            TestBuildPlanner(settings, FakeGitHub()).plan(cpu=False, avx2=False, cuda=False)

    def test_rejects_an_unknown_system(self, settings):
        with pytest.raises(ReleaseError, match="Systems must be"):
            TestBuildPlanner(settings, FakeGitHub()).plan(systems="macos")

    def test_rejects_a_non_boolean_switch(self, settings):
        with pytest.raises(ReleaseError, match="must be booleans"):
            TestBuildPlanner(settings, FakeGitHub()).plan(cpu="yes")

    def test_rejects_unknown_selections(self, settings):
        planner = TestBuildPlanner(settings, FakeGitHub())
        with pytest.raises(ReleaseError, match="Invalid Python versions"):
            planner.plan(python_versions="3.8")
        with pytest.raises(ReleaseError, match="Invalid CUDA channels"):
            planner.plan(cuda=True, cuda_channels="cu999")

    def test_rejects_duplicate_selections(self, settings):
        with pytest.raises(ReleaseError, match="Duplicate"):
            TestBuildPlanner(settings, FakeGitHub()).plan(python_versions="3.13,3.13")

    def test_rejects_an_invalid_upstream_version(self, settings):
        with pytest.raises(ValueError):
            TestBuildPlanner(settings, FakeGitHub()).plan(version="nope")


# ---------------------------------------------------------------------------
# Receipts
# ---------------------------------------------------------------------------


class TestReceiptCollector:
    def test_a_complete_set_produces_a_gate(self, tmp_path, settings):
        plan, prepared, artifacts = prepared_build(tmp_path, settings)
        receipts = tmp_path / "receipts"
        manifest = receipts_for(settings, plan, prepared, artifacts, receipts)
        gate = ReceiptCollector(settings).collect(plan, manifest, receipts)
        assert set(gate.channels) == {"cpu", "avx2"}
        assert len(gate.channels["cpu"]) == 2 * len(plan.python_versions)

    def test_a_missing_receipt_is_refused(self, tmp_path, settings):
        plan, prepared, artifacts = prepared_build(tmp_path, settings)
        receipts = tmp_path / "receipts"
        manifest = receipts_for(settings, plan, prepared, artifacts, receipts)
        name = cpu_artifact(settings.package, "", Channel("cpu"), Platform.LINUX)
        (receipts / f"{name}.json").unlink()
        with pytest.raises(ReleaseError, match="Missing validation receipt"):
            ReceiptCollector(settings).collect(plan, manifest, receipts)

    def test_a_foreign_receipt_is_refused(self, tmp_path, settings):
        plan, prepared, artifacts = prepared_build(tmp_path, settings)
        receipts = tmp_path / "receipts"
        manifest = receipts_for(settings, plan, prepared, artifacts, receipts)
        name = cpu_artifact(settings.package, "", Channel("cpu"), Platform.LINUX)
        path = receipts / f"{name}.json"
        document = read_json(path)
        document["recipe_commit"] = "c" * 40
        write_json(path, document)
        with pytest.raises(ReleaseError, match="Receipt identity mismatch"):
            ReceiptCollector(settings).collect(plan, manifest, receipts)

    def test_an_incomplete_matrix_is_refused(self, tmp_path, settings):
        plan, prepared, artifacts = prepared_build(tmp_path, settings)
        receipts = tmp_path / "receipts"
        manifest = receipts_for(settings, plan, prepared, artifacts, receipts)
        path = (
            receipts / f"{cpu_artifact(settings.package, '', Channel('cpu'), Platform.LINUX)}.json"
        )
        document = read_json(path)
        document["wheels"] = document["wheels"][:-1]
        write_json(path, document)
        with pytest.raises(ReleaseError, match="Incomplete wheel matrix"):
            ReceiptCollector(settings).collect(plan, manifest, receipts)

    def test_a_bad_checksum_is_refused(self, tmp_path, settings):
        plan, prepared, artifacts = prepared_build(tmp_path, settings)
        receipts = tmp_path / "receipts"
        manifest = receipts_for(settings, plan, prepared, artifacts, receipts)
        path = (
            receipts / f"{cpu_artifact(settings.package, '', Channel('cpu'), Platform.LINUX)}.json"
        )
        document = read_json(path)
        document["wheels"][0]["sha256"] = "zz"
        write_json(path, document)
        with pytest.raises(ReleaseError, match="Invalid wheel checksum/size"):
            ReceiptCollector(settings).collect(plan, manifest, receipts)

    def test_a_manifest_from_another_plan_is_refused(self, tmp_path, settings):
        plan, prepared, artifacts = prepared_build(tmp_path, settings)
        receipts = tmp_path / "receipts"
        manifest = receipts_for(settings, plan, prepared, artifacts, receipts)
        other = make_plan(settings, ["cpu", "avx2"])
        with pytest.raises(ReleaseError, match="does not match the release plan"):
            ReceiptCollector.check_plan(other, manifest)


# ---------------------------------------------------------------------------
# Publishing
# ---------------------------------------------------------------------------


@pytest.fixture
def publishing(tmp_path, settings, monkeypatch):
    """A staged, gated, ready-to-publish release with a writable API."""
    monkeypatch.setenv("GITHUB_REPOSITORY", settings.repository)
    plan, prepared, artifacts = prepared_build(tmp_path, settings)
    plan = replace_commit(plan, SHA_B)
    write_json(tmp_path / "plan.json", plan.to_mapping())
    monkeypatch.chdir(tmp_path)
    receipts = tmp_path / "receipts"
    manifest = receipts_for(settings, plan, prepared, artifacts, receipts)
    gate = ReceiptCollector(settings).collect(plan, manifest, receipts)
    api = PublishingAPI()
    publisher = ReleasePublisher(settings, api)
    return plan, prepared, artifacts, gate, api, publisher


def replace_commit(plan, commit: str):
    """Return `plan` with an immutable recipe commit."""
    from dataclasses import replace

    return replace(plan, recipe_commit=commit)


class TestReleasePublisher:
    def test_the_release_body_carries_the_provenance(self, settings):
        plan = make_plan(settings)
        body = ReleasePublisher(settings, None).release_body(plan, Channel("cpu"), True)
        assert "Original upstream notes" in body
        assert "guanaco-py 0.3.49" in body
        public, marker = body.split("<!-- ")[0], body[body.index("<!-- ") :]
        assert SHA_A in public
        assert json.loads(marker[marker.index("{") : marker.rindex("}") + 1])["complete"] is True

    def test_preflight_accepts_empty_destinations(self, publishing):
        plan, _, _, _, _, publisher = publishing
        assert set(publisher.preflight(plan).values()) == {None}

    def test_preflight_refuses_a_foreign_release(self, publishing, settings):
        plan, _, _, _, api, publisher = publishing
        other = make_plan(settings, ["cpu"])
        api.items[1] = {**owned_release(settings, other, "cpu"), "id": 1, "tag_name": "v0.3.49"}
        api.items[1]["body"] = "foreign body without a marker"
        with pytest.raises(ReleaseError, match="Refusing to replace unrelated release"):
            publisher.preflight(plan)

    def test_preflight_refuses_an_incomplete_published_release(self, publishing, settings):
        plan, _, _, _, api, publisher = publishing
        payload = owned_release(settings, plan, "cpu")
        payload["assets"] = payload["assets"][:1]
        api.items[1] = {**payload, "id": 1}
        with pytest.raises(ReleaseError, match="incomplete"):
            publisher.preflight(plan)

    def test_preflight_refuses_to_move_a_git_tag(self, publishing, settings):
        plan, _, _, _, api, publisher = publishing
        api.tag_commits["v0.3.49"] = "0" * 40
        with pytest.raises(ReleaseError, match="tags are never moved"):
            publisher.preflight(plan)

    def test_a_rehearsal_cannot_be_staged(self, settings, tmp_path):
        plan = make_plan(settings, test_only=True)
        with pytest.raises(ReleaseError, match="cannot be staged"):
            ReleasePublisher(settings, None).stage(plan, tmp_path, tmp_path, tmp_path / "out")

    def test_stage_builds_a_complete_folder(self, publishing, tmp_path, settings):
        plan, prepared, artifacts, _, _, publisher = publishing
        manifest, folders = publisher.stage(plan, prepared, artifacts, tmp_path / "staged")
        assert set(folders) == {Channel("cpu"), Channel("avx2")}
        cpu = folders[Channel("cpu")]
        names = {path.name for path in cpu.iterdir()}
        assert "guanaco-build.json" in names and "SHA256SUMS" in names
        assert "guanaco-source-0.3.49.tar.gz" in names
        assert any(name.endswith("manylinux_2_34_x86_64.whl") for name in names)
        assert any(name.endswith("win_amd64.whl") for name in names)
        assert len(names) == 2 + 2 * len(plan.python_versions) + 2

    def test_stage_can_narrow_to_one_channel(self, publishing, tmp_path):
        plan, prepared, artifacts, _, _, publisher = publishing
        only_avx2 = tmp_path / "avx2-only"
        for source in artifacts.glob("*avx2*"):
            shutil.copytree(source, only_avx2 / source.name)
        _, folders = publisher.stage(
            plan, prepared, only_avx2, tmp_path / "staged", channel=Channel("avx2")
        )
        assert set(folders) == {Channel("avx2")}

    def test_stage_refuses_an_unrequested_channel(self, publishing, tmp_path):
        plan, prepared, artifacts, _, _, publisher = publishing
        with pytest.raises(ReleaseError, match="Cannot stage a channel absent"):
            publisher.stage(
                plan, prepared, artifacts, tmp_path / "staged", channel=Channel("cu124")
            )

    def test_stage_rejects_an_unexpected_artifact(self, publishing, tmp_path, settings):
        plan, prepared, artifacts, _, _, publisher = publishing
        manifest = SourceManifest.from_mapping(read_json(prepared / "build-manifest.json"))
        write_wheel(
            artifacts / "unexpected-artifact",
            manifest,
            {"llama_cpp/__init__.py": b'__version__ = "0.3.49"\n'},
            "cpu",
            "linux",
            "cp313",
        )
        with pytest.raises(ReleaseError, match="Unexpected or unrequested"):
            publisher.stage(plan, prepared, artifacts, tmp_path / "staged")

    def test_publish_creates_verified_releases(self, publishing, tmp_path):
        plan, prepared, artifacts, gate, api, publisher = publishing
        _, folders = publisher.stage(plan, prepared, artifacts, tmp_path / "staged")
        publisher.publish(plan, folders, uploader=api.upload_assets)
        for channel in plan.missing_channels:
            tag = channel.release_tag(plan.version)
            release = api.release(plan.repository, tag)
            assert release is not None and not release.draft
            assert release.assets

    def test_publish_marks_the_cpu_release_complete(self, publishing, tmp_path):
        plan, prepared, artifacts, gate, api, publisher = publishing
        _, folders = publisher.stage(plan, prepared, artifacts, tmp_path / "staged")
        publisher.publish(plan, folders, uploader=api.upload_assets)
        release = api.release(plan.repository, "v0.3.49")
        state = Provenance.from_body(release.body, "v0.3.49")
        assert state.complete is True
        assert is_complete(release, state, "guanaco-py", plan.python_versions)

    def test_publish_keeps_a_complete_release(self, publishing, tmp_path, settings):
        plan, prepared, artifacts, gate, api, publisher = publishing
        api.items[1] = {**owned_release(settings, plan, "cpu"), "id": 1}
        api.tag_commits["v0.3.49"] = SHA_B
        _, folders = publisher.stage(plan, prepared, artifacts, tmp_path / "staged")
        publisher.publish(plan, folders, uploader=api.upload_assets)
        created = [data["tag_name"] for method, _, data in api.calls if method == "POST"]
        assert "v0.3.49" not in created
        assert "v0.3.49-avx2" in created

    def test_publish_requires_an_immutable_recipe_commit(self, publishing, tmp_path):
        plan, prepared, artifacts, gate, api, publisher = publishing
        _, folders = publisher.stage(plan, prepared, artifacts, tmp_path / "staged")
        unstaged = replace_commit(plan, "local-working-tree")
        with pytest.raises(ReleaseError, match="immutable automation commit"):
            publisher.publish(unstaged, folders, uploader=api.upload_assets)

    def test_publish_requires_the_expected_repository(self, publishing, tmp_path, monkeypatch):
        plan, prepared, artifacts, gate, api, publisher = publishing
        _, folders = publisher.stage(plan, prepared, artifacts, tmp_path / "staged")
        monkeypatch.setenv("GITHUB_REPOSITORY", "someone/else")
        with pytest.raises(ReleaseError, match="does not match GITHUB_REPOSITORY"):
            publisher.publish(plan, folders, uploader=api.upload_assets)

    def test_stage_cross_checks_the_gate(self, publishing, tmp_path):
        plan, prepared, artifacts, gate, _, publisher = publishing
        _, folders = publisher.stage(plan, prepared, artifacts, tmp_path / "staged", gate=gate)
        assert (folders[Channel("cpu")] / "guanaco-build.json").is_file()

    def test_stage_refuses_a_wheel_that_drifted_from_the_gate(self, publishing, tmp_path):
        plan, prepared, artifacts, gate, _, publisher = publishing
        records = dict(gate.channels["cpu"])
        name = sorted(records)[0]
        records[name] = {"size": 1, "sha256": "0" * 64}
        foreign = replace(gate, channels={**gate.channels, "cpu": records})
        with pytest.raises(ReleaseError, match="differs from the globally validated receipt"):
            publisher.stage(plan, prepared, artifacts, tmp_path / "staged", gate=foreign)

    def test_stage_refuses_a_gate_without_every_channel(self, publishing, tmp_path):
        plan, prepared, artifacts, gate, _, publisher = publishing
        foreign = replace(gate, channels={"cpu": gate.channels["cpu"]})
        with pytest.raises(ReleaseError, match="missing requested channels"):
            publisher.stage(plan, prepared, artifacts, tmp_path / "staged", gate=foreign)

    def test_stage_refuses_a_gate_with_an_incomplete_matrix(self, publishing, tmp_path):
        plan, prepared, artifacts, gate, _, publisher = publishing
        records = dict(gate.channels["cpu"])
        del records[sorted(records)[0]]
        foreign = replace(gate, channels={**gate.channels, "cpu": records})
        with pytest.raises(ReleaseError, match="incomplete matrix"):
            publisher.stage(plan, prepared, artifacts, tmp_path / "staged", gate=foreign)

    def test_stage_refuses_a_gate_from_another_plan(self, publishing, tmp_path, settings):
        plan, prepared, artifacts, gate, _, publisher = publishing
        foreign = replace(gate, plan=make_plan(settings, ["cpu"]))
        with pytest.raises(ReleaseError, match="does not match the prepared build"):
            publisher.stage(plan, prepared, artifacts, tmp_path / "staged", gate=foreign)

    def test_publish_leaves_a_failed_upload_as_a_draft(self, publishing, tmp_path):
        plan, prepared, artifacts, gate, api, publisher = publishing

        def broken(plan, tag, files):
            """Upload nothing at all."""
            del plan, tag, files

        _, folders = publisher.stage(plan, prepared, artifacts, tmp_path / "staged")
        with pytest.raises(ReleaseError, match="Unexpected or duplicate release assets"):
            publisher.publish(plan, folders, uploader=broken)
        assert api.release(plan.repository, "v0.3.49").draft is True
