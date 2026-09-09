"""The JSON documents the pipeline exchanges: plans, manifests, receipts, provenance."""

from __future__ import annotations

import json

import pytest
from helpers import SHA_A, SHA_B, make_plan, manifest_for, owned_release, upstream_payload

from guanaco.models import (
    Channel,
    DocumentError,
    PatchRecord,
    Plan,
    Platform,
    Provenance,
    Receipt,
    Release,
    ReleaseAsset,
    Snapshot,
    SourceManifest,
    UpstreamOrigin,
    expected_asset_names,
    is_complete,
    read_json,
    source_asset,
    source_notes,
    wheel_prefix,
    write_json,
    write_outputs,
)

# ---------------------------------------------------------------------------
# Documents
# ---------------------------------------------------------------------------


class TestPlan:
    def test_round_trips_through_json(self, settings, tmp_path):
        plan = make_plan(settings)
        path = tmp_path / "plan.json"
        write_json(path, plan.to_mapping())
        restored = Plan.from_mapping(read_json(path))
        assert restored.to_mapping() == plan.to_mapping()
        assert restored.version == "0.3.49"
        assert restored.origin.commit == SHA_A

    def test_needs_build_only_when_something_is_missing(self, settings):
        assert make_plan(settings, ["cpu"]).needs_build
        assert not make_plan(settings, []).needs_build

    def test_release_plans_target_both_platforms(self, settings):
        assert make_plan(settings).build_platforms() == (Platform.LINUX, Platform.WINDOWS)

    def test_test_plans_can_narrow_platforms(self, settings):
        plan = make_plan(settings, ["cpu"], test_only=True, platforms=(Platform.LINUX,))
        assert plan.build_platforms() == (Platform.LINUX,)
        assert plan.to_mapping()["platforms"] == ["linux"]

    def test_release_plans_omit_the_platform_key(self, settings):
        assert "platforms" not in make_plan(settings).to_mapping()

    def test_rejects_a_wrong_schema(self, settings):
        document = make_plan(settings).to_mapping()
        document["schema"] = 99
        with pytest.raises(DocumentError):
            Plan.from_mapping(document)

    def test_rejects_a_missing_field(self, settings):
        document = make_plan(settings).to_mapping()
        del document["version"]
        with pytest.raises(DocumentError):
            Plan.from_mapping(document)

    def test_rejects_a_non_boolean_test_flag(self, settings):
        document = make_plan(settings).to_mapping()
        document["test_only"] = "yes"
        with pytest.raises(DocumentError):
            Plan.from_mapping(document)

    def test_delegates_to_the_matrix(self, settings):
        plan = make_plan(settings)
        assert plan.package == settings.package
        assert plan.python_versions == settings.python_versions
        assert plan.cuda == settings.cuda


class TestSourceManifest:
    def test_keeps_every_plan_field(self, settings):
        plan = make_plan(settings)
        manifest, _ = manifest_for(plan)
        document = manifest.to_mapping()
        for key, value in plan.to_mapping().items():
            assert document[key] == value

    def test_round_trips(self, settings):
        plan = make_plan(settings)
        manifest, _ = manifest_for(plan)
        assert (
            SourceManifest.from_mapping(manifest.to_mapping()).to_mapping() == manifest.to_mapping()
        )

    def test_rejects_a_missing_field(self, settings):
        manifest, _ = manifest_for(make_plan(settings))
        document = manifest.to_mapping()
        del document["native_commit"]
        with pytest.raises(DocumentError):
            SourceManifest.from_mapping(document)


class TestReceipt:
    def test_round_trips(self, settings):
        plan = make_plan(settings)
        receipt = Receipt(
            version=plan.version,
            channel=Channel("cpu"),
            platform=Platform.LINUX,
            recipe_commit=SHA_B,
            source_archive_sha256="2" * 64,
            wheels=(),
        )
        restored = Receipt.from_mapping(receipt.to_mapping())
        assert restored.channel == receipt.channel
        assert restored.platform == receipt.platform

    def test_rejects_a_wrong_schema(self):
        with pytest.raises(DocumentError):
            Receipt.from_mapping({"schema": 7, "version": "0.3.49"})


class TestSnapshotsAndPatches:
    def test_snapshot_round_trips(self):
        snapshot = Snapshot(".", "owner/repo", SHA_A, "3" * 64, "https://example.test")
        assert Snapshot.from_mapping(snapshot.to_mapping()) == snapshot

    def test_patch_record_round_trips(self):
        record = PatchRecord("x.patch", "4" * 64, ("llama_cpp/llama.py",), {}, {})
        assert PatchRecord.from_mapping(record.to_mapping()).files == ("llama_cpp/llama.py",)


# ---------------------------------------------------------------------------
# Releases
# ---------------------------------------------------------------------------


class TestRelease:
    def test_reads_a_github_payload(self):
        release = Release.from_mapping(upstream_payload())
        assert release.identifier == 49
        assert str(release.version) == "0.3.49"
        assert not release.draft

    def test_assets_report_whether_they_arrived(self):
        complete = ReleaseAsset.from_mapping({"name": "w", "state": "uploaded", "size": 10})
        pending = ReleaseAsset.from_mapping({"name": "w", "state": "uploaded", "size": 0})
        assert complete.is_uploaded and not pending.is_uploaded

    def test_unpublished_releases_are_never_complete(self, settings):
        plan = make_plan(settings)
        payload = owned_release(settings, plan, "cpu")
        payload["draft"] = True
        assert not Release.from_mapping(payload).is_complete(settings.package, plan.python_versions)


class TestExpectedAssets:
    def test_cpu_channel_ships_the_source_archive(self, settings):
        names = expected_asset_names(settings.package, "0.3.49", Channel("cpu"), ("3.13",))
        assert "guanaco-source-0.3.49.tar.gz" in names
        assert "packaging.patch" in names
        assert guanaco_wheel("cp313", "manylinux_2_34_x86_64") in names
        assert guanaco_wheel("cp313", "win_amd64") in names

    def test_cuda_channels_skip_the_source_archive(self, settings):
        names = expected_asset_names(settings.package, "0.3.49", Channel("cu124"), ("3.13",))
        assert "guanaco-source-0.3.49.tar.gz" not in names
        assert guanaco_wheel("cp313", "linux_x86_64") in names

    def test_prefix_follows_the_package_name(self):
        assert wheel_prefix("guanaco-py") == "guanaco_py"
        assert source_asset("guanaco-py", "0.3.49") == "guanaco-source-0.3.49.tar.gz"


def guanaco_wheel(python: str, platform: str) -> str:
    """Build an expected wheel filename."""
    return f"guanaco_py-0.3.49-{python}-{python}-{platform}.whl"


class TestIsComplete:
    def test_a_foreign_release_is_never_complete(self, settings):
        plan = make_plan(settings)
        release = Release.from_mapping(owned_release(settings, plan, "cpu"))
        assert not is_complete(release, None, settings.package, plan.python_versions)

    def test_duplicate_assets_make_a_release_incomplete(self, settings):
        plan = make_plan(settings)
        payload = owned_release(settings, plan, "cpu")
        payload["assets"] = payload["assets"][:1] * 2
        release = Release.from_mapping(payload)
        assert not release.is_complete(settings.package, plan.python_versions)


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------


class TestProvenance:
    def test_marker_survives_a_round_trip(self, settings):
        plan = make_plan(settings)
        payload = owned_release(settings, plan, "cpu")
        release = Release.from_mapping(payload)
        state = release.provenance(settings.upstream, settings.package)
        assert state is not None
        assert state.complete is True
        assert state.channel.name == "cpu"
        assert reparse(state.render()) == state.to_document(hash_notes=True)

    def test_a_foreign_body_has_no_provenance(self):
        assert Provenance.from_body("plain notes", "v0.3.49") is None

    def test_an_ambiguous_marker_is_rejected(self, settings):
        plan = make_plan(settings)
        body = owned_release(settings, plan, "cpu")["body"]
        doubled = body + "\n" + body[body.index("<!-- ") :]
        with pytest.raises(DocumentError):
            Provenance.from_body(doubled, "v0.3.49")

    def test_a_marker_belonging_to_another_upstream_is_rejected(self, settings):
        plan = make_plan(settings)
        body = owned_release(settings, plan, "cpu")["body"]
        with pytest.raises(DocumentError):
            Provenance.from_body(body, "v0.3.49", "someone/else")

    def test_a_bad_note_hash_is_rejected(self, settings, monkeypatch):
        plan = make_plan(settings)
        body = owned_release(settings, plan, "cpu")["body"]
        import guanaco.models as models

        monkeypatch.setattr(models, "source_notes", lambda text: "tampered notes")
        with pytest.raises(DocumentError):
            Provenance.from_body(body, "v0.3.49")

    def test_notes_are_split_at_the_separator(self, settings):
        plan = make_plan(settings)
        body = owned_release(settings, plan, "cpu")["body"]
        assert "Original upstream notes" in source_notes(body)

    def test_a_body_without_a_separator_cannot_be_split(self):
        with pytest.raises(DocumentError):
            source_notes("no separator here")

    def test_rejects_a_bad_recipe_commit(self, settings):
        plan = make_plan(settings)
        document = Provenance.from_body(
            owned_release(settings, plan, "cpu")["body"], "v0.3.49"
        ).to_document(hash_notes=True)
        document["recipe_commit"] = "nope"
        with pytest.raises(DocumentError):
            Provenance.from_document(document, "v0.3.49", "", "", settings.package)

    def test_rejects_a_bad_release_id(self, settings):
        plan = make_plan(settings)
        document = Provenance.from_body(
            owned_release(settings, plan, "cpu")["body"], "v0.3.49"
        ).to_document(hash_notes=True)
        document["upstream_release_id"] = 0
        with pytest.raises(DocumentError):
            Provenance.from_document(document, "v0.3.49", "", "", settings.package)

    def test_rejects_a_non_boolean_complete(self, settings):
        plan = make_plan(settings)
        document = Provenance.from_body(
            owned_release(settings, plan, "cpu")["body"], "v0.3.49"
        ).to_document(hash_notes=True)
        document["complete"] = "yes"
        with pytest.raises(DocumentError):
            Provenance.from_document(document, "v0.3.49", "", "", settings.package)

    def test_escapes_a_closing_comment_sequence(self, settings):
        plan = make_plan(settings)
        state = Provenance.from_body(owned_release(settings, plan, "cpu")["body"], "v0.3.49")
        marker = state.render()
        assert marker.count("<!-- ") == 1
        assert marker.rstrip().endswith("-->")


def reparse(marker: str) -> dict:
    """Decode a rendered marker the way the release body would carry it."""

    from guanaco.models import PROVENANCE_PATTERN

    return json.loads(PROVENANCE_PATTERN.findall(marker)[0].replace("--\\u003e", "-->"))


class TestUpstreamOrigin:
    def test_validates_its_identifiers(self):
        origin = UpstreamOrigin(
            repository="owner/repo",
            release_id=1,
            tag="v0.3.49",
            commit=SHA_A,
            release_url="https://example.test",
            release_name="v0.3.49",
            published_at=None,
            body="notes",
            zip_url="https://example.test/zip",
        )
        assert origin.with_body_hash()["body_sha256"]
        assert "body" not in origin.with_body_hash()

    def test_rejects_a_short_commit(self):
        with pytest.raises(DocumentError):
            UpstreamOrigin(
                repository="owner/repo",
                release_id=1,
                tag="v0.3.49",
                commit="abc123",
                release_url="u",
                release_name="n",
                published_at=None,
                body="",
                zip_url="z",
            ).validate()


class TestJsonHelpers:
    def test_write_then_read(self, tmp_path):
        path = tmp_path / "nested" / "file.json"
        write_json(path, {"b": 1, "a": 2})
        assert read_json(path) == {"a": 2, "b": 1}

    def test_reading_a_non_object_is_an_error(self, tmp_path):
        path = tmp_path / "list.json"
        path.write_text("[1, 2]", encoding="utf-8")
        with pytest.raises(DocumentError):
            read_json(path)

    def test_reading_a_missing_file_is_an_error(self, tmp_path):
        with pytest.raises(DocumentError):
            read_json(tmp_path / "absent.json")

    def test_outputs_are_validated_before_anything_is_written(self, tmp_path):
        path = tmp_path / "output.txt"
        with pytest.raises(DocumentError):
            write_outputs(path, **{"bad-key": "value"})
        with pytest.raises(DocumentError):
            write_outputs(path, ok="two\nlines")
        assert not path.exists()

    def test_outputs_are_appended_and_formatted(self, tmp_path):
        path = tmp_path / "output.txt"
        path.write_text("", encoding="utf-8")
        write_outputs(path, flag=True, items=[1, 2], platform=Platform.LINUX)
        assert path.read_text(encoding="utf-8") == "flag=true\nitems=[1,2]\nplatform=linux\n"
