"""The PEP 503 wheel index: safety, filtering and rendering."""

from __future__ import annotations

import json

import pytest
from helpers import owned_release, upstream_payload

from guanaco.catalog import IndexError, IndexGenerator, check_ownership, generated_site
from guanaco.models import Release


@pytest.fixture
def generator(settings):
    """An index generator wired to the synthetic repository."""
    return IndexGenerator(settings)


def published(settings, plan, channels=("cpu",)):
    """Return releases that look fully published for `channels`."""
    return [Release.from_mapping(owned_release(settings, plan, name)) for name in channels]


class TestOwnership:
    def test_an_empty_directory_is_fine(self, tmp_path):
        check_ownership(tmp_path / "site")

    def test_a_missing_directory_is_fine(self, tmp_path):
        check_ownership(tmp_path / "absent")

    def test_a_file_is_refused(self, tmp_path):
        path = tmp_path / "site"
        path.write_text("x", encoding="utf-8")
        with pytest.raises(IndexError, match="must be a directory"):
            check_ownership(path)

    def test_a_symlink_is_refused(self, tmp_path):
        target = tmp_path / "target"
        target.mkdir()
        (tmp_path / "site").symlink_to(target)
        with pytest.raises(IndexError, match="must be a directory"):
            check_ownership(tmp_path / "site")

    def test_an_unowned_directory_is_refused(self, tmp_path):
        site = tmp_path / "site"
        site.mkdir()
        (site / "index.html").write_text("hi", encoding="utf-8")
        with pytest.raises(IndexError, match="unowned index directory"):
            check_ownership(site)

    def test_a_foreign_generator_is_refused(self, tmp_path):
        site = tmp_path / "site"
        site.mkdir()
        (site / ".guanaco-wheel-index.json").write_text(
            json.dumps({"generator": "someone-else", "files": ["index.html"]}), encoding="utf-8"
        )
        with pytest.raises(IndexError, match="ownership record"):
            check_ownership(site)

    def test_an_illegal_page_name_is_refused(self, tmp_path):
        site = tmp_path / "site"
        site.mkdir()
        (site / ".guanaco-wheel-index.json").write_text(
            json.dumps({"generator": "guanaco-wheel-index-v1", "files": ["../../etc/passwd"]}),
            encoding="utf-8",
        )
        with pytest.raises(IndexError, match="ownership record"):
            check_ownership(site)

    def test_an_unlisted_file_is_refused(self, tmp_path):
        site = tmp_path / "site"
        site.mkdir()
        (site / ".guanaco-wheel-index.json").write_text(
            json.dumps({"generator": "guanaco-wheel-index-v1", "files": ["index.html"]}),
            encoding="utf-8",
        )
        (site / "index.html").write_text("hi", encoding="utf-8")
        (site / "secret.txt").write_text("nope", encoding="utf-8")
        with pytest.raises(IndexError, match="unowned files or symlinks"):
            check_ownership(site)

    def test_a_known_site_is_accepted(self, tmp_path):
        site = tmp_path / "site"
        (site / "whl" / "cpu").mkdir(parents=True)
        (site / ".guanaco-wheel-index.json").write_text(
            json.dumps({"generator": "guanaco-wheel-index-v1", "files": ["whl/cpu/index.html"]}),
            encoding="utf-8",
        )
        (site / "whl" / "cpu" / "index.html").write_text("hi", encoding="utf-8")
        check_ownership(site)


class TestGeneratedSite:
    def test_the_site_appears_only_on_success(self, tmp_path):
        output = tmp_path / "site"
        with generated_site(output) as workspace:
            (workspace / "index.html").write_text("ok", encoding="utf-8")
            assert not output.exists()
        assert (output / "index.html").read_text(encoding="utf-8") == "ok"

    def test_a_failure_leaves_the_previous_site(self, tmp_path):
        output = tmp_path / "site"
        with generated_site(output) as workspace:
            (workspace / "index.html").write_text("first", encoding="utf-8")
        with pytest.raises(RuntimeError):
            with generated_site(output):
                raise RuntimeError("boom")
        assert (output / "index.html").read_text(encoding="utf-8") == "first"

    def test_regeneration_replaces_stale_pages(self, tmp_path):
        output = tmp_path / "site"
        with generated_site(output) as workspace:
            stale = workspace / "whl" / "cpu"
            stale.mkdir(parents=True)
            (stale / "index.html").write_text("old", encoding="utf-8")
        with generated_site(output) as workspace:
            (workspace / "index.html").write_text("new", encoding="utf-8")
        assert not (output / "whl" / "cpu" / "index.html").exists()
        assert (output / "index.html").is_file()

    def test_the_marker_lists_every_generated_file(self, tmp_path):
        output = tmp_path / "site"
        with generated_site(output) as workspace:
            (workspace / "index.html").write_text("ok", encoding="utf-8")
        state = json.loads((output / ".guanaco-wheel-index.json").read_text(encoding="utf-8"))
        assert state["generator"] == "guanaco-wheel-index-v1"
        assert "index.html" in state["files"]


class TestGeneration:
    def test_indexes_a_complete_release(self, tmp_path, settings, generator):
        from helpers import make_plan

        plan = make_plan(settings)
        output = generator.generate(published(settings, plan), tmp_path / "site")
        page = (output / "whl" / "cpu" / settings.package / "index.html").read_text(
            encoding="utf-8"
        )
        assert "guanaco_py-0.3.49-cp313-cp313-manylinux_2_34_x86_64.whl" in page
        assert "sha256=" in page

    def test_creates_the_landing_page_and_the_channel_root(self, tmp_path, settings, generator):
        from helpers import make_plan

        plan = make_plan(settings)
        output = generator.generate(published(settings, plan), tmp_path / "site")
        assert (output / "index.html").is_file()
        assert (output / "whl" / "index.html").is_file()
        assert (output / "whl" / "cpu" / "index.html").is_file()

    def test_skips_drafts_and_legacy_releases(self, tmp_path, settings, generator):
        payload = owned_release(settings, _plan(settings), "cpu")
        payload["draft"] = True
        draft = Release.from_mapping(payload)
        legacy = Release.from_mapping(upstream_payload("v0.3.49"))
        output = generator.generate([draft, legacy], tmp_path / "site")
        page = (output / "whl" / "cpu" / settings.package / "index.html").read_text(
            encoding="utf-8"
        )
        assert "No wheels published yet" in page

    def test_skips_an_incomplete_release(self, tmp_path, settings, generator):
        from helpers import make_plan

        plan = make_plan(settings)
        payload = owned_release(settings, plan, "cpu")
        payload["assets"] = payload["assets"][:3]
        output = generator.generate([Release.from_mapping(payload)], tmp_path / "site")
        page = (output / "whl" / "cpu" / settings.package / "index.html").read_text(
            encoding="utf-8"
        )
        assert "No wheels published yet" in page

    def test_skips_assets_from_another_repository(self, tmp_path, settings, generator):
        from helpers import make_plan

        plan = make_plan(settings)
        payload = owned_release(settings, plan, "cpu")
        for asset in payload["assets"]:
            asset["browser_download_url"] = "https://evil.example/" + asset["name"]
        output = generator.generate([Release.from_mapping(payload)], tmp_path / "site")
        page = (output / "whl" / "cpu" / settings.package / "index.html").read_text(
            encoding="utf-8"
        )
        assert "evil.example" not in page

    def test_every_channel_of_a_family_is_listed(self, tmp_path, settings, generator):
        from helpers import make_plan

        plan = make_plan(settings)
        releases = published(settings, plan, ("cpu", "avx2", "cu124"))
        output = generator.generate(releases, tmp_path / "site")
        root = (output / "whl" / "index.html").read_text(encoding="utf-8")
        assert "CPU (portable)" in root
        assert "CPU (AVX2)" in root
        assert "CUDA 12.4" in root

    def test_wheels_are_sorted_newest_first(self, tmp_path, settings, generator):
        from helpers import make_plan

        plan = make_plan(settings)
        releases = published(settings, plan)
        output = generator.generate(releases, tmp_path / "site")
        page = (output / "whl" / "cpu" / settings.package / "index.html").read_text(
            encoding="utf-8"
        )
        for python in ("cp39", "cp310", "cp313", "cp314"):
            assert python in page
        assert page.index("cp39") < page.index("cp313")

    def test_generating_twice_is_safe(self, tmp_path, settings, generator):
        from helpers import make_plan

        plan = make_plan(settings)
        releases = published(settings, plan)
        generator.generate(releases, tmp_path / "site")
        generator.generate(releases, tmp_path / "site")
        assert (tmp_path / "site" / "index.html").is_file()

    def test_channel_labels_fall_back_gracefully(self, generator):
        assert generator._label("cpu") == "CPU (portable)"
        assert generator._label("avx2") == "CPU (AVX2)"
        assert generator._label("cu124") == "CUDA 12.4"
        assert generator._label("other") == "other"

    def test_channels_are_ordered_cpu_first(self, generator):
        names = ["cu128", "avx2", "cu124", "cpu"]
        assert sorted(names, key=generator.order) == ["cpu", "avx2", "cu124", "cu128"]

    def test_the_version_key_reads_pep_427_names(self, generator):
        assert generator._version_key("guanaco_py-0.3.49-cp313-cp313-linux_x86_64.whl") == (
            0,
            3,
            49,
        )
        assert generator._version_key("unrelated.txt") == (0,)


def _plan(settings):
    """A minimal plan, imported lazily to keep the fixtures readable."""
    from helpers import make_plan

    return make_plan(settings)
