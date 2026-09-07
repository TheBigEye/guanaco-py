import pytest
from check_upstream import make_plan, select_release
from helpers import SHA_A, SHA_B, FakeGitHub, owned, plan, upstream
from publish_release import release_body, state_for
from release_common import (
    CHANNELS,
    CONFIG,
    UPSTREAM,
    GitHub,
    complete,
    provenance,
    version_key,
)


def test_numeric_version_order_not_lexicographic_or_release_creation_order():
    releases = [upstream("v0.3.9-cu124-win-20990101"), upstream("v0.3.10-cu124-win-20260101")]
    assert select_release(releases)["tag_name"].startswith("v0.3.10-")


def test_groups_backend_tags_and_chooses_latest_published_snapshot():
    a = upstream("v0.3.49-Metal-macos-20260831", published_at="2026-08-31T09:00:00Z")
    b = upstream()
    assert select_release([b, a]) == b


def test_ignores_drafts_prereleases_and_non_version_tags():
    good = upstream()
    assert (
        select_release(
            [
                upstream("v9.0.0", draft=True),
                upstream("v8.0.0", prerelease=True),
                upstream("nightly"),
                good,
            ]
        )
        == good
    )


def test_explicit_version_can_backfill_without_promoting_latest():
    old = upstream("v0.3.48-Metal-macos-20260821", id=48)
    p = make_plan(FakeGitHub([upstream(), old]), "TheBigEye/guanaco-py", "0.3.48")
    assert p["version"] == "0.3.48" and not p["promote_latest"]


@pytest.mark.parametrize("value", ["1.2", "1.2.3;echo bad", "01.2.3", "1.2.3\n4", "v0.3.49"])
def test_rejects_invalid_requested_versions(value):
    with pytest.raises(ValueError):
        version_key(value)


def test_missing_requested_version_fails():
    with pytest.raises(ValueError, match="No published stable"):
        select_release([upstream()], "0.3.47")


def test_first_run_plans_all_channels_and_pins_release_not_main():
    api = FakeGitHub()
    p = make_plan(api, "TheBigEye/guanaco-py")
    assert p["missing_channels"] == CHANNELS
    assert api.commit_calls == [(UPSTREAM, upstream()["tag_name"])]
    assert p["upstream"]["zip_url"].endswith(SHA_A)


def test_completed_version_does_not_rebuild_for_new_backend_tag():
    newer = upstream("v0.3.49-cu131-win-20260906", id=50, published_at="2026-09-06T10:00:00Z")
    api = FakeGitHub([upstream(), newer], [owned(c) for c in CHANNELS])
    p = make_plan(api, "TheBigEye/guanaco-py")
    assert p["missing_channels"] == [] and not p["build"]
    assert not api.commit_calls
    assert p["upstream"]["tag"] == upstream()["tag_name"]


def test_partial_family_resumes_same_commit_and_only_missing_channels():
    later = upstream("v0.3.49-cu131-win-20260906", id=50, published_at="2026-09-06T10:00:00Z")
    api = FakeGitHub([upstream(), later], [owned("cpu"), owned("avx2", draft=True, finished=False)])
    p = make_plan(api, "TheBigEye/guanaco-py")
    assert p["upstream"]["commit"] == SHA_A
    assert p["upstream"]["release_id"] == upstream()["id"]
    assert p["missing_channels"] == CHANNELS[1:]


def test_conflicting_source_commits_fail_closed():
    other = plan()
    other["upstream"]["commit"] = SHA_B
    with pytest.raises(ValueError, match="Mixed upstream commits"):
        make_plan(
            FakeGitHub(existing=[owned("cpu"), owned("avx2", base=other)]), "TheBigEye/guanaco-py"
        )


def test_does_not_overwrite_legacy_or_manual_release():
    legacy = {"tag_name": "v0.3.49", "body": "My manually uploaded release"}
    with pytest.raises(ValueError, match="without Guanaco provenance"):
        make_plan(FakeGitHub(existing=[legacy]), "TheBigEye/guanaco-py")


def test_old_personal_1x_releases_do_not_hide_new_03_upstream():
    legacy = {"tag_name": "v1.0.1", "body": "Old personal-fork release"}
    assert make_plan(FakeGitHub(existing=[legacy]), "TheBigEye/guanaco-py")["build"]


@pytest.mark.parametrize(
    "change", ["draft", "prerelease", "missing_asset", "empty_asset", "pending"]
)
def test_requires_complete_published_asset_inventory(change):
    r = owned("cpu")
    if change in ("draft", "prerelease"):
        r[change] = True
    elif change == "missing_asset":
        r["assets"].pop()
    elif change == "empty_asset":
        r["assets"][0]["size"] = 0
    else:
        r["body"] = release_body(plan(), "cpu", False)
    assert not complete(r, provenance(r), CONFIG["python_versions"])


def test_old_release_remains_complete_if_future_matrix_adds_python():
    r = owned("cpu")
    assert complete(r, provenance(r), CONFIG["python_versions"] + ["3.15"])


def test_notes_preserved_verbatim_and_provenance_is_parseable():
    p = plan()
    body = release_body(p, "cpu", True)
    assert body.startswith(p["upstream"]["body"])
    assert provenance({"tag_name": "v0.3.49", "body": body}) == state_for(p, "cpu", True)


def test_api_paginates_all_releases(monkeypatch):
    api = GitHub(token="")
    calls = []

    def request(path):
        calls.append(path)
        return [upstream()] * (100 if len(calls) == 1 else 2)

    monkeypatch.setattr(api, "request", request)
    assert len(api.releases(UPSTREAM)) == 102
    assert calls[-1].endswith("page=2")


def test_readonly_api_cannot_publish():
    with pytest.raises(ValueError, match="read-only"):
        GitHub(token="").request("/repos/test/repo/releases", method="POST", data={})


def test_truncated_git_tree_is_not_treated_as_complete(monkeypatch):
    api = GitHub(token="")
    monkeypatch.setattr(api, "request", lambda _: {"truncated": True, "tree": []})
    with pytest.raises(ValueError, match="Truncated"):
        api.tree(UPSTREAM, SHA_A)


@pytest.mark.parametrize("tag", ["v9.0.0-rc1", "v9.0.0-preview", "v9.0.0-dev.2", "v9.0.0-nightly"])
def test_prerelease_tag_is_not_mirrored_even_if_mislabelled_stable(tag):
    assert select_release([upstream(tag), upstream()])["tag_name"] == upstream()["tag_name"]


def test_draft_lookup_falls_back_when_tag_endpoint_returns_404(monkeypatch):
    api = GitHub(token="")
    draft = owned("cpu", draft=True, finished=False)
    monkeypatch.setattr(api, "request", lambda *a, **kw: None)
    monkeypatch.setattr(api, "releases", lambda _: [draft])
    assert api.release("TheBigEye/guanaco-py", "v0.3.49") == draft
