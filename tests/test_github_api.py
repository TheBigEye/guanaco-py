"""The GitHub REST client: pagination, retries, read-only safety and writes."""

from __future__ import annotations

import io
import json
import urllib.error

import pytest
from helpers import SHA_A, SHA_B, upstream_payload

from guanaco.github_api import GithubClient, GithubError
from guanaco.models import Release
from guanaco.settings import ConfigurationError


class FakeResponse(io.BytesIO):
    """A response-like object carrying a JSON body."""

    def __init__(self, payload: object) -> None:
        """Encode `payload` as the response body."""
        super().__init__(json.dumps(payload).encode() if payload is not None else b"")


class Recorder:
    """Records every request and answers with a scripted response queue."""

    def __init__(self, *responses) -> None:
        """Store the responses to return, in order."""
        self.responses = list(responses)
        self.calls: list[tuple[str, str, dict | None]] = []

    def __call__(self, request, timeout=60):
        """Record the call and return the next scripted response."""
        del timeout
        data = json.loads(request.data) if request.data else None
        self.calls.append((request.get_method(), request.full_url, data))
        answer = self.responses.pop(0) if self.responses else None
        if isinstance(answer, Exception):
            raise answer
        return FakeResponse(answer)

    @property
    def paths(self) -> list[str]:
        """The paths that were requested."""
        return [url.split("api.github.com", 1)[1] for _, url, _ in self.calls]


@pytest.fixture
def recorder(monkeypatch):
    """Install a recorder in place of the network opener."""

    def install(*responses):
        """Return a recorder answering with these responses, in order."""
        fake = Recorder(*responses)
        monkeypatch.setattr("guanaco.github_api.open_url", fake)
        return fake

    return install


class TestSafety:
    def test_the_token_comes_from_the_environment(self, monkeypatch):
        monkeypatch.setenv("GH_TOKEN", "secret")
        assert GithubClient().token == "secret"
        monkeypatch.delenv("GH_TOKEN")
        monkeypatch.setenv("GITHUB_TOKEN", "fallback")
        assert GithubClient().token == "fallback"

    def test_an_explicit_token_wins(self, monkeypatch):
        monkeypatch.setenv("GH_TOKEN", "secret")
        assert GithubClient(token="explicit").token == "explicit"

    def test_a_token_with_a_newline_is_refused(self):
        with pytest.raises(GithubError, match="Invalid GitHub token"):
            GithubClient(token="bad\ntoken")

    def test_only_repository_paths_are_allowed(self):
        client = GithubClient(token="")
        with pytest.raises(GithubError, match="Only GitHub repository API paths"):
            client.request("/users/octocat")

    def test_a_path_with_a_newline_is_refused(self):
        with pytest.raises(GithubError):
            GithubClient(token="").request("/repos/o/r\nx-injected")

    def test_writes_need_a_writable_client(self):
        client = GithubClient(token="")
        with pytest.raises(GithubError, match="read-only"):
            client.request("/repos/o/r/releases", method="POST", data={})

    def test_a_bad_repository_is_refused(self):
        with pytest.raises(ConfigurationError):
            GithubClient(token="").releases("not-a-repository")


class TestRequests:
    def test_a_get_returns_the_decoded_body(self, recorder):
        fake = recorder({"ok": True})
        assert GithubClient(token="").request("/repos/o/r") == {"ok": True}
        assert fake.paths == ["/repos/o/r"]

    def test_an_empty_body_returns_none(self, recorder):
        recorder(None)
        assert GithubClient(token="").request("/repos/o/r") is None

    def test_a_missing_resource_returns_none_on_get(self, recorder):
        recorder(urllib.error.HTTPError("u", 404, "missing", {}, None))
        assert GithubClient(token="").request("/repos/o/r") is None

    def test_a_permanent_error_is_not_retried(self, recorder):
        fake = recorder(urllib.error.HTTPError("u", 403, "denied", {}, None))
        with pytest.raises(GithubError, match="HTTP 403"):
            GithubClient(token="").request("/repos/o/r")
        assert len(fake.calls) == 1

    def test_a_transient_error_is_retried(self, recorder, monkeypatch):
        monkeypatch.setattr("guanaco.github_api.pause_before_retry", lambda attempt, error: None)
        fake = recorder(
            urllib.error.HTTPError("u", 500, "boom", {}, None),
            {"recovered": True},
        )
        assert GithubClient(token="").request("/repos/o/r") == {"recovered": True}
        assert len(fake.calls) == 2

    def test_an_oversized_response_is_refused(self, recorder, monkeypatch):
        monkeypatch.setattr("guanaco.github_api.MAX_API_BYTES", 10)
        recorder({"a" * 100: 1})
        with pytest.raises(GithubError, match="unexpectedly large"):
            GithubClient(token="").request("/repos/o/r")

    def test_the_authorization_header_is_added(self, recorder):
        fake = recorder({})
        GithubClient(token="secret").request("/repos/o/r")
        assert fake.calls[0][0] == "GET"


class TestReading:
    def test_releases_paginate_until_a_short_page(self, recorder):
        recorder([upstream_payload()] * 100, [upstream_payload(id=2)])
        releases = GithubClient(token="").releases("o/r")
        assert len(releases) == 101
        assert all(isinstance(item, Release) for item in releases)

    def test_a_malformed_release_list_is_refused(self, recorder):
        recorder({"unexpected": True})
        with pytest.raises(GithubError, match="must be a list of objects"):
            GithubClient(token="").releases("o/r")

    def test_a_release_is_found_by_tag(self, recorder):
        recorder(upstream_payload())
        assert GithubClient(token="").release("o/r", "v0.3.49-cu124-win-20260831").identifier == 49

    def test_a_missing_release_falls_back_to_the_list(self, recorder):
        recorder(None, [upstream_payload()])
        found = GithubClient(token="").release("o/r", "v0.3.49-cu124-win-20260831")
        assert found is not None and found.identifier == 49

    def test_an_ambiguous_tag_is_refused(self, recorder):
        recorder(None, [upstream_payload(), upstream_payload(id=50)])
        with pytest.raises(GithubError, match="Multiple draft releases"):
            GithubClient(token="").release("o/r", "v0.3.49-cu124-win-20260831")

    def test_a_missing_tag_returns_none(self, recorder):
        recorder(None, [])
        assert GithubClient(token="").release("o/r", "v9.9.9") is None

    def test_a_commit_is_resolved_and_validated(self, recorder):
        recorder({"sha": SHA_A})
        assert GithubClient(token="").commit("o/r", "v0.3.49") == SHA_A

    def test_an_invalid_commit_is_refused(self, recorder):
        recorder({"sha": "short"})
        with pytest.raises(GithubError, match="invalid commit SHA"):
            GithubClient(token="").commit("o/r", "v0.3.49")

    def test_a_missing_tag_returns_no_commit(self, recorder):
        recorder(None)
        assert GithubClient(token="").tag_commit("o/r", "v0.3.49") is None

    def test_a_lightweight_tag_returns_its_commit(self, recorder):
        recorder({"object": {"type": "commit", "sha": SHA_A}})
        assert GithubClient(token="").tag_commit("o/r", "v0.3.49") == SHA_A

    def test_an_annotated_tag_is_followed(self, recorder):
        recorder(
            {"object": {"type": "tag", "sha": SHA_B}},
            {"object": {"type": "commit", "sha": SHA_A}},
        )
        assert GithubClient(token="").tag_commit("o/r", "v0.3.49") == SHA_A

    def test_a_tag_pointing_elsewhere_is_refused(self, recorder):
        recorder({"object": {"type": "blob", "sha": SHA_A}})
        with pytest.raises(GithubError, match="does not point to a commit"):
            GithubClient(token="").tag_commit("o/r", "v0.3.49")

    def test_an_excessively_nested_tag_is_refused(self, recorder):
        recorder(*[{"object": {"type": "tag", "sha": SHA_B}} for _ in range(10)])
        with pytest.raises(GithubError, match="Excessively nested"):
            GithubClient(token="").tag_commit("o/r", "v0.3.49")

    def test_a_tree_is_returned(self, recorder):
        recorder({"truncated": False, "tree": [{"path": "a", "mode": "160000"}]})
        assert GithubClient(token="").tree("o/r", SHA_A)[0]["path"] == "a"

    def test_a_truncated_tree_is_refused(self, recorder):
        recorder({"truncated": True, "tree": []})
        with pytest.raises(GithubError, match="Truncated Git tree"):
            GithubClient(token="").tree("o/r", SHA_A)

    def test_a_tree_needs_a_full_commit(self):
        with pytest.raises(GithubError, match="immutable commit SHA"):
            GithubClient(token="").tree("o/r", "v0.3.49")


class TestWriting:
    def test_creating_a_release_sends_the_payload(self, recorder):
        fake = recorder(upstream_payload())
        client = GithubClient(token="", writable=True)
        release = client.create_release(
            "o/r",
            tag="v0.3.49",
            commit=SHA_A,
            name="v0.3.49",
            body="notes",
            draft=True,
            latest=False,
        )
        assert release.identifier == 49
        method, _, data = fake.calls[0]
        assert method == "POST"
        assert data["tag_name"] == "v0.3.49"
        assert data["make_latest"] == "false"
        assert data["draft"] is True

    def test_updating_a_release_only_sends_the_given_fields(self, recorder):
        fake = recorder(upstream_payload())
        client = GithubClient(token="", writable=True)
        client.update_release("o/r", 49, body="new notes", draft=False, latest=True)
        _, url, data = fake.calls[0]
        assert url.endswith("/releases/49")
        assert data == {
            "prerelease": False,
            "body": "new notes",
            "draft": False,
            "make_latest": "true",
        }

    def test_a_bare_update_still_clears_the_preview_flag(self, recorder):
        fake = recorder(upstream_payload())
        GithubClient(token="", writable=True).update_release("o/r", 49)
        assert fake.calls[0][2] == {"prerelease": False}
