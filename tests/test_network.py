"""No network access: scripted HTTPS responses exercise retries and credential boundaries."""

import hashlib
import http.client
import io
import json
import urllib.error
import urllib.request

import download_utils as net
import pytest
import release_common as common
from helpers import SHA_A, SHA_B


class Response(io.BytesIO):
    def __init__(self, data=b"ok", headers=None):
        super().__init__(data)
        self.headers = headers or {}


def responses(monkeypatch, module, sequence):
    pending = iter(sequence)
    calls = []

    def open_request(request, **kwargs):
        calls.append(request)
        result = next(pending)
        if isinstance(result, Exception):
            raise result
        return result

    monkeypatch.setattr(module, "open_url", open_request)
    monkeypatch.setattr(module, "pause_before_retry", lambda *_: None)
    return calls


def http_error(status, headers=None):
    return urllib.error.HTTPError(
        "https://api.github.com/test", status, "test failure", headers or {}, io.BytesIO()
    )


@pytest.mark.parametrize(
    "url",
    [
        "http://example.com/a",
        "ftp://example.com/a",
        "https://user:secret@example.com/a",
        "/relative",
    ],
)
def test_download_requires_credential_free_https(tmp_path, url):
    with pytest.raises(ValueError, match="HTTPS"):
        net.download(url, tmp_path / "asset")


def test_successful_download_is_checked_and_sends_no_token(tmp_path, monkeypatch):
    monkeypatch.setenv("GH_TOKEN", "must-not-leak-to-codeload")
    calls = responses(monkeypatch, net, [Response(b"wheel", {"Content-Length": "5"})])
    destination = tmp_path / "nested" / "wheel"
    digest = hashlib.sha256(b"wheel").hexdigest()
    assert (
        net.download(
            "https://codeload.github.com/test/repo/zip/sha", destination, expected_sha256=digest
        )
        == digest
    )
    assert destination.read_bytes() == b"wheel"
    assert not calls[0].has_header("Authorization")
    assert not list(destination.parent.glob(".download-*"))


def test_failed_checksum_preserves_previous_file_and_removes_partial_download(
    tmp_path, monkeypatch
):
    path = tmp_path / "asset"
    path.write_bytes(b"previous")
    responses(monkeypatch, net, [Response(b"corrupt")])
    with pytest.raises(ValueError, match="Checksum"):
        net.download("https://example.com/asset", path, expected_sha256="a" * 64)
    assert path.read_bytes() == b"previous"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["asset"]


@pytest.mark.parametrize("headers", [{"Content-Length": "8"}, {}])
def test_download_limits_declared_and_actual_size(tmp_path, monkeypatch, headers):
    responses(monkeypatch, net, [Response(b"too-long", headers)])
    with pytest.raises(ValueError, match="allowed size"):
        net.download("https://example.com/asset", tmp_path / "asset", max_bytes=4)
    assert not list(tmp_path.iterdir())


def test_truncated_response_retries_from_scratch(tmp_path, monkeypatch):
    calls = responses(
        monkeypatch,
        net,
        [Response(b"a", {"Content-Length": "2"}), Response(b"ab", {"Content-Length": "2"})],
    )
    net.download("https://example.com/asset", tmp_path / "asset")
    assert len(calls) == 2
    assert (tmp_path / "asset").read_bytes() == b"ab"
    assert len(list(tmp_path.iterdir())) == 1


def test_transient_failure_retries_but_404_does_not(tmp_path, monkeypatch):
    calls = responses(monkeypatch, net, [http_error(503), Response(b"success")])
    net.download("https://example.com/asset", tmp_path / "asset")
    assert len(calls) == 2
    calls = responses(monkeypatch, net, [http_error(404)])
    with pytest.raises(urllib.error.HTTPError):
        net.download("https://example.com/missing", tmp_path / "missing")
    assert len(calls) == 1


def test_retry_budget_is_finite(tmp_path, monkeypatch):
    calls = responses(monkeypatch, net, [TimeoutError()] * net.ATTEMPTS)
    with pytest.raises(TimeoutError):
        net.download("https://example.com/asset", tmp_path / "asset")
    assert len(calls) == net.ATTEMPTS
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize(
    "error,expected",
    [
        (http_error(500), True),
        (http_error(403), False),
        (http_error(403, {"Retry-After": "10"}), True),
        (http.client.IncompleteRead(b"x"), True),
        (ValueError(), False),
    ],
)
def test_retry_policy(error, expected):
    assert net.retryable(error) is expected


def test_retry_after_is_bounded_and_malformed_headers_are_tolerated(monkeypatch):
    sleeps = []
    monkeypatch.setattr(net.time, "sleep", sleeps.append)
    net.pause_before_retry(0, http_error(429, {"Retry-After": "90"}))
    net.pause_before_retry(2, http_error(429, {"Retry-After": "later"}))
    assert sleeps == [30, 4]


@pytest.mark.parametrize(
    "url",
    [
        "http://api.github.com/next",
        "https://other.example/next",
        "https://user:pass@api.github.com/next",
    ],
)
def test_authenticated_redirect_cannot_change_origin_or_downgrade(url):
    request = urllib.request.Request(
        "https://api.github.com/repos/old/repo", headers={"Authorization": "Bearer secret"}
    )
    with pytest.raises(ValueError, match="redirect"):
        net.SafeRedirect().redirect_request(request, None, 302, "Found", {}, url)


def test_same_host_redirect_keeps_api_auth_and_public_asset_redirect_has_none():
    handler = net.SafeRedirect()
    request = urllib.request.Request(
        "https://api.github.com/repos/old/repo", headers={"Authorization": "Bearer secret"}
    )
    redirected = handler.redirect_request(
        request, None, 301, "Moved", {}, "https://api.github.com/repos/new/repo"
    )
    assert redirected.get_header("Authorization") == "Bearer secret"
    request = urllib.request.Request("https://github.com/test/repo/releases/download/asset")
    redirected = handler.redirect_request(
        request, None, 302, "Found", {}, "https://release-assets.githubusercontent.com/asset"
    )
    assert not redirected.has_header("Authorization")


def test_opener_installs_the_safe_redirect_handler(monkeypatch):
    class Opener:
        def open(self, request, timeout):
            return request.full_url, timeout

    handlers = []
    monkeypatch.setattr(
        urllib.request, "build_opener", lambda handler: (handlers.append(handler), Opener())[1]
    )
    assert net.open_url(urllib.request.Request("https://example.com"), timeout=7) == (
        "https://example.com",
        7,
    )
    assert isinstance(handlers[0], net.SafeRedirect)


def test_api_get_retries_and_writes_have_explicit_permission(monkeypatch):
    calls = responses(monkeypatch, common, [http_error(503), Response(b'{"ok": true}')])
    assert common.GitHub(token="token").request("/repos/test/repo") == {"ok": True}
    assert len(calls) == 2 and calls[-1].get_header("Authorization") == "Bearer token"
    calls = responses(monkeypatch, common, [http_error(503)])
    with pytest.raises(RuntimeError, match="HTTP 503"):
        common.GitHub(token="", writable=True).request(
            "/repos/test/repo/releases", method="POST", data={"draft": True}
        )
    assert len(calls) == 1
    assert json.loads(calls[0].data) == {"draft": True}


def test_api_missing_ok_empty_response_and_oversized_response(monkeypatch):
    responses(monkeypatch, common, [http_error(404)])
    assert common.GitHub(token="").request("/repos/test/repo", missing_ok=True) is None
    responses(monkeypatch, common, [Response(b"")])
    assert common.GitHub(token="").request("/repos/test/repo") is None
    monkeypatch.setattr(common, "MAX_API_BYTES", 2)
    responses(monkeypatch, common, [Response(b"too long")])
    with pytest.raises(ValueError, match="unexpectedly large"):
        common.GitHub(token="").request("/repos/test/repo")


def test_api_rejects_header_and_path_injection():
    with pytest.raises(ValueError, match="token"):
        common.GitHub(token="x\r\ny")
    with pytest.raises(ValueError, match="paths"):
        common.GitHub(token="").request("https://example.com/steal")


def test_commit_resolution_validates_sha_and_quotes_ref(monkeypatch):
    api = common.GitHub(token="")
    paths = []
    monkeypatch.setattr(api, "request", lambda path: (paths.append(path), {"sha": SHA_A})[1])
    assert api.commit("test/repo", "tag/with/slashes") == SHA_A
    assert paths[-1].endswith("tag%2Fwith%2Fslashes")
    monkeypatch.setattr(api, "request", lambda path: {"sha": "main"})
    with pytest.raises(ValueError, match="SHA"):
        api.commit("test/repo", "tag")


@pytest.mark.parametrize(
    "kind", ["missing", "lightweight", "annotated", "tree", "cycle", "invalid_sha"]
)
def test_tag_resolution_handles_annotated_tags_and_rejects_bad_targets(monkeypatch, kind):
    api = common.GitHub(token="")
    count = 0

    def request(path, **kwargs):
        nonlocal count
        count += 1
        if kind == "missing":
            return None
        if kind == "invalid_sha":
            return {"object": {"type": "commit", "sha": "bad"}}
        target = (
            "tree"
            if kind == "tree"
            else ("tag" if kind == "cycle" or kind == "annotated" and count == 1 else "commit")
        )
        return {"object": {"type": target, "sha": SHA_A if count == 1 else SHA_B}}

    monkeypatch.setattr(api, "request", request)
    if kind in ("tree", "cycle", "invalid_sha"):
        with pytest.raises(ValueError):
            api.tag_commit("test/repo", "v1.2.3")
    else:
        assert (
            api.tag_commit("test/repo", "v1.2.3")
            == {"missing": None, "lightweight": SHA_A, "annotated": SHA_B}[kind]
        )
