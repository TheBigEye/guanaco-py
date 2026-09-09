"""A small, read-mostly client for the GitHub REST API.

The client is deliberately read-only unless it is constructed with
``writable=True``: the discovery and validation jobs in CI are pure readers, and
only the publication job is allowed to create or modify a release.

Mutating requests are never retried automatically -- retrying a release upload
is a decision for a human, not for a backoff loop.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.parse
import urllib.request

from .models import COMMIT_PATTERN, Release
from .settings import repository_name
from .transfer import DEFAULT_ATTEMPTS, is_retryable, open_url, pause_before_retry

MAX_API_BYTES = 32 * 1024**2
API_VERSION = "2022-11-28"


class GithubError(RuntimeError):
    """Raised when the API refuses a request or returns something unexpected."""


class GithubClient:
    """Reads releases, tags, commits and trees; writes only when asked.

    Attributes:
        writable: Whether mutating requests (POST/PATCH) are permitted.
    """

    def __init__(self, token: str | None = None, *, writable: bool = False) -> None:
        """Create a client.

        Args:
            token: GitHub token. Defaults to ``GH_TOKEN`` then ``GITHUB_TOKEN``.
            writable: Allow POST/PATCH. Most callers leave this off.
        """
        self.token = (
            token if token is not None else (os.getenv("GH_TOKEN") or os.getenv("GITHUB_TOKEN", ""))
        )
        if any(char in self.token for char in "\r\n"):
            raise GithubError("Invalid GitHub token format")
        self.writable = writable

    # -- Low level ----------------------------------------------------------

    def request(self, path: str, *, method: str = "GET", data: dict | None = None):
        """Call the API and return the decoded JSON body.

        Args:
            path: An API path starting with ``/repos/``.
            method: HTTP method. Anything but GET requires a writable client.
            data: JSON body for mutating requests.

        Raises:
            GithubError: If the request is not allowed or the API fails.
        """
        if not path.startswith("/repos/") or any(char in path for char in "\r\n"):
            raise GithubError("Only GitHub repository API paths are allowed")
        if method != "GET" and not self.writable:
            raise GithubError("This GitHub client is read-only")
        headers = {
            "Accept": "application/vnd.github+json",
            "User-Agent": "guanaco-py-builder",
            "X-GitHub-Api-Version": API_VERSION,
        }
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        encoded = None
        if data is not None:
            encoded = json.dumps(data).encode("utf-8")
            headers["Content-Type"] = "application/json"
        request = urllib.request.Request(
            "https://api.github.com" + path, data=encoded, headers=headers, method=method
        )
        for attempt in range(DEFAULT_ATTEMPTS):
            try:
                with open_url(request, timeout=60) as response:
                    content = response.read(MAX_API_BYTES + 1)
                if len(content) > MAX_API_BYTES:
                    raise GithubError("GitHub API response is unexpectedly large")
                return json.loads(content) if content else None
            except Exception as error:
                if isinstance(error, urllib.error.HTTPError):
                    if error.code == 404 and method == "GET":
                        return None
                    if method == "GET" and attempt + 1 < DEFAULT_ATTEMPTS and is_retryable(error):
                        pause_before_retry(attempt, error)
                        continue
                    raise GithubError(
                        f"GitHub {method} {path}: HTTP {error.code}; check permissions/rate limits"
                    ) from error
                raise
        raise GithubError("GitHub retry loop exhausted")

    # -- Reading ------------------------------------------------------------

    def releases(self, repository: str) -> list[Release]:
        """Return every release of `repository`, oldest to newest."""
        repository_name(repository)
        result: list[Release] = []
        page = 1
        while True:
            batch = self.request(f"/repos/{repository}/releases?per_page=100&page={page}")
            if not isinstance(batch, list) or any(not isinstance(item, dict) for item in batch):
                raise GithubError("GitHub releases response must be a list of objects")
            result.extend(Release.from_mapping(item) for item in batch)
            if len(batch) < 100:
                return result
            page += 1

    def release(self, repository: str, tag: str) -> Release | None:
        """Return the release for `tag`, or ``None`` if there is none.

        Falls back to scanning the release list, because the direct lookup does
        not see draft releases without push access.

        Raises:
            GithubError: If more than one release shares the tag.
        """
        quoted = urllib.parse.quote(tag, safe="")
        result = self.request(f"/repos/{repository_name(repository)}/releases/tags/{quoted}")
        if result is not None:
            return Release.from_mapping(result)
        candidates = [item for item in self.releases(repository) if item.tag == tag]
        if len(candidates) > 1:
            raise GithubError(f"Multiple draft releases share {tag}; resolve them before retrying")
        return candidates[0] if candidates else None

    def commit(self, repository: str, ref: str) -> str:
        """Resolve a tag, branch or commit to a full commit SHA."""
        quoted = urllib.parse.quote(ref, safe="")
        value = self.request(f"/repos/{repository_name(repository)}/commits/{quoted}")["sha"]
        if not COMMIT_PATTERN.fullmatch(value):
            raise GithubError("GitHub returned an invalid commit SHA")
        return value

    def tag_commit(self, repository: str, tag: str) -> str | None:
        """Return the commit a Git tag points at, or ``None`` if it is absent.

        Both lightweight and annotated tags are handled; a tag is never moved.
        """
        quoted = urllib.parse.quote(tag, safe="")
        reference = self.request(f"/repos/{repository_name(repository)}/git/ref/tags/{quoted}")
        if reference is None:
            return None
        target = reference["object"]
        for _ in range(8):
            if not COMMIT_PATTERN.fullmatch(target.get("sha", "")):
                raise GithubError("Invalid tag target SHA")
            if target["type"] == "commit":
                return target["sha"]
            if target["type"] != "tag":
                raise GithubError("Release tag does not point to a commit")
            target = self.request(f"/repos/{repository}/git/tags/{target['sha']}")["object"]
        raise GithubError("Excessively nested annotated tag")

    def tree(self, repository: str, commit: str) -> list[dict]:
        """Return the recursive Git tree at `commit`.

        Raises:
            GithubError: If the tree is truncated, because a truncated tree
                could hide a submodule we must not silently skip.
        """
        if not COMMIT_PATTERN.fullmatch(commit):
            raise GithubError("A full immutable commit SHA is required")
        result = self.request(
            f"/repos/{repository_name(repository)}/git/trees/{commit}?recursive=1"
        )
        if result.get("truncated"):
            raise GithubError("Truncated Git tree: refusing to miss a submodule")
        return result["tree"]

    # -- Writing ------------------------------------------------------------

    def create_release(
        self,
        repository: str,
        *,
        tag: str,
        commit: str,
        name: str,
        body: str,
        draft: bool,
        latest: bool,
    ) -> Release:
        """Create a release. Requires a writable client."""
        payload = self.request(
            f"/repos/{repository_name(repository)}/releases",
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

    def update_release(
        self,
        repository: str,
        identifier: int,
        *,
        body: str | None = None,
        draft: bool | None = None,
        latest: bool | None = None,
    ) -> Release:
        """Patch a release. Only the given fields are changed."""
        data = {"prerelease": False}
        if body is not None:
            data["body"] = body
        if draft is not None:
            data["draft"] = draft
        if latest is not None:
            data["make_latest"] = "true" if latest else "false"
        payload = self.request(
            f"/repos/{repository_name(repository)}/releases/{identifier}",
            method="PATCH",
            data=data,
        )
        return Release.from_mapping(payload)


__all__ = ["GithubClient", "GithubError"]
