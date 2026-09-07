"""Release policy, validated provenance and a small GitHub REST client."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from download_utils import ATTEMPTS, open_url, pause_before_retry, retryable
from download_utils import sha256 as sha256

CONFIG = json.loads(
    (Path(__file__).resolve().parents[1] / "build-matrix.json").read_text(encoding="utf-8")
)
UPSTREAM = CONFIG["upstream"]
PACKAGE = CONFIG["package"]
CHANNELS = ["cpu", "avx2", *CONFIG["cuda"]]
VERSION = re.compile(r"(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)")
UPSTREAM_TAG = re.compile(
    r"v?((?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*))(?:[-+][A-Za-z0-9_.+\-]+)?"
)
PRERELEASE = re.compile(
    r"(?:^|[-+.])(?:alpha|beta|rc|dev|pre|preview|nightly|snapshot)(?:[0-9]*|[.-][0-9]+)?(?:$|[-+.])",
    re.I,
)
SHA = re.compile(r"[0-9a-f]{40}")
MARKER = re.compile(r"<!-- guanaco-upstream-build-v1\r?\n(.*?)\r?\n-->\s*$", re.S)
PROVENANCE_SEPARATOR = "\n\n---\n\n### Guanaco build provenance\n\n"
SNAPSHOT_FIELDS = ("upstream", "python_versions", "channels", "cuda")
MAX_API_BYTES = 32 * 1024**2


def version_key(version: str) -> tuple[int, int, int]:
    if not isinstance(version, str) or not VERSION.fullmatch(version):
        raise ValueError(f"Expected a stable X.Y.Z version, got {version!r}")
    major, minor, patch = map(int, version.split("."))
    return major, minor, patch


def version_from_tag(tag: str) -> str | None:
    if not isinstance(tag, str):
        return None
    match = UPSTREAM_TAG.fullmatch(tag)
    return match.group(1) if match and not PRERELEASE.search(tag) else None


def repository_name(value: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", value):
        raise ValueError(f"Invalid GitHub repository: {value!r}")
    if any(part in (".", "..") for part in value.split("/")):
        raise ValueError("Repository paths cannot contain dot segments")
    return value


def release_tag(version: str, channel: str) -> str:
    version_key(version)
    if not isinstance(channel, str) or not re.fullmatch(r"cpu|avx2|cu[0-9]+", channel):
        raise ValueError(f"Unknown channel: {channel}")
    return f"v{version}" + ("" if channel == "cpu" else f"-{channel}")


def validate_python_versions(versions: list[str]) -> None:
    if not isinstance(versions, list) or not versions:
        raise ValueError("Python matrix must be a nonempty list")
    if any(
        not isinstance(v, str) or not re.fullmatch(r"3\.(?:9|[1-9][0-9]+)", v) for v in versions
    ):
        raise ValueError("Python matrix must use supported 3.9+ version strings")
    if len(set(versions)) != len(versions):
        raise ValueError("Python matrix contains duplicate versions")


def validate_build_matrix(matrix: dict) -> None:
    validate_python_versions(matrix.get("python_versions"))
    channels, cuda = matrix.get("channels"), matrix.get("cuda")
    if not isinstance(channels, list) or not isinstance(cuda, dict):
        raise ValueError("Missing channel/CUDA build matrix")
    if (
        channels[:2] != ["cpu", "avx2"]
        or len(channels) != len(set(channels))
        or set(channels[2:]) != set(cuda)
    ):
        raise ValueError("Channel list does not match the CUDA matrix")
    for channel, settings in cuda.items():
        if not re.fullmatch(r"cu[0-9]+", channel) or not isinstance(settings, dict):
            raise ValueError("Invalid CUDA channel configuration")
        version_key(settings.get("toolkit"))
        major, minor, _ = settings["toolkit"].split(".")
        if channel != f"cu{major}{minor}":
            raise ValueError("CUDA channel and toolkit version disagree")
        if not isinstance(settings.get("architectures"), str) or not re.fullmatch(
            r"[0-9]+[af]?(?:;[0-9]+[af]?)*", settings["architectures"]
        ):
            raise ValueError("Invalid CUDA architecture list")
        if type(settings.get("legacy_msvc")) is not bool:
            raise ValueError("legacy_msvc must be a boolean")


def build_snapshot(plan: dict) -> dict:
    """Fields that must not drift during a partially published version family."""
    validate_build_matrix(plan)
    return copy.deepcopy({key: plan[key] for key in SNAPSHOT_FIELDS})


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(value, indent=2, sort_keys=True) + "\n"
    fd, name = tempfile.mkstemp(prefix=".json-", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(content)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def outputs(path: str | Path | None, **values) -> None:
    """Validate the entire batch before appending workflow output records."""
    records = []
    for key, value in values.items():
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
            raise ValueError("Invalid workflow output key")
        if isinstance(value, bool):
            value = "true" if value else "false"
        elif isinstance(value, (dict, list)):
            value = json.dumps(value, separators=(",", ":"))
        value = str(value)
        if "\n" in value or "\r" in value:
            raise ValueError("Workflow output must be a single line")
        records.append(f"{key}={value}\n")
    if path:
        with Path(path).open("a", encoding="utf-8", newline="\n") as stream:
            stream.writelines(records)


class GitHub:
    """Read-only by default. Mutating requests are never retried implicitly."""

    def __init__(self, token: str | None = None, *, writable: bool = False):
        self.token = (
            token if token is not None else (os.getenv("GH_TOKEN") or os.getenv("GITHUB_TOKEN", ""))
        )
        if any(char in self.token for char in "\r\n"):
            raise ValueError("Invalid GitHub token format")
        self.writable = writable

    def request(self, path: str, *, method="GET", data=None, missing_ok=False):
        if not path.startswith("/repos/") or any(char in path for char in "\r\n"):
            raise ValueError("Only GitHub repository API paths are allowed")
        if method != "GET" and not self.writable:
            raise ValueError("This GitHub client is read-only")
        headers = {
            "Accept": "application/vnd.github+json",
            "User-Agent": "guanaco-py-builder",
            "X-GitHub-Api-Version": "2022-11-28",
        }
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        encoded = None if data is None else json.dumps(data).encode("utf-8")
        if encoded is not None:
            headers["Content-Type"] = "application/json"
        request = urllib.request.Request(
            "https://api.github.com" + path, data=encoded, headers=headers, method=method
        )
        for attempt in range(ATTEMPTS):
            try:
                with open_url(request, timeout=60) as response:
                    content = response.read(MAX_API_BYTES + 1)
                if len(content) > MAX_API_BYTES:
                    raise ValueError("GitHub API response is unexpectedly large")
                return json.loads(content) if content else None
            except Exception as error:
                if isinstance(error, urllib.error.HTTPError) and error.code == 404 and missing_ok:
                    return None
                if method == "GET" and attempt + 1 < ATTEMPTS and retryable(error):
                    pause_before_retry(attempt, error)
                    continue
                if isinstance(error, urllib.error.HTTPError):
                    raise RuntimeError(
                        f"GitHub {method} {path}: HTTP {error.code}; check permissions/rate limits"
                    ) from error
                raise
        raise RuntimeError("GitHub retry loop exhausted")

    def releases(self, repository: str) -> list[dict]:
        repository_name(repository)
        result = []
        page = 1
        while True:
            batch = self.request(f"/repos/{repository}/releases?per_page=100&page={page}")
            if not isinstance(batch, list) or any(not isinstance(item, dict) for item in batch):
                raise ValueError("GitHub releases response must be a list of objects")
            result.extend(batch)
            if len(batch) < 100:
                return result
            page += 1

    def release(self, repository: str, tag: str) -> dict | None:
        result = self.request(
            f"/repos/{repository_name(repository)}/releases/tags/{urllib.parse.quote(tag, safe='')}",
            missing_ok=True,
        )
        if result is not None:
            return result
        candidates = [r for r in self.releases(repository) if r["tag_name"] == tag]
        if len(candidates) > 1:
            raise ValueError(f"Multiple draft releases share {tag}; resolve them before retrying")
        return candidates[0] if candidates else None

    def commit(self, repository: str, ref: str) -> str:
        value = self.request(
            f"/repos/{repository_name(repository)}/commits/{urllib.parse.quote(ref, safe='')}"
        )["sha"]
        if not SHA.fullmatch(value):
            raise ValueError("GitHub returned an invalid commit SHA")
        return value

    def tag_commit(self, repository: str, tag: str) -> str | None:
        """Resolve lightweight and annotated tags without moving them."""
        reference = self.request(
            f"/repos/{repository_name(repository)}/git/ref/tags/{urllib.parse.quote(tag, safe='')}",
            missing_ok=True,
        )
        if reference is None:
            return None
        obj = reference["object"]
        for _ in range(8):
            if not SHA.fullmatch(obj.get("sha", "")):
                raise ValueError("Invalid tag target SHA")
            if obj["type"] == "commit":
                return obj["sha"]
            if obj["type"] != "tag":
                raise ValueError("Release tag does not point to a commit")
            obj = self.request(f"/repos/{repository}/git/tags/{obj['sha']}")["object"]
        raise ValueError("Excessively nested annotated tag")

    def tree(self, repository: str, commit: str) -> list[dict]:
        if not SHA.fullmatch(commit):
            raise ValueError("A full immutable commit SHA is required")
        result = self.request(
            f"/repos/{repository_name(repository)}/git/trees/{commit}?recursive=1"
        )
        if result.get("truncated"):
            raise ValueError("Truncated Git tree: refusing to miss a submodule")
        return result["tree"]


def provenance_offset(body: str) -> int:
    separators = (PROVENANCE_SEPARATOR, PROVENANCE_SEPARATOR.replace("\n", "\r\n"))
    return max(body.rfind(separator) for separator in separators)


def source_notes(body: str) -> str:
    """Extract the exact upstream preamble, even when line endings are mixed."""
    boundary = provenance_offset(body)
    if boundary < 0:
        raise ValueError(
            "Legacy provenance has no upstream note snapshot; manual recovery required"
        )
    return body[:boundary]


def provenance(release: dict) -> dict | None:
    body = release.get("body") or ""
    boundary = provenance_offset(body)
    matches = MARKER.findall(body[boundary:] if boundary >= 0 else body)
    if not matches:
        return None
    if len(matches) != 1 or matches[0].count("<!-- guanaco-upstream-build-v1"):
        raise ValueError("Ambiguous provenance marker")
    try:
        state = json.loads(matches[0])
        version_key(state["version"])
        if state.get("upstream_repository") != UPSTREAM or not SHA.fullmatch(
            state.get("upstream_commit", "")
        ):
            raise ValueError("Invalid upstream provenance")
        if state["tag"] != release_tag(state["version"], state["channel"]) or state[
            "tag"
        ] != release.get("tag_name"):
            raise ValueError("Invalid channel provenance")
        if version_from_tag(state.get("upstream_tag")) != state["version"]:
            raise ValueError("Provenance does not match the upstream version")
        if not SHA.fullmatch(state.get("recipe_commit", "")):
            raise ValueError("Invalid build recipe provenance")
        if type(state.get("complete")) is not bool:
            raise ValueError("Provenance complete must be a boolean")
        if type(state.get("upstream_release_id")) is not int or state["upstream_release_id"] <= 0:
            raise ValueError("Invalid upstream release ID")
        validate_python_versions(state.get("python_versions"))
        if "snapshot" in state:
            snapshot = state["snapshot"]
            validate_build_matrix(snapshot)
            origin = snapshot["upstream"]
            if "body_sha256" in origin:
                notes = source_notes(release["body"])
                if hashlib.sha256(notes.encode("utf-8")).hexdigest() != origin.pop("body_sha256"):
                    raise ValueError("Upstream note snapshot checksum mismatch")
                if "body" in origin:
                    raise ValueError("Ambiguous note snapshot")
                origin["body"] = notes
            if not isinstance(origin.get("body"), str):
                raise ValueError("Invalid frozen release notes")
            if (
                origin["repository"] != UPSTREAM
                or origin["commit"] != state["upstream_commit"]
                or origin["tag"] != state["upstream_tag"]
                or origin["release_id"] != state["upstream_release_id"]
                or snapshot["python_versions"] != state["python_versions"]
            ):
                raise ValueError("Frozen snapshot does not match release provenance")
        return state
    except (KeyError, TypeError, AttributeError, json.JSONDecodeError) as error:
        raise ValueError("Malformed Guanaco release provenance") from error


def expected_assets(version: str, channel: str, python_versions: list[str]) -> set[str]:
    release_tag(version, channel)
    validate_python_versions(python_versions)
    linux = "manylinux_2_34_x86_64" if channel in ("cpu", "avx2") else "linux_x86_64"
    names = {
        f"guanaco_py-{version}-cp{v.replace('.', '')}-cp{v.replace('.', '')}-{platform}.whl"
        for v in python_versions
        for platform in (linux, "win_amd64")
    }
    names.update({"guanaco-build.json", "SHA256SUMS"})
    if channel == "cpu":
        names.update({f"guanaco-source-{version}.tar.gz", "packaging.patch"})
    return names


def complete(release: dict, state: dict | None, python_versions: list[str]) -> bool:
    if (
        not state
        or release.get("draft")
        or release.get("prerelease")
        or state.get("complete") is not True
    ):
        return False
    assets = release.get("assets", [])
    names = [asset.get("name") for asset in assets]
    if len(names) != len(set(names)):
        return False
    uploaded = {
        a["name"]
        for a in assets
        if a.get("state") == "uploaded" and type(a.get("size")) is int and a["size"] > 0
    }
    return (
        expected_assets(
            state["version"], state["channel"], state.get("python_versions", python_versions)
        )
        <= uploaded
    )
