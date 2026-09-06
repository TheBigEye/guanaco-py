"""Small synthetic releases/wheels for offline tests. Native fixtures are not executable."""

import base64
import copy
import csv
import hashlib
import io
import json
import struct
import zipfile

import validate_receipts as gate
import verify_wheels
from publish_release import release_body
from release_common import (
    CHANNELS,
    CONFIG,
    UPSTREAM,
    expected_assets,
    release_tag,
    sha256,
    write_json,
)

SHA_A, SHA_B = "a" * 40, "b" * 40


def upstream(tag="v0.3.49-cu124-win-20260831", **overrides):
    return {
        "id": 49,
        "tag_name": tag,
        "published_at": "2026-08-31T10:00:00Z",
        "draft": False,
        "prerelease": False,
        "body": "Original upstream notes.\r\nKeep the changelog.",
        "html_url": f"https://github.com/{UPSTREAM}/releases/tag/{tag}",
        "name": tag,
        **overrides,
    }


def plan(channels=None):
    r = upstream()
    return copy.deepcopy(
        {
            "schema": 1,
            "repository": "TheBigEye/guanaco-py",
            "version": "0.3.49",
            "channels": CHANNELS,
            "missing_channels": channels if channels is not None else CHANNELS,
            "python_versions": CONFIG["python_versions"],
            "cuda": CONFIG["cuda"],
            "recipe_commit": SHA_B,
            "promote_latest": True,
            "run_url": "https://github.com/TheBigEye/guanaco-py/actions/runs/1",
            "upstream": {
                "repository": UPSTREAM,
                "release_id": r["id"],
                "tag": r["tag_name"],
                "commit": SHA_A,
                "release_url": r["html_url"],
                "body": r["body"],
                "release_name": r["name"],
                "published_at": r["published_at"],
            },
        }
    )


def owned(channel, *, finished=True, draft=False, base=None):
    p = base or plan()
    tag = release_tag(p["version"], channel)
    assets = [
        {
            "name": name,
            "state": "uploaded",
            "size": 100,
            "browser_download_url": f"https://github.com/{p['repository']}/releases/download/{tag}/{name}",
            "digest": "sha256:" + "d" * 64,
        }
        for name in sorted(expected_assets(p["version"], channel, p["python_versions"]))
    ]
    return {
        "id": 100 + CHANNELS.index(channel),
        "tag_name": tag,
        "body": release_body(p, channel, finished),
        "draft": draft,
        "prerelease": False,
        "assets": assets,
    }


class FakeGitHub:
    def __init__(self, upstream_releases=None, existing=None):
        self.upstream_releases = (
            upstream_releases if upstream_releases is not None else [upstream()]
        )
        self.existing = existing or []
        self.commit_calls = []

    def releases(self, repository):
        return self.upstream_releases if repository == UPSTREAM else self.existing

    def commit(self, repository, ref):
        self.commit_calls.append((repository, ref))
        return SHA_A

    def request(self, path, **kwargs):
        raise AssertionError(f"Unexpected API call: {path}")


def manifest_for(p):
    runtime = {
        "llama_cpp/__init__.py": f'__version__ = "{p["version"]}"\n'.encode(),
        "llama_cpp/py.typed": b"",
    }
    return {
        **p,
        "package": "guanaco-py",
        "runtime_sha256": {
            name: hashlib.sha256(data).hexdigest() for name, data in runtime.items()
        },
    }, runtime


def native_header(platform):
    # Minimal architecture fixtures, NOT loadable runtime libraries.
    header = bytearray(256)
    if platform == "linux":
        header[:7] = b"\x7fELF\x02\x01\x01"
        struct.pack_into("<HH", header, 16, 3, 62)
    else:
        header[:2] = b"MZ"
        struct.pack_into("<I", header, 60, 128)
        header[128:134] = b"PE\x00\x00\x64\x86"
    return bytes(header)


def zip_contents(path, contents, *, record=True):
    if record:
        record_path = (
            next(name for name in contents if name.endswith(".dist-info/METADATA")).removesuffix(
                "METADATA"
            )
            + "RECORD"
        )
        rows = []
        for name, data in contents.items():
            if name == record_path:
                continue
            digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).decode().rstrip("=")
            rows.append([name, "sha256=" + digest, str(len(data))])
        rows.append([record_path, "", ""])
        output = io.StringIO(newline="")
        csv.writer(output).writerows(rows)
        contents[record_path] = output.getvalue().encode()
    with zipfile.ZipFile(path, "w") as wheel:
        for name, data in contents.items():
            wheel.writestr(name, data)


def write_wheel(
    directory,
    manifest,
    runtime,
    channel,
    platform,
    cp="cp313",
    *,
    metadata_name="guanaco-py",
    native=True,
    altered=False,
    extra=False,
    raw=False,
):
    directory.mkdir(parents=True, exist_ok=True)
    policy = (
        "win_amd64"
        if platform == "windows"
        else ("linux_x86_64" if channel.startswith("cu") or raw else "manylinux_2_34_x86_64")
    )
    version = manifest["version"]
    path = directory / f"guanaco_py-{version}-{cp}-{cp}-{policy}.whl"
    contents = {
        name: data + b"# altered" if altered and name.endswith(".py") else data
        for name, data in runtime.items()
    }
    if extra:
        contents["llama_cpp/additional.py"] = b"pass"
    info = f"guanaco_py-{version}.dist-info/"
    contents[info + "METADATA"] = (
        f"Metadata-Version: 2.3\nName: {metadata_name}\nVersion: {version}\n".encode()
    )
    contents[info + "WHEEL"] = (
        f"Wheel-Version: 1.0\nRoot-Is-Purelib: false\nTag: {cp}-{cp}-{policy}\n".encode()
    )
    contents[info + "licenses/LICENSE.md"] = b"MIT notice"
    if native:
        contents["llama_cpp/lib/" + ("llama.dll" if platform == "windows" else "libllama.so")] = (
            native_header(platform)
        )
    zip_contents(path, contents)
    return path


def prepared_build(tmp_path, missing=False):
    p = plan(["cpu", "avx2"])
    p["python_versions"] = ["3.12", "3.13"]
    m, runtime = manifest_for(p)
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    (prepared / "source.tar.gz").write_bytes(b"prepared archive fixture")
    (prepared / "packaging.patch").write_text("metadata patch fixture")
    m["source_archive_sha256"] = sha256(prepared / "source.tar.gz")
    m["packaging_patch_sha256"] = sha256(prepared / "packaging.patch")
    (prepared / "build-manifest.json").write_text(json.dumps(m))
    artifacts = tmp_path / "artifacts"
    for channel in p["missing_channels"]:
        for platform in ("linux", "windows"):
            for cp in ("cp312", "cp313"):
                if missing and channel == "avx2" and platform == "windows" and cp == "cp313":
                    continue
                write_wheel(
                    artifacts / f"guanaco-py-{channel}-{platform}-x64",
                    m,
                    runtime,
                    channel,
                    platform,
                    cp,
                )
    return p, prepared, artifacts


class PublishingAPI:
    def __init__(self):
        self.items = {}
        self.calls = []

    def release(self, repo, tag):
        return next((r for r in self.items.values() if r["tag_name"] == tag), None)

    def tag_commit(self, repo, tag):
        return None

    def request(self, path, method="GET", data=None):
        self.calls.append((method, path, copy.deepcopy(data)))
        if method == "POST":
            number = len(self.items) + 1
            self.items[number] = {"id": number, "assets": [], **data}
            return copy.deepcopy(self.items[number])
        number = int(path.split("/")[-1])
        if method == "PATCH":
            self.items[number].update(data)
        return copy.deepcopy(self.items[number])

    def upload(self, tag, files):
        release = self.release("", tag)
        release["assets"] = [
            {
                "name": p.name,
                "state": "uploaded",
                "size": p.stat().st_size,
                "digest": "sha256:" + sha256(p),
            }
            for p in files
        ]


def wheel_data(wheel):
    with zipfile.ZipFile(wheel) as archive:
        return {name: archive.read(name) for name in archive.namelist()}


def receipts_for(p, prepared, artifacts, directory):
    manifest = json.loads((prepared / "build-manifest.json").read_text())
    for artifact, job in gate.artifact_specs(p).items():
        wheels = sorted((artifacts / artifact).glob("*.whl"))
        write_json(
            directory / f"{artifact}.json",
            verify_wheels.receipt(wheels, manifest, job["channel"], job["platform"]),
        )
    return manifest
