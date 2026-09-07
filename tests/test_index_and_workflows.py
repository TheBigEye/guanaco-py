import json
from pathlib import Path

import wheel_index
import yaml
from helpers import owned, plan

ROOT = Path(__file__).resolve().parents[1]


def test_index_lists_only_completed_managed_releases_and_keeps_channels_separate(tmp_path):
    releases = [
        owned("cpu"),
        owned("avx2"),
        owned("cu124"),
        owned("cu121", draft=True, finished=False),
        {"tag_name": "v1.0.1", "body": "legacy", "assets": []},
    ]
    # gh api --paginate --slurp returns pages, not a single flat list.
    input_file = tmp_path / "releases.json"
    input_file.write_text(json.dumps([releases[:2], releases[2:]]))
    site = tmp_path / "site"
    wheel_index.generate(input_file, site)
    cpu = (site / "whl/cpu/guanaco-py/index.html").read_text()
    avx2 = (site / "whl/avx2/guanaco-py/index.html").read_text()
    cuda = (site / "whl/cu124/guanaco-py/index.html").read_text()
    assert "/v0.3.49/" in cpu and "v0.3.49-avx2" not in cpu
    assert "/v0.3.49-avx2/" in avx2 and "/v0.3.49-cu124/" in cuda
    assert "#sha256=" in cpu
    assert not (site / "whl/cu121").exists()
    assert "v1.0.1" not in cpu
    assert "data:image/svg+xml;base64," in (site / "index.html").read_text()
    assert "guanaco-source-" not in cpu


def test_empty_index_still_has_valid_pep503_channel_roots(tmp_path):
    data = tmp_path / "empty.json"
    data.write_text("[]")
    wheel_index.generate(data, tmp_path / "site")
    assert 'href="guanaco-py/"' in (tmp_path / "site/whl/cpu/index.html").read_text()
    assert (
        "No wheels published yet" in (tmp_path / "site/whl/cpu/guanaco-py/index.html").read_text()
    )


def test_index_does_not_hide_historical_versions_after_matrix_changes(tmp_path):
    p = plan()
    p["python_versions"] = ["3.12"]
    release = owned("cpu", base=p)
    data = tmp_path / "releases.json"
    data.write_text(json.dumps([release]))
    wheel_index.generate(data, tmp_path / "site")
    assert "cp312" in (tmp_path / "site/whl/cpu/guanaco-py/index.html").read_text()


def workflow(name):
    return yaml.load((ROOT / ".github/workflows" / name).read_text(), Loader=yaml.BaseLoader)


def test_schedule_and_explicit_followup_workflows():
    main = workflow("build-release.yaml")
    assert main["on"]["schedule"][0]["cron"] == "0 10 * * *"
    assert set(main["on"]) == {"schedule", "workflow_dispatch"}
    assert main["concurrency"]["cancel-in-progress"] == "false"
    jobs = main["jobs"]
    assert jobs["pages-index"]["uses"].endswith("deploy-pages.yaml")
    assert jobs["docker"]["uses"].endswith("build-docker.yaml")
    assert "publish" in jobs["docker"]["needs"]
    assert "always()" in jobs["validate"]["if"]
    assert "needs.source.result == 'success'" in jobs["validate"]["if"]
    assert main["permissions"]["contents"] == "read"
    assert jobs["publish"]["permissions"]["contents"] == "write"
    assert jobs["docker"]["permissions"]["packages"] == "write"


def test_every_reusable_builder_uses_prepared_source_not_checkout_submodules():
    for name in ["build-wheel-cpu.yml", "build-wheel-cuda.yaml"]:
        text = (ROOT / ".github/workflows" / name).read_text()
        assert "submodules: recursive" not in text
        assert "name: guanaco-source" in text and "unpack_source.py" in text
        assert "work/source" in text and "verify_wheels.py" in text
        assert set(workflow(name)["on"]) == {"workflow_call"}
    assert workflow("build-wheel-avx2.yml")["jobs"]["avx2"]["uses"].endswith("build-wheel-cpu.yml")
    assert "release" not in workflow("build-docker.yaml")["on"]


def test_repository_contains_recipes_not_vendored_bindings():
    for name in ["llama_cpp", "vendor", ".gitmodules", "CMakeLists.txt", "pyproject.toml"]:
        assert not (ROOT / name).exists()
    assert (ROOT / "docs/icon.svg").exists()
    assert not (ROOT / ".github/workflows/update-submodule.yaml").exists()
    assert "make build" not in (ROOT / "docker/simple/run.sh").read_text()


def test_all_publishers_wait_for_the_global_gate_and_download_one_channel_only():
    jobs = workflow("build-release.yaml")["jobs"]
    assert jobs["publish"]["needs"] == ["plan", "validate"]
    assert "needs.validate.result == 'success'" in jobs["publish"]["if"]
    publish_steps = jobs["publish"]["steps"]
    downloads = [
        step["with"]
        for step in publish_steps
        if step.get("uses", "").startswith("actions/download-artifact@")
    ]
    assert any(item.get("name") == "validated-build" for item in downloads)
    assert [item["pattern"] for item in downloads if "pattern" in item] == [
        "${{ matrix.artifact_pattern }}"
    ]
    command = publish_steps[-1]["run"]
    assert "--gate work/validated-build.json" in command and "--channel" in command
    assert any("--preflight" in step.get("run", "") for step in jobs["validate"]["steps"])


def test_cuda_reads_frozen_manifest_and_cpu_avx2_share_one_recipe():
    cuda = workflow("build-wheel-cuda.yaml")["jobs"]
    steps = cuda["config"]["steps"]
    assert any(step.get("with", {}).get("name") == "guanaco-source" for step in steps)
    command = steps[-1]["run"]
    assert "configure_build.py cuda" in command and "--manifest" in command
    assert "check_upstream.py" not in command
    avx2 = workflow("build-wheel-avx2.yml")["jobs"]["avx2"]
    assert avx2["uses"].endswith("build-wheel-cpu.yml") and avx2["with"]["channel"] == "avx2"
    cpu = (ROOT / ".github/workflows/build-wheel-cpu.yml").read_text()
    assert "auditwheel repair --only-plat" in cpu
    assert "llama_print_system_info" in cpu


def test_artifact_reruns_can_replace_temporary_artifacts_not_releases():
    for path in (ROOT / ".github/workflows").iterdir():
        parsed = workflow(path.name)
        for job in parsed["jobs"].values():
            for step in job.get("steps", []):
                if step.get("uses", "").startswith("actions/upload-artifact@"):
                    assert step["with"]["overwrite"] == "true", path.name


def test_dockerfiles_copy_the_shared_helpers_and_all_local_copy_sources_exist():
    import shlex

    for dockerfile in (ROOT / "docker").glob("*/Dockerfile"):
        text = dockerfile.read_text()
        assert ".github/scripts/download_utils.py" in text
        if "unpack_source.py" in text:
            assert ".github/scripts/archive_utils.py" in text
        for line in text.splitlines():
            if line.startswith("COPY ") and "--from=" not in line:
                for source in shlex.split(line)[1:-1]:
                    assert (ROOT / source).is_file(), f"{dockerfile}: {source}"
    checkout = workflow("deploy-pages.yaml")["jobs"]["pages-index"]["steps"][0]
    assert checkout["with"]["ref"] == "${{ github.event.repository.default_branch }}"
