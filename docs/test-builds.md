# Manual wheel test builds

**Test Wheel Build (no release)** compiles selected wheels using the same reusable CPU/AVX2/CUDA recipes as the release workflow, but uploads **only GitHub Actions artifacts**.

It does not create releases or Git tags, deploy Pages, build/push Docker images, or promote a version to latest. The workflow uses `contents: read`; its plans carry `test_only: true` and are rejected by the release publisher.

## Run a test

1. Put the workflow and its supporting script changes on the default branch so GitHub displays the **Run workflow** button.
2. Open **Actions → Test Wheel Build (no release) → Run workflow**.
3. Choose the recipe branch, upstream version and desired build subset.
4. Wait for the selected jobs and the final report. Download files from the run's **Artifacts** section.

| Input | Selection |
|---|---|
| `version` | Exact upstream `X.Y.Z`; empty selects the newest stable upstream release |
| `cpu` | Checkbox for portable CPU; enabled by default |
| `avx2` | Checkbox for AVX2; disabled by default |
| `cuda` | Checkbox for CUDA; disabled by default |
| `cuda_channels` | Comma-separated channels such as `cu124,cu128`, or `all`; ignored when CUDA is disabled |
| `python_versions` | Comma-separated versions such as `3.9,3.13,3.14`, or `all`; default `3.13` |
| `systems` | `linux`, `windows` or `both`; default `both` |

Choose at least one backend checkbox. Versions/channels must be present in [the build matrix](../.github/build-matrix.json). Duplicate or unsupported selections fail before starting expensive builds. Targets remain **x86-64**, as in the real release pipeline; this does not add macOS or ARM support.

The default is a small test: **CPU × Python 3.13 × Linux/Windows = 2 wheels**. Selecting every backend, CUDA toolkit, Python version and OS requests the full **108-wheel matrix** and can take considerable time and Actions resources.

### Examples

- **Small CPU check:** `cpu=true`, `avx2=false`, `cuda=false`, Python `3.13`, systems `both`.
- **Windows SIMD check:** CPU and AVX2 enabled, CUDA disabled, Python `3.9,3.14`, systems `windows` → 4 wheels.
- **One CUDA toolchain:** only CUDA enabled, channels `cu124`, Python `3.13`, systems `linux` → 1 wheel.
- **Compare CPU and two CUDA toolchains:** CPU and CUDA enabled, channels `cu124,cu128`, Python `3.13`, systems `both` → 6 wheels.

## Test a version that is already published

Unlike **Check Upstream and Release**, this workflow does not look for missing Guanaco channels or skip a completed version. You can test `0.3.49` repeatedly with the current recipe and local patches without replacing its published assets.

It uses the release workflow's **upstream selection rules** and pins the selected release tag to a commit SHA. It deliberately uses a fresh test plan, not a historical Guanaco family's frozen plan. Compare the source SHA/tag, recipe revision, selected matrix and patch report before drawing conclusions about a later production run.

`prepare_source.py` is reused unchanged, including the current `.github/patches/*.patch` mechanism. This feature does not change patch application or introduce a new patch policy.

## Downloadable artifacts

| Artifact | Contents |
|---|---|
| `test-guanaco-py-cpu-<system>-x64` | CPU wheels for the selected Python versions |
| `test-guanaco-py-avx2-<system>-x64` | AVX2 wheels for the selected Python versions |
| `test-guanaco-py-cuda-<system>-x64-<channel>-py<version>` | One CUDA wheel for that matrix cell |
| `inspection-test-guanaco-py-*` | `wheels.json`: filename, SHA256, sizes, ZIP file inventory, and METADATA/WHEEL text |
| `receipt-test-guanaco-py-*` | Small receipts emitted only after the existing wheel validation succeeds |
| `test-build-plan` | Exact selected upstream source, recipe and requested matrix |
| `guanaco-source` | Shared `source.tar.gz`, `build-manifest.json` and `packaging.patch` |
| `test-build-inspection` | Input patches, preparation log, recorded applied patches, final affected source files and build settings |
| `test-build-report` | Final PASS/FAIL summary, expected/received receipts, job results and test validation data |

Artifacts are retained for **14 days**, subject to repository/organization retention limits. Each workflow run has its own artifact namespace; the test workflow does not download artifacts from a release run.

The `.whl` filenames and package version are unchanged so the wheels can be installed locally. **Use an isolated virtual environment:** test and released wheels can share the same package name/version while containing different patches. Do not infer release status from a wheel filename.

### On failure

- Source preparation output is captured in `preparation.log`; diagnostics are retained when a patch does not apply or another preparation step fails.
- If compilation or verification fails, any wheel files already produced by that **test** job are still uploaded for debugging. They are **not necessarily validated**.
- The final job checks both required job outcomes and **every requested receipt**, respecting the selected Python/OS/backend subset. Missing files, failed jobs or failed artifact downloads produce a failed result, not an empty green run.
- Wheel metadata inspection itself does not execute the package and is not a validation verdict. Consult the receipt, job logs and final report.
- A cancelled run may stop before uploading artifacts; cancellation is not a preservation guarantee.

The final job downloads only small receipts, not the whole multi-GB CUDA wheel matrix into one runner. Download the wheel artifacts individually or use the CLI pattern below.

## Optional GitHub CLI usage

```bash
gh workflow run build-test.yaml --ref main \
  -f version=0.3.49 \
  -F cpu=true -F avx2=false -F cuda=true \
  -f cuda_channels=cu124,cu128 \
  -f python_versions=3.13 \
  -f systems=both

gh run list --workflow build-test.yaml
gh run download RUN_ID --pattern 'test-guanaco-py-*' --dir test-wheels
```

## What a passing test does and does not mean

CPU/AVX2 use the same cibuildwheel build, manylinux repair, installed-wheel import and native API smoke check as the release workflow. CUDA uses the same toolkit, compiler flags and existing wheel-content verification.

A successful selected matrix does not certify unselected Python versions/platforms, GPU inference, every model, or every integration. Hosted CUDA jobs do not provide a GPU inference guarantee. The patch records are the existing mechanism's diagnostics, not a new semantic proof of every patch.

This is a rehearsal, **not promotion**. Run the real release workflow separately when ready; it still builds/validates its production matrix and enforces its publication rules. Test plans and their gates cannot be passed to the release publisher.
