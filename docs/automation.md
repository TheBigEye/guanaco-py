# Automation & maintenance

Guanaco is a build/distribution repository. The binding source of truth is a **published release of JamePeng/llama-cpp-python**, not this repository and not upstream `main`.

## Architecture

```text
07:00 Argentina / 10:00 UTC, or workflow_dispatch
                    |
          python -m guanaco plan
       stable X.Y.Z + frozen family plan
       + early read-only tag preflight
                    |
       python -m guanaco prepare-source
       immutable ZIPs + exact gitlinks
       + .github/patches/ (strict, no fuzz)
       + distribution/build metadata
                    |
       one checksummed source.tar.gz
             /      |      \
           CPU     AVX2    CUDA matrix
             \      |      /
   python -m guanaco verify-wheels (each job)
       binaries + small validation receipts
                    |
     python -m guanaco validate-receipts
       full matrix gate + global preflight
                    |
       python -m guanaco publish (per channel)
       recheck receipt hashes and binaries
       draft -> upload -> verify -> publish
                    |
               /         \
        GitHub Pages      Docker / GHCR
```

Every step above is one subcommand of a single entry point, `python -m guanaco`.
The whole pipeline lives in the `guanaco/` package, one module per job; see
[Module map](#module-map) below.

No bindings, CMake project or Git submodule are stored in the working tree. Downloaded source, compiled objects, wheels and generated indexes go into ignored temporary/output directories.

## First-run checklist

1. Put this repository's files on the **default branch**, normally `main`. Enable GitHub Actions.
2. In **Settings → Pages**, select **GitHub Actions** as the source. Keep the `github-pages` environment available to the default branch and avoid an approval rule if unattended daily deployment is desired.
3. Allow the workflow's scoped token permissions. Publication needs `contents: write`, Pages needs `pages: write` / `id-token: write`, and Docker needs `packages: write`.
4. If a GHCR package already exists, grant this repository access to it. Recreating a repository with the same name does not necessarily preserve its old package/environment permissions.
5. Run **Test Distribution Automation**.
6. Run **Check Upstream and Release** manually. Leave `version` empty to select the latest stable upstream version. A fresh repository has no wheel assets until that build succeeds.
7. Verify all channel releases, their manifests/checksums, the `/whl/cpu/guanaco-py/` index and the CPU Docker image.
8. Test installation on your actual CPU/GPU and target platforms before depending on the new version in production.

No personal access token is required for the public upstream: workflows use the repository's `GITHUB_TOKEN`. Source ZIP downloads never forward that token to archive hosts.

The schedule is `0 10 * * *` (07:00 at UTC-3). It is not an exact-time guarantee: GitHub can delay scheduled runs and may disable schedules in inactive public repositories. Check the Actions tab if automatic checks stop.

## What counts as a new version?

JamePeng publishes tags such as:

```text
v0.3.49-Metal-macos-20260831
v0.3.49-cu124-linux-20260831
v0.3.49-cu124-win-20260831
```

These are one package version: `0.3.49`.

The checker:

- Paginates releases; it does not rely on the single `/releases/latest` result or API ordering.
- Rejects drafts, releases flagged as prereleases, non-version tags, and preview/RC/dev-style tags.
- Compares the three version components numerically: `0.3.10` is newer than `0.3.9`.
- Selects the highest stable version. Within that version, the most recently published eligible tag wins; the tag name is the deterministic tie-breaker.
- Resolves the selected tag to a full commit SHA before downloading source.
- Does not follow new commits on `main`, nor recompile a complete version merely because another backend/date tag appears.

The default run handles **the newest stable version only**, not every version ever released. Use the manual `version` input to backfill a specific older `X.Y.Z`. Backfilling does not promote an older version to GitHub/GHCR `latest`.

## Channel tags and package identity

The existing Guanaco channel scheme is retained:

| Channel | Guanaco release tag |
|---|---|
| Portable CPU | `vX.Y.Z` |
| CPU AVX2 | `vX.Y.Z-avx2` |
| CUDA | `vX.Y.Z-cu121`, `-cu122`, `-cu123`, `-cu124`, `-cu126`, `-cu128`, `-cu131` |

The wheel distribution is `guanaco-py`, its normalized filename prefix is `guanaco_py`, and the version is exactly upstream's `X.Y.Z`. No CUDA/AVX2 suffix is added to the Python package version. Separate releases/index channels prevent same-filename wheels for different backends from colliding.

The Guanaco tag points to the **build-recipe commit**, not an upstream source commit that is absent from this Git history. The selected upstream tag/commit is recorded separately and unambiguously.

## Immutable source preparation

`python -m guanaco prepare-source` downloads:

```text
https://codeload.github.com/JamePeng/llama-cpp-python/zip/<resolved-commit>
```

GitHub source ZIPs do not contain submodule contents. The preparer reads `.gitmodules`, asks the Git tree API for `160000` gitlink entries at that same commit, and downloads each referenced repository at its exact SHA. It repeats this for nested submodules.

There is no fallback to `main`, `master`, a guessed release branch or the newest llama.cpp commit. Missing commits, truncated trees or changed/unrecognized layouts fail the build instead of silently mixing revisions.

Archive extraction and source preparation are transactional: failures leave no partial destination and do not overwrite an existing nonempty source directory. Path checks reject traversal, links/devices, Windows aliases/alternate data streams, duplicate names and case-colliding directories. Member-count and decompressed-size limits are enforced. A future source-layout change may require updating the preparer.

Only public HTTPS GitHub submodule URLs are supported. Downloads stream to a temporary file, check length/checksums where available, and replace the destination only after success. Transient GET failures have a bounded retry budget; mutating API calls are not retried blindly. Authenticated API redirects cannot move the token to another host, and codeload/release-asset requests carry no GitHub token.

### Automatic metadata adaptation

Only the downloaded `pyproject.toml` is adapted:

- Distribution name, description and maintainer identity.
- Dependencies that refer to the distribution itself, such as `llama_cpp_python[server,test]`.
- Homepage, packaging issue tracker and upstream documentation/changelog links.
- Inclusion of upstream/native license notices in the wheel.
- `LLAMA_BUILD_COMMIT` set to the pinned native commit and `LLAMA_BUILD_NUMBER=0` for an archive build without Git history. This prevents CMake from accidentally reporting the automation repository's Git commit/count.

The source's `__version__` must match the selected `X.Y.Z`. Authors, runtime dependencies, license text and the version provider are retained. TOML is reparsed and checked after serialization.

Every runtime source file is hashed right after download, and again after any local patches are applied (see below); wheel validation checks the wheel's actual bytes against that second, post-patch hash. There is no global search-and-replace through Python code, no fuzzy patching, and no cache/inference patch queue beyond the reviewed, file-scoped patches described next.

### Local patches: a narrow, explicit, provenance-tracked exception

`.github/patches/*.patch` holds small, hand-written, hand-reviewed unified diffs, applied in filename order by `python -m guanaco prepare-source` immediately after the upstream download and before metadata adaptation. Each one:

- Must apply with `git apply` **exactly**, full context match, no fuzz. If upstream changes the surrounding code enough that a patch no longer applies, the build fails right there with the patch's name and `git apply`'s own error, instead of silently dropping the fix, silently skipping the file, or fuzzy-applying it onto code nobody has reviewed. Refresh or retire the patch, then re-run.
- Must declare its own target file(s) as normal `+++ b/<path>` diff headers; a patch aimed at a file upstream no longer ships also fails closed rather than being silently skipped.
- Is hashed before and after, and its filename, its own SHA-256, its target file(s), and their before/after hashes are all recorded in `build-manifest.json` under `applied_patches`. `runtime_sha256` (what `python -m guanaco verify-wheels` checks the built wheel against) reflects the tree **after** patches are applied; the untouched download is kept separately as `upstream_runtime_sha256` purely so a patched file stays easy to diff against the pristine release it came from.
- Is exactly as invisible to the metadata-adaptation invariant as any other runtime file: `adapt_metadata()` still may not change a single byte under `llama_cpp/`, patched or not, and preparation still fails if it does.

This is a deliberate, narrow carve-out for hand-reviewed behavioral fixes upstream hasn't (yet) merged, not a general patch queue: keep `.github/patches/` small, keep each patch scoped to one concern, and prefer upstreaming the fix instead when practical, since every patch is one more thing that can drift and demand attention on the next import.

### Boundary of the no-unreviewed-changes policy

Changing a distribution name is not a universal compatibility alias. Upstream code or downstream applications that call `importlib.metadata.version("llama-cpp-python")` will not discover `guanaco-py` under that name.

For example, upstream `0.3.49` uses that lookup in a version guard for optional Windows OpenMP preloading. This code is intentionally not patched by the metadata step, and no blanket patch papers over every such name assumption. Import checks are necessary but do not cover every embedded/ComfyUI/OpenMP integration scenario. If a name assumption causes an integration failure, investigate it explicitly, add a scoped patch under `.github/patches/` if it's worth carrying, or fix it upstream; do not assume that byte-identical Python files guarantee identical runtime behavior under different packaging/build options.

## Artifacts and provenance

One prepared source artifact is shared by all jobs:

```text
source.tar.gz
build-manifest.json
packaging.patch
```

If `.github/patches/` contains any patches, the manifest additionally records their provenance under `applied_patches` (patch name, patch hash, target file(s), before/after hashes of each), and `runtime_sha256` reflects the patched tree while `upstream_runtime_sha256` keeps the untouched download's hashes for comparison.

The manifest records the upstream release ID, tag, commit, original release-note text, source ZIP URLs/hashes, recursive submodule commits/hashes, native commit, automation revision/run, Python matrix and runtime file hashes.

Published assets:

| Every channel | CPU channel additionally |
|---|---|
| Its full wheel matrix | `guanaco-source-X.Y.Z.tar.gz` |
| `guanaco-build.json` | `packaging.patch` |
| `SHA256SUMS` | |

The CPU source archive is the reconstructed, buildable source with pinned native contents and adapted packaging metadata. GitHub's automatically generated source ZIP for a Guanaco tag contains **only the distribution repository**.

The release body begins with the original upstream note text and then adds a clearly separate Guanaco provenance section. Its hidden marker freezes the upstream identity, Python/CUDA matrix and a hash of the preceding notes, without duplicating a long changelog in the body. Editing upstream's notes later does not change the already selected family's notes. It does not generate a changelog from automation commits or copy upstream binary assets.

## Validation and publication rules

The default matrix is configured in `.github/build-matrix.json`:

- Python 3.9–3.14.
- Linux and Windows x86-64.
- Portable CPU, AVX2 and seven CUDA channels.
- Twelve wheels per channel: six Python versions × two platforms; 108 wheels for a complete new nine-channel family.

CPU/AVX2 use the same parametrized reusable workflow; the AVX2 file is a small wrapper. `python -m guanaco configure` resolves their fixed SIMD flags and the CUDA toolkit/architectures/Python versions from the **prepared manifest**, not from today's config file. Existing compiler/toolkit choices are retained. `auditwheel --only-plat` keeps the intended single manylinux filename/tag rather than adding a second, older compatibility tag. Toolchains may still need maintenance if upstream changes its requirements.

Checks include:

- Correct `guanaco_py` filename and exact package/version metadata.
- Required `METADATA`, `WHEEL` and `RECORD`; coherent Python ABI/platform tags and a native, non-pure-Python layout.
- Streaming RECORD hash/size checks and ZIP CRC verification, including native library payloads.
- No missing, duplicate or incorrectly attributed Python/platform matrix entry.
- No leftover self-dependency on `llama-cpp-python`.
- Unchanged upstream Python files; no untracked Python code, startup/bytecode files or wheel relocation schemes that could override the verified package.
- Presence of license notices and the main native library; x86-64 ELF/PE header checks. These are structural checks, not a substitute for loading the library on the target OS.
- Source archive/metadata-patch checksums and equality of **every plan field**, including the recipe commit.
- Uploaded release asset names, sizes and GitHub SHA256 digests when available.

All requested build jobs must succeed and produce a complete set of matching validation receipts before any publisher starts. The global gate also inspects **all** release/tag destinations before any write. A complete default build produces 88 small receipts for 108 wheels; the gate does not download all wheel binaries into one runner.

Each publisher downloads **one channel only**, validates its entire matrix again, and compares its actual hashes/sizes with the global gate. Staging uses hardlinks when possible, avoiding another multi-GB copy. This bounds runner disk usage without dropping the global validation gate. The gate file is not an external signature or a substitute for validating downloaded binaries.

Each release is created or resumed as a **draft**, populated and verified, then made public. Publication is not atomic across nine GitHub releases: a network failure can leave some public channels and others as drafts. Successful channels stay published; Pages/Docker follow-ups wait for all requested publishers to succeed.

A managed channel is complete only when its provenance marker contains the boolean `true`, it is public/non-prerelease, and its required assets are uploaded, nonempty and not duplicated. Its recorded Python matrix is used so historical wheels do not vanish just because the current matrix later changes.

Complete published channels are never silently overwritten. A **damaged public release is not turned back into a draft or repaired automatically**. Only a managed draft with matching provenance/recipe can be resumed. Duplicate drafts, legacy/manual releases without provenance, mixed snapshots and Git tags pointing to another recipe cause errors. Lightweight and annotated tags are checked; tags are never moved automatically.

## Retries and interrupted builds

- **Compilation failure:** no publication job runs. Retry failed jobs while their artifacts remain available (14 days). GitHub reruns use the original workflow SHA; to use a changed recipe, start a new run. Named artifacts use `overwrite: true` so a rerun can replace its own temporary artifacts, not published release assets.
- **Interrupted upload:** the incomplete release stays a draft. A later run discovers missing channels and reuses the frozen source, upstream notes, Python versions and CUDA settings. A draft from a **different recipe commit** is deliberately blocked: rerun the original recipe, or inspect/resolve only that unfinished draft and any conflicting tag manually. Do not delete a published channel to make a retry pass.
- **Another upstream backend tag appears, or notes/config change:** a complete version remains a no-op. A partial family uses its frozen snapshot rather than silently changing origin, notes or matrix.
- **A newer upstream version appears before an old partial family finishes:** the default check selects the newest stable version. Run manually with the old version to finish that older family if you still need it.
- **Pages fails:** run **Deploy Pages**; do not rebuild all wheels merely to regenerate the index.
- **Docker fails:** run **Build Docker Image** with the published `X.Y.Z`; set `promote_latest` only when appropriate.

For original v1 markers that predate the full snapshot, the checker recovers the notes from the existing Guanaco body and keeps the recorded source/Python versions. It logs that the old marker cannot recover a historical CUDA matrix and uses the current one. Inconsistent legacy/snapshot families fail closed rather than guessing.

A global non-cancelling workflow concurrency group prevents scheduled and manual release runs from racing each other. Discovery and global preflight have `contents: write` because GitHub exposes draft releases to callers with push access; their Python API clients are nevertheless read-only. Source preparation/build jobs have read-only repository permissions. Only the publisher performs release writes, and the script requires an explicit `--publish` flag.

## Pages and Docker are explicit follow-up jobs

Events created with the repository `GITHUB_TOKEN` generally do not start other workflows. Therefore the orchestrator calls both reusable workflows directly; it does not assume its own `release: published` event will trigger Docker or Pages.

The index keeps the existing channel URLs and styling/icon. CSS is maintained in `docs/wheel-index.css` and embedded in the generated HTML; the SVG icon is embedded too. Only complete managed releases are indexed, legacy personal-fork versions are excluded, and asset SHA256 fragments are included when available.

Generation builds a fresh site and replaces the previous **owned** site only after success, so deleted channel pages cannot linger. The `.guanaco-wheel-index.json` ownership record prevents deleting unrelated files. An old output without that marker, or one containing user-added files, is refused: choose a fresh/empty output directory rather than deleting unrelated content. The generator never deletes remote releases or assets. Pages explicitly checks out the default branch so a release event cannot select an obsolete generator from the release tag.

Docker fetches a pinned wheel directly from its release and checks `SHA256SUMS`, avoiding a Pages/CDN propagation race. The default automatic image is CPU, as before. CUDA and OpenBLAS Dockerfiles remain available for explicit builds; OpenBLAS compiles the reconstructed source archive because it is not a separate published wheel channel.

## Migrating or recreating the repository

Recreating GitHub is optional; replacing the working tree is sufficient for the new architecture. Deleting a repository can also lose releases, issues, stars and configuration. Back up anything you want to retain first.

If using a fresh repository:

- Upload the distribution files, including `.github/` and `docs/icon.svg`; do not upload a previous `.git` directory or downloaded build source.
- Reconfigure Actions, Pages and GHCR access.
- Keep the same owner/name if you want the existing index URLs to remain valid.
- Run the first release manually; old release assets are not recreated automatically.

If retaining the current repository:

- Old `1.x` personal-fork releases are not automatically deleted, but the new index deliberately excludes them. They should not outrank the new upstream-aligned `0.3.x` wheels in pip resolution.
- Existing environments may need a fresh virtual environment or an explicit version reinstall; `pip -U` alone does not imply a downgrade from `1.x`.
- Old Git history still contains the former binding source. The clean export/new repository route removes that history; deleting files in an ordinary commit does not erase it.

## Manual compilation rehearsal (no release)

Before running the publisher, use **Test Wheel Build (no release)** to select an upstream version, CPU/AVX2/CUDA backends, Python versions and Linux/Windows targets. It builds even when the version is already published, but does not touch existing releases or run Pages/Docker follow-ups. Default selection: CPU, Python 3.13, both systems.

It reuses the current source preparation and reusable builders. The shared builders now read an optional test platform selection from the prepared manifest; normal release manifests still build both systems. Test plans are marked `test_only`, use `test-` artifact prefixes, and are explicitly refused by release staging/publication.

The test run retains wheels and inspection data as Actions artifacts. Its final report checks the selected job outcomes and validation receipts; partial or failed runs are not labelled successful merely because some wheel files are downloadable. Input patches, reported application results, final affected source files and preparation logs are included for debugging without changing the patch mechanism.

See [Manual test builds](test-builds.md) for the seven inputs, examples and the artifact/download guide. A test does not promote its artifacts: the real release workflow remains a separate action.

## Local development

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q --cov=guanaco --cov=docker --cov-report=term-missing --cov-fail-under=85
python -m ruff check guanaco docker tests
python -m ruff format --check guanaco docker tests
python -m compileall -q guanaco docker

# Read-only discovery (GH_TOKEN is optional locally, useful for API rate limits).
python -m guanaco plan --output work/plan.json

# Download and prepare exactly the selected release; does not publish anything.
python -m guanaco prepare-source --plan work/plan.json --output work/prepared
python -m guanaco unpack-source work/prepared work/source

# Show every value the configuration resolves to.
python -m guanaco explain
```

`make check` runs tests, Ruff lint/format checks, compilation checks and `git diff --check`; `make format` applies the Python formatter. CI tests Python 3.9, 3.13 and 3.14 on Linux/Windows, enforces an 85% line-coverage floor, and validates workflows with actionlint/ShellCheck.

`python -m guanaco publish --plan work/plan.json --preflight` only reads GitHub state. Staging is also a dry run unless `--publish` is supplied. Single-channel publication additionally requires the full-matrix `--gate` artifact. Do not use a local raw `linux_x86_64` CPU build as a substitute for the manylinux-certified CI artifact: release validation intentionally rejects that tag for CPU/AVX2.

Unit tests are offline and do not download models. They exercise release selection, pagination, version/tag mismatch, source pinning, safe extraction, metadata adaptation, archive integrity, wheel identity, incomplete matrices, draft recovery, repeat-run idempotence, index isolation and workflow wiring. They do not replace real Windows/CUDA or model-inference testing.

## Module map

| Module | Responsibility |
|---|---|
| `guanaco/channels.py` | Build channels (`cpu`, `avx2`, `cuNNN`) and target platforms: release tags and wheel platform tags |
| `guanaco/settings.py` | Load and validate `build-matrix.json`; the only place a name or a version is configured |
| `guanaco/models.py` | The documents the pipeline exchanges: plan, manifest, receipt, gate and provenance |
| `guanaco/transfer.py` | Bounded HTTPS downloads and transactional, path-safe extraction |
| `guanaco/github_api.py` | A read-mostly GitHub REST client; writes only when constructed as writable |
| `guanaco/source.py` | Download, patch and package the frozen upstream source |
| `guanaco/toolchain.py` | Turn a plan into compiler flags, job settings and container tags |
| `guanaco/wheels.py` | Validate wheels against a manifest and emit receipts |
| `guanaco/releases.py` | Upstream discovery, release planning, receipt gating and conservative publication |
| `guanaco/catalog.py` | Render the PEP 503 index and safely replace an owned previous site |
| `guanaco/reports.py` | The diagnostics a rehearsal produces: source, wheels and result |
| `guanaco/cli.py` | The single `python -m guanaco` entry point and its argument parser |
| `docker/fetch_release.py` | Resolve and install an exact, checksummed release asset |

Dependencies flow one way: `channels` → `settings` → `models` → the rest, so no
module imports a layer above it.

Nothing the configuration already knows is written twice. `configure cpu` emits
`package` and `artifact`; `configure cuda` emits `package`, `artifact_linux` and
`artifact_windows`; the workflows interpolate those outputs instead of spelling
out a distribution name. Renaming the distribution or forking the repository is
therefore a change to `build-matrix.json` and nothing else, and a unit test
fails if any workflow file ever mentions the package or repository name again.

### Command map

| Command | Replaces |
|---|---|
| `python -m guanaco plan` | `check_upstream.py` |
| `python -m guanaco plan-test` | `plan_test_build.py` |
| `python -m guanaco prepare-source` | `prepare_source.py` |
| `python -m guanaco unpack-source` | `unpack_source.py` |
| `python -m guanaco configure {cpu,cuda,matrix,docker}` | `configure_build.py` |
| `python -m guanaco verify-wheels` | `verify_wheels.py` |
| `python -m guanaco validate-receipts` | `validate_receipts.py` |
| `python -m guanaco publish` | `publish_release.py` |
| `python -m guanaco build-index` | `generate-wheel-index.py` |
| `python -m guanaco inspect {source,wheels,result}` | `inspect_test_build.py` |
| `python -m guanaco explain` | (new) prints the resolved configuration |

Every module above is stdlib-only. Only `prepare-source` needs a third-party
package (`tomlkit`, from `requirements-ci.txt`) because it is the one step that
rewrites `pyproject.toml`; planning, validation, publication and indexing run on
a bare interpreter. Tests use shared synthetic fixtures under `tests/`; their
native headers are intentionally not executable libraries. Actual wheel
installation, inference, Docker execution and GPU/Windows compatibility require
integration testing in the relevant environment.
