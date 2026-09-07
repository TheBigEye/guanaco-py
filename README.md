<p align="center">
  <img src="https://raw.githubusercontent.com/TheBigEye/guanaco-py/main/docs/icon.svg" style="height: 16rem; width: 16rem">
</p>

# Guanaco-py - Python Bindings for [`llama.cpp`](https://github.com/ggml-org/llama.cpp)

**A personal, wheels-first distribution of [JamePeng/llama-cpp-python](https://github.com/JamePeng/llama-cpp-python), built from upstream releases instead of a maintained copy of the bindings**

[![Forked from abetlen/llama-cpp-python](https://img.shields.io/badge/forked%20from-abetlen/llama--cpp--python-blue)](https://github.com/abetlen/llama-cpp-python)
[![Tests](https://github.com/TheBigEye/guanaco-py/actions/workflows/build-testing.yaml/badge.svg?branch=main)](https://github.com/TheBigEye/guanaco-py/actions/workflows/build-testing.yaml)
[![Github All Releases](https://img.shields.io/github/downloads/TheBigEye/guanaco-py/total.svg?label=Github%20Downloads)](https://github.com/TheBigEye/guanaco-py/releases)

The bindings you already know, still using `import llama_cpp`, but **prebuilt and ready to install**, including dedicated wheels for **pure-CPU machines**. Pick the wheel for your hardware, run `pip install`, done: no compiler, no CMake, no local rebuild on every machine.

---

## Why this repository exists

A bit of context on how we got here:

* **[abetlen/llama-cpp-python](https://github.com/abetlen/llama-cpp-python)** - the original project these bindings come from.
* **[JamePeng/llama-cpp-python](https://github.com/JamePeng/llama-cpp-python)** - the upstream maintained source and release version this distribution follows.
* Guanaco keeps **CPU portable, CPU AVX2 and CUDA channels separate**. JamePeng's CUDA wheels include CPU backends too; the distinction here is a dedicated CPU-only distribution, not an absence of CPU support upstream.

`guanaco-py` is **[@TheBigEye](https://github.com/TheBigEye)'s personal distribution**, shared for anyone who finds it useful. Its job is now deliberately narrow:

* **Watch upstream releases**, rather than manually sync or patch the bindings.
* **Download the release's source ZIP and its pinned submodules**, build the existing CPU/AVX2/CUDA matrix, and package it as `guanaco-py`.
* **Publish channel releases, Docker images and a GitHub Pages wheel index**, keeping the upstream version and release notes.

> [!NOTE]
> There is no `llama_cpp/`, `vendor/llama.cpp`, package `pyproject.toml` or CMake project checked into this repository. The source exists only in temporary build directories and reconstructed source archives attached to releases. This repository itself is **not pip-installable**.

### Versioning & upstream relationship

* **The package version comes from upstream.** If JamePeng releases `X.Y.Z`, these wheels use `guanaco-py==X.Y.Z` and retain `llama_cpp.__version__ == "X.Y.Z"`. No independent Guanaco version bump is needed.
* **Tags identify our build channels.** `vX.Y.Z` is portable CPU, `vX.Y.Z-avx2` is AVX2, and `vX.Y.Z-cu124` is CUDA 12.4. The upstream tag may include a backend, OS and date; it is recorded in the release notes and manifest, not mistaken for a separate package version.
* **Releases, not `main`.** A new upstream backend tag or a commit on `main` does not rebuild an already complete `X.Y.Z`. Fixes that have not reached the selected release are intentionally not included.
* **Distribution metadata changes; binding code does not.** The name, self-referencing extras, package links, license inclusion and native build identity are adapted automatically. The Python source is checked byte-for-byte against the downloaded release. Logger names and other upstream identifiers stay upstream's.

> [!IMPORTANT]
> This is not an official upstream distribution, nor a promise of identical behavior on every backend. Compile options differ. Both distributions install `llama_cpp`: use separate environments when comparing them. `guanaco-py` does not satisfy a dependency explicitly named `llama-cpp-python`.

## How releases are built

Every day at **07:00 Argentina time** (`10:00 UTC`), [Check Upstream and Release](.github/workflows/build-release.yaml):

1. Lists upstream releases, ignores drafts/prereleases, and compares **numeric `X.Y.Z` versions** across their backend tags.
2. Selects the latest stable version, or the explicit version requested in a manual run. The first run builds the latest version, not the entire historical catalog.
3. Checks provenance, assets and Git tags. Complete channels are left alone; a partial family keeps its source, notes and build matrix frozen for retries.
4. Downloads the source ZIP at the **resolved commit SHA**, then resolves and downloads the exact Git submodule commits. It never substitutes `main` for a missing source revision.
5. Adapts packaging metadata once and shares the checksummed source snapshot. Builders check wheel identity, ABI/platform tags, `WHEEL`/`RECORD`, file hashes, native headers, licenses and unchanged Python code.
6. Checks validation receipts for the **entire requested matrix** before any publication. Publishers download one channel each, recheck its binaries, then upload to **draft releases** and verify the uploads before making them public.
7. Explicitly runs the Pages and Docker workflows. Releases created with `GITHUB_TOKEN` do not automatically trigger other release-event workflows.

Each channel includes the original upstream release-note text, provenance, `guanaco-build.json` and `SHA256SUMS`. The CPU release also includes `guanaco-source-X.Y.Z.tar.gz` and `packaging.patch`.

> [!NOTE]
> GitHub may start scheduled jobs late. Schedules run on the default branch and can be disabled after repository inactivity. This removes manual binding synchronization, **not** the occasional need to maintain compilers, dependencies and build workflows when upstream changes its requirements.

See [Automation & maintenance](docs/automation.md) for selection rules, retry behavior, permissions and the first-run checklist.

## Installation

Wheels are served through the [GitHub Pages PEP 503 index](https://thebigeye.github.io/guanaco-py/whl/), backed by GitHub release assets. They are not published to PyPI.

### Choosing your wheel

Pick **one** channel matching your hardware:

| Hardware | Channel | Index |
|---|---|---|
| **CPU, portable** (x86-64 without an AVX2 requirement) | `cpu` | `https://thebigeye.github.io/guanaco-py/whl/cpu/` |
| **CPU, AVX2** | `avx2` | `https://thebigeye.github.io/guanaco-py/whl/avx2/` |
| **CUDA 12.1 / 12.2 / 12.3 / 12.4** | `cu121` – `cu124` | `https://thebigeye.github.io/guanaco-py/whl/cu121/` … |
| **CUDA 12.6 / 12.8 / 13.1** | `cu126` / `cu128` / `cu131` | `https://thebigeye.github.io/guanaco-py/whl/cu126/` … |

> [!IMPORTANT]
> Append `--extra-index-url https://pypi.org/simple` so pip can fetch dependencies (`numpy`, `jinja2`, `diskcache`, ...). The Guanaco index only carries `guanaco-py`. Run the initial release workflow before expecting wheels in a newly created repository.

**CPU (portable):**

```bash
pip install guanaco-py \
  --index-url https://thebigeye.github.io/guanaco-py/whl/cpu/ \
  --extra-index-url https://pypi.org/simple
```

**CPU (AVX2):**

```bash
pip install guanaco-py \
  --index-url https://thebigeye.github.io/guanaco-py/whl/avx2/ \
  --extra-index-url https://pypi.org/simple
```

> [!WARNING]
> AVX2 wheels require AVX2/FMA/F16C/SSE4.2/BMI2 support. Using an incompatible wheel can cause an illegal-instruction crash. When in doubt, choose portable `cpu`.

**CUDA:**

```bash
# Example: CUDA 12.4
pip install guanaco-py \
  --index-url https://thebigeye.github.io/guanaco-py/whl/cu124/ \
  --extra-index-url https://pypi.org/simple
```

Choose a supported CUDA channel and a compatible NVIDIA driver/runtime. `nvidia-smi` reports the driver's CUDA compatibility, not necessarily the locally installed toolkit version.

> [!NOTE]
> The configured matrix targets **Windows and Linux x86-64**, **CPython 3.9–3.14**. CPU/AVX2 Linux wheels use `manylinux_2_34_x86_64` (glibc ≥ 2.34). CUDA Linux wheels are built on Ubuntu 22.04 and tagged `linux_x86_64`; they are **not** advertised as manylinux-certified. No prebuilt Metal, macOS/ARM, Vulkan, ROCm or SYCL wheels are provided here.

**Upgrading:** use the same channel with `-U`. To pin an environment, use `guanaco-py==X.Y.Z` with that channel's index. Completed managed releases remain in the generated index; legacy personal-fork builds are intentionally not mixed into this version line.

<details>
<summary>Moving from the old personal-fork versions</summary>

The old `1.x` versions and upstream's `0.3.x` versions are different numbering schemes. `pip install -U` will not necessarily downgrade an existing environment. Prefer a new virtual environment, or uninstall the old distribution and install an explicit published upstream-aligned version.

Do not keep both `guanaco-py` and `llama-cpp-python` installed in the same environment: they share `llama_cpp`.

</details>

<details>
<summary>Installing from source</summary>

Do **not** use `pip install git+https://github.com/TheBigEye/guanaco-py.git`: this repo now contains build recipes, not the package source.

Download `guanaco-source-X.Y.Z.tar.gz` and `SHA256SUMS` from the **CPU release** `vX.Y.Z`, verify the archive checksum, extract it into an empty directory, and run `pip install .` there. The archive already contains the pinned submodules and adjusted Guanaco metadata. A C/C++ compiler and CMake are still required for a source build.

GitHub's automatically generated "Source code (zip)" for a **Guanaco** tag contains these build recipes; it is not the reconstructed bindings source archive.

</details>

<details>
<summary>Installing with uv</summary>

```toml
[[tool.uv.index]]
name = "guanaco-cpu"
url = "https://thebigeye.github.io/guanaco-py/whl/cpu/"
explicit = true

[tool.uv.sources]
guanaco-py = { index = "guanaco-cpu" }
```

</details>

## Quick start

```python
from llama_cpp import Llama

llm = Llama(
    model_path="path/to/model.gguf",
    n_ctx=4096,
    chat_format="llama-3",  # use the template your model was trained on
)

response = llm.create_chat_completion(
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user",   "content": "Say hello in one short sentence."},
    ]
)
print(response["choices"][0]["message"]["content"])
```

### Documentation & wiki

For the bindings' APIs and features, use **JamePeng's upstream documentation**:

* **[Documentation index](https://github.com/JamePeng/llama-cpp-python/blob/main/docs/wiki/index.md)** - source-aligned guides under `docs/wiki`.
* **[GitHub Wiki](https://github.com/JamePeng/llama-cpp-python/wiki)** - upstream's wiki entry point.
* **[Llama API reference](https://github.com/JamePeng/llama-cpp-python/blob/main/docs/wiki/core/Llama.md)**, **[source-build guide](https://github.com/JamePeng/llama-cpp-python/blob/main/docs/wiki/install.md)** and **[examples](https://github.com/JamePeng/llama-cpp-python/tree/main/examples)**.
* **[Discussions](https://github.com/JamePeng/llama-cpp-python/discussions)** - feature announcements and usage notes.

> [!NOTE]
> Documentation on upstream's `main` may describe features newer than your installed release. For Guanaco's package name, wheel channels and builds, use this README and the matching release manifest. The [original project's documentation](https://llama-cpp-python.readthedocs.io/en/latest/) remains a useful complementary reference.

## Docker & server

Docker support stays. The default CPU and CUDA images install a **version-pinned, checksummed Guanaco release wheel**, not a checkout, and do not compile anything at startup.

The bundled `llama_cpp.server` follows upstream. Guanaco no longer carries a separate plan to remove it: install the `server` extra when you need it. See [Docker instructions](docker/README.md) for CPU, CUDA, OpenBLAS and the retained GGUF convenience image.

## Development

Changes here should focus on **build recipes, packaging, release automation, Docker and the wheel index**. Binding fixes belong upstream; no local runtime patch queue is maintained.

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
python -m ruff check .github/scripts docker tests
python -m ruff format --check .github/scripts docker tests
```

The supported toolchain matrix is in [`.github/build-matrix.json`](.github/build-matrix.json). Offline tests cover the automation on Python 3.9, 3.13 and 3.14 in CI, with lint/format checks and an 85% coverage floor. CPU and AVX2 share one parametrized builder. Wheel jobs validate package contents, and CPU/AVX2 jobs import the installed wheel and call its native API. CUDA jobs validate wheel contents but do not claim GPU inference coverage on GPU-less runners.

## License & credits

* [MIT](LICENSE.md)
* [llama.cpp](https://github.com/ggml-org/llama.cpp) - the inference engine, by [@ggerganov](https://github.com/ggerganov) and contributors
* [abetlen/llama-cpp-python](https://github.com/abetlen/llama-cpp-python) - the original bindings, by Andrei Betlen
* [JamePeng/llama-cpp-python](https://github.com/JamePeng/llama-cpp-python) - the upstream source, versioning and release notes this distribution follows

Upstream copyright and license notices are retained in the source archives and included in the wheels.
