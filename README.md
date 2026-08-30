<p align="center">
  <img src="https://raw.githubusercontent.com/TheBigEye/guanaco-py/main/docs/icon.svg" style="height: 16rem; width: 16rem">
</p>

# Guanaco-py - Python Bindings for [`llama.cpp`](https://github.com/ggml-org/llama.cpp)

**A simplified, wheels-first fork, kept in sync with [JamePeng/llama-cpp-python](https://github.com/JamePeng/llama-cpp-python)**

[![Forked from abetlen/llama-cpp-python](https://img.shields.io/badge/forked%20from-abetlen/llama--cpp--python-blue)](https://github.com/abetlen/llama-cpp-python)
[![Tests](https://github.com/TheBigEye/guanaco-py/actions/workflows/test.yaml/badge.svg?branch=main)](https://github.com/TheBigEye/guanaco-py/actions/workflows/test.yaml)
[![Github All Releases](https://img.shields.io/github/downloads/TheBigEye/guanaco-py/total.svg?label=Github%20Downloads)]()

The bindings you already know `import llama_cpp`, API-compatible with upstream but **prebuilt and ready to install**, including on **pure-CPU machines**, the one configuration upstream does not publish wheels for. Pick the wheel for your hardware, run `pip install`, done: no compiler, no CMake, no rebuilding llama.cpp from source on every machine and every update.

---

## Why this repository exists
A bit of context on how we got here:

* **[abetlen/llama-cpp-python](https://github.com/abetlen/llama-cpp-python)** - the original project has not had a release since August 2025, while llama.cpp itself moves forward essentially every day.
* **[JamePeng/llama-cpp-python](https://github.com/JamePeng/llama-cpp-python)** is currently the only actively maintained continuation of the bindings. **This repository is updated against that upstream**, so it stays current with modern llama.cpp (new chat templates, GGUF changes, performance work, bug fixes).
* JamePeng's repository, however, ships **no CPU builds**: its releases carry prebuilt wheels for CUDA and other platforms, but if you run on CPU — as every ordinary PC, laptop and shared box does, installing it means having a compiler toolchain on every machine and sitting through a full CMake build on every install or upgrade.

`guanaco-py` exists to simplify exactly that, and deliberately nothing more. This fork's job is:

* **Track JamePeng's upstream**, keeping the bindings current with llama.cpp.
* **Build and publish prebuilt wheels**, and only for two configurations: **CPU and CUDA**. CPU is the gap upstream leaves; CUDA rides along so wheel users can stick to a single, explicit index instead of mixing sources.

> [!NOTE]
> Support is intentionally limited to **CPU and CUDA** builds (no prebuilt Metal, Vulkan, HIP/ROCm, SYCL, RPC, or macOS/ARM wheels). That narrow focus is not a lack of ambition, it is what keeps the builds tested, reliable and publishable on time instead of rotting across a giant untested matrix. The source tree can still build those backends the same way upstream does; there just won't be prebuilt wheels for them here.

> [!IMPORTANT]
> If you need Metal, macOS, or other backends beyond CPU/CUDA, use the upstream repositories directly: [JamePeng/llama-cpp-python](https://github.com/JamePeng/llama-cpp-python) (actively maintained) or [abetlen/llama-cpp-python](https://github.com/abetlen/llama-cpp-python) (the original). Many thanks to **Andrei Betlen** for the original work!

## Installation

Wheels are served from a small [PEP 503 index hosted on GitHub Pages](https://thebigeye.github.io/guanaco-py/whl/) - plain PyPI is not, and will never be, an option (see the last section for why).

### Choosing your wheel

Pick **one** channel, the one matching your hardware, and install:

| Hardware | Channel | Index |
|---|---|---|
| **CPU, portable** (runs on any x86-64) | `cpu` | `https://thebigeye.github.io/guanaco-py/whl/cpu/` |
| **CPU, AVX2** (most CPUs since ~2013) | `avx2` | `https://thebigeye.github.io/guanaco-py/whl/avx2/` |
| **CUDA 12.1 / 12.2 / 12.3 / 12.4** | `cu121` – `cu124` | `https://thebigeye.github.io/guanaco-py/whl/cu121/` … |
| **CUDA 12.6 / 12.8 / 13.1** | `cu126` / `cu128` / `cu131` | `https://thebigeye.github.io/guanaco-py/whl/cu126/` … |

> [!IMPORTANT]
> In all cases, append `--extra-index-url https://pypi.org/simple` so pip can still fetch the pure-Python dependencies (`numpy`, `jinja2`, `diskcache`, ...) from PyPI. The GitHub Pages index only carries `guanaco-py` itself.

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
> The AVX2 wheels require a CPU with **AVX2** support (Intel Haswell+, AMD Excavator+/Zen+). They will crash with an illegal instruction on the first inference if your processor is older. To check beforehand: run `grep -m1 avx2 /proc/cpuinfo` on Linux; on Windows, *Task Manager → Performance → CPU* (or CPU-Z). When in doubt, use the portable `cpu` channel, it always works.

**CUDA:**

```bash
# Example: CUDA 12.4
pip install guanaco-py \
  --index-url https://thebigeye.github.io/guanaco-py/whl/cu124/ \
  --extra-index-url https://pypi.org/simple
```

Where the CUDA version in the URL is one of `cu121`, `cu122`, `cu123`, `cu124`, `cu126`, `cu128` or `cu131` (CUDA 12.1, 12.2, 12.3, 12.4, 12.6, 12.8 and 13.1 respectively). Matching your installed CUDA toolkit's major.minor is what matters; you can check yours with `nvidia-smi`.

> [!NOTE]
> All wheels are built for **x86-64 only**, for both **Windows** and **Linux**. Linux wheels use the `manylinux_2_34_x86_64` policy, i.e. they need glibc ≥ 2.34 (Ubuntu 22.04+, Debian 12+, Fedora 35+, and anything newer). Every supported build covers **CPython 3.9 through 3.14**.

> [!TIP]
> Occasionally pip resolves its own cached indexes and claims there is no matching wheel. Adding `--only-binary=:all:` nudges it to pick the wheel from the custom index:
> `pip install guanaco-py --only-binary=:all: --index-url https://thebigeye.github.io/guanaco-py/whl/cpu/`

**Upgrading:** run the same command with `-U`. Every historical version stays published on the index, so pinned environments keep working.

<details>
<summary>Installing from a release tag (git source)</summary>

If you need a source build (custom `CMAKE_ARGS`, other backends, bleeding edge), install straight from git, this compiles llama.cpp locally, so you need a C compiler and CMake 3.21+:

```bash
pip install -U "guanaco-py @ git+https://github.com/TheBigEye/guanaco-py.git"
```

or pinned to a release, for example `v0.5.0`:

```bash
pip install -U git+https://github.com/TheBigEye/guanaco-py@v0.5.0
```

</details>

<details>
<summary>Installing with uv</summary>

The same indexes work with `uv pip` using `--index-url`, or declared once in `pyproject.toml`:

```toml
[[tool.uv.index]]
name = "guanaco-cpu"
url = "https://thebigeye.github.io/guanaco-py/whl/avx2/"
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

Everything else — plain text completion, chat formats, grammars/JSON mode, embeddings, speculative decoding, function calling, the low-level `ctypes` API, works exactly the same as upstream, because it *is* the same code.

> [!NOTE]
> Documentation is currently shared with the original project: [https://llama-cpp-python.readthedocs.io/en/latest](https://llama-cpp-python.readthedocs.io/en/latest). As this fork diverges further, separate documentation may be created.

## The server module will be removed

> [!CAUTION]
> **Heads up:** the bundled OpenAI-compatible web server (`llama_cpp.server`) will be **removed entirely** in a future release.

The server is a poor fit for a wheels-only distribution: it drags in a whole web-framework dependency stack (`fastapi`, `uvicorn`, `pydantic-settings`, ...) that most callers never use, and it duplicates a job that dedicated servers already do better.

If you serve models over HTTP today, plan accordingly:

* **Pin your current version** while you need the embedded server, or
* **Migrate to [llama.cpp's own server](https://github.com/ggml-org/llama.cpp/tree/master/tools/server)**, which is the better-maintained OpenAI-compatible endpoint and also ships prebuilt binaries.

This notice is the deprecation window: the current release still ships the server module, but it will disappear with a future version bump.

## Why is it not on PyPI (and never will be)?

Not a limitation, a deliberate and permanent choice, for three reasons that cannot be fixed on PyPI's side:

1. **pip cannot pick a backend for you.**
   PyPI has no notion of "this machine has CUDA 12.4" or "this CPU lacks AVX2". A package on PyPI means exactly one default build per platform, so GPU users would silently download the CPU build (hundreds of extra megabytes) and wonder why nothing is fast. Per-backend indexes make that choice **explicit**, and impossible to get wrong by accident.

2. **CUDA wheels do not fit PyPI's limits in practice.**
   PyPI caps file sizes at 100 MB by default, with raises granted case-by-case. A CUDA-version x Python-version wheel matrix is exactly the workload those caps are not sized for, it's the same reason PyTorch, JAX and every serious CUDA project self-host their package indexes. Publishing only the small CPU wheels on PyPI would just recreate the problem as two diverging install paths for the same package.

3. **This index *is* the product.**
   The whole point of `guanaco-py` is that the right wheel for your hardware is one explicit line away, and stays installable forever, because old versions are never pruned. A partial PyPI entry would only add a slower, more confusing second door, and would attract bug reports against a build that does not represent this repository.

> [!TIP]
> If `pip install guanaco-py` fails on your machine, the answer is never "wait for PyPI" - it is "point pip at the index shown in [Installation](#installation)".

## Development

This repository periodically syncs from [JamePeng/llama-cpp-python](https://github.com/JamePeng/llama-cpp-python) and builds the wheel matrix in CI. Issues and PRs about **packaging, wheels and documentation** are welcome here; bugs in the bindings' behavior or in model inference should be reported to [JamePeng's repo](https://github.com/JamePeng/llama-cpp-python) or to [llama.cpp](https://github.com/ggml-org/llama.cpp) directly.

## License & credits

* [MIT](LICENSE.md)
* [llama.cpp](https://github.com/ggml-org/llama.cpp) - the inference engine these bindings wrap, by [@ggerganov](https://github.com/ggerganov) and contributors
* [abetlen/llama-cpp-python](https://github.com/abetlen/llama-cpp-python) - the original project these bindings come from Andrei Betlen
* [JamePeng/llama-cpp-python](https://github.com/JamePeng/llama-cpp-python) - the actively maintained upstream this repository tracks
