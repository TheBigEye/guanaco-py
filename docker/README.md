# Docker

Docker support is retained, but the repository no longer contains a package source checkout. Images use a **published Guanaco release**, and a version is required explicitly.

The examples below use `0.3.49`; substitute a version already published by **your Guanaco repository**. The first upstream-release build must finish before these images can fetch its assets.

## CPU (portable or AVX2)

Run from the repository root:

```bash
docker build --platform linux/amd64 \
  -f docker/simple/Dockerfile \
  --build-arg GUANACO_VERSION=0.3.49 \
  --build-arg GUANACO_CHANNEL=cpu \
  -t guanaco-py:0.3.49 .

docker run --rm -p 8000:8000 \
  -v /absolute/path/to/models:/models:ro \
  guanaco-py:0.3.49 --model /models/model.gguf
```

Use `GUANACO_CHANNEL=avx2` only on a CPU supporting that instruction set. The default image uses Python 3.12 on Debian Bookworm; override `IMAGE` only with a compatible base and a Python version for which the selected release has a wheel.

The entrypoint directly runs `python -m llama_cpp.server`. There is no `make build`, compiler invocation or implicit source download at startup.

## CUDA

Keep the image's CUDA runtime and wheel channel compatible:

```bash
docker build --platform linux/amd64 \
  -f docker/cuda_simple/Dockerfile \
  --build-arg GUANACO_VERSION=0.3.49 \
  --build-arg GUANACO_CHANNEL=cu128 \
  --build-arg CUDA_IMAGE=12.8.1-runtime-ubuntu22.04 \
  -t guanaco-py:0.3.49-cu128 .

docker run --rm --gpus all -p 8000:8000 \
  -v /absolute/path/to/models:/models:ro \
  guanaco-py:0.3.49-cu128 \
  --model /models/model.gguf --n_gpu_layers -1
```

Requires a compatible NVIDIA driver and NVIDIA Container Toolkit on the host. Ubuntu 22.04 supplies Python 3.10 for this image. The Dockerfile installs a checksummed CUDA wheel; it does not compile CUDA locally.

## OpenBLAS source-build image

OpenBLAS remains an optional Docker configuration, **not a published wheel channel**:

```bash
docker build --platform linux/amd64 \
  -f docker/openblas_simple/Dockerfile \
  --build-arg GUANACO_VERSION=0.3.49 \
  -t guanaco-py:0.3.49-openblas .
```

The builder downloads `guanaco-source-0.3.49.tar.gz` and its manifest/checksums from the CPU release. It compiles that pinned, reconstructed source with OpenBLAS. The final image contains the wheel and runtime dependencies, not the source checkout or compiler toolchain. Invoke it with `--model /models/model.gguf` and a model volume as in the CPU example.

## Retained OpenLlama/GGUF convenience image

`docker/open_llama/` is retained, but obsolete interactive `.bin`/GGML downloads and baked-in models are replaced by an explicit GGUF mount:

```bash
GUANACO_VERSION=0.3.49 sh docker/open_llama/build.sh
GUANACO_VERSION=0.3.49 MODEL_DIR=/absolute/path/to/models \
  sh docker/open_llama/start.sh
```

Place a compatible model at `MODEL_DIR/model.gguf`, or run the image yourself with a different `MODEL` environment variable. This convenience image defaults to CPU; use `cuda_simple` for GPU serving.

The optional `hug_model.py` helper requires `huggingface-hub` and an explicit repository/filename. It downloads only when invoked by you, never as part of an image build. For reproducibility, provide a pinned Hugging Face revision:

```bash
python docker/open_llama/hug_model.py OWNER/MODEL MODEL.gguf \
  --revision COMMIT_SHA --directory models
```

## Distribution provenance

All Dockerfiles accept `GUANACO_REPOSITORY` (default `TheBigEye/guanaco-py`). `fetch_release.py` constructs the requested channel tag, downloads `SHA256SUMS`, fetches the exact Python/platform asset, and verifies it before installation. There is no fallback to PyPI's `llama-cpp-python` or to upstream `main`.

The Dockerfiles copy the same `download_utils.py` used by CI; the OpenBLAS builder also copies the shared archive extractor. HTTPS reads have bounded retries and size limits. Downloads are verified in a temporary file, so a checksum failure does not replace an existing good file with corrupt data.

The `server` extra and its dependencies follow the selected upstream release. Python dependencies still come from PyPI; pinning the Guanaco wheel is not a lockfile for the complete container environment.

## Automatic GHCR image

After channel releases are published, the main workflow explicitly calls **Build Docker Image**. It builds the CPU image for `linux/amd64` and pushes:

```text
ghcr.io/thebigeye/guanaco-py:vX.Y.Z
ghcr.io/thebigeye/guanaco-py:latest
```

`latest` is updated only when requested by the orchestrator for the newest stable upstream version, or explicitly enabled in a manual Docker run. For another repository owner, the image name follows `GITHUB_REPOSITORY` in lowercase.

The workflow needs `packages: write` and access to an existing GHCR package of that name. If you recreate the GitHub repository, recheck that package's Actions access settings.
