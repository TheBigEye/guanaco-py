# Optional source build: OpenBLAS is not a published wheel channel.
FROM python:3.12-slim-bookworm AS builder
ARG GUANACO_VERSION
ARG GUANACO_REPOSITORY=TheBigEye/guanaco-py
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential libopenblas-dev pkg-config \
    && rm -rf /var/lib/apt/lists/*
RUN python -m pip install --no-cache-dir build cmake ninja
COPY docker/fetch_release.py /tmp/
COPY guanaco /tmp/guanaco/
RUN python /tmp/fetch_release.py source --version "${GUANACO_VERSION}" \
    --repository "${GUANACO_REPOSITORY}" --directory /tmp/prepared \
    && PYTHONPATH=/tmp python -m guanaco unpack-source /tmp/prepared /src --version "${GUANACO_VERSION}" \
    && rm -rf /tmp/guanaco /tmp/fetch_release.py /tmp/__pycache__
ENV CMAKE_BUILD_PARALLEL_LEVEL=2
RUN CMAKE_ARGS="-DGGML_CUDA=OFF -DGGML_NATIVE=OFF -DGGML_BLAS=ON -DGGML_BLAS_VENDOR=OpenBLAS" \
    python -m build --wheel --outdir /wheels /src

FROM python:3.12-slim-bookworm
RUN apt-get update && apt-get install -y --no-install-recommends libopenblas0-pthread libgomp1 \
    && rm -rf /var/lib/apt/lists/*
COPY --from=builder /wheels /tmp/wheels
RUN set -eu; wheel="$(find /tmp/wheels -name '*.whl')"; \
    python -m pip install --no-cache-dir "${wheel}[server]" && rm -rf /tmp/wheels
ENV HOST=0.0.0.0 PORT=8000
EXPOSE 8000
ENTRYPOINT ["python", "-m", "llama_cpp.server"]
