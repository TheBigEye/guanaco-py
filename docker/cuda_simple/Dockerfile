ARG CUDA_IMAGE=12.8.1-runtime-ubuntu22.04
FROM nvidia/cuda:${CUDA_IMAGE}

ARG GUANACO_VERSION
ARG GUANACO_CHANNEL=cu128
ARG GUANACO_REPOSITORY=TheBigEye/guanaco-py

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip ca-certificates libgomp1 \
    && rm -rf /var/lib/apt/lists/*
COPY docker/fetch_release.py .github/scripts/download_utils.py /tmp/
RUN python3 /tmp/fetch_release.py wheel --version "${GUANACO_VERSION}" \
    --channel "${GUANACO_CHANNEL}" --repository "${GUANACO_REPOSITORY}" \
    && rm -rf /tmp/guanaco-release /tmp/fetch_release.py /tmp/download_utils.py /tmp/__pycache__

ENV HOST=0.0.0.0 PORT=8000
WORKDIR /app
EXPOSE 8000
ENTRYPOINT ["python3", "-m", "llama_cpp.server"]
