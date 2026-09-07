ARG IMAGE=python:3.12-slim-bookworm
FROM ${IMAGE}

ARG GUANACO_VERSION
ARG GUANACO_CHANNEL=cpu
ARG GUANACO_REPOSITORY=TheBigEye/guanaco-py

RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*
COPY docker/fetch_release.py .github/scripts/download_utils.py /tmp/
RUN python /tmp/fetch_release.py wheel --version "${GUANACO_VERSION}" \
    --channel "${GUANACO_CHANNEL}" --repository "${GUANACO_REPOSITORY}" \
    && rm -rf /tmp/guanaco-release /tmp/fetch_release.py /tmp/download_utils.py /tmp/__pycache__

WORKDIR /app
COPY docker/simple/run.sh /app/run.sh
ENV HOST=0.0.0.0 PORT=8000
EXPOSE 8000
ENTRYPOINT ["/bin/sh", "/app/run.sh"]
