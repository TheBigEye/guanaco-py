#!/bin/sh
set -eu
: "${GUANACO_VERSION:?Set GUANACO_VERSION}"
: "${MODEL_DIR:?Set MODEL_DIR to a directory containing model.gguf}"
exec docker run --rm --platform linux/amd64 -p 8000:8000 \
    -v "$MODEL_DIR:/models:ro" "guanaco-open-llama:$GUANACO_VERSION"
