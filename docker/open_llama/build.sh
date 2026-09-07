#!/bin/sh
set -eu
: "${GUANACO_VERSION:?Set GUANACO_VERSION to a published upstream X.Y.Z}"
ROOT="$(CDPATH='' cd -- "$(dirname -- "$0")/../.." && pwd)"
exec docker build --platform linux/amd64 -f "$ROOT/docker/open_llama/Dockerfile" \
    --build-arg GUANACO_VERSION="$GUANACO_VERSION" -t "guanaco-open-llama:$GUANACO_VERSION" "$ROOT"
