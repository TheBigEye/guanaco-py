#!/bin/sh
set -eu
# No source checkout, make or compilation at container startup.
exec python -m llama_cpp.server --host "${HOST:-0.0.0.0}" --port "${PORT:-8000}" "$@"
