#!/bin/sh
set -eu
exec python -m llama_cpp.server --host "${HOST:-0.0.0.0}" --port "${PORT:-8000}" \
    --model "${MODEL:-/models/model.gguf}" --n_gpu_layers "${N_GPU_LAYERS:-0}" "$@"
