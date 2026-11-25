#!/usr/bin/env bash
# Download the checkpoint and the Engram tables at the pinned revisions.
#   ./download.sh       (MODELS_DIR=./models by default; ~167 GB checkpoint + ~203 GB Engram tables)
# The checkpoint repo includes the INT8 dense weights the engine serves (int8/dense.safetensors). The Engram
# tables are DeepSeek's own files (shards 47-48 of deepseek-ai/DeepSeek-V4.1-Flash-1017 and its index); the
# engine reads them from disk while serving, so keep them on a fast SSD.
set -euo pipefail
cd "$(dirname "$0")"
. ./versions.env
[ -f .env ] && . ./.env
IMAGE=${IMAGE:-${ENGINE_IMAGE:-$ENGINE_IMAGE_LOCAL}}
MODELS_DIR=${MODELS_DIR:-$PWD/models}
MODEL_DIR=${MODEL_DIR:-$MODELS_DIR/DeepSeek-V4.1-Flash-1017-A4NX-2.2bpw}
ENGRAM_DIR=${ENGRAM_DIR:-$MODELS_DIR/DeepSeek-V4.1-Flash-1017-engram}
mkdir -p "$MODEL_DIR" "$ENGRAM_DIR"
docker image inspect "$IMAGE" >/dev/null 2>&1 || docker pull "$IMAGE"
run() { docker run --rm --user "$(id -u):$(id -g)" -e HOME=/tmp -e HF_TOKEN="${HF_TOKEN:-}" \
          --entrypoint python3 "$@"; }

run -v "$MODEL_DIR:/model" "$IMAGE" -c "
from huggingface_hub import snapshot_download
snapshot_download('$MODEL_REPO', revision='$MODEL_REVISION', local_dir='/model')
print('checkpoint ready: $MODEL_DIR')"

run -v "$ENGRAM_DIR:/engram" "$IMAGE" -c "
from huggingface_hub import snapshot_download
snapshot_download('$ENGRAM_REPO', revision='$ENGRAM_REVISION', local_dir='/engram',
                  allow_patterns=['model.safetensors.index.json', 'model-00047-of-00048.safetensors',
                                  'model-00048-of-00048.safetensors', 'LICENSE'])
print('Engram tables ready: $ENGRAM_DIR')"
