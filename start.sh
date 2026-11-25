#!/usr/bin/env bash
# Start DeepSeek-V4.1-Flash-1017 (A4NX routed experts, 2.2 bpw) pipeline-parallel on CMP 170HX cards.
#   ./start.sh          start the container, wait for /health
# Settings: .env (see .env.example) or environment variables. Engine settings are in docker/serve.sh, mounted
# over the image's copy.
set -euo pipefail
cd "$(dirname "$0")"
. ./versions.env
[ -f .env ] && . ./.env

IMAGE=${IMAGE:-${ENGINE_IMAGE:-$ENGINE_IMAGE_LOCAL}}
MODELS_DIR=${MODELS_DIR:-$PWD/models}
MODEL_DIR=${MODEL_DIR:-$MODELS_DIR/DeepSeek-V4.1-Flash-1017-A4NX-2.2bpw}
ENGRAM_DIR=${ENGRAM_DIR:-$MODELS_DIR/DeepSeek-V4.1-Flash-1017-engram}
CACHE_DIR=${CACHE_DIR:-$PWD/cache}
GPUS=${GPUS:-0,1,2}
HOST=${HOST:-127.0.0.1}
PORT=${PORT:-8000}
NAME=${NAME:-cmp170hx-deepseek-v41-flash}
READY_TIMEOUT=${READY_TIMEOUT:-1800}

die() { echo "start.sh: $*" >&2; exit 1; }
[ -f "$MODEL_DIR/config.json" ] || die "no model at $MODEL_DIR (run ./download.sh)"
[ -f "$MODEL_DIR/int8/dense.safetensors" ] || die "no INT8 dense weights at $MODEL_DIR/int8 (run ./download.sh)"
for f in model.safetensors.index.json model-00047-of-00048.safetensors model-00048-of-00048.safetensors; do
  [ -f "$ENGRAM_DIR/$f" ] || die "no Engram file $ENGRAM_DIR/$f (run ./download.sh)"
done
PP_SIZE=$(echo "$GPUS" | tr ',' '\n' | grep -c .)
case "$PP_SIZE" in
  3) LAYER_PARTITION=${LAYER_PARTITION:-15,10,15} ;;   # 64 / 40 / 64 GB: list the 40 GB card second
  2) LAYER_PARTITION=${LAYER_PARTITION:-20,20}
     GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.99} ;;   # two 80 GB cards
  *) die "GPUS must name two or three cards, e.g. 0,2,1" ;;
esac
docker image inspect "$IMAGE" >/dev/null 2>&1 || docker pull "$IMAGE"
mkdir -p "$CACHE_DIR"

docker rm -f "$NAME" >/dev/null 2>&1 || true
docker run -d --name "$NAME" --gpus "\"device=$GPUS\"" -e CUDA_VISIBLE_DEVICES="$GPUS" --shm-size 16g \
  --user "$(id -u):$(id -g)" -e HOME=/cache -e HF_HUB_OFFLINE=1 \
  -e PP_SIZE="$PP_SIZE" -e LAYER_PARTITION="$LAYER_PARTITION" \
  -e GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.97}" \
  -e MAX_MODEL_LEN="${MAX_MODEL_LEN:-262144}" -e MAX_NUM_SEQS="${MAX_NUM_SEQS:-4}" \
  -p "$HOST:$PORT:8000" \
  -v "$MODEL_DIR:/model:ro" \
  -v "$ENGRAM_DIR:/engram:ro" \
  -v "$CACHE_DIR:/cache" \
  -v "$PWD/docker/serve.sh:/opt/cmp170hx/serve.sh:ro" \
  "$IMAGE" >/dev/null

echo "started $NAME ($IMAGE) on $HOST:$PORT; loading (the first start compiles kernels)..."
deadline=$(( $(date +%s) + READY_TIMEOUT ))
until curl -fs "http://$HOST:$PORT/health" >/dev/null 2>&1; do
  if ! docker ps -q -f "name=^$NAME$" | grep -q .; then
    docker logs --tail 50 "$NAME" >&2 || true
    die "container exited during startup"
  fi
  [ "$(date +%s)" -lt "$deadline" ] || die "not ready after ${READY_TIMEOUT}s (docker logs $NAME)"
  sleep 5
done
echo "ready: http://$HOST:$PORT/v1 (model deepseek-v4.1-flash-a4nx-2.2bpw)"
