#!/usr/bin/env bash
# Host-venv runner: same code as the image, no rebuild round-trip.
# Usage: ./scripts/dev.sh frus manifest        (add --gpus via FRUS_GPU=1)
set -euo pipefail
cd "$(dirname "$0")/.."

HUB=/home/haoche_nitro_v15/.cache/huggingface/hub
SNAP=$(cat "$HUB/models--BAAI--bge-m3/refs/main")

GPU_ARGS=()
[[ "${FRUS_GPU:-0}" == "1" ]] && GPU_ARGS=(--gpus all)

exec docker run --rm -i "${GPU_ARGS[@]}" \
  --network host \
  -u "$(id -u):$(id -g)" -e HOME=/tmp \
  -v "$PWD:/workspace" -w /workspace \
  -v "$HUB:/models/hf-hub:ro" \
  -e PATH="/workspace/.venv/bin:/usr/local/bin:/usr/bin:/bin" \
  -e PYTHONPATH=/workspace/src \
  -e FRUS_DATA_DIR=/workspace/data \
  -e FRUS_RAW_DIR=/workspace/data/raw/frus/volumes \
  -e FRUS_LANCEDB_URI=/workspace/data/index/lancedb \
  -e FRUS_REPORTS_DIR=/workspace/reports \
  -e FRUS_BGE_MODEL_PATH="/models/hf-hub/models--BAAI--bge-m3/snapshots/$SNAP" \
  -e FRUS_EMBED_DEVICE="${FRUS_EMBED_DEVICE:-cpu}" \
  -e FRUS_OLLAMA_HOST="${FRUS_OLLAMA_HOST:-http://localhost:11434}" \
  -e PHOENIX_COLLECTOR_ENDPOINT="${PHOENIX_COLLECTOR_ENDPOINT:-http://localhost:6006}" \
  -e PHOENIX_PROJECT="${PHOENIX_PROJECT:-frus-agentic-rag}" \
  -e PHOENIX_DISABLED="${PHOENIX_DISABLED:-}" \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e TOKENIZERS_PARALLELISM=false \
  --entrypoint "" \
  jobshift:latest "$@"
