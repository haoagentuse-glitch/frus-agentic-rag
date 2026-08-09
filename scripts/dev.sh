#!/usr/bin/env bash
# Host-venv runner: same code as the image, no rebuild round-trip.
# The image supplies the OS and the CUDA plumbing; PATH below puts the host
# venv first, so a source edit needs no rebuild.
# Usage: ./scripts/dev.sh frus manifest        (add --gpus via FRUS_GPU=1)
set -euo pipefail
cd "$(dirname "$0")/.."

HUB=/home/haoche_nitro_v15/.cache/huggingface/hub
SNAP=$(cat "$HUB/models--BAAI--bge-m3/refs/main")

# Reranker is optional: an unset path disables cross-encoder reranking and the
# pipeline falls back to RRF order.
RERANK=""
RR_REF="$HUB/models--BAAI--bge-reranker-v2-m3/refs/main"
if [[ -f "$RR_REF" ]]; then
  RERANK="/models/hf-hub/models--BAAI--bge-reranker-v2-m3/snapshots/$(cat "$RR_REF")"
fi

GPU_ARGS=()
[[ "${FRUS_GPU:-0}" == "1" ]] && GPU_ARGS=(--gpus all)

# Any other FRUS_* the caller exported comes along, so a single setting can be
# flipped for one run without editing this file — `FRUS_FOCUS_SENTENCES=false
# ./scripts/dev.sh frus eval` is the paired half of an ablation. Listed first on
# purpose: docker keeps the last -e for a name, so the container paths below
# still win and no override can point the run at the wrong index.
PASSTHROUGH=()
while IFS='=' read -r name _; do
  [[ -n "$name" ]] && PASSTHROUGH+=(-e "$name")
done < <(env | grep -E '^FRUS_' || true)

exec docker run --rm -i "${GPU_ARGS[@]}" \
  "${PASSTHROUGH[@]}" \
  --network host \
  -u "$(id -u):$(id -g)" -e HOME=/tmp \
  -v "$PWD:/workspace" -w /workspace \
  -v "$HUB:/models/hf-hub:ro" \
  -e PATH="/workspace/.venv/bin:/usr/local/bin:/usr/bin:/bin" \
  -e PYTHONPATH=/workspace/src \
  -e PYTHONUNBUFFERED=1 \
  -e FRUS_DATA_DIR=/workspace/data \
  -e FRUS_RAW_DIR=/workspace/data/raw/frus/volumes \
  -e FRUS_LANCEDB_URI=/workspace/data/index/lancedb \
  -e FRUS_REPORTS_DIR=/workspace/reports \
  -e FRUS_BGE_MODEL_PATH="/models/hf-hub/models--BAAI--bge-m3/snapshots/$SNAP" \
  -e FRUS_RERANKER_MODEL_PATH="${FRUS_RERANKER_MODEL_PATH-$RERANK}" \
  -e FRUS_RERANKER_DEVICE="${FRUS_RERANKER_DEVICE:-cpu}" \
  -e FRUS_EMBED_DEVICE="${FRUS_EMBED_DEVICE:-cpu}" \
  -e FRUS_OLLAMA_HOST="${FRUS_OLLAMA_HOST:-http://localhost:11434}" \
  -e PHOENIX_COLLECTOR_ENDPOINT="${PHOENIX_COLLECTOR_ENDPOINT:-http://localhost:6006}" \
  -e PHOENIX_PROJECT="${PHOENIX_PROJECT:-frus-agentic-rag}" \
  -e PHOENIX_DISABLED="${PHOENIX_DISABLED:-}" \
  -e DEEPSEEK_API_KEY="${DEEPSEEK_API_KEY:-$(grep -m1 '^DEEPSEEK_API_KEY=' .env 2>/dev/null | cut -d= -f2- | tr -d '\r')}" \
  -e DEEPSEEK_MODEL="${DEEPSEEK_MODEL:-}" \
  -e GEMINI_API_KEY="${GEMINI_API_KEY:-$(grep -m1 '^GEMINI_API_KEY=' .env 2>/dev/null | cut -d= -f2- | tr -d '\r')}" \
  -e GEMINI_MODEL="${GEMINI_MODEL:-$(grep -m1 '^GEMINI_MODEL=' .env 2>/dev/null | cut -d= -f2- | tr -d '\r')}" \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e TOKENIZERS_PARALLELISM=false \
  --entrypoint "" \
  "${FRUS_IMAGE:-frus-agentic-rag:latest}" "$@"
