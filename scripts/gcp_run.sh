#!/usr/bin/env bash
# Run the pending experiments on a GCP L4 instance.
#
# Why this shape: the 7.4GB LanceDB index is the only thing that has to leave
# this machine, and a home uplink is the whole cost of the job. It goes to GCS
# once; the VM pulls it at datacentre speed. Everything else the VM fetches
# itself — the BGE checkpoints come from HuggingFace faster than they would
# upload from here, and Ollama installs in seconds.
#
# The VM self-provisions from a startup script and shuts itself down when done,
# so nothing depends on holding an SSH session open. Progress and results land
# in GCS; `./scripts/gcp_run.sh fetch` brings them back.
#
# Usage:
#   ./scripts/gcp_run.sh preflight     # check auth, project, quota — do this first
#   ./scripts/gcp_run.sh upload        # push index + source to GCS (the slow step)
#   ./scripts/gcp_run.sh launch        # create the VM; it runs and self-deletes
#   ./scripts/gcp_run.sh logs          # tail the run
#   ./scripts/gcp_run.sh fetch         # pull reports/ back into this repo
#   ./scripts/gcp_run.sh kill          # delete the VM now
set -euo pipefail
cd "$(dirname "$0")/.."

PROJECT="${GCP_PROJECT:-$(gcloud config get-value project 2>/dev/null | tr -d '\r')}"
BUCKET="${GCP_BUCKET:-gs://${PROJECT}-frus-rag}"
ZONE="${GCP_ZONE:-us-central1-a}"
VM="${GCP_VM:-frus-rag-run}"
# L4 24GB: Ollama and the cross-encoder coexist without the 8GB juggling this
# work needed locally. g2 is the only family that carries it.
MACHINE="${GCP_MACHINE:-g2-standard-8}"
GPU="${GCP_GPU:-nvidia-l4}"
# Spot is ~65% cheaper and the ablation resumes from its own checkpoint, so a
# preemption costs minutes. Set GCP_SPOT=0 for a run that must not be
# interrupted.
SPOT="${GCP_SPOT:-1}"

die() { echo "ERROR: $*" >&2; exit 1; }

cmd_preflight() {
  echo "== account =="; gcloud auth list --filter=status:ACTIVE --format='value(account)'
  [[ -n "$PROJECT" ]] || die "no project set. Run: gcloud config set project YOUR_PROJECT"
  echo "== project == $PROJECT"
  echo "== L4 quota in ${ZONE%-*} =="
  gcloud compute regions describe "${ZONE%-*}" --project "$PROJECT" \
    --format="value(quotas.filter(metric:NVIDIA_L4_GPUS).limit)" 2>/dev/null || true
  echo "== APIs =="
  gcloud services list --enabled --project "$PROJECT" \
    --filter="name:(compute.googleapis.com OR storage.googleapis.com)" \
    --format='value(config.name)'
  echo
  echo "If compute.googleapis.com is missing:"
  echo "  gcloud services enable compute.googleapis.com storage.googleapis.com --project $PROJECT"
  echo "If the L4 quota is 0, request it or set GCP_GPU=nvidia-tesla-t4 GCP_MACHINE=n1-standard-8."
}

cmd_upload() {
  [[ -d data/index/lancedb ]] || die "no local index at data/index/lancedb"
  gcloud storage buckets describe "$BUCKET" --project "$PROJECT" >/dev/null 2>&1 ||
    gcloud storage buckets create "$BUCKET" --project "$PROJECT" --location "${ZONE%-*}"

  # Source goes as a tarball rather than a git clone: it does not depend on the
  # GitHub repo being reachable or public from the VM, and it is a few MB.
  tar czf /tmp/frus-src.tgz \
    --exclude='.git' --exclude='data' --exclude='reports' --exclude='.venv' \
    --exclude='__pycache__' --exclude='*.pyc' .
  gcloud storage cp /tmp/frus-src.tgz "$BUCKET/src/frus-src.tgz"

  echo "uploading index (7.4GB — this is the slow part)"
  gcloud storage rsync -r data/index/lancedb "$BUCKET/data/index/lancedb"
  gcloud storage rsync -r data/processed "$BUCKET/data/processed"
  gcloud storage cp eval/gold_cases.jsonl "$BUCKET/eval/gold_cases.jsonl"
  echo "upload complete"
}

# L4 capacity is regional and moves hour to hour, and spot capacity is scarcer
# than on-demand. Picking one zone by hand turns that into a manual retry loop,
# so sweep instead: every zone on spot first, then the same list on-demand.
# US zones lead because the bucket is created in the region of $ZONE and pulling
# 7.4GB across regions is both slower and billed.
ZONES_DEFAULT="us-central1-a us-central1-b us-central1-c us-east1-c us-east1-d \
us-east4-a us-east4-c us-west1-a us-west1-b us-west4-a europe-west4-a europe-west4-b"
ZONE_STATE=".gcp_zone"

cmd_launch() {
  read -r -a zones <<<"${GCP_ZONES:-$ZONE $ZONES_DEFAULT}"
  local modes=("spot" "ondemand")
  [[ "$SPOT" == "1" ]] || modes=("ondemand")

  for mode in "${modes[@]}"; do
    local spot_args=()
    [[ "$mode" == "spot" ]] && spot_args=(--provisioning-model=SPOT --instance-termination-action=DELETE)
    for z in "${zones[@]}"; do
      echo "== trying $MACHINE + $GPU in $z ($mode) =="
      if try_create "$z" "${spot_args[@]}"; then
        printf '%s\n' "$z" > "$ZONE_STATE"
        echo
        echo "VM $VM created in $z ($mode). It provisions itself, runs, uploads to"
        echo "$BUCKET/reports/, then deletes itself."
        echo "  ./scripts/gcp_run.sh logs"
        return 0
      fi
    done
    echo "-- no capacity in any zone on $mode --"
  done
  die "no capacity anywhere. Try a T4: GCP_GPU=nvidia-tesla-t4 GCP_MACHINE=n1-standard-8 $0 launch"
}

try_create() {
  local z=$1; shift
  local out
  # Capacity and quota failures are expected while sweeping and must not stop
  # the loop; anything else is a real error and is surfaced immediately.
  if out=$(do_create "$z" "$@" 2>&1); then
    return 0
  fi
  if grep -qE "ZONE_RESOURCE_POOL_EXHAUSTED|stockout|does not have enough resources|QUOTA_EXCEEDED|Quota .* exceeded" <<<"$out"; then
    echo "   unavailable here"
    return 1
  fi
  echo "$out" >&2
  die "instance create failed for a reason other than capacity"
}

do_create() {
  local z=$1; shift
  local spot_args=("$@")

  # Deep Learning VM: CUDA and the driver are already in the image, so there is
  # no driver install to fail and no container to build. `common-` rather than
  # `pytorch-`: uv installs torch from uv.lock into its own venv either way, and
  # the pytorch image's preinstalled copy would only be a second version to
  # confuse the picture. Google retires these families, so if creation fails
  # with "resource not found", list what exists now:
  #   gcloud compute images list --project deeplearning-platform-release \
  #     --no-standard-images --format='value(family)' | sort -u
  gcloud compute instances create "$VM" \
    --project "$PROJECT" --zone "$z" \
    --machine-type "$MACHINE" \
    --accelerator "type=$GPU,count=1" \
    --image-family="${GCP_IMAGE_FAMILY:-common-cu129-ubuntu-2204-nvidia-580}" \
    --image-project=deeplearning-platform-release \
    --boot-disk-size=200GB --boot-disk-type=pd-balanced \
    --metadata-from-file=startup-script=<(render_startup) \
    --metadata="install-nvidia-driver=True,bucket=$BUCKET" \
    --scopes=cloud-platform \
    --maintenance-policy=TERMINATE \
    "${spot_args[@]}"
}

# The launch sweep may land anywhere, so the other subcommands read back where.
resolved_zone() {
  if [[ -n "${GCP_ZONE_OVERRIDE:-}" ]]; then echo "$GCP_ZONE_OVERRIDE"
  elif [[ -f "$ZONE_STATE" ]]; then cat "$ZONE_STATE"
  else echo "$ZONE"; fi
}

render_startup() {
  cat <<'STARTUP'
#!/bin/bash
# Runs as root on first boot. Every step logs to /var/log/frus-run.log and is
# mirrored to GCS, so a failure is readable without an SSH session.
set -x
exec > >(tee -a /var/log/frus-run.log) 2>&1

BUCKET=$(curl -sf -H 'Metadata-Flavor: Google' \
  http://metadata.google.internal/computeMetadata/v1/instance/attributes/bucket)
WORK=/opt/frus
push_log() { gcloud storage cp /var/log/frus-run.log "$BUCKET/logs/frus-run.log" || true; }
trap push_log EXIT

mkdir -p "$WORK" && cd "$WORK"
gcloud storage cp "$BUCKET/src/frus-src.tgz" . && tar xzf frus-src.tgz

mkdir -p data/index data/processed eval reports
gcloud storage rsync -r "$BUCKET/data/index/lancedb" data/index/lancedb
gcloud storage rsync -r "$BUCKET/data/processed" data/processed
gcloud storage cp "$BUCKET/eval/gold_cases.jsonl" eval/gold_cases.jsonl

# Python env. uv resolves the locked deps directly; no Docker on the VM.
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="/root/.local/bin:$PATH"
uv sync --frozen || uv sync

# Models from HuggingFace — faster here than uploading from a home connection.
export HF_HOME=/opt/hf
uv run python - <<'PY'
from huggingface_hub import snapshot_download
for repo in ("BAAI/bge-m3", "BAAI/bge-reranker-v2-m3"):
    print("downloading", repo, flush=True)
    snapshot_download(repo)
PY
BGE=$(uv run python -c "from huggingface_hub import snapshot_download as d; print(d('BAAI/bge-m3'))")
RR=$(uv run python -c "from huggingface_hub import snapshot_download as d; print(d('BAAI/bge-reranker-v2-m3'))")

curl -fsSL https://ollama.com/install.sh | sh
systemctl start ollama && sleep 10
ollama pull qwen3:4b-instruct

export FRUS_DATA_DIR=$WORK/data
export FRUS_LANCEDB_URI=$WORK/data/index/lancedb
export FRUS_REPORTS_DIR=$WORK/reports
export FRUS_BGE_MODEL_PATH="$BGE"
export FRUS_RERANKER_MODEL_PATH="$RR"
export FRUS_RERANKER_DEVICE=cuda
export FRUS_EMBED_DEVICE=cuda
export PYTHONPATH=$WORK/src
export PYTHONUNBUFFERED=1
# No Phoenix on the VM: its exporter was inflating latency up to 19x, and
# latency is one of the things this run has to measure.
export PHOENIX_DISABLED=1
unset PHOENIX_COLLECTOR_ENDPOINT

sync_reports() { gcloud storage rsync -r "$WORK/reports" "$BUCKET/reports" || true; }

# 1. Score distribution over the full gold set. Retrieval only, no LLM, minutes.
#    Everything downstream depends on these numbers, so it runs first.
uv run python scripts/score_distribution.py
sync_reports

# 2. The three filtering strategies from docs/HANDOVER.md 3.2, fixed top-k
#    first as the baseline the adaptive rules have to beat.
run_variant() {  # name, then FRUS_* assignments
  local name=$1; shift
  ( export "$@"; uv run frus eval --no-resume ) 2>&1 | tail -40
  cp reports/agent_ablation.json "reports/ablation_${name}.json" 2>/dev/null || true
  cp reports/ablation_runs.jsonl "reports/runs_${name}.jsonl" 2>/dev/null || true
  rm -f reports/ablation_runs.jsonl
  sync_reports
}
run_variant A_fixed_topk   FRUS_SCORE_MIN_PER_HOP=5 FRUS_SCORE_MAX_PER_HOP=5
run_variant B_min_margin   FRUS_SCORE_MIN_PER_HOP=3 FRUS_SCORE_KEEP_MARGIN=3.0
run_variant C_min_margin_max FRUS_SCORE_MIN_PER_HOP=3 FRUS_SCORE_KEEP_MARGIN=3.0 FRUS_SCORE_MAX_PER_HOP=8

# 3. Sentence focus is currently applied to every system, so it is only
#    measurable as a paired run against itself with focus off.
run_variant C_focus_off FRUS_SCORE_MIN_PER_HOP=3 FRUS_SCORE_KEEP_MARGIN=3.0 \
  FRUS_SCORE_MAX_PER_HOP=8 FRUS_FOCUS_SENTENCES=false

echo "ALL RUNS COMPLETE"
sync_reports
push_log
# Self-delete so a finished job stops costing money.
NAME=$(curl -sf -H 'Metadata-Flavor: Google' http://metadata.google.internal/computeMetadata/v1/instance/name)
ZONE=$(curl -sf -H 'Metadata-Flavor: Google' http://metadata.google.internal/computeMetadata/v1/instance/zone | awk -F/ '{print $NF}')
gcloud compute instances delete "$NAME" --zone "$ZONE" --quiet
STARTUP
}

cmd_logs() {
  local z; z=$(resolved_zone)
  echo "-- serial console (provisioning), zone $z --"
  gcloud compute instances get-serial-port-output "$VM" --zone "$z" --project "$PROJECT" 2>/dev/null | tail -30 || true
  echo "-- run log from GCS --"
  gcloud storage cat "$BUCKET/logs/frus-run.log" 2>/dev/null | tail -40 || echo "(not uploaded yet)"
}

cmd_fetch() {
  gcloud storage rsync -r "$BUCKET/reports" reports
  echo "results in reports/"
  ls -la reports/ | grep -E "ablation_|score_dist" || true
}

cmd_kill() {
  gcloud compute instances delete "$VM" --zone "$(resolved_zone)" --project "$PROJECT" --quiet
  rm -f "$ZONE_STATE"
}

case "${1:-}" in
  preflight|upload|launch|logs|fetch|kill) "cmd_$1" ;;
  *) sed -n '3,22p' "$0"; exit 1 ;;
esac
