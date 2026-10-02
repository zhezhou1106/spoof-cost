#!/usr/bin/env bash
# Portable paper-pipeline runner. No Slurm.
#
#   ./scripts/run.sh e0
#   ./scripts/run.sh e1
#   SEED=0 MONITOR=scl ./scripts/run.sh e2
#   SEED=0 MONITOR=e0 ./scripts/run.sh e4
#   SEED=0 MONITOR=scl ./scripts/run.sh epsi
#   SEED=0 ./scripts/run.sh e5a
#   SEED=0 ./scripts/run.sh e5
#   ./scripts/run.sh aggregate
#
# MODE=gpu (default) starts vLLM on GPU 1 and trains/evals on GPU 0.
# MODE=dry uses FakeGenerator and needs no CUDA.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
export LD_PRELOAD="${LD_PRELOAD:-}"
export PATH="${PATH:-}"
if [[ -n "${CUDA_HOME:-}" ]]; then
  export PATH="${CUDA_HOME}/bin:${PATH}"
  export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}"
fi

export PYTHONUNBUFFERED=1
export TQDM_MININTERVAL="${TQDM_MININTERVAL:-10}"
export TOKENIZERS_PARALLELISM=false
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"

export PROJECT_NAME="${PROJECT_NAME:-Spoof Cost}"
export WANDB_PROJECT="${WANDB_PROJECT:-spoof-cost}"
export SEED="${SEED:-0}"
export DATA_SEED="${DATA_SEED:-0}"

E0_ROOT="${E0_ROOT:-results/e0}"
E1_ROOT="${E1_ROOT:-results/e1}"
E1_PROBE="${E1_PROBE:-${E1_ROOT}/indomain_probe/probe.json}"
E1_METRICS="${E1_METRICS:-${E1_ROOT}/metrics.json}"
SEED_ROOT="${SEED_ROOT:-results/seed_${SEED}}"

export HF_HOME="${HF_HOME:-$ROOT/artifacts/hf_cache}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"

if [[ -f "$ROOT/.env" ]]; then
  # shellcheck disable=SC1091
  source "$ROOT/.env"
fi

export PROJECT_NAME="${PROJECT_NAME:-Spoof Cost}"
export WANDB_PROJECT="${WANDB_PROJECT:-spoof-cost}"
export SEED="${SEED:-0}"
export DATA_SEED="${DATA_SEED:-0}"
SEED_ROOT="${SEED_ROOT:-results/seed_${SEED}}"
mkdir -p "$HF_HOME" artifacts logs results "$SEED_ROOT"

MODE="${MODE:-gpu}"
VLLM_PID=""
VLLM_GPU="${VLLM_GPU:-1}"
TRAIN_GPU="${TRAIN_GPU:-0}"
VLLM_MODEL="${VLLM_MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
VLLM_HOST="${VLLM_HOST:-127.0.0.1}"
export VLLM_HOST

_pick_free_vllm_port() {
  if [[ -n "${VLLM_PORT:-}" ]]; then
    echo "${VLLM_PORT}"
    return 0
  fi
  export _VLLM_BIND_HOST="${VLLM_HOST}"
  uv run python - <<'PY'
import hashlib
import os
import socket

host = os.environ.get("_VLLM_BIND_HOST", "127.0.0.1")
job = str(os.getpid())
node = os.environ.get("HOSTNAME", "local")
digest = int(hashlib.sha256(f"{node}:{job}:spoof-cost-vllm".encode()).hexdigest()[:8], 16)
base = 51337 + (digest % 12000)
for port in range(base, base + 400):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        sock.bind((host, port))
    except OSError:
        continue
    finally:
        sock.close()
    print(port)
    break
else:
    raise SystemExit("no free vLLM port in scan range")
PY
}

start_vllm() {
  if [[ "$MODE" != "gpu" ]]; then
    return 0
  fi
  VLLM_PORT="$(_pick_free_vllm_port)"
  export VLLM_PORT
  mkdir -p "${ROOT}/logs"
  local log="${ROOT}/logs/vllm-${VLLM_PORT}.out"
  echo "Starting vLLM on ${VLLM_HOST}:${VLLM_PORT} GPU ${VLLM_GPU} → ${log}"
  CUDA_VISIBLE_DEVICES="${VLLM_GPU}" env -u VLLM_PORT uv run python -m afterburner.generation.vllm.server \
    --model "${VLLM_MODEL}" \
    --host "${VLLM_HOST}" \
    --port "${VLLM_PORT}" \
    --max-lora-rank 128 \
    --gpu-memory-utilization "${VLLM_GPU_MEM_UTIL:-0.90}" \
    --tensor-parallel-size 1 \
    >"$log" 2>&1 &
  VLLM_PID=$!
  export CUDA_VISIBLE_DEVICES="${TRAIN_GPU}"
  local url="http://${VLLM_HOST}:${VLLM_PORT}/health"
  local max_iters="${VLLM_WAIT_ITERS:-360}"
  local i
  for i in $(seq 1 "${max_iters}"); do
    if curl -sf "$url" >/dev/null 2>&1; then
      echo "vLLM ready (pid=${VLLM_PID}) after $((i * 5))s"
      return 0
    fi
    if ! kill -0 "${VLLM_PID}" 2>/dev/null; then
      echo "vLLM exited early; see ${log}" >&2
      return 1
    fi
    if (( i % 12 == 0 )); then
      echo "still waiting for vLLM… $((i * 5))s / $((max_iters * 5))s"
    fi
    sleep 5
  done
  echo "vLLM failed to become ready within $((max_iters * 5))s; see ${log}" >&2
  return 1
}

stop_vllm() {
  if [[ -n "${VLLM_PID}" ]] && kill -0 "${VLLM_PID}" 2>/dev/null; then
    echo "Stopping vLLM pid=${VLLM_PID}"
    kill "${VLLM_PID}" 2>/dev/null || true
    wait "${VLLM_PID}" 2>/dev/null || true
  fi
}

trap stop_vllm EXIT

build_overrides() {
  local extras=("$@")
  OVERRIDES=()
  if [[ "$MODE" == "gpu" ]]; then
    OVERRIDES+=("dry_run=false")
  else
    OVERRIDES+=("dry_run=true")
  fi
  OVERRIDES+=("seed=${SEED}")
  OVERRIDES+=("data_seed=${DATA_SEED}")
  if [[ -n "${SEEDS:-}" ]]; then
    OVERRIDES+=("seeds=[${SEEDS}]")
  else
    OVERRIDES+=("seeds=[${SEED}]")
  fi
  if [[ ${#extras[@]} -gt 0 ]]; then
    OVERRIDES+=("${extras[@]}")
  fi
  if [[ -n "${EXTRA_OVERRIDES:-}" ]]; then
    # shellcheck disable=SC2206
    OVERRIDES+=($EXTRA_OVERRIDES)
  fi
}

log_header() {
  echo "=============================================="
  echo "${PROJECT_NAME} stage: ${1:-unknown}"
  echo "host=$(hostname) MODE=$MODE SEED=$SEED DATA_SEED=$DATA_SEED"
  echo "ROOT=$ROOT SEED_ROOT=$SEED_ROOT VLLM=${VLLM_HOST}:${VLLM_PORT:-}"
  echo "HF_HOME=$HF_HOME WANDB_PROJECT=${WANDB_PROJECT}"
  echo "date=$(date -Is)"
  echo "=============================================="
  if [[ "$MODE" == "gpu" ]]; then
    nvidia-smi || true
  fi
}

run_uv() {
  echo "+ uv run python $*"
  uv run python "$@"
}

archive_e0_responses() {
  local out_dir="$1"
  uv run python - <<PY
from pathlib import Path
import shutil

out = Path("${out_dir}")
default = out / "default"
if not default.is_dir():
    print("[e0] WARNING: no default/ under", out, "; skip responses archive")
    raise SystemExit(0)
cands = sorted(default.glob("trigger_none_model_*"), key=lambda p: p.stat().st_mtime, reverse=True)
if not cands:
    print("[e0] WARNING: no trigger_none_model_* under", default, "; skip responses archive")
    raise SystemExit(0)
responses = cands[0] / "checkpoints" / "responses"
archive = out / "responses_archive"
files = sorted(responses.glob("step_*.jsonl")) if responses.is_dir() else []
if not files:
    print("[e0] WARNING: no step_*.jsonl under", responses, "; skip responses archive")
    raise SystemExit(0)
archive.mkdir(parents=True, exist_ok=True)
for src in files:
    dest = archive / src.name
    if not dest.exists():
        shutil.copy2(src, dest)
print(f"[e0] archived {len(files)} response files → {archive}")
PY
}

stage_e0() {
  export EXPERIMENT_NAME="track_a/e0"
  export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
  export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
  export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
  export VLLM_GPU_MEM_UTIL="${VLLM_GPU_MEM_UTIL:-0.90}"
  log_header "e0"
  start_vllm
  local out_dir="${OUT_DIR:-${E0_ROOT}}"
  build_overrides \
    "monitor=none" \
    "beta=1.0e-4" \
    "alpha=0.0" \
    "output_dir=${out_dir}" \
    "wandb_run_name_prefix=e0" \
    "reference_baseline_path=${out_dir}/reference_baseline.json"
  run_uv scripts/run_e0.py "${OVERRIDES[@]}"
  archive_e0_responses "${out_dir}"
}

stage_e1() {
  export EXPERIMENT_NAME="track_a/e1"
  export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
  export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
  export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
  export VLLM_GPU_MEM_UTIL="${VLLM_GPU_MEM_UTIL:-0.90}"
  log_header "e1"
  start_vllm
  local out_dir="${OUT_DIR:-${E1_ROOT}}"
  local ckpt_dir="${CKPT_DIR:-${E0_ROOT}}"
  build_overrides \
    "checkpoint_dir=${ckpt_dir}" \
    "output_dir=${out_dir}" \
    "reference_baseline_path=${ckpt_dir}/reference_baseline.json" \
    "wandb_run_name_prefix=e1"
  run_uv scripts/run_e1.py "${OVERRIDES[@]}"
}

stage_e2() {
  export EXPERIMENT_NAME="track_a/e2"
  export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
  export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
  export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
  export VLLM_GPU_MEM_UTIL="${VLLM_GPU_MEM_UTIL:-0.90}"
  local monitor="${MONITOR:-scl}"
  local alpha="${ALPHA:-1.0}"
  local out_dir="${OUT_DIR:-${SEED_ROOT}/e2_${monitor}}"
  local act_cache="${INDOMAIN_ACTIVATION_CACHE:-artifacts/indomain_probe/activations_seed_${SEED}_${monitor}}"
  log_header "e2 monitor=${monitor}"
  start_vllm
  build_overrides \
    "monitor=${monitor}" \
    "beta=1.0e-4" \
    "alpha=${alpha}" \
    "output_dir=${out_dir}" \
    "wandb_run_name_prefix=e2-s${SEED}-${monitor}" \
    "reference_baseline_path=" \
    "indomain_probe_path=${INDOMAIN_PROBE_PATH:-${E1_PROBE}}" \
    "indomain_activation_cache=${act_cache}"
  run_uv scripts/run_e2.py "${OVERRIDES[@]}"
}

stage_e4() {
  export EXPERIMENT_NAME="track_a/e4"
  export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
  export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
  export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
  export VLLM_GPU_MEM_UTIL="${VLLM_GPU_MEM_UTIL:-0.90}"
  local monitor="${MONITOR:-scl}"
  local ckpt_dir out_dir
  if [[ "${monitor}" == "e0" || "${monitor}" == "none" ]]; then
    ckpt_dir="${CKPT_DIR:-${E0_ROOT}}"
    out_dir="${OUT_DIR:-results/e4_e0}"
  else
    ckpt_dir="${CKPT_DIR:-${SEED_ROOT}/e2_${monitor}}"
    out_dir="${OUT_DIR:-${SEED_ROOT}/e4_${monitor}}"
  fi
  log_header "e4 ckpt=${ckpt_dir}"
  start_vllm
  local override_args=(
    "checkpoint_dir=${ckpt_dir}"
    "prefix_fraction=${PREFIX_FRACTION:-0.3}"
    "ablation_middle_fraction=${ABLATION_MIDDLE:-0.4}"
    "num_prompts=${NUM_PROMPTS:-128}"
    "output_dir=${out_dir}"
  )
  if [[ -n "${LORA_PATH:-}" ]]; then
    override_args+=("lora_path=${LORA_PATH}")
  fi
  build_overrides "${override_args[@]}"
  run_uv scripts/run_e4.py "${OVERRIDES[@]}"
}

stage_epsi() {
  export EXPERIMENT_NAME="track_a/endpoint_psi"
  export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
  export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
  export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
  export VLLM_GPU_MEM_UTIL="${VLLM_GPU_MEM_UTIL:-0.90}"
  local monitor="${MONITOR:-scl}"
  local ckpt_dir out_dir
  if [[ "${monitor}" == "e0" || "${monitor}" == "none" ]]; then
    ckpt_dir="${CKPT_DIR:-${E0_ROOT}}"
    out_dir="${OUT_DIR:-results/endpoint_psi_e0}"
  else
    ckpt_dir="${CKPT_DIR:-${SEED_ROOT}/e2_${monitor}}"
    out_dir="${OUT_DIR:-${SEED_ROOT}/e2_${monitor}/endpoint_psi}"
  fi
  log_header "endpoint_psi ckpt=${ckpt_dir}"
  start_vllm
  local override_args=(
    "checkpoint_dir=${ckpt_dir}"
    "num_prompts=${NUM_PROMPTS:-128}"
    "output_dir=${out_dir}"
  )
  if [[ -n "${LORA_PATH:-}" ]]; then
    override_args+=("lora_path=${LORA_PATH}")
  fi
  build_overrides "${override_args[@]}"
  run_uv scripts/run_endpoint_psi.py "${OVERRIDES[@]}"
}

stage_e5a() {
  export EXPERIMENT_NAME="track_a/e5_probe_scores"
  export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
  export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
  export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
  if [[ "$MODE" == "gpu" ]]; then
    export CUDA_VISIBLE_DEVICES="${TRAIN_GPU}"
  fi
  local out_dir="${OUT_DIR:-${SEED_ROOT}/e5_probe_scores}"
  local cells="${E0_ROOT},${SEED_ROOT}/e2_probe,${SEED_ROOT}/e2_scl,${SEED_ROOT}/e2_cprobe"
  log_header "e5a"
  local override_args=(
    "cells=[${cells}]"
    "probe_path=${PROBE_PATH:-${E1_PROBE}}"
    "batch_size=${BATCH_SIZE:-4}"
    "output_dir=${out_dir}"
  )
  if [[ -n "${REFERENCE_LORA_PATH:-}" ]]; then
    override_args+=("reference_lora_path=${REFERENCE_LORA_PATH}")
  fi
  build_overrides "${override_args[@]}"
  run_uv scripts/run_e5_probe_scores.py "${OVERRIDES[@]}"
}

stage_e5() {
  export EXPERIMENT_NAME="track_a/e5"
  local out_dir="${OUT_DIR:-${SEED_ROOT}/e5}"
  local cells="${E0_ROOT},${SEED_ROOT}/e2_probe,${SEED_ROOT}/e2_scl,${SEED_ROOT}/e2_cprobe"
  log_header "e5"
  build_overrides \
    "cells=[${cells}]" \
    "output_dir=${out_dir}" \
    "e1_metrics_path=${E1_METRICS}" \
    "probe_scores_path=${SEED_ROOT}/e5_probe_scores/metrics.json"
  run_uv scripts/run_e5.py "${OVERRIDES[@]}"
}

stage_aggregate() {
  log_header "aggregate"
  local seeds="${SEEDS:-0 1 2}"
  run_uv scripts/aggregate_seeds.py --seeds ${seeds}
  run_uv scripts/plot_training_curves.py || true
  run_uv scripts/plot_prefix_vs_hack.py || true
}

usage() {
  cat <<'EOF'
Usage: ./scripts/run.sh <stage>

Stages:
  e0          Shared no-monitor GRPO baseline
  e1          Offline gate + in-domain probe fit
  e2          One monitor cell (MONITOR=probe|scl|cprobe, SEED=0|1|2)
  e4          Prefix / ablation diagnostic (MONITOR=e0|probe|scl|cprobe)
  epsi        Endpoint trained prefix score (MONITOR=e0|probe|scl|cprobe)
  e5a         Two-reader probe re-score
  e5          Outcome taxonomy
  aggregate   Multi-seed summary + figures

Environment:
  MODE=gpu|dry          default gpu
  SEED                  default 0
  MONITOR               e2/e4/epsi arm
  TRAIN_GPU / VLLM_GPU  default 0 / 1
  HF_HOME               Hugging Face cache
EOF
}

STAGE="${1:-}"
if [[ -z "${STAGE}" ]]; then
  usage
  exit 2
fi
shift || true

case "${STAGE}" in
  e0) stage_e0 ;;
  e1) stage_e1 ;;
  e2) stage_e2 ;;
  e4) stage_e4 ;;
  epsi|endpoint_psi) stage_epsi ;;
  e5a|e5_probe_scores) stage_e5a ;;
  e5) stage_e5 ;;
  aggregate|agg) stage_aggregate ;;
  -h|--help|help) usage ;;
  *)
    echo "unknown stage: ${STAGE}" >&2
    usage
    exit 2
    ;;
esac
