#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

export PYTHONPATH="${REPO_ROOT}/analysis/active/ssl_experiments:${PYTHONPATH:-}"

CACHE_ROOT="${CACHE_ROOT:-/workspace/utah_ssl/data/cache_v1}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/workspace/utah_ssl/outputs/ssl_experiments/willett_s5_tx_sbp}"
RUN_NAME="${RUN_NAME:-willett_s5_tx_sbp_seed${SEED:-7}_60k}"
MAX_STEPS="${MAX_STEPS:-60000}"
BATCH_SIZE="${BATCH_SIZE:-64}"
SEED="${SEED:-7}"
VAL_EVERY_STEPS="${VAL_EVERY_STEPS:-100}"
CHECKPOINT_EVERY_STEPS="${CHECKPOINT_EVERY_STEPS:-500}"
CHECKPOINT_KEEP_LAST="${CHECKPOINT_KEEP_LAST:-4}"
LEARNING_RATE="${LEARNING_RATE:-1e-3}"
MIN_LEARNING_RATE="${MIN_LEARNING_RATE:-1e-5}"
WARMUP_STEPS="${WARMUP_STEPS:-1000}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-5}"
ADAM_EPSILON="${ADAM_EPSILON:-1e-8}"
MAX_GRAD_NORM="${MAX_GRAD_NORM:-10.0}"
PRECOMPUTED_SPLIT_STATS_PATH="${PRECOMPUTED_SPLIT_STATS_PATH:-}"
RECOMPUTE_SPLIT_STATS="${RECOMPUTE_SPLIT_STATS:-1}"
SPLIT_POLICY="${SPLIT_POLICY:-competition_train_test}"
CV_NUM_FOLDS="${CV_NUM_FOLDS:-5}"
CV_FOLD_INDEX="${CV_FOLD_INDEX:-0}"
AUTO_STOP_POD_ON_EXIT="${AUTO_STOP_POD_ON_EXIT:-1}"
RUNPOD_STOP_API_BASE="${RUNPOD_STOP_API_BASE:-https://rest.runpod.io/v1}"

export CACHE_ROOT

mkdir -p "${OUTPUT_ROOT}"

echo "repo: ${REPO_ROOT}"
echo "cache_root: ${CACHE_ROOT}"
echo "output_root: ${OUTPUT_ROOT}"
echo "run_name: ${RUN_NAME}"
echo "max_steps: ${MAX_STEPS}"
echo "seed: ${SEED}"
echo "learning_rate: ${LEARNING_RATE}"
echo "adam_epsilon: ${ADAM_EPSILON}"
echo "split_policy: ${SPLIT_POLICY}"
echo "cv_num_folds: ${CV_NUM_FOLDS}"
echo "cv_fold_index: ${CV_FOLD_INDEX}"
echo "recompute_split_stats: ${RECOMPUTE_SPLIT_STATS}"
echo "auto_stop_pod_on_exit: ${AUTO_STOP_POD_ON_EXIT}"

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi
fi

python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda_available:", torch.cuda.is_available())
print("cuda_device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")
PY

if [[ ! -f "${CACHE_ROOT}/brain2text24/manifest.jsonl" ]]; then
  echo "Missing cache manifest: ${CACHE_ROOT}/brain2text24/manifest.jsonl" >&2
  echo "Set CACHE_ROOT to the directory containing brain2text24/manifest.jsonl." >&2
  exit 2
fi

python - <<'PY'
import json
import os
from pathlib import Path

cache_root = Path(os.environ["CACHE_ROOT"])
dataset_root = cache_root / "brain2text24"
metadata = json.loads((dataset_root / "metadata.json").read_text())
layout = metadata.get("feature_layout") or {}
expected = {
    "n_total_features": 256,
    "n_tx_features": 128,
    "n_sbp_features": 128,
}
problems = [
    f"{key}={layout.get(key)!r} expected {value}"
    for key, value in expected.items()
    if int(layout.get(key, -1)) != value
]
with (dataset_root / "manifest.jsonl").open() as handle:
    first_row = json.loads(next(line for line in handle if line.strip()))
if not bool(first_row.get("has_sbp")):
    problems.append("manifest first row has_sbp is not true")
if str(first_row.get("feature_modalities")) != "tx+sbp":
    problems.append(f"manifest feature_modalities={first_row.get('feature_modalities')!r} expected 'tx+sbp'")
if problems:
    raise SystemExit("Cache preflight failed:\n- " + "\n- ".join(problems))
print(
    "cache preflight ok: "
    f"{layout.get('n_tx_features')} TX + {layout.get('n_sbp_features')} SBP "
    f"features, {metadata.get('total_examples')} examples"
)
PY

if [[ "${RECOMPUTE_SPLIT_STATS}" == "1" && "${SPLIT_POLICY}" == "competition_train_test" ]]; then
  echo "recomputing canonical raw tx_sbp split stats for uploaded cache"
  python analysis/active/ssl_experiments/recompute_split_feature_stats.py \
    --cache-root "${CACHE_ROOT}" \
    --dataset brain2text24 \
    --feature-mode tx_sbp \
    --boundary-key-mode session \
    --overwrite
elif [[ "${RECOMPUTE_SPLIT_STATS}" == "1" ]]; then
  echo "skipping canonical split-stats recompute for split_policy=${SPLIT_POLICY}; fold stats are computed from fold train rows"
fi

auto_stop_pod() {
  if [[ "${AUTO_STOP_POD_ON_EXIT}" != "1" ]]; then
    echo "auto-stop disabled; leaving pod running"
    return 0
  fi
  if [[ -z "${RUNPOD_POD_ID:-}" ]]; then
    echo "RUNPOD_POD_ID is not set; cannot auto-stop pod"
    return 0
  fi
  if [[ -z "${RUNPOD_API_KEY:-}" ]]; then
    echo "RUNPOD_API_KEY is not set; cannot auto-stop pod"
    return 0
  fi
  if ! command -v curl >/dev/null 2>&1; then
    echo "curl is not available; cannot auto-stop pod"
    return 0
  fi

  local stop_url="${RUNPOD_STOP_API_BASE}/pods/${RUNPOD_POD_ID}/stop"
  echo "requesting pod stop via ${stop_url}"
  if curl --fail --silent --show-error \
    --request POST \
    --url "${stop_url}" \
    --header "Authorization: Bearer ${RUNPOD_API_KEY}"; then
    echo
    echo "stop request accepted for pod ${RUNPOD_POD_ID}"
  else
    local status=$?
    echo
    echo "failed to stop pod ${RUNPOD_POD_ID}; curl exit status ${status}" >&2
    return "${status}"
  fi
}

CMD=(
  python -m willett_reconstruction.train
  --cache-root "${CACHE_ROOT}"
  --output-root "${OUTPUT_ROOT}"
  --run-name "${RUN_NAME}"
  --dataset brain2text24
  --feature-mode tx_sbp
  --boundary-key-mode session
  --split-policy "${SPLIT_POLICY}"
  --cv-num-folds "${CV_NUM_FOLDS}"
  --cv-fold-index "${CV_FOLD_INDEX}"
  --normalization-mode global
  --decoder-backbone-type s5
  --batch-size "${BATCH_SIZE}"
  --max-steps "${MAX_STEPS}"
  --learning-rate "${LEARNING_RATE}"
  --min-learning-rate "${MIN_LEARNING_RATE}"
  --warmup-steps "${WARMUP_STEPS}"
  --weight-decay "${WEIGHT_DECAY}"
  --adam-epsilon "${ADAM_EPSILON}"
  --max-grad-norm "${MAX_GRAD_NORM}"
  --val-every-steps "${VAL_EVERY_STEPS}"
  --checkpoint-every-steps "${CHECKPOINT_EVERY_STEPS}"
  --checkpoint-keep-last "${CHECKPOINT_KEEP_LAST}"
  --seed "${SEED}"
  --resume-latest
)

if [[ -n "${PRECOMPUTED_SPLIT_STATS_PATH}" ]]; then
  CMD+=(--precomputed-split-stats-path "${PRECOMPUTED_SPLIT_STATS_PATH}")
fi

echo "launching:"
printf ' %q' "${CMD[@]}"
echo

set +e
"${CMD[@]}"
cmd_status=$?
set -e

echo "training command exit status: ${cmd_status}"
auto_stop_pod || true
exit "${cmd_status}"
