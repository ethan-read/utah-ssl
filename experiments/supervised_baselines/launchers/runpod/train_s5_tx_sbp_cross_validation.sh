#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
cd "${REPO_ROOT}"

CV_NUM_FOLDS="${CV_NUM_FOLDS:-5}"
CV_START_FOLD="${CV_START_FOLD:-0}"
CV_STOP_FOLD="${CV_STOP_FOLD:-$((CV_NUM_FOLDS - 1))}"
SEED="${SEED:-7}"
RUN_NAME_PREFIX="${RUN_NAME_PREFIX:-willett_s5_tx_sbp_cv${CV_NUM_FOLDS}_seed${SEED}_60k}"
AUTO_STOP_POD_ON_EXIT="${AUTO_STOP_POD_ON_EXIT:-1}"
RUNPOD_STOP_API_BASE="${RUNPOD_STOP_API_BASE:-https://rest.runpod.io/v1}"

if (( CV_NUM_FOLDS < 2 )); then
  echo "CV_NUM_FOLDS must be at least 2" >&2
  exit 2
fi
if (( CV_START_FOLD < 0 || CV_START_FOLD >= CV_NUM_FOLDS )); then
  echo "CV_START_FOLD must satisfy 0 <= CV_START_FOLD < CV_NUM_FOLDS" >&2
  exit 2
fi
if (( CV_STOP_FOLD < CV_START_FOLD || CV_STOP_FOLD >= CV_NUM_FOLDS )); then
  echo "CV_STOP_FOLD must satisfy CV_START_FOLD <= CV_STOP_FOLD < CV_NUM_FOLDS" >&2
  exit 2
fi

auto_stop_pod() {
  if [[ "${AUTO_STOP_POD_ON_EXIT}" != "1" ]]; then
    echo "auto-stop disabled; leaving pod running"
    return 0
  fi
  if [[ -z "${RUNPOD_POD_ID:-}" || -z "${RUNPOD_API_KEY:-}" ]]; then
    echo "RUNPOD_POD_ID or RUNPOD_API_KEY is not set; cannot auto-stop pod"
    return 0
  fi
  if ! command -v curl >/dev/null 2>&1; then
    echo "curl is not available; cannot auto-stop pod"
    return 0
  fi

  local stop_url="${RUNPOD_STOP_API_BASE}/pods/${RUNPOD_POD_ID}/stop"
  echo "requesting pod stop via ${stop_url}"
  curl --fail --silent --show-error \
    --request POST \
    --url "${stop_url}" \
    --header "Authorization: Bearer ${RUNPOD_API_KEY}" || return $?
  echo
  echo "stop request accepted for pod ${RUNPOD_POD_ID}"
}

overall_status=0
for fold_index in $(seq "${CV_START_FOLD}" "${CV_STOP_FOLD}"); do
  fold_run_name="${RUN_NAME_PREFIX}_fold${fold_index}"
  echo "starting CV fold ${fold_index}/${CV_NUM_FOLDS} as ${fold_run_name}"

  set +e
  AUTO_STOP_POD_ON_EXIT=0 \
  SPLIT_POLICY=competition_train_kfold \
  CV_NUM_FOLDS="${CV_NUM_FOLDS}" \
  CV_FOLD_INDEX="${fold_index}" \
  SEED="${SEED}" \
  RUN_NAME="${fold_run_name}" \
  "${SCRIPT_DIR}/train_s5_tx_sbp.sh"
  fold_status=$?
  set -e

  if (( fold_status != 0 )); then
    echo "CV fold ${fold_index} failed with status ${fold_status}" >&2
    overall_status="${fold_status}"
    break
  fi
  echo "finished CV fold ${fold_index}/${CV_NUM_FOLDS}"
done

echo "CV batch exit status: ${overall_status}"
auto_stop_pod || true
exit "${overall_status}"
