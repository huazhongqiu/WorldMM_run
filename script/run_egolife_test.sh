#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKERS="${WORLDMM_EGOLIFE_WORKERS:-4}"
MAX_ROUNDS="${WORLDMM_EGOLIFE_MAX_ROUNDS:-5}"
OUTPUT_DIR="${WORLDMM_EGOLIFE_OUTPUT_ROOT:-/myworkspace/projects/output/worldmm/egolife}/test-$(date +%Y%m%d_%H%M%S)"

PREPARE=""
OUTPUT_DIR_EXPLICIT=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --prepare-openie-only|--prepare-index-only) PREPARE="$1"; shift ;;
    --workers) WORKERS="$2"; shift 2 ;;
    --output-dir) OUTPUT_DIR="$2"; OUTPUT_DIR_EXPLICIT=1; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

source /opt/conda/etc/profile.d/conda.sh
conda activate worldmm

export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}"
export WORLDMM_LMDEPLOY_BASE_URL="${WORLDMM_LMDEPLOY_BASE_URL:-http://127.0.0.1:23333/v1}"
export WORLDMM_LMDEPLOY_MODEL="${WORLDMM_LMDEPLOY_MODEL:-Qwen3.5-4B}"
export WORLDMM_LMDEPLOY_MAX_TOKENS="${WORLDMM_LMDEPLOY_MAX_TOKENS:-4096}"
export WORLDMM_TEXT_EMBEDDING_MODEL="${WORLDMM_TEXT_EMBEDDING_MODEL:-/myworkspace/models/Qwen/Qwen3-Embedding-4B}"
export WORLDMM_VISUAL_EMBEDDING_MODEL="${WORLDMM_VISUAL_EMBEDDING_MODEL:-/myworkspace/projects/experiment/worldmm/models/VLM2Vec}"
export WORLDMM_VLM_BACKBONE_MODEL="${WORLDMM_VLM_BACKBONE_MODEL:-/myworkspace/projects/experiment/worldmm/models/Qwen2-VL-2B-Instruct}"
export WORLDMM_TEXT_ATTENTION="${WORLDMM_TEXT_ATTENTION:-flash_attention_2}"
export WORLDMM_VLM_ATTENTION="${WORLDMM_VLM_ATTENTION:-flash_attention_2}"
export WORLDMM_EMBEDDING_DEVICE="${WORLDMM_EMBEDDING_DEVICE:-cuda:1}"

ARTIFACTS_DIR="${WORLDMM_EGOLIFE_ARTIFACTS_DIR:-/myworkspace/projects/worldmm_needed/egolife/prebuilt/A1_JAKE}"
if [[ -n "$PREPARE" && "$OUTPUT_DIR_EXPLICIT" == "0" ]]; then
  OUTPUT_DIR="$ARTIFACTS_DIR/runs/${PREPARE#--}-$(date +%Y%m%d_%H%M%S)"
fi
if [[ -z "$PREPARE" && ! -d "$ARTIFACTS_DIR/snapshots" ]]; then
  echo "ERROR: Prepare OpenIE and indexes first: $0 --prepare-openie-only, then $0 --prepare-index-only" >&2
  exit 2
fi
mkdir -p "${OUTPUT_DIR}"
LMDEPLOY_PROFILE="${WORLDMM_LMDEPLOY_PROFILE:-formal-2gpu}"
AUTO_START_SERVICE="${WORLDMM_EGOLIFE_AUTO_START_SERVICE:-1}"
STARTUP_TIMEOUT_SECONDS="${WORLDMM_LMDEPLOY_STARTUP_TIMEOUT_SECONDS:-600}"
LMDEPLOY_LOG="${OUTPUT_DIR}/lmdeploy.log"
LMDEPLOY_PID=""

is_lmdeploy_ready() {
  curl --fail --silent --show-error "${WORLDMM_LMDEPLOY_BASE_URL}/models" >/dev/null 2>&1
}

cleanup_started_service() {
  local exit_code=$?
  if [[ -n "${LMDEPLOY_PID}" ]] && kill -0 "${LMDEPLOY_PID}" 2>/dev/null; then
    echo "========== 服务停止 =========="
    echo "pid=${LMDEPLOY_PID} reason=evaluation_finished"
    kill "${LMDEPLOY_PID}" 2>/dev/null || true
    wait "${LMDEPLOY_PID}" 2>/dev/null || true
  fi
  exit "${exit_code}"
}

wait_for_lmdeploy() {
  local elapsed=0
  while (( elapsed < STARTUP_TIMEOUT_SECONDS )); do
    if is_lmdeploy_ready; then
      echo "status=ready source=started pid=${LMDEPLOY_PID} waited=${elapsed}s"
      return 0
    fi
    if ! kill -0 "${LMDEPLOY_PID}" 2>/dev/null; then
      echo "ERROR: LMDeploy exited during startup; recent log follows:" >&2
      tail -n 80 "${LMDEPLOY_LOG}" >&2 || true
      return 1
    fi
    if (( elapsed == 0 || elapsed % 10 == 0 )); then
      echo "status=waiting elapsed=${elapsed}s timeout=${STARTUP_TIMEOUT_SECONDS}s log=${LMDEPLOY_LOG}"
    fi
    sleep 1
    elapsed=$((elapsed + 1))
  done
  echo "ERROR: LMDeploy did not become ready within ${STARTUP_TIMEOUT_SECONDS}s; recent log follows:" >&2
  tail -n 80 "${LMDEPLOY_LOG}" >&2 || true
  return 1
}

if [[ "$PREPARE" == "--prepare-openie-only" ]]; then
  LMDEPLOY_PROFILE=debug-1gpu
fi
if [[ "$PREPARE" == "--prepare-index-only" ]]; then
  export WORLDMM_EMBEDDING_DEVICE="${WORLDMM_PREBUILD_DEVICE:-cuda:0}"
fi

if [[ "$PREPARE" != "--prepare-index-only" ]]; then
echo "========== 服务检查 =========="
if is_lmdeploy_ready; then
  echo "status=ready source=existing"
else
  if [[ "${AUTO_START_SERVICE}" != "1" ]]; then
    echo "ERROR: LMDeploy is unavailable at ${WORLDMM_LMDEPLOY_BASE_URL}; set WORLDMM_EGOLIFE_AUTO_START_SERVICE=1 to start it." >&2
    exit 2
  fi
  echo "========== 服务启动 =========="
  echo "profile=${LMDEPLOY_PROFILE} log=${LMDEPLOY_LOG}"
  bash "${PROJECT_ROOT}/script/serve_lmdeploy.sh" "${LMDEPLOY_PROFILE}" >"${LMDEPLOY_LOG}" 2>&1 &
  LMDEPLOY_PID=$!
  trap cleanup_started_service EXIT
  wait_for_lmdeploy
fi

fi

echo "========== 评测配置 =========="
echo "query_days=5,6,7 questions=170 workers=${WORKERS} max_rounds=${MAX_ROUNDS} output=${OUTPUT_DIR}"
echo "日志将显示每个 worker 的索引、检索轮次与最终答案。"
export PYTHONUNBUFFERED=1
export TQDM_DISABLE=1


python "${PROJECT_ROOT}/eval/eval_egolife.py" \
  --subject A1_JAKE \
  --data-dir /myworkspace/data/EgoLife/lmms-lab_EgoLife \
  --caption-dir "${PROJECT_ROOT}/data/EgoLife/caption.zip" \
  --metadata-dir /myworkspace/projects/worldmm_needed/egolife \
  --video-root /myworkspace/data/EgoLife/lmms-lab_EgoLife \
  --output-dir "${OUTPUT_DIR}" \
  --cache-dir "${OUTPUT_DIR}/cache" \
  --retriever-model "${WORLDMM_LMDEPLOY_MODEL}" \
  --respond-model "${WORLDMM_LMDEPLOY_MODEL}" \
  --query-days 5 6 7 \
  --episodic-artifacts-dir "${ARTIFACTS_DIR}" \
  --workers "${WORKERS}" \
  --max-rounds "${MAX_ROUNDS}" \
  --mode test ${PREPARE:+"$PREPARE"}
