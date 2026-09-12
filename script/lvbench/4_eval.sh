#!/usr/bin/env bash
# ============================================================================
# Standalone LVBench evaluation with WorldMM (after preprocessing is done).
#
# Usage:
#   bash /myworkspace/projects/WorldMM/script/lvbench/4_eval.sh
#
# Starts an LMDeploy server (videospy config, first GPU) if none is running,
# loads embeddings on the last GPU, then runs eval/eval.py on the 315 questions.
# ============================================================================
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

GPU_LIST="${GPU_LIST:-0,1}"
MODEL="${MODEL:-Qwen3.5-4B}"
MODEL_PATH="${MODEL_PATH:-/myworkspace/models/Qwen/${MODEL}}"
BASE_PORT="${BASE_PORT:-23333}"
STARTUP_TIMEOUT="${STARTUP_TIMEOUT:-900}"

WORLDMM_NEEDED="${WORLDMM_NEEDED:-/myworkspace/projects/worldmm_needed}"
LVBENCH_ROOT="${WORLDMM_LVBENCH_ROOT:-${WORLDMM_NEEDED}/lvbench}"
OUTPUT_DIR="${WORLDMM_LVBENCH_OUTPUT:-/myworkspace/projects/output/worldmm/lvbench}"

WORLDMM_PYTHON="${WORLDMM_PYTHON:-/opt/conda/envs/worldmm/bin/python}"
LMDEPLOY_BIN="${LMDEPLOY_BIN:-/opt/conda/envs/llm_deploy/bin/lmdeploy}"

IFS=',' read -r -a GPUS <<< "${GPU_LIST}"
BLUE='\033[1;34m'; GREEN='\033[1;32m'; NC='\033[0m'
log() { echo -e "${BLUE}[lvbench-eval]${NC} $*"; }
ok()  { echo -e "${GREEN}[lvbench-eval]${NC} $*"; }

SERVER_PID=""
cleanup() {
    local exit_code=$?
    if [[ -n "${SERVER_PID}" ]] && kill -0 "${SERVER_PID}" 2>/dev/null; then
        log "Stopping LMDeploy (pid=${SERVER_PID})"
        kill "${SERVER_PID}" 2>/dev/null || true
        wait "${SERVER_PID}" 2>/dev/null || true
    fi
    exit "${exit_code}"
}
trap cleanup EXIT INT TERM

is_ready() { curl -fs "http://127.0.0.1:${BASE_PORT}/v1/models" >/dev/null 2>&1; }

mkdir -p "${OUTPUT_DIR}/cache" "${OUTPUT_DIR}/logs"
if is_ready; then
    ok "Existing LMDeploy instance found on port ${BASE_PORT} (reusing)"
else
    log "Starting LMDeploy (videospy config) on GPU ${GPUS[0]}, port ${BASE_PORT} ..."
    CUDA_VISIBLE_DEVICES="${GPUS[0]}" "${LMDEPLOY_BIN}" serve api_server "${MODEL_PATH}" \
        --backend pytorch \
        --tp 1 \
        --server-name 0.0.0.0 \
        --server-port "${BASE_PORT}" \
        --model-name "${MODEL}" \
        --max-batch-size 8 \
        --cache-max-entry-count 0.8 \
        --trust-remote-code \
        --reasoning-parser default \
        --tool-call-parser qwen3coder \
        --log-level REQUEST \
        >"${OUTPUT_DIR}/logs/lmdeploy_eval.log" 2>&1 &
    SERVER_PID=$!
    elapsed=0
    while (( elapsed < STARTUP_TIMEOUT )); do
        if is_ready; then ok "LMDeploy ready after ${elapsed}s"; break; fi
        if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
            echo "ERROR: LMDeploy exited during startup:" >&2
            tail -n 80 "${OUTPUT_DIR}/logs/lmdeploy_eval.log" >&2 || true
            exit 1
        fi
        sleep 2; elapsed=$((elapsed + 2))
    done
    is_ready || { echo "ERROR: LMDeploy not ready within ${STARTUP_TIMEOUT}s" >&2; exit 1; }
fi

export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}"
export WORLDMM_LMDEPLOY_BASE_URL="${WORLDMM_LMDEPLOY_BASE_URL:-http://127.0.0.1:${BASE_PORT}/v1}"
export WORLDMM_LMDEPLOY_MODEL="${MODEL}"
export WORLDMM_LMDEPLOY_MAX_TOKENS="${WORLDMM_LMDEPLOY_MAX_TOKENS:-4096}"
export WORLDMM_LMDEPLOY_API_KEY="${WORLDMM_LMDEPLOY_API_KEY:-EMPTY}"
export WORLDMM_TEXT_EMBEDDING_MODEL="${WORLDMM_TEXT_EMBEDDING_MODEL:-/myworkspace/mymodels/Qwen3-Embedding-4B}"
export WORLDMM_VISUAL_EMBEDDING_MODEL="${WORLDMM_VISUAL_EMBEDDING_MODEL:-/myworkspace/mymodels/VLM2Vec}"
export WORLDMM_VLM_BACKBONE_MODEL="${WORLDMM_VLM_BACKBONE_MODEL:-/myworkspace/mymodels/Qwen2-VL-2B-Instruct}"
export WORLDMM_TEXT_ATTENTION="${WORLDMM_TEXT_ATTENTION:-flash_attention_2}"
export WORLDMM_VLM_ATTENTION="${WORLDMM_VLM_ATTENTION:-flash_attention_2}"
export WORLDMM_EMBEDDING_DEVICE=cuda:1
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

log "Running eval: qa=${LVBENCH_ROOT}/qa/lvbench_test.json, caption=${LVBENCH_ROOT}/caption, metadata=${LVBENCH_ROOT}"
CUDA_VISIBLE_DEVICES="${GPUS[0]},${GPUS[-1]}" WORLDMM_LVBENCH_USAGE_FILE="${OUTPUT_DIR}/token_usage.json" \
    "${WORLDMM_PYTHON}" "${PROJECT_ROOT}/data/LVBench/utils/run_eval.py" \
    --eval-json "${LVBENCH_ROOT}/qa/lvbench_test.json" \
    --caption-dir "${LVBENCH_ROOT}/caption" \
    --metadata-dir "${LVBENCH_ROOT}" \
    --retriever-model "${MODEL}" \
    --respond-model "${MODEL}" \
    --episodic-cache-dir "${OUTPUT_DIR}/cache" \
    --output-dir "${OUTPUT_DIR}" \
    --eval-name lvbench \
    2>&1 | tee "${OUTPUT_DIR}/logs/eval_$(date +%Y%m%d_%H%M%S).log"

model_dir="${MODEL//-/_}"
"${WORLDMM_PYTHON}" "${PROJECT_ROOT}/data/LVBench/utils/make_videospy_report.py" \
    --eval-json "${OUTPUT_DIR}/${model_dir}_${model_dir}/lvbench_eval.json" \
    --report-dir "${OUTPUT_DIR}/report" \
    --agent WorldMM --mode test --model "${MODEL}" \
    --usage-json "${OUTPUT_DIR}/token_usage.json" \
    2>&1 | tee "${OUTPUT_DIR}/logs/report_$(date +%Y%m%d_%H%M%S).log"

ok "videospy-style report: ${OUTPUT_DIR}/report/report.md"
ok "Done. Results under ${OUTPUT_DIR}"
