#!/usr/bin/env bash
# ============================================================================
# Standalone MedVidBench inference + official scoring (after preprocessing).
#
# Usage:
#   bash /myworkspace/projects/WorldMM/script/medvidbench/4_eval.sh
#   SKIP_LLM_JUDGE=1 EVAL_WORKERS=8 bash .../4_eval.sh
#
# Starts an LMDeploy server (videospy config, first GPU) if none is running,
# runs eval/eval_medvidbench.py (WorldMM open-ended inference), then scores
# submission.json with the official leaderboard evaluator (judge = the same
# local Qwen3.5-4B endpoint, as in videospy) and renders the report.
# ============================================================================
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

GPU_LIST="${GPU_LIST:-0,1}"
MODEL="${MODEL:-Qwen3.5-4B}"
MODEL_PATH="${MODEL_PATH:-/myworkspace/models/Qwen/${MODEL}}"
BASE_PORT="${BASE_PORT:-23333}"
STARTUP_TIMEOUT="${STARTUP_TIMEOUT:-900}"
EVAL_WORKERS="${EVAL_WORKERS:-4}"
SKIP_LLM_JUDGE="${SKIP_LLM_JUDGE:-0}"
EVAL_NAME="${EVAL_NAME:-medvidbench}"
MAX_ROUNDS="${WORLDMM_MEDVIDBENCH_MAX_ROUNDS:-5}"
EVAL_ATTEMPTS="${EVAL_ATTEMPTS:-2}"

WORLDMM_NEEDED="${WORLDMM_NEEDED:-/myworkspace/projects/worldmm_needed}"
MEDVIDBENCH_ROOT="${WORLDMM_MEDVIDBENCH_ROOT:-${WORLDMM_NEEDED}/medvidbench}"
OUTPUT_DIR="${WORLDMM_MEDVIDBENCH_OUTPUT:-/myworkspace/projects/output/worldmm/medvidbench}"
TRAINVAL_JSON="${TRAINVAL_JSON:-/myworkspace/data/MedVidBench/MedVidU_ECCV2026_TrainVal/medvidu_eccv2026_trainval.json}"
LEADERBOARD_DIR="${LEADERBOARD_DIR:-/myworkspace/data/MedVidBench/MedVidBench-Leaderboard}"

WORLDMM_PYTHON="${WORLDMM_PYTHON:-/opt/conda/envs/worldmm/bin/python}"
LMDEPLOY_BIN="${LMDEPLOY_BIN:-/opt/conda/envs/llm_deploy/bin/lmdeploy}"

IFS=',' read -r -a GPUS <<< "${GPU_LIST}"
CACHE_RATIO="${CACHE_RATIO:-0.8}"
COLOCATED_CACHE_RATIO="${COLOCATED_CACHE_RATIO:-0.35}"
COLOCATED_EVAL_RATIO="${COLOCATED_EVAL_RATIO:-0.30}"
BLUE='\033[1;34m'; GREEN='\033[1;32m'; NC='\033[0m'
log() { echo -e "${BLUE}[medvidbench-eval]${NC} $*"; }
ok()  { echo -e "${GREEN}[medvidbench-eval]${NC} $*"; }
section() {
    printf "\n============================================================\n %s\n============================================================\n" "$*"
}
banner() {
    echo ""
    echo "=============================================================="
    echo "  [medvidbench-eval] $*  ($(date '+%Y-%m-%d %H:%M:%S'))"
    echo "=============================================================="
}

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

endpoint_responding() { curl -fs "http://127.0.0.1:${BASE_PORT}/v1/models" >/dev/null 2>&1; }
served_model_matches() {
    curl -fs "http://127.0.0.1:${BASE_PORT}/v1/models" 2>/dev/null | \
        "${WORLDMM_PYTHON}" -c 'import json,sys; target=sys.argv[1]; data=json.load(sys.stdin); raise SystemExit(0 if any(row.get("id") == target for row in data.get("data", [])) else 1)' "${MODEL}"
}
is_ready() { endpoint_responding && served_model_matches; }

mkdir -p "${OUTPUT_DIR}/cache" "${OUTPUT_DIR}/logs"
[[ -x "${WORLDMM_PYTHON}" ]] || { echo "ERROR: WorldMM Python is not executable: ${WORLDMM_PYTHON}" >&2; exit 2; }
[[ -x "${LMDEPLOY_BIN}" ]] || { echo "ERROR: LMDeploy is not executable: ${LMDEPLOY_BIN}" >&2; exit 2; }
[[ -d "${MODEL_PATH}" ]] || { echo "ERROR: model directory is missing: ${MODEL_PATH}" >&2; exit 2; }
[[ -f "${TRAINVAL_JSON}" ]] || { echo "ERROR: ground-truth JSON is missing: ${TRAINVAL_JSON}" >&2; exit 2; }
[[ -d "${LEADERBOARD_DIR}" ]] || { echo "ERROR: leaderboard evaluator is missing: ${LEADERBOARD_DIR}" >&2; exit 2; }
section "OFFLINE PREFLIGHT"
"${WORLDMM_PYTHON}" "${PROJECT_ROOT}/eval/validate_precomputed.py" \
    --eval-json "${MEDVIDBENCH_ROOT}/qa/medvidbench_test.json" \
    --root "${MEDVIDBENCH_ROOT}" \
    --model "${MODEL}"
banner "MedVidBench eval — gpus=${GPUS[*]}, model=${MODEL}"
if (( ${#GPUS[@]} == 1 )); then
    log "Single-GPU layout: LLM server + text embedding share GPU ${GPUS[0]} (cache ratio ${COLOCATED_CACHE_RATIO})"
fi
section "LLM DEPLOY"
if endpoint_responding; then
    if served_model_matches; then
        ok "Existing LMDeploy instance found on port ${BASE_PORT} (reusing)"
    else
        echo "ERROR: port ${BASE_PORT} already responds but does not serve ${MODEL}" >&2
        exit 2
    fi
else
    local_ratio="${CACHE_RATIO}"
    (( ${#GPUS[@]} == 1 )) && local_ratio="${COLOCATED_EVAL_RATIO}"
    log "Starting LMDeploy (videospy config) on GPU ${GPUS[0]}, port ${BASE_PORT}, cache-max-entry-count ${local_ratio} ..."
    CUDA_VISIBLE_DEVICES="${GPUS[0]}" "${LMDEPLOY_BIN}" serve api_server "${MODEL_PATH}" \
        --backend pytorch \
        --tp 1 \
        --server-name 0.0.0.0 \
        --server-port "${BASE_PORT}" \
        --model-name "${MODEL}" \
        --max-batch-size 8 \
        --cache-max-entry-count "${local_ratio}" \
        --trust-remote-code \
        --reasoning-parser default \
        --tool-call-parser qwen3coder \
        --log-level WARNING \
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
export WORLDMM_LMDEPLOY_ENABLE_THINKING="${WORLDMM_LMDEPLOY_ENABLE_THINKING:-false}"
export WORLDMM_TEXT_EMBEDDING_MODEL="${WORLDMM_TEXT_EMBEDDING_MODEL:-/myworkspace/mymodels/Qwen3-Embedding-4B}"
export WORLDMM_VISUAL_EMBEDDING_MODEL="${WORLDMM_VISUAL_EMBEDDING_MODEL:-/myworkspace/mymodels/VLM2Vec}"
export WORLDMM_VLM_BACKBONE_MODEL="${WORLDMM_VLM_BACKBONE_MODEL:-/myworkspace/mymodels/Qwen2-VL-2B-Instruct}"
export WORLDMM_TEXT_ATTENTION="${WORLDMM_TEXT_ATTENTION:-flash_attention_2}"
export WORLDMM_VLM_ATTENTION="${WORLDMM_VLM_ATTENTION:-flash_attention_2}"
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

section "INFERENCE"
log "Running eval: qa=${MEDVIDBENCH_ROOT}/qa/medvidbench_test.json, caption=${MEDVIDBENCH_ROOT}/caption, metadata=${MEDVIDBENCH_ROOT}"
eval_cvd="${GPUS[0]},${GPUS[-1]}"
eval_emb_device="cuda:1"
eval_vis_device="cuda:1"
if (( ${#GPUS[@]} == 1 )); then
    eval_cvd="${GPUS[0]}"
    eval_emb_device="cuda:0"
    eval_vis_device="cpu"
fi
# With a single GPU the vision encoder (VLM2Vec) runs on CPU so the LLM
# server, the text embedding model and the vision encoder all fit together.
run_inference() {
    local attempt="$1"
    CUDA_VISIBLE_DEVICES="${eval_cvd}" WORLDMM_EMBEDDING_DEVICE="${eval_emb_device}" WORLDMM_VIS_EMBEDDING_DEVICE="${eval_vis_device}" \
        "${WORLDMM_PYTHON}" "${PROJECT_ROOT}/eval/eval_medvidbench.py" \
        --eval-json "${MEDVIDBENCH_ROOT}/qa/medvidbench_test.json" \
        --ground-truth-json "${TRAINVAL_JSON}" \
        --caption-dir "${MEDVIDBENCH_ROOT}/caption" \
        --metadata-dir "${MEDVIDBENCH_ROOT}" \
        --retriever-model "${MODEL}" \
        --respond-model "${MODEL}" \
        --episodic-cache-dir "${OUTPUT_DIR}/cache" \
        --output-dir "${OUTPUT_DIR}" \
        --eval-name "${EVAL_NAME}" \
        --workers "${EVAL_WORKERS}" \
        --max-rounds "${MAX_ROUNDS}" \
        --require-complete \
        2>&1 | tee "${OUTPUT_DIR}/logs/eval_attempt${attempt}_$(date +%Y%m%d_%H%M%S).log"
}

inference_complete=0
for ((attempt=1; attempt<=EVAL_ATTEMPTS; attempt++)); do
    if run_inference "${attempt}"; then
        inference_complete=1
        break
    fi
    if (( attempt < EVAL_ATTEMPTS )); then
        log "Inference attempt ${attempt}/${EVAL_ATTEMPTS} incomplete; retrying only failed/missing questions from records.jsonl"
    fi
done
(( inference_complete == 1 )) || { echo "ERROR: MedVidBench inference remains incomplete after ${EVAL_ATTEMPTS} attempts" >&2; exit 1; }

judge_args=()
[[ "${SKIP_LLM_JUDGE}" == "1" ]] && judge_args+=(--skip-llm-judge)
section "EVALUATION"
"${WORLDMM_PYTHON}" "${PROJECT_ROOT}/data/MedVidBench/utils/evaluate_medvidbench.py" \
    --run-dir "${OUTPUT_DIR}" \
    --leaderboard-dir "${LEADERBOARD_DIR}" \
    "${judge_args[@]+"${judge_args[@]}"}" \
    2>&1 | tee "${OUTPUT_DIR}/logs/official_eval_$(date +%Y%m%d_%H%M%S).log"

section "REPORT"
"${WORLDMM_PYTHON}" "${PROJECT_ROOT}/data/MedVidBench/utils/make_report.py" \
    --run-dir "${OUTPUT_DIR}" \
    2>&1 | tee "${OUTPUT_DIR}/logs/report_$(date +%Y%m%d_%H%M%S).log"

ok "MedVidBench report: ${OUTPUT_DIR}/report/report.md"
ok "Done. Results under ${OUTPUT_DIR}"
