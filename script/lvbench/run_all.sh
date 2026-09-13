#!/usr/bin/env bash
# ============================================================================
# WorldMM x LVBench preprocessing + optional eval — single-command orchestrator.
#
# Usage (from anywhere):
#   bash /myworkspace/projects/WorldMM/script/lvbench/run_all.sh
#
# Real-time progress:
#   tail -f /workspace/worldmm/lvbench/logs/run_all_<ts>.log   (or the tee'd stream)
#   watch -n 30 bash /myworkspace/projects/WorldMM/script/lvbench/check_progress.sh
#
# Everything resumes: re-running skips already-completed artifacts.
# Model serving matches videospy exactly (lmdeploy 0.14.0, pytorch backend,
# tp=1, max-batch-size 8, cache-max-entry-count 0.8) — one instance per GPU.
# Single-GPU (e.g. 1x4090) works too: GPU_LIST=0 automatically lowers the KV
# cache ratio when the text embedding model must share the card (semantic/eval).
# ============================================================================
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# ============================ knobs (env-overridable) ============================
GPU_LIST="${GPU_LIST:-0,1}"                    # comma list, e.g. 0,1 / 0,1,2,3 / auto
MODEL="${MODEL:-Qwen3.5-4B}"
MODEL_PATH="${MODEL_PATH:-/myworkspace/models/Qwen/${MODEL}}"
VIDEO_ROOT="${VIDEO_ROOT:-/myworkspace/data/LVBench/AIWinter-LVBench/all_videos}"
META_JSONL="${META_JSONL:-/myworkspace/data/LVBench/zai-org-LVBench/video_info.meta.jsonl}"
TEST_MANIFEST="${TEST_MANIFEST:-/myworkspace/projects/videospy/experiments/data_splits/main/lvbench/test.json}"
WORLDMM_NEEDED="${WORLDMM_NEEDED:-/myworkspace/projects/worldmm_needed}"
LVBENCH_ROOT="${WORLDMM_LVBENCH_ROOT:-${WORLDMM_NEEDED}/lvbench}"
SCRATCH="${WORLDMM_LVBENCH_SCRATCH:-/workspace/worldmm/lvbench}"
OUTPUT_DIR="${WORLDMM_LVBENCH_OUTPUT:-/myworkspace/projects/output/worldmm/lvbench}"

SAMPLE_FPS="${SAMPLE_FPS:-1.0}"                # caption frame sampling rate (0.5 = faster)
MAX_FRAME_EDGE="${MAX_FRAME_EDGE:-0}"          # caption frame resize (longest edge; 0 = native resolution, same as original source code)
NUM_FRAMES="${NUM_FRAMES:-16}"                 # frames per 10s clip for VLM2Vec visual memory
CAPTION_WORKERS="${CAPTION_WORKERS:-16}"       # concurrent segment requests per caption client
NUM_SHARDS="${NUM_SHARDS:-0}"                  # 0 = auto (2 per server instance)
CAPTION_ATTEMPTS="${CAPTION_ATTEMPTS:-2}"      # passes over failed videos
BASE_PORT="${BASE_PORT:-23333}"
CACHE_RATIO="${CACHE_RATIO:-0.8}"              # KV cache ratio for LLM-only phases (multi-GPU default = videospy config)
COLOCATED_CACHE_RATIO="${COLOCATED_CACHE_RATIO:-0.35}"  # KV ratio when the embedding model shares the only GPU (semantic/eval on 1 GPU)
STARTUP_TIMEOUT="${STARTUP_TIMEOUT:-900}"
WITH_EVAL="${WITH_EVAL:-0}"                    # 1 = also run eval on the 315 questions after preprocessing
SMOKE="${SMOKE:-0}"                            # 1 = 3-minute clip + 3 questions, scratch outputs only
SMOKE_DURATION="${SMOKE_DURATION:-180}"
PROGRESS_INTERVAL="${PROGRESS_INTERVAL:-300}"

WORLDMM_PYTHON="${WORLDMM_PYTHON:-/opt/conda/envs/worldmm/bin/python}"
LMDEPLOY_BIN="${LMDEPLOY_BIN:-/opt/conda/envs/llm_deploy/bin/lmdeploy}"
# ================================== derived =====================================
LOG_DIR="${SCRATCH}/logs"
SHARD_DIR="${SCRATCH}/shards"
RUN_TS="$(date +%Y%m%d_%H%M%S)"
BLUE='\033[1;34m'; GREEN='\033[1;32m'; RED='\033[1;31m'; NC='\033[0m'

# runtime roots (switched to smoke scratch when SMOKE=1)
ROOT_QA="${LVBENCH_ROOT}/qa"
ROOT_CAPTION="${LVBENCH_ROOT}/caption"
ROOT_METADATA="${LVBENCH_ROOT}"          # episodic_memory/ semantic_memory/ visual_memory/
EVAL_OUT="${OUTPUT_DIR}"
EVAL_CACHE="${OUTPUT_DIR}/cache"
EVAL_NAME="lvbench"

SERVER_PIDS=()
SERVER_PORTS=()
WATCHER_PID=""

log()  { echo -e "${BLUE}[run_all]${NC} $*"; }
ok()   { echo -e "${GREEN}[run_all]${NC} $*"; }
fail() { echo -e "${RED}[run_all]${NC} $*" >&2; }
banner() {
    # Prominent phase separator, visible even in interleaved pod/kubectl logs.
    echo ""
    echo "=============================================================="
    echo "  [run_all] $*  ($(date '+%Y-%m-%d %H:%M:%S'))"
    echo "=============================================================="
}

cleanup() {
    local exit_code=$?
    if [[ -n "${WATCHER_PID}" ]] && kill -0 "${WATCHER_PID}" 2>/dev/null; then
        kill "${WATCHER_PID}" 2>/dev/null || true
    fi
    stop_all_servers
    exit "${exit_code}"
}
trap cleanup EXIT INT TERM

# ============================ GPU handling ======================================
resolve_gpus() {
    local list="${GPU_LIST}"
    if [[ "${list}" == "auto" ]]; then
        local n; n="$(nvidia-smi -L 2>/dev/null | wc -l)"
        if (( n < 1 )); then
            fail "No GPU detected"
            exit 2
        fi
        list="$(seq -s, 0 $((n - 1)))"
    fi
    IFS=',' read -r -a GPUS <<< "${list}"
    if (( ${#GPUS[@]} < 1 )); then
        fail "No usable GPU in GPU_LIST=${GPU_LIST}"
        exit 2
    fi
    if (( ${#GPUS[@]} == 1 )); then
        log "Single-GPU mode (${GPUS[0]}): LLM server and embedding model will share the card during semantic/eval (KV cache ratio ${COLOCATED_CACHE_RATIO})"
    fi
}

# ============================ LMDeploy servers (videospy config) ================
is_port_ready() {
    curl -fs "http://127.0.0.1:$1/v1/models" >/dev/null 2>&1
}

start_server_on_gpu() {
    # $1 = gpu id, $2 = port, $3 = KV cache ratio (optional, default CACHE_RATIO)
    # sets SERVER_MODE to "started" or "reused"
    local gpu="$1" port="$2" ratio="${3:-${CACHE_RATIO}}"
    if is_port_ready "${port}"; then
        log "Port ${port} already serving ${MODEL} — reusing existing instance"
        SERVER_MODE="reused"
        return 0
    fi
    log "Starting LMDeploy (videospy config) on GPU ${gpu}, port ${port}, cache-max-entry-count ${ratio} ..."
    CUDA_VISIBLE_DEVICES="${gpu}" "${LMDEPLOY_BIN}" serve api_server "${MODEL_PATH}" \
        --backend pytorch \
        --tp 1 \
        --server-name 0.0.0.0 \
        --server-port "${port}" \
        --model-name "${MODEL}" \
        --max-batch-size 8 \
        --cache-max-entry-count "${ratio}" \
        --trust-remote-code \
        --reasoning-parser default \
        --tool-call-parser qwen3coder \
        --log-level REQUEST \
        >"${LOG_DIR}/lmdeploy_gpu${gpu}_port${port}.log" 2>&1 &
    SERVER_PIDS+=("$!")
    SERVER_PORTS+=("${port}")
    SERVER_MODE="started"
}

start_all_servers() {
    local n=${#GPUS[@]}
    for i in "${!GPUS[@]}"; do
        local port=$((BASE_PORT + i))
        start_server_on_gpu "${GPUS[$i]}" "${port}"
        SERVER_MODES+=("${SERVER_MODE}")
    done
    for i in "${!GPUS[@]}"; do
        local port=$((BASE_PORT + i))
        if [[ "${SERVER_MODES[$i]}" == "reused" ]]; then continue; fi
        wait_for_port "${port}" "${STARTUP_TIMEOUT}" || return 1
    done
    ok "All ${n} LMDeploy instance(s) ready: ports ${BASE_PORT}..$((BASE_PORT + n - 1))"
}

wait_for_port() {
    local port="$1" timeout="$2" elapsed=0
    while (( elapsed < timeout )); do
        if is_port_ready "${port}"; then
            ok "Port ${port} ready after ${elapsed}s"
            return 0
        fi
        if (( elapsed > 0 && elapsed % 30 == 0 )); then
            log "Waiting for port ${port} ... ${elapsed}s/${timeout}s"
        fi
        sleep 2; elapsed=$((elapsed + 2))
    done
    fail "Port ${port} did not become ready within ${timeout}s"
    return 1
}

stop_server_by_port() {
    local port="$1" idx
    for idx in "${!SERVER_PORTS[@]}"; do
        if [[ "${SERVER_PORTS[$idx]}" == "${port}" ]]; then
            local pid="${SERVER_PIDS[$idx]:-}"
            if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
                log "Stopping LMDeploy instance (pid=${pid}, port=${port})"
                kill "${pid}" 2>/dev/null || true
                wait "${pid}" 2>/dev/null || true
            fi
            SERVER_PIDS[$idx]=""
        fi
    done
}

stop_all_servers() {
    local idx
    for idx in "${!SERVER_PORTS[@]}"; do
        local port="${SERVER_PORTS[$idx]}"
        [[ -n "${port}" ]] && stop_server_by_port "${port}"
    done
}

base_url_for() { echo "http://127.0.0.1:$1/v1"; }

# ============================ sharding helpers ==================================
make_shards() {
    # $1 = test_videos.json, $2 = output dir, $3 = num shards ; echoes actual num shards written
    "${WORLDMM_PYTHON}" - "$1" "$2" "$3" <<'PYEOF'
import json, os, sys
list_path, out_dir, num = sys.argv[1], sys.argv[2], int(sys.argv[3])
videos = json.load(open(list_path))
num = max(1, min(num, len(videos)))
os.makedirs(out_dir, exist_ok=True)
per = (len(videos) + num - 1) // num
for i in range(num):
    chunk = videos[i * per:(i + 1) * per]
    with open(os.path.join(out_dir, f"shard_{i}.json"), "w") as f:
        json.dump(chunk, f)
print(num)
PYEOF
}

make_symlink_farm() {
    # $1 = caption dir, $2 = shard json, $3 = farm dir
    local caption_dir="$1" shard_json="$2" farm="$3"
    rm -rf "${farm}"; mkdir -p "${farm}"
    "${WORLDMM_PYTHON}" - "$caption_dir" "$shard_json" "$farm" <<'PYEOF'
import json, os, sys
caption_dir, shard_json, farm = sys.argv[1], sys.argv[2], sys.argv[3]
for v in json.load(open(shard_json)):
    os.symlink(os.path.join(caption_dir, v), os.path.join(farm, v))
PYEOF
}

# ============================ phases ============================================
phase_prepare() {
    banner "PHASE 0/5 — prepare (QA conversion)"
    mkdir -p "${LOG_DIR}" "${SHARD_DIR}" "${ROOT_QA}"
    if [[ ! -f "${ROOT_QA}/lvbench_test.json" || "${FORCE_PREPARE:-0}" == "1" ]]; then
        log "Phase prepare: converting LVBench QA (videospy test split) ..."
        "${WORLDMM_PYTHON}" "${PROJECT_ROOT}/data/LVBench/utils/prepare_lvbench.py" \
            --meta-jsonl "${META_JSONL}" \
            --test-manifest "${TEST_MANIFEST}" \
            --video-root "${VIDEO_ROOT}" \
            --output-dir "${ROOT_QA}" \
            2>&1 | tee "${LOG_DIR}/prepare_${RUN_TS}.log"
    else
        ok "Phase prepare: QA already present at ${ROOT_QA}/lvbench_test.json (skip)"
    fi
}

setup_smoke() {
    local smoke_dir="${SCRATCH}/smoke"
    mkdir -p "${smoke_dir}/videos" "${smoke_dir}/qa" "${smoke_dir}/caption" "${smoke_dir}/cache" "${smoke_dir}/eval"
    local ffmpeg_bin="${FFMPEG_BIN:-$(command -v ffmpeg 2>/dev/null || true)}"
    if [[ -z "${ffmpeg_bin}" ]]; then
        ffmpeg_bin="$("${WORLDMM_PYTHON}" -c 'import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())' 2>/dev/null || true)"
    fi
    if [[ -z "${ffmpeg_bin}" ]]; then
        fail "ffmpeg not found (install ffmpeg, or: ${WORLDMM_PYTHON} -m pip install imageio-ffmpeg)"
        exit 2
    fi
    local smoke_key
    smoke_key="$("${WORLDMM_PYTHON}" -c "import json; print(json.load(open('${ROOT_QA}/test_videos.json'))[0])")"
    if [[ ! -f "${smoke_dir}/videos/${smoke_key}.mp4" ]]; then
        log "Smoke: trimming ${SMOKE_DURATION}s from ${smoke_key}.mp4 ..."
        mkdir -p "${smoke_dir}/videos"
        if ! "${ffmpeg_bin}" -y -loglevel error -i "${VIDEO_ROOT}/${smoke_key}.mp4" -t "${SMOKE_DURATION}" -c copy \
                -movflags +faststart "${smoke_dir}/videos/${smoke_key}.mp4" \
                2>"${LOG_DIR}/ffmpeg_${RUN_TS}.log"; then
            log "ffmpeg stream-copy failed, re-encoding instead ..."
            "${ffmpeg_bin}" -y -loglevel error -i "${VIDEO_ROOT}/${smoke_key}.mp4" -t "${SMOKE_DURATION}" \
                -c:v libx264 -preset veryfast -an -movflags +faststart \
                "${smoke_dir}/videos/${smoke_key}.mp4" 2>>"${LOG_DIR}/ffmpeg_${RUN_TS}.log"
        fi
    fi
    "${WORLDMM_PYTHON}" - "${ROOT_QA}/lvbench_test.json" "${smoke_key}" "${smoke_dir}/qa" <<'PYEOF'
import json, os, sys
rows, key, out_dir = json.load(open(sys.argv[1])), sys.argv[2], sys.argv[3]
subset = [r for r in rows if r["video_id"] == key][:3]
if not subset:
    raise SystemExit(f"no QA rows for smoke video {key}")
json.dump(subset, open(os.path.join(out_dir, "lvbench_test.json"), "w"), indent=2, ensure_ascii=False)
json.dump([key], open(os.path.join(out_dir, "test_videos.json"), "w"))
print(f"smoke QA subset: {len(subset)} questions from {key}")
PYEOF
    ROOT_QA="${smoke_dir}/qa"
    ROOT_CAPTION="${smoke_dir}/caption"
    ROOT_METADATA="${smoke_dir}"   # episodic_memory/ semantic_memory/ visual_memory/ directly under smoke_dir
    VIDEO_ROOT="${smoke_dir}/videos"
    EVAL_OUT="${smoke_dir}/eval"
    EVAL_CACHE="${smoke_dir}/cache"
    EVAL_NAME="lvbench_smoke"
    WORLDMM_LVBENCH_ROOT="${smoke_dir}"
    ok "Smoke mode: video=${smoke_key} (${SMOKE_DURATION}s), QA subset=3, outputs under ${smoke_dir}"
}

run_sharded_clients() {
    # $1 = phase name; $2 = num shards; $3 = log prefix; then per-shard command via function cmd_for_shard
    local phase="$1" num="$2" log_prefix="$3"
    local pids=() names=() i
    for i in $(seq 0 $((num - 1))); do
        local port=$((BASE_PORT + i % ${#GPUS[@]}))
        local log_file="${LOG_DIR}/${log_prefix}_shard${i}_${RUN_TS}.log"
        (
            export WORLDMM_LMDEPLOY_BASE_URL="$(base_url_for "${port}")"
            cmd_for_shard "${phase}" "${i}" "${log_file}"
        ) &
        pids+=($!); names+=("shard${i}")
    done
    local failed=0
    for i in "${!pids[@]}"; do
        if wait "${pids[$i]}"; then
            ok "${phase} ${names[$i]} done"
        else
            fail "${phase} ${names[$i]} FAILED (see ${LOG_DIR}/${log_prefix}_shard${i}_${RUN_TS}.log)"
            failed=1
        fi
    done
    return "${failed}"
}

cmd_for_shard() {
    local phase="$1" i="$2" log_file="$3"
    {
        echo "=============================================================="
        echo "=== ${phase} shard${i} started at $(date '+%Y-%m-%d %H:%M:%S')"
        echo "=============================================================="
        case "${phase}" in
            caption)
                "${WORLDMM_PYTHON}" "${PROJECT_ROOT}/data/LVBench/utils/generate_fine_caption.py" \
                    --video-path "${VIDEO_ROOT}" \
                    --video-list "${SHARD_DIR}/caption/shard_${i}.json" \
                    --output-path "${ROOT_CAPTION}" \
                    --model "${MODEL}" \
                    --unit-time 10 \
                    --sample-fps "${SAMPLE_FPS}" \
                    --max-frame-longest-edge "${MAX_FRAME_EDGE}" \
                    --max-workers "${CAPTION_WORKERS}" \
                    --retries 3
                ;;
            multiscale)
                local farm="${SHARD_DIR}/multiscale/farm_${i}"
                make_symlink_farm "${ROOT_CAPTION}" "${SHARD_DIR}/multiscale/shard_${i}.json" "${farm}"
                "${WORLDMM_PYTHON}" -m worldmm.memory.episodic.multiscale \
                    --caption_dir "${farm}" \
                    --model "${MODEL}" \
                    --windows "30,180,600" \
                    --granularity_names "30sec,3min,10min" \
                    --perspective general
                ;;
            episodic)
                local farm="${SHARD_DIR}/episodic/farm_${i}"
                make_symlink_farm "${ROOT_CAPTION}" "${SHARD_DIR}/episodic/shard_${i}.json" "${farm}"
                "${WORLDMM_PYTHON}" "${PROJECT_ROOT}/preprocess/build_memory.py" \
                    --caption-dir "${farm}" \
                    --output-dir "${ROOT_METADATA}" \
                    --model "${MODEL}" \
                    --step episodic
                ;;
            *) echo "unknown phase ${phase}" >&2; return 1 ;;
        esac
    } 2>&1 | tee "${log_file}"
}

phase_caption() {
    banner "PHASE 1/5 — captions (${SAMPLE_FPS} fps, longest-edge ${MAX_FRAME_EDGE})"
    local num="${NUM_SHARDS}"
    [[ "${num}" == "0" ]] && num=$((2 * ${#GPUS[@]}))
    mkdir -p "${SHARD_DIR}/caption"
    num="$(make_shards "${ROOT_QA}/test_videos.json" "${SHARD_DIR}/caption" "${num}")"
    log "Caption shards: ${num}"
    local attempt=1
    while (( attempt <= CAPTION_ATTEMPTS )); do
        if run_sharded_clients caption "${num}" "caption_a${attempt}"; then
            ok "Phase captions complete"; return 0
        fi
        if (( attempt < CAPTION_ATTEMPTS )); then
            fail "Some videos failed captioning; retrying failed ones (attempt $((attempt + 1))/${CAPTION_ATTEMPTS}) ..."
        fi
        attempt=$((attempt + 1))
    done
    fail "Caption generation still failing after ${CAPTION_ATTEMPTS} attempts"
    return 1
}

phase_multiscale() {
    banner "PHASE 2/5 — multiscale summaries (30sec/3min/10min)"
    local num="${NUM_SHARDS}"
    [[ "${num}" == "0" ]] && num=$((2 * ${#GPUS[@]}))
    mkdir -p "${SHARD_DIR}/multiscale"
    num="$(make_shards "${ROOT_QA}/test_videos.json" "${SHARD_DIR}/multiscale" "${num}")"
    log "Multiscale shards: ${num}"
    run_sharded_clients multiscale "${num}" "multiscale"
}

phase_episodic() {
    banner "PHASE 3/5 — episodic memory (NER + OpenIE triples)"
    local num="${NUM_SHARDS}"
    [[ "${num}" == "0" ]] && num=$((2 * ${#GPUS[@]}))
    mkdir -p "${SHARD_DIR}/episodic"
    num="$(make_shards "${ROOT_QA}/test_videos.json" "${SHARD_DIR}/episodic" "${num}")"
    log "Episodic shards: ${num}"
    run_sharded_clients episodic "${num}" "episodic"
}

phase_semantic() {
    banner "PHASE 4/5 — semantic memory (extraction + consolidation)"
    if (( ${#GPUS[@]} == 1 )); then
        # Single GPU: the LLM server and the text embedding model must share the
        # card, so restart the server with a smaller KV cache to avoid OOM.
        stop_all_servers
        start_server_on_gpu "${GPUS[0]}" "${BASE_PORT}" "${COLOCATED_CACHE_RATIO}"
        if [[ "${SERVER_MODE}" == "started" ]]; then
            wait_for_port "${BASE_PORT}" "${STARTUP_TIMEOUT}"
        elif [[ "${SERVER_MODE}" == "reused" ]]; then
            log "WARNING: reusing an external server with its own KV cache setting; if the GPU is tight, restart it with --cache-max-entry-count ${COLOCATED_CACHE_RATIO}"
        fi
        log "Single-GPU layout: LLM server + text embedding share GPU ${GPUS[0]} (cache ratio ${COLOCATED_CACHE_RATIO})"
    else
        # keep server on GPUS[0] only; load text embedding on the last GPU
        log "Multi-GPU layout: LLM on port ${BASE_PORT} (GPU ${GPUS[0]}), embedding on GPU ${GPUS[-1]}"
        local i
        for i in "${!GPUS[@]}"; do
            if (( i > 0 )); then stop_server_by_port "$((BASE_PORT + i))"; fi
        done
    fi
    CUDA_VISIBLE_DEVICES="${GPUS[-1]}" WORLDMM_EMBEDDING_DEVICE=cuda:0 \
        "${WORLDMM_PYTHON}" "${PROJECT_ROOT}/preprocess/build_memory.py" \
            --caption-dir "${ROOT_CAPTION}" \
            --output-dir "${ROOT_METADATA}" \
            --model "${MODEL}" \
            --step semantic \
            2>&1 | tee "${LOG_DIR}/semantic_${RUN_TS}.log"
}

phase_visual() {
    banner "PHASE 5/5 — visual memory (VLM2Vec, ${NUM_FRAMES} frames/clip)"
    stop_all_servers
    "${WORLDMM_PYTHON}" "${PROJECT_ROOT}/preprocess/build_memory.py" \
        --caption-dir "${ROOT_CAPTION}" \
        --output-dir "${ROOT_METADATA}" \
        --model "${MODEL}" \
        --step visual \
        --gpu "$(IFS=,; echo "${GPUS[*]}")" \
        --num-frames "${NUM_FRAMES}" \
        2>&1 | tee "${LOG_DIR}/visual_${RUN_TS}.log"
}

phase_eval() {
    banner "PHASE eval — LVBench (${EVAL_NAME}, $(grep -c video_id "${ROOT_QA}/lvbench_test.json" 2>/dev/null || echo '?') questions)"
    stop_all_servers
    local eval_ratio="${CACHE_RATIO}"
    if (( ${#GPUS[@]} == 1 )); then
        eval_ratio="${COLOCATED_CACHE_RATIO}"
        log "Single-GPU layout: LLM server + text embedding share GPU ${GPUS[0]} (cache ratio ${eval_ratio})"
    fi
    start_server_on_gpu "${GPUS[0]}" "${BASE_PORT}" "${eval_ratio}"
    if [[ "${SERVER_MODE}" == "started" ]]; then
        wait_for_port "${BASE_PORT}" "${STARTUP_TIMEOUT}"
    fi
    local eval_cvd="${GPUS[0]},${GPUS[-1]}" eval_emb_device="cuda:1"
    if (( ${#GPUS[@]} == 1 )); then
        eval_cvd="${GPUS[0]}"
        eval_emb_device="cuda:0"
    fi
    CUDA_VISIBLE_DEVICES="${eval_cvd}" WORLDMM_EMBEDDING_DEVICE="${eval_emb_device}" \
        WORLDMM_LVBENCH_USAGE_FILE="${EVAL_OUT}/token_usage.json" \
        "${WORLDMM_PYTHON}" "${PROJECT_ROOT}/data/LVBench/utils/run_eval.py" \
            --eval-json "${ROOT_QA}/lvbench_test.json" \
            --caption-dir "${ROOT_CAPTION}" \
            --metadata-dir "${ROOT_METADATA}" \
            --retriever-model "${MODEL}" \
            --respond-model "${MODEL}" \
            --episodic-cache-dir "${EVAL_CACHE}" \
            --output-dir "${EVAL_OUT}" \
            --eval-name "${EVAL_NAME}" \
            2>&1 | tee "${LOG_DIR}/eval_${RUN_TS}.log"

    local model_dir="${MODEL//-/_}"
    "${WORLDMM_PYTHON}" "${PROJECT_ROOT}/data/LVBench/utils/make_videospy_report.py" \
        --eval-json "${EVAL_OUT}/${model_dir}_${model_dir}/${EVAL_NAME}_eval.json" \
        --report-dir "${EVAL_OUT}/report" \
        --agent WorldMM --mode test --model "${MODEL}" \
        --usage-json "${EVAL_OUT}/token_usage.json" \
        2>&1 | tee "${LOG_DIR}/report_${RUN_TS}.log"
    ok "videospy-style report: ${EVAL_OUT}/report/report.md"
}

progress_watcher() {
    (
        while true; do
            sleep "${PROGRESS_INTERVAL}"
            echo "[PROGRESS] $(WORLDMM_LVBENCH_ROOT="${WORLDMM_LVBENCH_ROOT}" MODEL="${MODEL}" bash "${PROJECT_ROOT}/script/lvbench/check_progress.sh" --oneline)"
        done
    ) &
    WATCHER_PID=$!
}

# ============================ main ==============================================
resolve_gpus
mkdir -p "${LOG_DIR}"

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
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

log "========== WorldMM x LVBench preprocessing =========="
log "gpus=${GPUS[*]} model=${MODEL} (${MODEL_PATH})"
log "cache_ratio=${CACHE_RATIO} colocated_cache_ratio=${COLOCATED_CACHE_RATIO}"
log "video_root=${VIDEO_ROOT}"
log "output_root=${LVBENCH_ROOT} scratch=${SCRATCH} eval_out=${OUTPUT_DIR}"
log "sample_fps=${SAMPLE_FPS} max_frame_edge=${MAX_FRAME_EDGE} num_frames=${NUM_FRAMES} with_eval=${WITH_EVAL} smoke=${SMOKE}"

phase_prepare
if [[ "${SMOKE}" == "1" ]]; then setup_smoke; fi

progress_watcher
banner "SERVERS — starting ${#GPUS[@]} LMDeploy instance(s)"
SERVER_MODES=()
start_all_servers

phase_caption && ok ">>> captions DONE" || { fail "Pipeline stopped: captions failed"; exit 1; }
phase_multiscale && ok ">>> multiscale DONE" || { fail "Pipeline stopped: multiscale failed"; exit 1; }
phase_episodic && ok ">>> episodic DONE" || { fail "Pipeline stopped: episodic failed"; exit 1; }
phase_semantic && ok ">>> semantic DONE" || { fail "Pipeline stopped: semantic failed"; exit 1; }
phase_visual && ok ">>> visual DONE" || { fail "Pipeline stopped: visual failed"; exit 1; }

if [[ "${WITH_EVAL}" == "1" || "${SMOKE}" == "1" ]]; then
    phase_eval
fi

banner "DONE — all requested phases finished"
MODEL="${MODEL}" WORLDMM_LVBENCH_ROOT="${WORLDMM_LVBENCH_ROOT}" bash "${PROJECT_ROOT}/script/lvbench/check_progress.sh"
if [[ "${SMOKE}" == "1" ]]; then
    ok "Smoke eval output: ${EVAL_OUT}"
elif [[ "${WITH_EVAL}" == "1" ]]; then
    ok "Eval output: ${EVAL_OUT}"
else
    ok "Preprocessing complete and ready for inference: bash ${PROJECT_ROOT}/script/lvbench/4_eval.sh"
fi
