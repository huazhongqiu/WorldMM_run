#!/usr/bin/env bash
# ============================================================================
# WorldMM x MedVidBench preprocessing + optional eval — single-command
# orchestrator (clone of script/lvbench/run_all.sh for the frame-only,
# open-ended MedVidBench test split).
#
# Usage (from anywhere):
#   bash /myworkspace/projects/WorldMM/script/medvidbench/run_all.sh
#   SMOKE=1 bash .../run_all.sh        # debug split (2 questions), scratch only
#
# Real-time progress:
#   tail -f /workspace/worldmm/medvidbench/logs/run_all_<ts>.log
#   watch -n 30 bash /myworkspace/projects/WorldMM/script/medvidbench/check_progress.sh
#
# Everything resumes: re-running skips already-completed artifacts.
# ============================================================================
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# ============================ knobs (env-overridable) ============================
GPU_LIST="${GPU_LIST:-0}"                      # comma list, e.g. 0 / 0,1 / auto
MODEL="${MODEL:-Qwen3.5-4B}"
MODEL_PATH="${MODEL_PATH:-/myworkspace/models/Qwen/${MODEL}}"
TRAINVAL_JSON="${TRAINVAL_JSON:-/myworkspace/data/MedVidBench/MedVidU_ECCV2026_TrainVal/medvidu_eccv2026_trainval.json}"
TEST_MANIFEST="${TEST_MANIFEST:-/myworkspace/projects/videospy/experiments/data_splits/main/medvidbench/test.json}"
FRAME_ROOT="${FRAME_ROOT:-/myworkspace/data/MedVidBench/MedVidU_ECCV2026_TrainVal/valdata}"
WORLDMM_NEEDED="${WORLDMM_NEEDED:-/myworkspace/projects/worldmm_needed}"
MEDVIDBENCH_ROOT="${WORLDMM_MEDVIDBENCH_ROOT:-${WORLDMM_NEEDED}/medvidbench}"
SCRATCH="${WORLDMM_MEDVIDBENCH_SCRATCH:-/workspace/worldmm/medvidbench}"
OUTPUT_DIR="${WORLDMM_MEDVIDBENCH_OUTPUT:-/myworkspace/projects/output/worldmm/medvidbench}"

SAMPLE_FPS="${SAMPLE_FPS:-1.0}"                # caption frame sampling rate
MAX_FRAME_EDGE="${MAX_FRAME_EDGE:-1280}"       # caption frame resize (longest edge; 0 = native)
NUM_FRAMES="${NUM_FRAMES:-16}"                 # frames per 10s clip for VLM2Vec visual memory
CAPTION_WORKERS="${CAPTION_WORKERS:-16}"       # concurrent segment requests per caption client
NUM_SHARDS="${NUM_SHARDS:-0}"                  # 0 = auto (2 per server instance)
CAPTION_ATTEMPTS="${CAPTION_ATTEMPTS:-2}"      # passes over failed videos
SYNTH_WORKERS="${SYNTH_WORKERS:-8}"            # parallel ffmpeg frame->mp4 jobs
BASE_PORT="${BASE_PORT:-23333}"
CACHE_RATIO="${CACHE_RATIO:-0.8}"              # KV cache ratio for LLM-only phases
COLOCATED_CACHE_RATIO="${COLOCATED_CACHE_RATIO:-0.35}"  # KV ratio when the embedding model shares the only GPU
COLOCATED_EVAL_RATIO="${COLOCATED_EVAL_RATIO:-0.30}"    # KV ratio during eval: text + vision embedding models share the card with the server
STARTUP_TIMEOUT="${STARTUP_TIMEOUT:-900}"
WITH_EVAL="${WITH_EVAL:-0}"                    # 1 = also run inference + official scoring
EVAL_WORKERS="${EVAL_WORKERS:-4}"              # concurrent segment workers during eval
SKIP_LLM_JUDGE="${SKIP_LLM_JUDGE:-0}"          # 1 = deterministic official metrics only
SMOKE="${SMOKE:-0}"                            # 1 = debug split (2 questions), scratch outputs only
PROGRESS_INTERVAL="${PROGRESS_INTERVAL:-60}"

WORLDMM_PYTHON="${WORLDMM_PYTHON:-/opt/conda/envs/worldmm/bin/python}"
LMDEPLOY_BIN="${LMDEPLOY_BIN:-/opt/conda/envs/llm_deploy/bin/lmdeploy}"
# ================================== derived =====================================
LOG_DIR="${SCRATCH}/logs"
SHARD_DIR="${SCRATCH}/shards"
RUN_TS="$(date +%Y%m%d_%H%M%S)"
BLUE='\033[1;34m'; GREEN='\033[1;32m'; RED='\033[1;31m'; NC='\033[0m'

# runtime roots (switched to smoke scratch when SMOKE=1)
ROOT_QA="${MEDVIDBENCH_ROOT}/qa"
ROOT_CAPTION="${MEDVIDBENCH_ROOT}/caption"
ROOT_METADATA="${MEDVIDBENCH_ROOT}"           # episodic_memory/ semantic_memory/ visual_memory/
VIDEO_ROOT="${SCRATCH}/videos"                # synthesized per-segment mp4s
SYNTH_MAPPING="${SCRATCH}/segment_videos.json"
EVAL_OUT="${OUTPUT_DIR}"
EVAL_CACHE="${OUTPUT_DIR}/cache"
EVAL_NAME="medvidbench"

SERVER_PIDS=()
SERVER_PORTS=()
WATCHER_PID=""
CURRENT_PHASE=""
CURRENT_PREFIX=""
CURRENT_SHARDS=0

log()  { echo -e "${BLUE}[run_all]${NC} $*"; }
ok()   { echo -e "${GREEN}[run_all]${NC} $*"; }
fail() { echo -e "${RED}[run_all]${NC} $*" >&2; }
banner() {
    echo ""
    echo "=============================================================="
    echo "  [run_all] $*  ($(date '+%Y-%m-%d %H:%M:%S'))"
    echo "=============================================================="
}

set_progress_context() {
    CURRENT_PHASE="$1"; CURRENT_PREFIX="$2"; CURRENT_SHARDS="$3"
    mkdir -p "${SCRATCH}" 2>/dev/null || true
    printf '%s\t%s\t%s\n' "$1" "$2" "$3" > "${SCRATCH}/current_phase.txt" 2>/dev/null || true
}

show_progress() {
    "${WORLDMM_PYTHON}" - \
        "${ROOT_QA}" "${ROOT_CAPTION}" "${ROOT_METADATA}" "${MODEL}" \
        "${SHARD_DIR}" "${LOG_DIR}" "${RUN_TS}" \
        "${SCRATCH}/current_phase.txt" <<'PYEOF' || true
import glob, json, os, re, sys
(qa_root, caption_root, meta_root, model, shard_dir, log_dir, run_ts,
 status_file) = sys.argv[1:9]

cur_phase, cur_prefix, cur_shards = "", "", 0
try:
    cur_phase, cur_prefix, cur_shards = open(status_file).read().split("\t")
    cur_shards = int(cur_shards)
except Exception:
    pass
if cur_phase == "caption":
    cur_phase = "captions"

W = 20
def bar(done, tot):
    tot = max(1, tot)
    done = max(0, min(done, tot))
    filled = done * W // tot
    return f"【{done:>4}/{tot:<4}】{done * 100 // tot:3d}% " + "█" * filled + "░" * (W - filled)

videos = json.load(open(os.path.join(qa_root, "test_videos.json")))
total = len(videos)

stages = [
    ("captions",   caption_root, "10sec.json"),
    ("multiscale", caption_root, "10min.json"),
    ("episodic",   os.path.join(meta_root, "episodic_memory"),
     f"episodic_triple_results_{model}.json"),
    ("semantic",   os.path.join(meta_root, "semantic_memory"),
     f"semantic_consolidation_results_{model}.json"),
    ("visual",     os.path.join(meta_root, "visual_memory"), "visual_embeddings.pkl"),
]
shard_subdir = {"captions": "caption", "multiscale": "multiscale", "episodic": "episodic"}

def count_done(root, marker, vid_list=None):
    if vid_list is None:
        vid_list = os.listdir(root) if os.path.isdir(root) else []
    return sum(1 for v in vid_list if os.path.isfile(os.path.join(root, v, marker)))

def tail(path, nbytes=262144):
    try:
        size = os.path.getsize(path)
        with open(path, errors="ignore") as f:
            if size > nbytes:
                f.seek(size - nbytes)
            return f.read()
    except OSError:
        return ""

def inflight(name, idx=None):
    def last(pattern, text, groups):
        m = None
        for m in re.finditer(pattern, text):
            pass
        return m.groups() if m else None
    if name in ("captions", "multiscale", "episodic"):
        lfs = sorted(glob.glob(os.path.join(log_dir, f"{cur_prefix}_shard{idx}_*.log")))
        if not lfs:
            return ""
        text = tail(lfs[-1])
        if name == "captions":
            g = last(r"Captioning (\S+?)\.mp4:\s+\d+%[^\n]*?\|\s*(\d+)/(\d+)", text, 3)
            return f"  进行中 {g[0]} {g[1]}/{g[2]} 段" if g else ""
        if name == "episodic":
            g = last(r"(NER|Extracting triples):\s+\d+%[^\n]*?\|\s*(\d+)/(\d+)", text, 3)
            return f"  进行中 {g[0]} {g[1]}/{g[2]} 块" if g else ""
        g = last(r"Multiscale memory:\s+\d+%[^\n]*?\|\s*(\d+)/(\d+)", text, 2)
        if g:
            v = last(r"video=(\S+)", text, 1)
            return f"  进行中 {v[0] if v else ''} {g[0]}/{g[1]} 视频"
        return ""
    text = tail(os.path.join(log_dir, f"{cur_prefix}_{run_ts}.log"))
    g = last(r"\[\s*\d+/\d+\] [^\n:]*?: (\S+)", text, 1)
    return f"  进行中 {g[0]}" if g else ""

for name, root, marker in stages:
    print(f"====== {name} ======")
    print("  total  " + bar(count_done(root, marker), total))
    if name == cur_phase and cur_shards > 0:
        sdir = os.path.join(shard_dir, shard_subdir[name])
        shard_files = sorted(glob.glob(os.path.join(sdir, "shard_*.json")),
                             key=lambda p: int(re.search(r"shard_(\d+)", p).group(1)))
        for sf in shard_files:
            try:
                vids = json.load(open(sf))
            except Exception:
                continue
            idx = re.search(r"shard_(\d+)", sf).group(1)
            line = f"  shard{idx} " + bar(count_done(root, marker, vids), len(vids))
            line += inflight(name, idx)
            print(line)
    elif name == cur_phase:
        detail = inflight(name)
        if detail:
            print(detail)
PYEOF
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
    banner "PHASE 0/7 — prepare (QA conversion + segment list)"
    mkdir -p "${LOG_DIR}" "${SHARD_DIR}" "${ROOT_QA}"
    if [[ ! -f "${ROOT_QA}/medvidbench_test.json" || "${FORCE_PREPARE:-0}" == "1" ]]; then
        log "Phase prepare: converting MedVidBench QA (manifest=${TEST_MANIFEST}) ..."
        "${WORLDMM_PYTHON}" "${PROJECT_ROOT}/data/MedVidBench/utils/prepare_medvidbench.py" \
            --trainval-json "${TRAINVAL_JSON}" \
            --test-manifest "${TEST_MANIFEST}" \
            --frame-root "${FRAME_ROOT}" \
            --output-dir "${ROOT_QA}" \
            2>&1 | tee "${LOG_DIR}/prepare_${RUN_TS}.log"
    else
        ok "Phase prepare: QA already present at ${ROOT_QA}/medvidbench_test.json (skip)"
    fi
}

phase_synthesize() {
    banner "PHASE 1/7 — synthesize segment mp4s from frames (${SYNTH_WORKERS} workers)"
    set_progress_context "synthesize" "synthesize" 0
    "${WORLDMM_PYTHON}" "${PROJECT_ROOT}/data/MedVidBench/utils/synthesize_segment_videos.py" \
        --segments-json "${ROOT_QA}/test_segments.json" \
        --output-dir "${VIDEO_ROOT}" \
        --mapping-json "${SYNTH_MAPPING}" \
        --workers "${SYNTH_WORKERS}" \
        2>&1 | tee "${LOG_DIR}/synthesize_${RUN_TS}.log"
}

setup_smoke() {
    local smoke_dir="${SCRATCH}/smoke"
    mkdir -p "${smoke_dir}/qa" "${smoke_dir}/videos"
    TEST_MANIFEST="${SMOKE_MANIFEST:-/myworkspace/projects/videospy/experiments/data_splits/debug/medvidbench/test.json}"
    ROOT_QA="${smoke_dir}/qa"
    ROOT_CAPTION="${smoke_dir}/caption"
    ROOT_METADATA="${smoke_dir}"
    VIDEO_ROOT="${smoke_dir}/videos"
    SYNTH_MAPPING="${smoke_dir}/segment_videos.json"
    EVAL_OUT="${smoke_dir}/eval"
    EVAL_CACHE="${smoke_dir}/cache"
    EVAL_NAME="medvidbench_smoke"
    ok "Smoke mode: manifest=${TEST_MANIFEST}, outputs under ${smoke_dir}"
}

run_sharded_clients() {
    local phase="$1" num="$2" log_prefix="$3"
    set_progress_context "${phase}" "${log_prefix}" "${num}"
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
    banner "PHASE 2/7 — captions (${SAMPLE_FPS} fps, longest-edge ${MAX_FRAME_EDGE})"
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
    banner "PHASE 3/7 — multiscale summaries (30sec/3min/10min)"
    local num="${NUM_SHARDS}"
    [[ "${num}" == "0" ]] && num=$((2 * ${#GPUS[@]}))
    mkdir -p "${SHARD_DIR}/multiscale"
    num="$(make_shards "${ROOT_QA}/test_videos.json" "${SHARD_DIR}/multiscale" "${num}")"
    log "Multiscale shards: ${num}"
    run_sharded_clients multiscale "${num}" "multiscale"
}

phase_episodic() {
    banner "PHASE 4/7 — episodic memory (NER + OpenIE triples)"
    local num="${NUM_SHARDS}"
    [[ "${num}" == "0" ]] && num=$((2 * ${#GPUS[@]}))
    mkdir -p "${SHARD_DIR}/episodic"
    num="$(make_shards "${ROOT_QA}/test_videos.json" "${SHARD_DIR}/episodic" "${num}")"
    log "Episodic shards: ${num}"
    run_sharded_clients episodic "${num}" "episodic"
}

phase_semantic() {
    banner "PHASE 5/7 — semantic memory (extraction + consolidation)"
    set_progress_context "semantic" "semantic" 0
    if (( ${#GPUS[@]} == 1 )); then
        stop_all_servers
        start_server_on_gpu "${GPUS[0]}" "${BASE_PORT}" "${COLOCATED_CACHE_RATIO}"
        if [[ "${SERVER_MODE}" == "started" ]]; then
            wait_for_port "${BASE_PORT}" "${STARTUP_TIMEOUT}"
        elif [[ "${SERVER_MODE}" == "reused" ]]; then
            log "WARNING: reusing an external server with its own KV cache setting; if the GPU is tight, restart it with --cache-max-entry-count ${COLOCATED_CACHE_RATIO}"
        fi
        log "Single-GPU layout: LLM server + text embedding share GPU ${GPUS[0]} (cache ratio ${COLOCATED_CACHE_RATIO})"
    else
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
    banner "PHASE 6/7 — visual memory (VLM2Vec, ${NUM_FRAMES} frames/clip)"
    set_progress_context "visual" "visual" 0
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
    banner "PHASE 7/7 — eval (${EVAL_NAME}, $(grep -c video_id "${ROOT_QA}/medvidbench_test.json" 2>/dev/null || echo '?') questions)"
    stop_all_servers
    local eval_ratio="${CACHE_RATIO}"
    if (( ${#GPUS[@]} == 1 )); then
        eval_ratio="${COLOCATED_EVAL_RATIO}"
        log "Single-GPU layout: LLM server + text/vision embedding models share GPU ${GPUS[0]} (KV ratio ${eval_ratio}, vision encoder on CPU)"
    fi
    start_server_on_gpu "${GPUS[0]}" "${BASE_PORT}" "${eval_ratio}"
    if [[ "${SERVER_MODE}" == "started" ]]; then
        wait_for_port "${BASE_PORT}" "${STARTUP_TIMEOUT}"
    fi
    local eval_cvd="${GPUS[0]},${GPUS[-1]}" eval_emb_device="cuda:1" eval_vis_device="cuda:1"
    if (( ${#GPUS[@]} == 1 )); then
        eval_cvd="${GPUS[0]}"
        eval_emb_device="cuda:0"
        eval_vis_device="cpu"
    fi
    CUDA_VISIBLE_DEVICES="${eval_cvd}" WORLDMM_EMBEDDING_DEVICE="${eval_emb_device}" WORLDMM_VIS_EMBEDDING_DEVICE="${eval_vis_device}" \
        "${WORLDMM_PYTHON}" "${PROJECT_ROOT}/eval/eval_medvidbench.py" \
            --eval-json "${ROOT_QA}/medvidbench_test.json" \
            --ground-truth-json "${TRAINVAL_JSON}" \
            --caption-dir "${ROOT_CAPTION}" \
            --metadata-dir "${ROOT_METADATA}" \
            --retriever-model "${MODEL}" \
            --respond-model "${MODEL}" \
            --episodic-cache-dir "${EVAL_CACHE}" \
            --output-dir "${EVAL_OUT}" \
            --eval-name "${EVAL_NAME}" \
            --mode "$([[ "${SMOKE}" == "1" ]] && echo smoke || echo test)" \
            --workers "${EVAL_WORKERS}" \
            2>&1 | tee "${LOG_DIR}/eval_${RUN_TS}.log"

    local judge_args=()
    [[ "${SKIP_LLM_JUDGE}" == "1" ]] && judge_args+=(--skip-llm-judge)
    "${WORLDMM_PYTHON}" "${PROJECT_ROOT}/data/MedVidBench/utils/evaluate_medvidbench.py" \
        --run-dir "${EVAL_OUT}" \
        --leaderboard-dir "${LEADERBOARD_DIR:-/myworkspace/data/MedVidBench/MedVidBench-Leaderboard}" \
        "${judge_args[@]+"${judge_args[@]}"}" \
        2>&1 | tee "${LOG_DIR}/official_eval_${RUN_TS}.log"

    "${WORLDMM_PYTHON}" "${PROJECT_ROOT}/data/MedVidBench/utils/make_report.py" \
        --run-dir "${EVAL_OUT}" \
        2>&1 | tee "${LOG_DIR}/report_${RUN_TS}.log"
    ok "MedVidBench report: ${EVAL_OUT}/report/report.md"
}

progress_watcher() {
    (
        show_progress
        while true; do
            sleep "${PROGRESS_INTERVAL}"
            show_progress
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

log "========== WorldMM x MedVidBench pipeline =========="
log "gpus=${GPUS[*]} model=${MODEL} (${MODEL_PATH})"
log "trainval=${TRAINVAL_JSON}"
log "manifest=${TEST_MANIFEST} frame_root=${FRAME_ROOT}"
log "output_root=${MEDVIDBENCH_ROOT} scratch=${SCRATCH} eval_out=${OUTPUT_DIR}"
log "sample_fps=${SAMPLE_FPS} max_frame_edge=${MAX_FRAME_EDGE} num_frames=${NUM_FRAMES} synth_workers=${SYNTH_WORKERS}"
log "with_eval=${WITH_EVAL} eval_workers=${EVAL_WORKERS} skip_llm_judge=${SKIP_LLM_JUDGE} smoke=${SMOKE}"

if [[ "${SMOKE}" == "1" ]]; then setup_smoke; fi
phase_prepare

phase_synthesize

set_progress_context "prepare" "" 0
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
MODEL="${MODEL}" WORLDMM_MEDVIDBENCH_ROOT="$(dirname "${ROOT_QA}")" bash "${PROJECT_ROOT}/script/medvidbench/check_progress.sh"
if [[ "${SMOKE}" == "1" ]]; then
    ok "Smoke eval output: ${EVAL_OUT}"
elif [[ "${WITH_EVAL}" == "1" ]]; then
    ok "Eval output: ${EVAL_OUT}"
else
    ok "Preprocessing complete and ready for inference: bash ${PROJECT_ROOT}/script/medvidbench/4_eval.sh"
fi
