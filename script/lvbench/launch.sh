#!/usr/bin/env bash
# ============================================================================
# Foreground launcher for the LVBench pipeline — safe to paste as a single
# job/command (e.g. k8s pod or cluster console).
#
#   GPU_LIST=auto bash /myworkspace/projects/WorldMM/script/lvbench/launch.sh
#
# What it does:
#   1. creates the scratch log directory up front (the outer-shell redirect
#      failure "No such file or directory" happens when it does not exist yet);
#   2. runs run_all.sh in the FOREGROUND and tees everything to a timestamped
#      log file, so the live console/pod-log shows PHASE banners + tqdm while
#      a full copy is kept on disk for `tail -f` from any other terminal.
# All run_all.sh env knobs still work (GPU_LIST, SAMPLE_FPS, MAX_FRAME_EDGE,
# WITH_EVAL, SMOKE, ...). A missing GPU_LIST means run_all.sh's own default.
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRATCH="${WORLDMM_LVBENCH_SCRATCH:-/workspace/worldmm/lvbench}"
LOG_DIR="${SCRATCH}/logs"
mkdir -p "${LOG_DIR}"

# The platform pod log only keeps the most recent lines and the scratch dir is
# destroyed with the pod, so tee a second copy straight to shared storage.
PERSIST_LOG_DIR="${WORLDMM_LVBENCH_PERSIST_LOGDIR:-/myworkspace/projects/output/worldmm/lvbench/logs}"
mkdir -p "${PERSIST_LOG_DIR}"

RUN_TS="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_DIR}/run_all_${RUN_TS}.log"
PERSIST_LOG_FILE="${PERSIST_LOG_DIR}/run_all_${RUN_TS}.log"
echo "[launch] live log also saved to: ${LOG_FILE}"
echo "[launch] persistent copy: ${PERSIST_LOG_FILE}"
echo "[launch] follow from another terminal with: tail -f ${LOG_FILE}"
echo ""

exec bash "${SCRIPT_DIR}/run_all.sh" 2>&1 | tee "${LOG_FILE}" "${PERSIST_LOG_FILE}"
