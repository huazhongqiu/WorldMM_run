#!/usr/bin/env bash
# ============================================================================
# Foreground launcher for the MedVidBench pipeline — safe to paste as a single
# job/command (e.g. k8s pod or cluster console).
#
#   GPU_LIST=auto bash /myworkspace/projects/WorldMM/script/medvidbench/launch.sh
#
# Creates the scratch log dir up front, runs run_all.sh in the FOREGROUND and
# tees everything to a timestamped log file. All run_all.sh env knobs still
# work (GPU_LIST, SAMPLE_FPS, WITH_EVAL, SMOKE, EVAL_WORKERS, ...).
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRATCH="${WORLDMM_MEDVIDBENCH_SCRATCH:-/workspace/worldmm/medvidbench}"
LOG_DIR="${SCRATCH}/logs"
mkdir -p "${LOG_DIR}"

LOG_FILE="${LOG_DIR}/run_all_$(date +%Y%m%d_%H%M%S).log"
echo "[launch] live log also saved to: ${LOG_FILE}"
echo "[launch] follow from another terminal with: tail -f ${LOG_FILE}"
echo ""

exec bash "${SCRIPT_DIR}/run_all.sh" 2>&1 | tee "${LOG_FILE}"
