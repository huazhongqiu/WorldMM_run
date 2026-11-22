#!/usr/bin/env bash
# Print an isolated experiment environment, or run a supplied command within it.
# Usage: bash script/run_embedding_variant.sh DATASET BACKEND [-- command args...]
# No preprocessing/evaluation is started without an explicit command.
set -euo pipefail
DATASET="${1:?dataset required: lvbench|egolife|videomme_long}"
BACKEND="${2:?backend required: original|freeret|wemm4b}"
shift 2
case "$DATASET" in lvbench|egolife|videomme_long) ;; *) echo 'Invalid dataset' >&2; exit 2 ;; esac
case "$BACKEND" in
  original) BACKEND_LABEL='Qwen3-Embedding-4B+VLM2Vec-V2' ;;
  freeret) BACKEND_LABEL='Qwen3.5-4B-FreeRet' ;;
  wemm4b) BACKEND_LABEL='WeMM-Embedding-4B' ;;
  *) echo 'Invalid backend' >&2; exit 2 ;;
esac
export WORLDMM_EMBEDDING_BACKEND="$BACKEND"
export WORLDMM_DATASET="$DATASET"
VIDEO_SPY_CONFIG_ROOT="${VIDEO_SPY_CONFIG_ROOT:-/myworkspace/projects/videospy/config/videospy}"
CONFIG_DATASET="$DATASET"
[[ "$CONFIG_DATASET" != videomme_long ]] || CONFIG_DATASET=videomme
case "$BACKEND" in
  original) retrieval_files="${VIDEO_SPY_CONFIG_ROOT}/lmdeploy.yml" ;;
  freeret) retrieval_files="${VIDEO_SPY_CONFIG_ROOT}/lmdeploy.yml,${VIDEO_SPY_CONFIG_ROOT}/${CONFIG_DATASET}_retrieval_best.yml" ;;
  wemm4b) retrieval_files="${VIDEO_SPY_CONFIG_ROOT}/lmdeploy.yml,${VIDEO_SPY_CONFIG_ROOT}/wemm4b_local.yml" ;;
esac
export WORLDMM_RETRIEVAL_CONFIG="${WORLDMM_RETRIEVAL_CONFIG:-$retrieval_files}"
IDENTITY_FILE="${WORLDMM_EMBEDDING_IDENTITY_FILE:-${WORLDMM_NEEDED:-/myworkspace/projects/worldmm_needed}/freeret-identity.env}"
if [[ "$BACKEND" != original && -r "$IDENTITY_FILE" ]]; then
  source "$IDENTITY_FILE"
fi
export WORLDMM_TEXT_EMBEDDING_MODEL="${WORLDMM_TEXT_EMBEDDING_MODEL:-/myworkspace/models/Qwen/Qwen3-Embedding-4B}"
export WORLDMM_VISUAL_EMBEDDING_MODEL="${WORLDMM_VISUAL_EMBEDDING_MODEL:-/myworkspace/projects/experiment/worldmm/models/VLM2Vec}"
export WORLDMM_VLM_BACKBONE_MODEL="${WORLDMM_VLM_BACKBONE_MODEL:-/myworkspace/projects/experiment/worldmm/models/Qwen2-VL-2B-Instruct}"
export WORLDMM_LMDEPLOY_MODEL=Qwen3.5-4B
export WORLDMM_LMDEPLOY_BASE_URL="${WORLDMM_LMDEPLOY_BASE_URL:-http://127.0.0.1:23333/v1}"
export WORLDMM_INFERENCE_MODEL='Qwen3.5-4B'
export MODEL='Qwen3.5-4B'
export MODEL_PATH="${QWEN35_MODEL_PATH:-/myworkspace/models/Qwen/Qwen3.5-4B}"
export WORLDMM_NEEDED="${WORLDMM_NEEDED:-/myworkspace/projects/worldmm_needed}"
export WORLDMM_DATASET_ROOT="${WORLDMM_NEEDED}/${DATASET}"
export WORLDMM_SHARED_ROOT="${WORLDMM_DATASET_ROOT}/shared"
export WORLDMM_CAPTION_ROOT="${WORLDMM_SHARED_ROOT}/captions"
export WORLDMM_MANIFEST_ROOT="${WORLDMM_SHARED_ROOT}/manifests"
export WORLDMM_EXTRACTION_ROOT="${WORLDMM_SHARED_ROOT}/extractions"
export WORLDMM_BACKEND_ROOT="${WORLDMM_DATASET_ROOT}/${BACKEND_LABEL}"
export WORLDMM_MEMORY_ROOT="${WORLDMM_BACKEND_ROOT}/memories"
export WORLDMM_INDEX_ROOT="${WORLDMM_BACKEND_ROOT}/indexes"
export WORLDMM_VIEW_ROOT="${WORLDMM_BACKEND_ROOT}/views"
export WORLDMM_REPORT_ROOT="${WORLDMM_BACKEND_ROOT}/reports"
export WORLDMM_RUN_ROOT="${WORLDMM_BACKEND_ROOT}/runs"
if [[ -n "${WORLDMM_SMOKE_ROOT:-}" ]]; then
  export WORLDMM_SHARED_ROOT="${WORLDMM_SMOKE_ROOT}/${DATASET}/shared"
  export WORLDMM_CAPTION_ROOT="${WORLDMM_SHARED_ROOT}/captions"
  export WORLDMM_EXTRACTION_ROOT="${WORLDMM_SHARED_ROOT}/extractions"
  export WORLDMM_BACKEND_ROOT="${WORLDMM_SMOKE_ROOT}/${DATASET}/${BACKEND_LABEL}"
  export WORLDMM_MEMORY_ROOT="${WORLDMM_BACKEND_ROOT}/memories"
  export WORLDMM_INDEX_ROOT="${WORLDMM_BACKEND_ROOT}/indexes"
  export WORLDMM_VIEW_ROOT="${WORLDMM_BACKEND_ROOT}/views"
  export WORLDMM_REPORT_ROOT="${WORLDMM_BACKEND_ROOT}/reports"
  export WORLDMM_RUN_ROOT="${WORLDMM_BACKEND_ROOT}/runs"
fi
export WORLDMM_OUTPUT_ROOT="${WORLDMM_OUTPUT_ROOT:-/myworkspace/projects/output/worldmm}/${DATASET}/${BACKEND_LABEL}"
# This wrapper deliberately does not set sample, debug, limit, or query-days.
# Consumer commands must explicitly bind the paths above to their CLI arguments.
if [[ $# -eq 0 ]]; then
  for name in WORLDMM_EMBEDDING_BACKEND WORLDMM_RETRIEVAL_CONFIG WORLDMM_INFERENCE_MODEL MODEL_PATH WORLDMM_CAPTION_ROOT WORLDMM_MANIFEST_ROOT WORLDMM_EXTRACTION_ROOT WORLDMM_MEMORY_ROOT WORLDMM_INDEX_ROOT WORLDMM_VIEW_ROOT WORLDMM_REPORT_ROOT WORLDMM_RUN_ROOT WORLDMM_OUTPUT_ROOT; do
    printf '%s=%q\n' "$name" "${!name}"
  done
  exit 0
fi
[[ "$1" == '--' ]] || { echo 'Expected -- before command' >&2; exit 2; }
shift
[[ $# -gt 0 ]] || { echo 'Command required after --' >&2; exit 2; }
exec "$@"
