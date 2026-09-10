#!/usr/bin/env bash
# Run a bounded A1_JAKE WorldMM smoke test using mounted EgoLife assets.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source /opt/conda/etc/profile.d/conda.sh
conda activate worldmm

export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}"
export WORLDMM_LMDEPLOY_BASE_URL="${WORLDMM_LMDEPLOY_BASE_URL:-http://127.0.0.1:23333/v1}"
export WORLDMM_LMDEPLOY_MODEL="${WORLDMM_LMDEPLOY_MODEL:-Qwen3.5-4B}"
export WORLDMM_LMDEPLOY_MAX_TOKENS="${WORLDMM_LMDEPLOY_MAX_TOKENS:-4096}"
export WORLDMM_TEXT_EMBEDDING_MODEL="${WORLDMM_TEXT_EMBEDDING_MODEL:-/myworkspace/mymodels/Qwen3-Embedding-4B}"
export WORLDMM_VISUAL_EMBEDDING_MODEL="${WORLDMM_VISUAL_EMBEDDING_MODEL:-/myworkspace/mymodels/VLM2Vec}"
export WORLDMM_VLM_BACKBONE_MODEL="${WORLDMM_VLM_BACKBONE_MODEL:-/myworkspace/mymodels/Qwen2-VL-2B-Instruct}"
export WORLDMM_TEXT_ATTENTION="${WORLDMM_TEXT_ATTENTION:-flash_attention_2}"
export WORLDMM_VLM_ATTENTION="${WORLDMM_VLM_ATTENTION:-flash_attention_2}"

python "${PROJECT_ROOT}/eval/eval_egolife.py" \
  --subject A1_JAKE \
  --data-dir /myworkspace/data/EgoLife/lmms-lab_EgoLife \
  --caption-dir /workspace/worldmm/egolife/data/EgoLife \
  --metadata-dir /myworkspace/projects/worldmm_needed/egolife \
  --video-root /myworkspace/data/EgoLife/lmms-lab_EgoLife \
  --output-dir /workspace/worldmm/eval \
  --retriever-model Qwen3.5-4B \
  --respond-model Qwen3.5-4B \
  --mode smoke \
  --limit "${1:-1}"
