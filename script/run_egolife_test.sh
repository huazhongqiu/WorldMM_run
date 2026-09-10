#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKERS="${WORLDMM_EGOLIFE_WORKERS:-4}"
OUTPUT_DIR="${WORLDMM_EGOLIFE_OUTPUT_ROOT:-/myworkspace/projects/output/worldmm/egolife}/test-$(date +%Y%m%d_%H%M%S)"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --workers) WORKERS="$2"; shift 2 ;;
    --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

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
export WORLDMM_EMBEDDING_DEVICE="${WORLDMM_EMBEDDING_DEVICE:-cuda:1}"

mkdir -p "${OUTPUT_DIR}"
echo "========== 服务检查 =========="
curl --fail --silent --show-error "${WORLDMM_LMDEPLOY_BASE_URL}/models" >/dev/null
echo "status=ready"
echo "========== 评测配置 =========="
echo "questions=95 workers=${WORKERS} output=${OUTPUT_DIR}"
echo "日志将显示每个 worker 的索引、检索轮次与最终答案。"
export PYTHONUNBUFFERED=1
export TQDM_DISABLE=1


python "${PROJECT_ROOT}/eval/eval_egolife.py" \
  --subject A1_JAKE \
  --data-dir /myworkspace/data/EgoLife/lmms-lab_EgoLife \
  --caption-dir /workspace/worldmm/egolife/data/EgoLife \
  --metadata-dir /myworkspace/projects/worldmm_needed/egolife \
  --video-root /myworkspace/data/EgoLife/lmms-lab_EgoLife \
  --output-dir "${OUTPUT_DIR}" \
  --cache-dir "${OUTPUT_DIR}/cache" \
  --retriever-model "${WORLDMM_LMDEPLOY_MODEL}" \
  --respond-model "${WORLDMM_LMDEPLOY_MODEL}" \
  --question-ids-file "${PROJECT_ROOT}/eval/splits/egolife_a1_jake_day6_day7_test.json" \
  --workers "${WORKERS}" \
  --mode test
