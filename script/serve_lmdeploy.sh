#!/usr/bin/env bash
set -euo pipefail

PROFILE="${1:-debug-1gpu}"
LMDEPLOY="${LMDEPLOY_BIN:-/opt/conda/envs/llm_deploy/bin/lmdeploy}"
MODEL="${WORLDMM_LMDEPLOY_MODEL_PATH:-/myworkspace/models/Qwen/Qwen3.5-4B}"
PORT="${WORLDMM_LMDEPLOY_PORT:-23333}"
HOST="${WORLDMM_LMDEPLOY_HOST:-0.0.0.0}"

fail() {
  echo "ERROR: $*" >&2
  exit 2
}

[[ -x "$LMDEPLOY" ]] || fail "LMDeploy executable not found: $LMDEPLOY"
[[ -f "$MODEL/config.json" ]] || fail "Model not found: $MODEL"

case "$PROFILE" in
  debug-1gpu)
    GPU="${WORLDMM_LMDEPLOY_GPU:-0}"
    CACHE="${WORLDMM_LMDEPLOY_CACHE:-0.1}"
    MAX_BATCH_SIZE="${WORLDMM_LMDEPLOY_MAX_BATCH_SIZE:-1}"
    ;;
  formal-2gpu)
    GPU="${WORLDMM_LMDEPLOY_GPU:-0}"
    CACHE="${WORLDMM_LMDEPLOY_CACHE:-0.8}"
    MAX_BATCH_SIZE="${WORLDMM_LMDEPLOY_MAX_BATCH_SIZE:-8}"
    echo "Formal profile: reserve physical GPU 1 for embeddings with WORLDMM_EMBEDDING_DEVICE=cuda:1." >&2
    ;;
  *)
    echo "Usage: $0 [debug-1gpu|formal-2gpu]" >&2
    exit 2
    ;;
esac

[[ "$CACHE" =~ ^0\.[0-9]+$|^1(\.0+)?$ ]] || fail "WORLDMM_LMDEPLOY_CACHE must be in (0, 1]"
[[ "$MAX_BATCH_SIZE" =~ ^[1-9][0-9]*$ ]] || fail "WORLDMM_LMDEPLOY_MAX_BATCH_SIZE must be a positive integer"

echo "Starting Qwen3.5-4B profile=$PROFILE gpu=$GPU host=$HOST cache=$CACHE max_batch_size=$MAX_BATCH_SIZE" >&2
exec env CUDA_VISIBLE_DEVICES="$GPU" "$LMDEPLOY" serve api_server "$MODEL" \
  --backend pytorch \
  --tp 1 \
  --server-name "$HOST" \
  --server-port "$PORT" \
  --model-name Qwen3.5-4B \
  --max-batch-size "$MAX_BATCH_SIZE" \
  --cache-max-entry-count "$CACHE" \
  --trust-remote-code \
  --reasoning-parser default \
  --tool-call-parser qwen3coder \
  --log-level REQUEST
