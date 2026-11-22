#!/usr/bin/env bash
# Write the FreeRet deployment identity sourced by script/run_embedding_variant.sh.
# Usage: bash script/make_freeret_identity.sh [OUTPUT_ENV] [MODEL_DIR]
# Model: config + weights. Processor: image/video preprocessors, chat template, tokenizer config.
# Tokenizer: the patched LMDeploy server's own tokenizer_fingerprint, which it returns per request.
set -euo pipefail

OUTPUT="${1:-${WORLDMM_EMBEDDING_IDENTITY_FILE:-/workspace/worldmm-freeret-setup/freeret-identity.env}}"
MODEL="${2:-${QWEN35_MODEL_PATH:-/myworkspace/models/Qwen/Qwen3.5-4B}}"
PYTHON="${WORLDMM_LMDEPLOY_PYTHON:-/opt/conda/envs/llm_deploy/bin/python}"

files_digest() {
  (cd "$MODEL" && sha256sum "$@") | sha256sum | cut -d' ' -f1
}

[[ -f "$MODEL/config.json" ]] || { echo "Model not found: $MODEL" >&2; exit 2; }
mapfile -t weights < <(cd "$MODEL" && ls model*.safetensors* | sort)
model_fingerprint=$(files_digest config.json "${weights[@]}")
processor_fingerprint=$(files_digest preprocessor_config.json video_preprocessor_config.json \
  chat_template.jinja tokenizer_config.json)
tokenizer_fingerprint=$("$PYTHON" -c '
import sys
from transformers import AutoTokenizer
from lmdeploy.pytorch.engine.model_agent.freeret import tokenizer_fingerprint
print(tokenizer_fingerprint(AutoTokenizer.from_pretrained(sys.argv[1], trust_remote_code=True)))
' "$MODEL")

mkdir -p "$(dirname "$OUTPUT")"
cat > "$OUTPUT" <<EOF
# Generated $(date -Iseconds) from $MODEL by script/make_freeret_identity.sh
export WORLDMM_FREERET_MODEL_FINGERPRINT=sha256:$model_fingerprint
export WORLDMM_FREERET_PROCESSOR_FINGERPRINT=sha256:$processor_fingerprint
export WORLDMM_FREERET_TOKENIZER_FINGERPRINT=$tokenizer_fingerprint
EOF
cat "$OUTPUT"
