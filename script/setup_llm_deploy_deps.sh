#!/usr/bin/env bash
# Add the WorldMM runtime dependencies to the patched LMDeploy environment, where the WeMM
# backend runs (Transformers 5). Every installed package is pinned so torch, transformers and
# lmdeploy stay untouched. Safe to rerun.
set -euo pipefail

PIP="${LLM_DEPLOY_PIP:-/opt/conda/envs/llm_deploy/bin/pip}"
constraints=$(mktemp)
trap 'rm -f "$constraints"' EXIT
"$PIP" freeze | grep -v -E '^-e|@ ' > "$constraints"
"$PIP" install -c "$constraints" python-igraph==1.0.0 tenacity==9.1.4 pysrt==1.1.2 pandas==3.0.1 \
  pyarrow==23.0.1 boto3==1.42.75 botocore==1.42.75 litellm==1.82.6
# HippoRAG imports GritLM at module load; its training dependencies are never used.
"$PIP" install --no-deps gritlm==1.0.2
