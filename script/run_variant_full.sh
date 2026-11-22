#!/usr/bin/env bash
# Build every index (script/run_variant_build.sh), then run the requested QA comparisons.
# Qwen3.5 uses the VideoSpy service settings throughout; see run_variant_build.sh for GPU layout.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

run_id="${1:?run ID required}"
[[ "$run_id" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]] || exit 2
run_root="${WORLDMM_PIPELINE_ROOT:-/workspace/worldmm-embedding-bench/full}/$run_id"
mkdir -p "$run_root"
exec 9>"$run_root/pipeline.lock"
flock -n 9 || { echo "Run $run_id is already active" >&2; exit 2; }
WORLDMM_PIPELINE_ROOT="$run_root" bash script/run_variant_build.sh build
exec > >(tee -a "$run_root/pipeline.log") 2>&1
export WORLDMM_LMDEPLOY_MAX_TOKENS="${WORLDMM_LMDEPLOY_MAX_TOKENS:-4096}"
export WORLDMM_LMDEPLOY_HOST=127.0.0.1
export WORLDMM_LMDEPLOY_GPU="${WORLDMM_LLM_GPU:-0}"
export WORLDMM_EMBEDDING_DEVICE="cuda:${WORLDMM_EMBEDDING_GPU:-1}"
export WORLDMM_VIS_EMBEDDING_DEVICE="cuda:${WORLDMM_EMBEDDING_GPU:-1}"
unset WORLDMM_LMDEPLOY_CACHE WORLDMM_LMDEPLOY_MAX_BATCH_SIZE WORLDMM_LMDEPLOY_SESSION_LEN WORLDMM_LMDEPLOY_EAGER
server_pid=

stop_server() {
  if [[ -n "$server_pid" ]]; then
    kill -TERM "$server_pid" 2>/dev/null || true
    wait "$server_pid" 2>/dev/null || true
    server_pid=
  fi
}

finish() {
  result=$?
  stop_server
  printf '%s\n' "$result" > "$run_root/exit_code"
  exit "$result"
}
trap finish EXIT
trap 'exit 130' INT TERM

start_server() {
  if curl -fsS http://127.0.0.1:23333/v1/models >/dev/null 2>&1; then
    echo "Port 23333 is occupied by an external service" >&2
    exit 2
  fi
  bash script/serve_lmdeploy.sh videospy > "$run_root/server-$(date +%Y%m%d_%H%M%S).log" 2>&1 &
  server_pid=$!
  for ((attempt=0; attempt<180; attempt++)); do
    if curl -fsS http://127.0.0.1:23333/v1/models >/dev/null 2>&1; then return; fi
    kill -0 "$server_pid" || { wait "$server_pid"; exit 1; }
    sleep 5
  done
  echo "Qwen3.5 service startup timed out" >&2
  exit 1
}

run_stage() {
  dataset=$1 backend=$2 stage=$3
  printf '%s %s %s %s\n' "$(date -Iseconds)" "$dataset" "$backend" "$stage" > "$run_root/current_stage"
  bash script/run_variant_benchmark.sh "$dataset" "$backend" "$stage" "${@:4}"
}

start_server
qa_failed=0
for backend in original freeret wemm4b; do
  for dataset in lvbench egolife videomme_long; do
    run_stage "$dataset" "$backend" eval Base "$run_id" || qa_failed=1
  done
done
for dataset in lvbench egolife videomme_long; do
  profiles=(C S R CS CR SR CSR)
  if [[ "$dataset" == videomme_long ]]; then profiles=(C S CS); fi
  for profile in "${profiles[@]}"; do
    run_stage "$dataset" freeret eval "$profile" "$run_id" || qa_failed=1
  done
done
"${WORLDMM_PYTHON:-/opt/conda/envs/worldmm/bin/python}" eval/summarize_variants.py \
  --needed-root "${WORLDMM_NEEDED:-/myworkspace/projects/worldmm_needed}" \
  --output-root "${WORLDMM_OUTPUT_ROOT:-/myworkspace/projects/output/worldmm}" \
  --run-id "$run_id" --report-dir "$run_root"
exit "$qa_failed"
