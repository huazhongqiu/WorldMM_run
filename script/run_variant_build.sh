#!/usr/bin/env bash
# Build shared inputs and every backend's indexes; no QA. Resumable: rerun the same RUN_ID.
# Qwen3.5 always runs with the VideoSpy service settings (serve_lmdeploy.sh videospy) on
# WORLDMM_LLM_GPU. The original and WeMM encoders run on WORLDMM_EMBEDDING_GPU, so they need a
# second GPU; FreeRet embeds through the service and needs only WORLDMM_LLM_GPU.
# Usage: bash script/run_variant_build.sh RUN_ID [DATASET ...]
#   WORLDMM_BUILD_BACKENDS  space-separated subset of "original freeret wemm4b", or "none"
#                           to stop after the shared stage
#   WORLDMM_BUILD_ARGS      extra build_variant arguments, e.g. smoke-test unit limits
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

run_id="${1:?run ID required}"
[[ "$run_id" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]] || exit 2
shift
datasets=("$@")
[[ ${#datasets[@]} -gt 0 ]] || datasets=(lvbench egolife videomme_long)
read -r -a backends <<< "${WORLDMM_BUILD_BACKENDS:-original freeret wemm4b}"
[[ "${backends[*]}" != none ]] || backends=()
read -r -a build_args <<< "${WORLDMM_BUILD_ARGS:-}"
llm_gpu="${WORLDMM_LLM_GPU:-0}"
embedding_gpu="${WORLDMM_EMBEDDING_GPU:-1}"
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  IFS=, read -r -a visible <<< "$CUDA_VISIBLE_DEVICES"
  gpus=${#visible[@]}
else
  gpus=$(nvidia-smi -L | wc -l)
fi
gpu_ok() { [[ "$1" =~ ^[0-9]+$ && "$1" -lt "$gpus" ]]; }
gpu_ok "$llm_gpu" || { echo "GPU $llm_gpu not available ($gpus detected)" >&2; exit 2; }
if [[ " ${backends[*]} " =~ " "(original|wemm4b)" " ]]; then
  if ! gpu_ok "$embedding_gpu" || [[ "$llm_gpu" == "$embedding_gpu" ]]; then
    echo "original/WeMM need the service and a local encoder on different GPUs;" \
         "set WORLDMM_EMBEDDING_GPU or WORLDMM_BUILD_BACKENDS=freeret" >&2
    exit 2
  fi
fi

run_root="${WORLDMM_PIPELINE_ROOT:-${WORLDMM_NEEDED:-/myworkspace/projects/worldmm_needed}/pipeline_runs/build}/$run_id"
mkdir -p "$run_root"
exec 9>"$run_root/pipeline.lock"
flock -n 9 || { echo "Run $run_id is already active" >&2; exit 2; }
git rev-parse HEAD > "$run_root/code_commit" 2>/dev/null || true
git status --porcelain > "$run_root/working_tree_status" 2>/dev/null || true
exec > >(tee -a "$run_root/pipeline.log") 2>&1
export WORLDMM_LMDEPLOY_MAX_TOKENS="${WORLDMM_LMDEPLOY_MAX_TOKENS:-4096}"
export WORLDMM_LMDEPLOY_HOST=127.0.0.1
export WORLDMM_LMDEPLOY_GPU="$llm_gpu"
export WORLDMM_EMBEDDING_DEVICE="cuda:$embedding_gpu"
export WORLDMM_VIS_EMBEDDING_DEVICE="cuda:$embedding_gpu"
# Never let leftover overrides change the VideoSpy service settings.
unset WORLDMM_LMDEPLOY_CACHE WORLDMM_LMDEPLOY_MAX_BATCH_SIZE WORLDMM_LMDEPLOY_SESSION_LEN WORLDMM_LMDEPLOY_EAGER
server_pid=

stop_server() {
  # The service runs in its own process group; stop LMDeploy's engine workers with it.
  if [[ -n "$server_pid" ]]; then
    kill -TERM -- "-$server_pid" 2>/dev/null || true
    for _ in {1..30}; do
      kill -0 -- "-$server_pid" 2>/dev/null || break
      sleep 1
    done
    kill -KILL -- "-$server_pid" 2>/dev/null || true
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
  if [[ -n "$server_pid" ]]; then return; fi
  if curl -fsS http://127.0.0.1:23333/v1/models >/dev/null 2>&1; then
    echo "Port 23333 is occupied by an external service" >&2
    exit 2
  fi
  setsid bash script/serve_lmdeploy.sh videospy > "$run_root/server-$(date +%Y%m%d_%H%M%S).log" 2>&1 &
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
  printf '%s %s %s %s\n' "$(date -Iseconds)" "$dataset" "$backend" "$stage" | tee "$run_root/current_stage"
  if [[ "$stage" == manifests ]]; then
    bash script/run_variant_benchmark.sh "$dataset" "$backend" "$stage"
  else
    bash script/run_variant_benchmark.sh "$dataset" "$backend" "$stage" "${build_args[@]}"
  fi
  printf '%s %s %s %s\n' "$(date -Iseconds)" "$dataset" "$backend" "$stage" >> "$run_root/completed_stages"
}

for dataset in "${datasets[@]}"; do
  run_stage "$dataset" original manifests
done
start_server
for dataset in "${datasets[@]}"; do
  run_stage "$dataset" original shared
done
for backend in "${backends[@]}"; do
  for stage in semantic semantic-index episodic visual verify; do
    for dataset in "${datasets[@]}"; do
      run_stage "$dataset" "$backend" "$stage"
    done
  done
done
echo "Build complete: ${datasets[*]} x ${backends[*]}"
