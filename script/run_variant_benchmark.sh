#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
dataset="${1:?dataset required}"
backend="${2:?backend required}"
action="${3:?action required: manifests|shared|semantic|semantic-index|episodic|visual|verify|eval}"
shift 3
export PYTHONPATH="$PWD:$PWD/src:$PWD/src/HippoRAG/src:$PWD/eval${PYTHONPATH:+:$PYTHONPATH}"
export WORLDMM_EPISODIC_EMBEDDING_BATCH_SIZE="${WORLDMM_EPISODIC_EMBEDDING_BATCH_SIZE:-4}"
python="${WORLDMM_PYTHON:-/opt/conda/envs/worldmm/bin/python}"
if [[ "$backend" == wemm4b && "$action" != manifests && "$action" != shared ]]; then
  python="${WORLDMM_WEMM_PYTHON:-/opt/conda/envs/llm_deploy/bin/python}"
  if [[ -n "${WORLDMM_WEMM_DEPENDENCIES:-}" ]]; then
    export PYTHONPATH="$PYTHONPATH:$WORLDMM_WEMM_DEPENDENCIES"
  fi
fi
if [[ "$action" == manifests ]]; then
  exec bash script/run_embedding_variant.sh "$dataset" "$backend" -- bash -c '
    exec "$1" -m preprocess.variant_inputs "$WORLDMM_DATASET" --root "$WORLDMM_DATASET_ROOT/shared" "${@:2}"
  ' _ "$python" "$@"
fi
if [[ "$action" != eval ]]; then
  exec bash script/run_embedding_variant.sh "$dataset" "$backend" -- \
    "$python" -m preprocess.build_variant --stage "$action" "$@"
fi

profile="${1:?retrieval profile required: Base|C|S|R|CS|CR|SR|CSR}"
run_id="${2:?run ID required}"
shift 2
case "$profile" in Base|C|S|R|CS|CR|SR|CSR) ;; *) exit 2 ;; esac
[[ "$run_id" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]] || exit 2
if [[ "$backend" != freeret && "$profile" != Base || "$dataset" == videomme_long && "$profile" == *R* ]]; then
  echo "Profile $profile is unavailable for $dataset/$backend" >&2
  exit 2
fi
for pair in C:CENTER S:SPARSE R:RERANK; do
  enabled=0
  [[ "$profile" != *"${pair%:*}"* ]] || enabled=1
  export "WORLDMM_RETRIEVAL_${pair#*:}=$enabled"
done
export WORLDMM_VIS_EMBEDDING_DEVICE="${WORLDMM_VIS_EMBEDDING_DEVICE:-cpu}"
exec bash script/run_embedding_variant.sh "$dataset" "$backend" -- bash -c '
  python=$1 profile=$2 run_id=$3
  shift 3
  output="$WORLDMM_OUTPUT_ROOT/$profile/$run_id"
  questions="$WORLDMM_MANIFEST_ROOT/questions.json"
  smoke_args=()
  if [[ -n "${WORLDMM_SMOKE_ROOT:-}" ]]; then
    output="$WORLDMM_BACKEND_ROOT/reports/qa/$profile/$run_id"
    questions="$WORLDMM_SHARED_ROOT/questions.json"
    smoke_args=(--visual-clips-file "$WORLDMM_CAPTION_ROOT/smoke_visual.json")
  fi
  mkdir -p "$output"
  if [[ "$WORLDMM_DATASET" == egolife ]]; then
    subject=A1_JAKE
    openie="$WORLDMM_EXTRACTION_ROOT/episodic_memory/$subject/openie_results_gpt-5-mini.json"
    exec "$python" eval/eval_egolife.py --subject "$subject" \
      --data-dir /myworkspace/data/EgoLife/lmms-lab_EgoLife \
      --eval-json "$questions" \
      --caption-dir "$WORLDMM_CAPTION_ROOT" --metadata-dir "$WORLDMM_MEMORY_ROOT" \
      --visual-index-dir "$WORLDMM_INDEX_ROOT" --openie-file "$openie" \
      --prepared-openie-dir "$WORLDMM_EXTRACTION_ROOT/openie" \
      --snapshot-dir "$WORLDMM_VIEW_ROOT/$subject/snapshots" \
      --cache-dir "$output/cache" --output-dir "$output" --require-complete "${smoke_args[@]}" "$@"
  else
    exec "$python" eval/eval.py --eval-json "$questions" \
      --caption-dir "$WORLDMM_CAPTION_ROOT" --metadata-dir "$WORLDMM_MEMORY_ROOT" \
      --extraction-dir "$WORLDMM_EXTRACTION_ROOT" --index-dir "$WORLDMM_INDEX_ROOT" \
      --view-dir "$WORLDMM_VIEW_ROOT" \
      --episodic-cache-dir "$WORLDMM_INDEX_ROOT" --eval-name "$WORLDMM_DATASET" \
      --output-dir "$output" --records-jsonl "$output/records.jsonl" --require-complete "$@"
  fi
' _ "$python" "$profile" "$run_id" "$@"
