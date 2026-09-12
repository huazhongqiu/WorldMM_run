#!/usr/bin/env bash
# Print LVBench preprocessing progress (artifact counts per stage).
# Usage: check_progress.sh [--oneline]
set -euo pipefail

LVBENCH_ROOT="${WORLDMM_LVBENCH_ROOT:-/myworkspace/projects/worldmm_needed/lvbench}"
MODEL="${MODEL:-Qwen3.5-4B}"
ONELINE=0
[[ "${1:-}" == "--oneline" ]] && ONELINE=1

PY="${WORLDMM_PYTHON:-python3}"
[[ -x /opt/conda/envs/worldmm/bin/python ]] && PY=/opt/conda/envs/worldmm/bin/python

EXPECTED=$("$PY" - <<PYEOF 2>/dev/null || echo 0
import json
print(len(json.load(open("${LVBENCH_ROOT}/qa/test_videos.json"))))
PYEOF
)
EXPECTED=${EXPECTED:-0}

count_files() {
    local root="$1" pattern="$2"
    if [[ ! -d "$root" ]]; then echo 0; return; fi
    find "$root" -name "$pattern" 2>/dev/null | wc -l
}

cap_10sec=$(count_files "${LVBENCH_ROOT}/caption" "10sec.json")
cap_multiscale=$(count_files "${LVBENCH_ROOT}/caption" "10min.json")
episodic=$(count_files "${LVBENCH_ROOT}/episodic_memory" "episodic_triple_results_${MODEL}.json")
semantic=$(count_files "${LVBENCH_ROOT}/semantic_memory" "semantic_consolidation_results_${MODEL}.json")
visual=$(count_files "${LVBENCH_ROOT}/visual_memory" "visual_embeddings.pkl")

pct() {
    if (( EXPECTED > 0 )); then echo "$(( $1 * 100 / EXPECTED ))%"; else echo "n/a"; fi
}

if (( ONELINE )); then
    echo "captions ${cap_10sec}/${EXPECTED} ($(pct "$cap_10sec")) | multiscale ${cap_multiscale}/${EXPECTED} | episodic ${episodic}/${EXPECTED} | semantic ${semantic}/${EXPECTED} | visual ${visual}/${EXPECTED}"
else
    echo "=========== LVBench preprocessing progress (root: ${LVBENCH_ROOT}) ==========="
    printf "%-14s %8s / %-8s %8s\n" "stage" "done" "total" "percent"
    printf "%-14s %8s / %-8s %8s\n" "qa+captions" "$cap_10sec" "$EXPECTED" "$(pct "$cap_10sec")"
    printf "%-14s %8s / %-8s %8s\n" "multiscale" "$cap_multiscale" "$EXPECTED" "$(pct "$cap_multiscale")"
    printf "%-14s %8s / %-8s %8s\n" "episodic" "$episodic" "$EXPECTED" "$(pct "$episodic")"
    printf "%-14s %8s / %-8s %8s\n" "semantic" "$semantic" "$EXPECTED" "$(pct "$semantic")"
    printf "%-14s %8s / %-8s %8s\n" "visual" "$visual" "$EXPECTED" "$(pct "$visual")"
    echo "================================================================"
fi
