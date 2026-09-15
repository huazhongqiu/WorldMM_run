from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def script(name: str) -> str:
    return (ROOT / "script" / name / "4_eval.sh").read_text(encoding="utf-8")


def test_launchers_preflight_before_allocating_gpu_or_starting_lmdeploy() -> None:
    for benchmark in ("lvbench", "medvidbench"):
        content = script(benchmark)
        preflight = '"${PROJECT_ROOT}/eval/validate_precomputed.py"'
        assert preflight in content
        assert content.index(preflight) < content.index('serve api_server "${MODEL_PATH}"')
        assert '[[ -x "${WORLDMM_PYTHON}" ]]' in content
        assert '[[ -x "${LMDEPLOY_BIN}" ]]' in content
        assert '[[ -d "${MODEL_PATH}" ]]' in content


def test_launchers_match_videospy_lmdeploy_contract() -> None:
    required = (
        "--backend pytorch",
        "--tp 1",
        "--server-name 0.0.0.0",
        '--model-name "${MODEL}"',
        "--max-batch-size 8",
        '--cache-max-entry-count "${local_ratio}"',
        "--trust-remote-code",
        "--reasoning-parser default",
        "--tool-call-parser qwen3coder",
    )
    for benchmark in ("lvbench", "medvidbench"):
        content = script(benchmark)
        for argument in required:
            assert argument in content
        assert 'CACHE_RATIO="${CACHE_RATIO:-0.8}"' in content
        assert "--session-len" not in content
        assert "--language-model-only" not in content
        assert 'WORLDMM_LMDEPLOY_ENABLE_THINKING="${WORLDMM_LMDEPLOY_ENABLE_THINKING:-false}"' in content


def test_launchers_reject_an_endpoint_serving_the_wrong_model() -> None:
    for benchmark in ("lvbench", "medvidbench"):
        content = script(benchmark)
        assert "served_model_matches" in content
        assert "already responds but does not serve ${MODEL}" in content


def test_launchers_are_resumable_fail_closed_and_retry_once() -> None:
    for benchmark in ("lvbench", "medvidbench"):
        content = script(benchmark)
        assert 'EVAL_ATTEMPTS="${EVAL_ATTEMPTS:-2}"' in content
        assert '--max-rounds "${MAX_ROUNDS}"' in content
        assert "--require-complete" in content
        assert "records.jsonl" in content
        assert "attempt < EVAL_ATTEMPTS" in content


def test_launchers_default_to_dual_gpu_split_and_durable_outputs() -> None:
    lvbench = script("lvbench")
    medvidbench = script("medvidbench")
    for content in (lvbench, medvidbench):
        assert 'GPU_LIST="${GPU_LIST:-0,1}"' in content
        assert 'WORLDMM_EMBEDDING_DEVICE="${eval_emb_device}"' in content
        assert "/myworkspace/projects/output/worldmm/" in content
    assert 'eval_cvd="${GPUS[0]},${GPUS[-1]}"' in lvbench
    assert 'eval_cvd="${GPUS[0]},${GPUS[-1]}"' in medvidbench
    assert 'WORLDMM_VIS_EMBEDDING_DEVICE="${eval_vis_device}"' in medvidbench


def test_launchers_do_not_emit_ansi_color_escapes() -> None:
    for benchmark in ("lvbench", "medvidbench"):
        content = script(benchmark)
        assert "\\033[" not in content
        assert "echo -e" not in content


def test_eval_caches_stay_in_workspace_not_the_durable_output_directory() -> None:
    expected = {
        "lvbench": 'EVAL_CACHE_DIR="${WORLDMM_LVBENCH_EVAL_CACHE:-/workspace/worldmm/lvbench/eval_cache}"',
        "medvidbench": 'EVAL_CACHE_DIR="${WORLDMM_MEDVIDBENCH_EVAL_CACHE:-/workspace/worldmm/medvidbench/eval_cache}"',
    }
    for benchmark, declaration in expected.items():
        content = script(benchmark)
        assert declaration in content
        assert '--episodic-cache-dir "${EVAL_CACHE_DIR}"' in content
