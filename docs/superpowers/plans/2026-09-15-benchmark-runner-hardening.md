# LVBench and MedVidBench Runner Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deliver resumable, fail-closed LVBench and MedVidBench inference runners that reuse completed WorldMM artifacts and can be launched unattended with one command on separate dual-GPU machines.

**Architecture:** Keep the existing dataset-specific evaluators and output formats. Add one shared read-only artifact validator, add checkpoint/resume primitives to the existing LVBench evaluation path, fix MedVidBench's shared embedding initialization race, and make both shell entrypoints preflight, retry incomplete inference once, and score only complete runs. LMDeploy remains a single TP=1 service on GPU 0 while embedding and visual retrieval use GPU 1.

**Tech Stack:** Python 3.12, pytest, Bash, LMDeploy 0.14.0 OpenAI-compatible API, Qwen3.5-4B, WorldMM, CUDA.

## Global Constraints

- Project code is `/myworkspace/projects/WorldMM`; durable results are `/myworkspace/projects/output`.
- Reuse offline artifacts from `/myworkspace/projects/worldmm_needed`; do not rebuild them.
- Put disposable caches and implementation work under `/workspace`; do not add generated data to `/myworkspace`.
- Match VideoSpy deployment exactly: PyTorch backend, TP 1, model name `Qwen3.5-4B`, max batch size 8, cache ratio 0.8, reasoning parser `default`, tool parser `qwen3coder`, no explicit session length.
- Both formal jobs must run in the foreground from one startup command, persist logs/results, checkpoint per sample, and require no later terminal interaction.
- Paper-critical retrieval limit is explicitly `max_rounds=5`.

---

### Task 1: Shared inference artifact preflight

**Files:**
- Create: `eval/validate_precomputed.py`
- Create: `tests/test_validate_precomputed.py`

**Interfaces:**
- Consumes: WorldMM QA JSON rows containing `ID` and `video_id`, persisted caption and memory roots, retriever model name.
- Produces: `validate_precomputed(eval_json: Path, root: Path, model: str) -> ValidationSummary` and a CLI that exits 0 only when every referenced video/segment has four captions, episodic files, semantic consolidation, and visual embeddings.

- [ ] **Step 1: Write failing tests** for a complete temporary artifact tree and for each missing artifact class. Assert exact question/unit counts and non-zero CLI behavior for incomplete data.
- [ ] **Step 2: Verify RED** with `/opt/conda/envs/worldmm/bin/python -m pytest tests/test_validate_precomputed.py -q`; expect import/file-not-found failure because the validator does not exist.
- [ ] **Step 3: Implement the validator** with `REQUIRED_GRANULARITIES = ("10sec", "30sec", "3min", "10min")`, deterministic missing-path reporting, duplicate-ID rejection, empty-QA rejection, and model-specific memory filenames.
- [ ] **Step 4: Verify GREEN** with the same pytest command; expect all validator tests to pass.

### Task 2: LVBench per-sample checkpoint and resume

**Files:**
- Modify: `eval/eval.py`
- Modify: `data/LVBench/utils/run_eval.py`
- Modify: `tests/test_lmdeploy_provider.py`
- Create: `tests/test_eval_checkpoint.py`

**Interfaces:**
- Consumes: existing LVBench QA rows and the current WorldMemory evaluation loop.
- Produces: `load_latest_results(path: str) -> dict[str, dict]`, `append_result(path: str, result: dict) -> None`, `merge_results(rows, latest) -> list[dict]`, `--records-jsonl`, and `--require-complete`.

- [ ] **Step 1: Write failing tests** proving malformed trailing JSONL is ignored, last record wins, only `status=success` resumes, merged results preserve QA order, incomplete runs return a failure status, and LMDeploy does not forward the local-only `fps` option.
- [ ] **Step 2: Verify RED** with `/opt/conda/envs/worldmm/bin/python -m pytest tests/test_eval_checkpoint.py tests/test_lmdeploy_provider.py -q`; expect missing checkpoint helpers and an unexpected `fps` request field.
- [ ] **Step 3: Implement minimal checkpointing**: append and flush one record after every answer/index error, skip only prior successes, retry prior errors, merge latest records into the existing eval JSON schema, record explicit `status`, and return non-zero after writing outputs when `--require-complete` finds errors/missing rows.
- [ ] **Step 4: Remove the LVBench-only `fps` constructor monkeypatch** after filtering `fps` in the LMDeploy provider; retain phase/token accounting in `run_eval.py`.
- [ ] **Step 5: Verify GREEN** with the targeted tests and then `/opt/conda/envs/worldmm/bin/python -m pytest tests -q`.

### Task 3: MedVidBench concurrency and completeness gate

**Files:**
- Modify: `eval/eval_medvidbench.py`
- Modify: `tests/test_medvidbench.py`

**Interfaces:**
- Consumes: current segment-level concurrent runner and shared `EmbeddingModel`.
- Produces: exactly-once thread-safe embedding initialization and `--require-complete` behavior after output/checkpoint creation.

- [ ] **Step 1: Write a failing concurrency test** that calls `Runner.ensure_embedding_model()` from eight threads and asserts one model construction/load, plus a failing completeness-summary test covering success/error/missing records.
- [ ] **Step 2: Verify RED** with `/opt/conda/envs/worldmm/bin/python -m pytest tests/test_medvidbench.py -q`; expect multiple model initializations or missing completeness API.
- [ ] **Step 3: Implement minimal locking and gating** with a dedicated initialization lock, a pure `completion_counts()` helper, `--require-complete`, output writing before failure, and no changes to prediction contracts.
- [ ] **Step 4: Verify GREEN** with targeted and full tests.

### Task 4: Unattended dual-GPU entrypoints

**Files:**
- Modify: `script/lvbench/4_eval.sh`
- Modify: `script/medvidbench/4_eval.sh`
- Modify: `script/lvbench/README.md`
- Modify: `script/medvidbench/README.md`
- Create: `tests/test_benchmark_launchers.py`

**Interfaces:**
- Consumes: validator CLI, checkpoint-aware evaluators, `/myworkspace` mount layout, `GPU_LIST=0,1`.
- Produces: foreground one-command jobs that preflight before GPU allocation, launch/reuse only the requested model, run up to `EVAL_ATTEMPTS=2`, pass `--max-rounds 5 --require-complete`, and score/report only after complete inference.

- [ ] **Step 1: Write failing shell-contract tests** asserting both scripts call the validator before `lmdeploy`, carry every VideoSpy deployment flag/value, reject a ready endpoint serving a different model ID, pass checkpoint/completeness/max-round flags, use GPU 0 for LMDeploy and GPU 1 for embeddings, and persist logs under `/myworkspace/projects/output`.
- [ ] **Step 2: Verify RED** with `/opt/conda/envs/worldmm/bin/python -m pytest tests/test_benchmark_launchers.py -q`.
- [ ] **Step 3: Implement the shell changes** without backgrounding the overall job. Keep LMDeploy parameters identical to `/myworkspace/projects/videospy/experiments/aicloud_submit.sh`; use `CACHE_RATIO=0.8` on dual GPU and do not add `--language-model-only` because WorldMM answer rounds may retrieve visual evidence.
- [ ] **Step 4: Document exact dual-GPU commands**, checkpoint semantics, output paths, failure behavior, and the GPU split.
- [ ] **Step 5: Verify GREEN** with launcher tests, `bash -n` for both scripts, and `git diff --check`.

### Task 5: Remote validation, timing evidence, and synchronization

**Files:**
- No new production files; inspect generated `/workspace` smoke artifacts and durable logs only.

**Interfaces:**
- Consumes: completed implementation and the current one-GPU `szu-worldmm` host.
- Produces: verified commits on `origin/main`, a clean `/myworkspace/projects/WorldMM`, two final startup commands, and an evidence-qualified dual-GPU runtime estimate.

- [ ] **Step 1: Run fresh full static verification**: all pytest tests, Python compileall for changed Python files, Bash syntax, artifact preflight against both real datasets, and `git diff --check`.
- [ ] **Step 2: Run bounded smoke/dry validation** on the current host without writing formal output; if one-GPU resources make end-to-end inference unsafe, record that boundary and use existing logs plus sample-level timings instead of claiming a dual-GPU measurement.
- [ ] **Step 3: Review the final diff and commit** only the planned files on `codex/benchmark-runner-hardening`.
- [ ] **Step 4: Fast-forward main, push `origin/main`, update `/myworkspace/projects/WorldMM`, and re-run the verification commands on the remote checkout.
- [ ] **Step 5: Report** the exact LVBench and MedVidBench launch commands, VideoSpy parameter-by-parameter alignment, dual-GPU execution strategy, current inference flow, expected time range with assumptions, outputs, and remaining risks.
