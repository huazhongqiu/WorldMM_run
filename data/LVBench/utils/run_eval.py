#!/usr/bin/env python3
"""Run ``eval/eval.py`` for LVBench without modifying WorldMM core code.

``eval/eval.py`` instantiates the respond LLM with ``fps=1`` — a sampling hint
that only the local qwen3vl backend understands. The LMDeploy provider stores
extra constructor kwargs verbatim and forwards them to the OpenAI-compatible
``chat.completions.create``, which rejects ``fps``. This wrapper patches
``LMDeployModel.__init__`` to drop provider-incompatible kwargs before the
model is constructed, then delegates everything else to ``eval.main()``.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]  # .../WorldMM

TOKEN_USAGE: dict[str, int] = {
    "input_tokens": 0,
    "output_tokens": 0,
    "total_tokens": 0,
    "llm_calls": 0,
}
# `indexing` = episodic HippoRAG graph construction (NER/OpenIE) triggered by
# WorldMemory.index; `answering` = retrieval rounds + answer generation (QA).
# Reported token usage counts ANSWERING ONLY.
PHASE = {"current": "answering"}
USAGE_BY_PHASE: dict[str, dict[str, int]] = {}


def _bucket(phase: str) -> dict[str, int]:
    return USAGE_BY_PHASE.setdefault(phase, {k: 0 for k in TOKEN_USAGE})


def patch_phase_tracking() -> None:
    from worldmm.memory import WorldMemory

    original_index = WorldMemory.index

    def indexed(self, *args, **kwargs):
        PHASE["current"] = "indexing"
        try:
            return original_index(self, *args, **kwargs)
        finally:
            PHASE["current"] = "answering"

    WorldMemory.index = indexed


def patch_usage_collection() -> None:
    from worldmm.llm import lmdeploy

    original_generate = lmdeploy.LMDeployModel.generate

    def patched_generate(self, prompt, text_format=None, **kwargs):
        # LMDeployModel.generate returns the message string, so we take the
        # delta of its own usage snapshot around the call instead.
        before = dict(self.usage)
        result = original_generate(self, prompt, text_format=text_format, **kwargs)
        after = self.usage
        bucket = _bucket(PHASE["current"])
        bucket["input_tokens"] += max(0, after["prompt_tokens"] - before["prompt_tokens"])
        bucket["output_tokens"] += max(0, after["completion_tokens"] - before["completion_tokens"])
        bucket["total_tokens"] += max(0, after["total_tokens"] - before["total_tokens"])
        bucket["llm_calls"] += 1
        if os.environ.get("WORLDMM_LVBENCH_USAGE_DEBUG") and bucket["llm_calls"] <= 3:
            print(f"[usage-debug] phase={PHASE['current']} before={before} after={after}", file=sys.stderr)
        return result

    lmdeploy.LMDeployModel.generate = patched_generate


def main() -> int:
    sys.path.insert(0, str(PROJECT_ROOT / "eval"))
    patch_phase_tracking()
    patch_usage_collection()
    import eval as lvbench_eval  # eval/eval.py is a script, not a package module

    sys.argv[0] = "eval.py"
    exit_code = lvbench_eval.main()

    usage_path = os.environ.get("WORLDMM_LVBENCH_USAGE_FILE")
    if usage_path:
        zero = {k: 0 for k in TOKEN_USAGE}
        answering = USAGE_BY_PHASE.get("answering", zero)
        payload = dict(answering)
        payload["phase"] = "answering_only (episodic indexing excluded)"
        payload["indexing"] = USAGE_BY_PHASE.get("indexing", zero)
        Path(usage_path).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"Token usage written to {usage_path}: {json.dumps(payload)}")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
