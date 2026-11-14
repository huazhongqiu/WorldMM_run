# Embedding variant integration

`script/run_embedding_variant.sh DATASET BACKEND` prints the environment without
creating directories or starting a job. Valid datasets are `lvbench`, `egolife`,
`videomme_long`; backends are `original`, `freeret`, `wemm4b`. Passing `-- command`
executes that command with the exported environment.

The retrieval configuration is the ordered VideoSpy YAML overlay
`lmdeploy.yml,<dataset>_retrieval_best.yml` for FreeRet, with the reader loading
sibling `default.yml` first. Original uses the base retrieval profile. WeMM uses
`lmdeploy.yml,wemm4b_local.yml` with sparse retrieval disabled. Override
`VIDEO_SPY_CONFIG_ROOT` to relocate the source directory. Retrieval values are
read from those files; the launcher does not copy C/S/R defaults.

Consumer CLI arguments must explicitly bind directory variables. For the current
per-video builder, this invocation uses shared captions and backend-specific output:

```bash
bash script/run_embedding_variant.sh lvbench freeret -- bash -c '
  exec python preprocess/build_memory.py \
    --caption-dir "$WORLDMM_CAPTION_ROOT" \
    --output-dir "$WORLDMM_MEMORY_ROOT" \
    --model "$WORLDMM_INFERENCE_MODEL" --step semantic
'
```

This is an operational command example, not an instruction to start a full build.
The builder requires existing per-video `10sec.json` captions and episodic OpenIE
under the output tree. The current builder skips missing OpenIE and skips existing
consolidation based on filename: neither behavior establishes full coverage or
validates an old consolidation's encoder. Use a fresh backend tree until an audited
resume manifest covers semantic products. Shared extraction linking/import is not
implemented by this wrapper. Do not copy old consolidation into the new tree.

For per-video evaluator integration, explicitly pass `--caption-dir` from
`WORLDMM_CAPTION_ROOT`, `--metadata-dir` from `WORLDMM_MEMORY_ROOT`,
`--episodic-cache-dir` from `WORLDMM_INDEX_ROOT`, and a configuration-specific
`--output-dir` beneath `WORLDMM_OUTPUT_ROOT`. EgoLife snapshot/view writers must use
`WORLDMM_VIEW_ROOT` and include encoder identity in both saved and expected metadata.
These variables do not override arbitrary hardcoded paths in legacy scripts.

WeMM uses its own local `SentenceTransformer(..., trust_remote_code=True)` runtime.
Its documented environment requires Transformers 5.2.0,
SentenceTransformers >=5.7.0 and qwen-vl-utils[decord] 0.0.14; keep this separate from
legacy Qwen3/VLM2Vec dependencies. Model path and pixel/frame budgets come from the
WeMM model YAML. Set `WORLDMM_WEMM_REVISION` to the audited weights revision before
claiming a reproducible identity. Local adapter tests do not validate model weights
or the installed remote-code package. WeMM is dense-only; requesting sparse tokens
fails explicitly. Its baseline must not be reported as FreeRet CSR.

HippoRAG namespaces explicit backend indexes by encoder identity and validates
manifests before graph/Parquet load. Missing manifests in populated new namespaces
are rejected. Implicit legacy-original construction retains existing paths.
No cache migration, deletion, full preprocessing or QA run is performed by setup.

On `szu-worldmm`, the patched service environment is
`/opt/conda/envs/worldmm_qwen35_server` and the WeMM client environment is
`/opt/conda/envs/worldmm_wemm4b`. The launcher reads deployment fingerprints from
`/workspace/worldmm-freeret-setup/freeret-identity.env`. Start the patched service
on the address configured in VideoSpy `lmdeploy.yml` before a FreeRet run.

The YAML supplies visual scoring and model options. WorldMM still consumes its
caption-defined clip boundaries and the visual builder's frame-count budget.
EgoLife R requires aligned subtitle metadata per clip, including explicit empty
subtitles for clips without speech. Existing metadata must be checked before a full run.
