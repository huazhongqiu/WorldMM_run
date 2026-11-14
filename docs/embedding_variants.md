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

Use the dataset runner for the fixed experiment. It reads shared captions and
extractions, then writes semantic consolidation, indexes and causal views under
the selected backend. Run the patched Qwen3.5 service before model-dependent
stages. WeMM also needs the WorldMM Python dependencies in its Transformers 5
environment. On the server, set
`WORLDMM_WEMM_DEPENDENCIES=/workspace/worldmm-embedding-bench/wemm-deps` to use the
dependency overlay prepared for the smoke run.

```bash
bash script/run_variant_benchmark.sh lvbench original manifests
bash script/run_variant_benchmark.sh lvbench original shared --legacy-only
bash script/run_variant_benchmark.sh lvbench original shared
for backend in original freeret wemm4b; do
  for stage in semantic semantic-index episodic visual verify; do
    bash script/run_variant_benchmark.sh lvbench "$backend" "$stage" || exit
  done
done
bash script/run_variant_benchmark.sh lvbench freeret eval Base run-01
```

Use `egolife` or `videomme_long` in the same runner. Omit `--legacy-only` for those
datasets. For a smoke test, use a separate `WORLDMM_SMOKE_ROOT` and repeat the same
selection at each stage. Use `--unit VIDEO_ID --smoke-clips 3` for LVBench and
Video-MME. Use `--smoke-questions 4` for EgoLife. Formal QA uses the frozen question
manifest. FreeRet profiles Base, C, S, R, CS, CR, SR and CSR use the same indexes.
Video-MME supports Base, C, S and CS.

Episodic memory uses the dataset-specific configurations in the WorldMM paper's
implementation details.

| Dataset | Episodic timescales |
| --- | --- |
| EgoLifeQA | 30 seconds, 3 minutes, 10 minutes, 1 hour |
| LVBench and Video-MME | 10 seconds, 30 seconds, 3 minutes, 10 minutes |

The explicit snapshot reader uses the same four scales as the corresponding
builder. The generic memory class also supports their combined five-scale list.
The paper describes both dataset configurations in its implementation appendix.
[WorldMM implementation details](https://arxiv.org/html/2512.02425v2)

For the serial 4090 pipeline, run
`bash script/run_variant_full.sh RUN_ID`. It generates shared materials once,
builds the three backends, runs their Base evaluations and then FreeRet component
comparisons. It owns its Qwen3.5 service, starts it with a small KV cache and
releases it during local visual and graph construction. Progress, exit status,
service logs and the final comparison table live under
`/workspace/worldmm-embedding-bench/full/RUN_ID`.

| Dataset | Formal scope | Shared inputs and legacy provenance |
| --- | --- | --- |
| LVBench | 1,549 questions, 103 videos | The legacy 315-question run covers 20 videos. Its basic and multiscale captions, Qwen3.5-4B OpenIE and unconsolidated semantic extraction can be imported. Original visual pickles sample at 1,536 dimensions and the old builder used VLM2Vec. Exact historical frame indices and model hashes are absent. |
| EgoLife | A1_JAKE, seven days, 500 questions, 432 distinct cutoffs | The mounted A1_JAKE captions match the four JSON files in the author's tracked `data/EgoLife/caption.zip` byte for byte and contain 6,223 source clips. That archive contains 24 caption JSON files across six participants. GPT-5-mini OpenIE and semantic extraction come from the separate author metadata release. The server prepared multiscale OpenIE separately, including 64 bounded hourly extractions. The 155 existing episodic snapshots cover a subset of the full cutoffs and identify the original Qwen3-Embedding-4B text model. |
| Video-MME long | 900 questions, 300 videos | The official SRT files cover 292 videos. Eight videos with 24 questions use visual-only captions and remain in the manifest. |

Share the QA manifests, videos, aligned subtitles, base and multiscale captions,
NER/OpenIE and unconsolidated semantic triples. Rebuild semantic consolidation,
HippoRAG graphs and embeddings, semantic vectors, visual vectors and causal views
per backend. The legacy MedVidBench directory belongs to a separate benchmark.

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

LVBench and Video-MME caption windows use the YAML `clip_seconds` (currently 10).
EgoLife keeps the native caption windows and aligns English subtitle cues from the
hourly SRTs to the source clips. The visual builder shares decoded frame indices
across backends, with an even frame budget of 16 and a two-frame minimum for
VLM2Vec. Per-clip frame indices and subtitle counts are saved beside the visual
index. All three visual encoders receive the same sampled images. Subtitle text
enters the shared caption construction and EgoLife R scoring. The visual
embedding input contains frames only.
