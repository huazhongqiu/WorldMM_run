# 🌏 WorldMM: Dynamic Multimodal Memory Agent for Long Video Reasoning

<a href="https://arxiv.org/abs/2512.02425"><img src="https://img.shields.io/badge/arXiv-2512.02425-b31b1b.svg?logo=arxiv&logoColor=white" alt="arXiv"></a>
<a href="https://worldmm.github.io"><img src="https://img.shields.io/badge/Project-Page-green.svg" alt="Project Page"></a>
<a href="https://www.python.org/downloads/release/python-3100/"><img src="https://img.shields.io/badge/python-3.10+-blue.svg" alt="Python 3.10+"></a>

**WorldMM** is a novel dynamic multimodal memory agent designed for long video reasoning. It constructs multimodal, multi-scale memories that capture both textual and visual information, and employs adaptive retrieval across multiple memories with reasoning.

<img src="assets/concept.webp" alt="WorldMM Concept">

---

## Get Started

To set up the environment, we recommend using [uv](https://docs.astral.sh/uv/) for fast and deterministic setup. All dependencies are specified in `pyproject.toml` and pinned in `uv.lock`.

### 1. Clone the Repository
```sh
git clone https://github.com/wgcyeo/WorldMM.git
cd WorldMM
```

### 2. Run the Setup Script
The setup script will:
- Install uv (if not already installed)
- Install all project dependencies
- Download required datasets
```sh
bash script/1_setup.sh
```

### 3. Set Environment Variables (Optional)
To use GPT-family models for preprocessing or evaluation, set your OpenAI API key:
```sh
export OPENAI_API_KEY="your_openai_api_key"
```

## Preprocessing

Before memory construction and evaluation, preprocess the [EgoLife dataset](https://huggingface.co/datasets/lmms-lab/EgoLife):
```sh
bash script/2_preprocess.sh
```

After preprocessing, the dataset directory is organized as follows:
```
data/EgoLife/
├── A1_JAKE/
│   ├── DAY1/                    # Video files
│   ├── DAY2/
│   └── ...
├── EgoLifeCap/
│   ├── DenseCaption/            # Fine-grained video captions (in Chinese)
│   │   └── translated/          # Machine-translated English captions
│   ├── Sync/                    # Synchronized transcripts + captions
│   └── Transcript/              # Audio transcripts
└── EgoLifeQA/
    └── EgoLifeQA_A1_JAKE.json   # QA annotations
```

## Memory Construction

WorldMM builds three memory modules—episodic, semantic, and visual—to support long-term reasoning, which can be constructed with:
```sh
bash script/3_build_memory.sh
```
To run a specific module only:
```sh
bash script/3_build_memory.sh --step [episodic|semantic|visual]
```
#### Options
```sh
--step <type>       # Memory type: episodic, semantic, visual, all
--gpu <ids>         # GPU IDs to use (default: 0,1,2,3)
--model <name>      # LLM model for memory construction (default: gpt-5-mini)
```

> [!TIP]
> To skip memory construction and use the prebuilt WorldMM EgoLife memory metadata, download it from [wgcyeo/WorldMM-EgoLife](https://huggingface.co/datasets/wgcyeo/WorldMM-EgoLife) into `output/metadata`.

## Evaluation

Run evaluation on EgoLifeQA with:
```sh
bash script/4_eval.sh --retriever-model gpt-5-mini --respond-model gpt-5
```
#### Options
```sh
--retriever-model <m>   # Model for retrieval process (default: gpt-5-mini)
--respond-model <m>     # Model for iterative reasoning and generating answers (default: gpt-5)
--max-rounds <n>        # Max retrieval rounds (default: 5)
```
WorldMM supports a variety of backbone models for retrieval and reasoning, including `gpt-5` and `qwen3vl-8b`.

## Using WorldMM on Other Video Benchmarks

Beyond evaluation on week-long videos, WorldMM also supports evaluation on general video benchmarks. We provide an example pipeline for [Video-MME](https://github.com/MME-Benchmarks/Video-MME):
```sh
bash script/videomme/1_setup.sh
bash script/videomme/2_preprocess.sh
bash script/videomme/3_build_memory.sh --model gpt-5-mini
bash script/videomme/4_eval.sh --retriever-model gpt-5-mini --respond-model gpt-5
```
For detailed information about each step, please refer to the scripts located in `script/videomme`.

## Citation

If you find WorldMM helpful, please consider citing our paper:
```bibtex
@inproceedings{yeo2026worldmm,
  title     = {WorldMM: Dynamic Multimodal Memory Agent for Long Video Reasoning},
  author    = {Yeo, Woongyeong and Kim, Kangsan and Yoon, Jaehong and Hwang, Sung Ju},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  month     = {June},
  year      = {2026},
  pages     = {25599-25609}
}
```

## Acknowledgments

Our implementation is built upon [EgoLife](https://github.com/EvolvingLMMs-Lab/EgoLife), [HippoRAG](https://github.com/OSU-NLP-Group/HippoRAG), and [VLM2Vec](https://github.com/TIGER-AI-Lab/VLM2Vec). We thank the authors for open-sourcing their code and dataset.

### Mounted EgoLife Day 5–7 evaluation

The existing `script/run_egolife_test.sh` selects A1_JAKE questions by **query_time** on days 5, 6 and 7 (170 questions). It reads the mounted caption ZIP without writing to `data/`. From the project root:

```bash
# Single GPU: start Qwen3.5-4B, reuse the provided 30sec OpenIE, extract missing coarser OpenIE.
bash script/run_egolife_test.sh --prepare-openie-only
# Single GPU: embedding on cuda:0, no LLM service needed; save time-scoped multiscale snapshots.
bash script/run_egolife_test.sh --prepare-index-only
# Later, with two GPUs: use the prepared snapshots for evaluation.
bash script/run_egolife_test.sh
```

Persistent OpenIE and snapshots are stored at `/myworkspace/projects/worldmm_needed/egolife/prebuilt/A1_JAKE`; override with `WORLDMM_EGOLIFE_ARTIFACTS_DIR`. Preprocessing logs, extracted caption copies and working caches are saved under its `runs/` subdirectory unless `--output-dir` is supplied. Each snapshot includes only captions ending before or at the question time. Workers load private copies; incompatible or missing snapshots fail rather than silently rebuilding. Identity checks cover caption timestamps/text, base and prepared OpenIE, and the embedding and retrieval model paths/names. Preparation resumes complete OpenIE batches and snapshots. Keep the same input files and model settings across both stages and evaluation; use a new artifact directory when changing them. Repeated timestamp snapshots and worker copies can consume substantial disk space. OpenIE generation uses the configured local Qwen model for coarser captions; provided base OpenIE remains GPT-generated.

The evaluator also exposes `--query-days`, `--episodic-artifacts-dir`, `--prepare-openie-only` and `--prepare-index-only` directly. Supplying an artifact directory during evaluation requires all selected timestamps to have complete snapshots. A caption ZIP may be passed to `--caption-dir`; extraction goes into the run output directory. These options do not change the legacy 10sec OpenIE policy used by other benchmarks.


OpenIE preparation uses schema-constrained deterministic generation for entities and triples. Long caption text is extracted in segments of at most 1200 characters, preferring paragraph/word boundaries, and triples are deduplicated under the original caption ID and timestamp. A truncated segment is split further rather than retrying the same overlong input. This changes extraction context but preserves the four retrieval granularities. Completed legacy OpenIE records remain reusable; newly extracted records include a `bounded-openie-v1` policy marker. Every successful caption is saved independently, including when another caption in the batch fails. Only complete JSON documents are accepted: explanatory prose outside a complete JSON document is ignored, but truncated output is rejected explicitly. These format settings apply only to OpenIE; QA sampling remains unchanged.

After an interrupted preparation run, resume with:

```bash
bash script/run_egolife_test.sh --prepare-openie-only &&
bash script/run_egolife_test.sh --prepare-index-only
```


To uniformly re-extract **all 64 hourly captions**, including the one ending after the last test question, use the existing launcher:

```bash
bash script/run_egolife_test.sh --reextract-hourly-openie
```

This ignores legacy hourly OpenIE and other granularities, resuming only matching `bounded-openie-v1` checkpoints in `hourly_reextract/bounded-v1/1h.json` under the artifact root. All hourly captions use the same 1200-character segmentation policy. After all captions succeed, the prior active `openie/1h.json` is preserved once as `hourly_reextract/legacy-1h.json`, and the complete new result atomically replaces it. Other granularities remain untouched. Existing index snapshots become incompatible when hourly OpenIE changes; rebuild snapshots in a new artifact root before evaluating if snapshots already exist. Query-day selection still governs evaluation, not this all-week hourly re-extraction.
