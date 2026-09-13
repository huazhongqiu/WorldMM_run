# WorldMM × MedVidBench

MedVidBench（MedVidU ECCV2026 TrainVal 子集，开放式生成问答、纯帧输入）在 WorldMM 上的
预处理 + 推理 + 官方评测流水线。编排结构与 `script/lvbench/` 完全一致，评测方式与
`/myworkspace/projects/videospy/benchmark/medvidbench/` 相同（官方 leaderboard 评测器 +
本地 Qwen3.5-4B judge）。

## 测试集

videospy 的 1,124 题固定 split：`/myworkspace/projects/videospy/experiments/data_splits/main/medvidbench/test.json`
（整数下标 → `medvidu_eccv2026_trainval.json`）。去重后 1,011 个 segment
（`video_id&&start&&end&&fps`），帧来自 `.../MedVidU_ECCV2026_TrainVal/valdata/`
（json 内 `/root/data/` 前缀自动重映射）。

## 一键全流程

```bash
# 冒烟（debug split 2 题，产物只在 /workspace）：
SMOKE=1 bash /myworkspace/projects/WorldMM/script/medvidbench/launch.sh

# 全量预处理（1,011 段）：
bash /myworkspace/projects/WorldMM/script/medvidbench/launch.sh

# 预处理完成后，推理 + 官方评测 + 报告：
GPU_LIST=0 EVAL_WORKERS=4 bash /myworkspace/projects/WorldMM/script/medvidbench/4_eval.sh
# 只算确定性指标（不跑 LLM judge）：
SKIP_LLM_JUDGE=1 bash .../4_eval.sh
```

进度：`bash script/medvidbench/check_progress.sh`，日志在
`/workspace/worldmm/medvidbench/logs/`。

## 阶段与产物

| 阶段 | 输入 → 输出 | 位置 |
| --- | --- | --- |
| 0 prepare | trainval + split → `qa/medvidbench_test.json`（WorldMM qa schema，开放式）、`qa/test_segments.json`、`qa/test_videos.json` | `worldmm_needed/medvidbench/qa/` |
| 1 synthesize | 每段帧列表 → `<segment_key>.mp4`（`-framerate metadata.fps`，帧 i 的本地时间 = i/fps，与官方评测器的秒→帧换算一致） | `/workspace/worldmm/medvidbench/videos/` |
| 2 captions | mp4 → `caption/<segment_key>/{10sec}.json`（复用 `data/LVBench/utils/generate_fine_caption.py`，纯视觉、无 ASR） | `worldmm_needed/medvidbench/caption/` |
| 3 multiscale | 10sec → 30sec/3min/10min（复用 `worldmm.memory.episodic.multiscale`） | 同上 |
| 4 episodic / 5 semantic / 6 visual | 复用 `preprocess/build_memory.py` 三步 | `worldmm_needed/medvidbench/{episodic,semantic,visual}_memory/` |
| 7 eval | `eval/eval_medvidbench.py`（`qa_medvidbench` 模板、choices=None、records.jsonl 断点续跑、submission.json）→ `data/MedVidBench/utils/evaluate_medvidbench.py`（官方 `evaluate_predictions.py` 子进程，judge=本地 Qwen3.5-4B）→ `make_report.py` | `/myworkspace/projects/output/worldmm/medvidbench/` |

RC（region caption）题在合成 mp4 时按 `RC_info.start_frame_bbox` 画绿框（同 videospy），
且使用独立 segment key，不会污染同段非 RC 题的记忆。

## 关键环境变量

`GPU_LIST`（默认 0）、`MODEL`（默认 Qwen3.5-4B）、`SAMPLE_FPS`（1.0）、
`MAX_FRAME_EDGE`（1280）、`NUM_FRAMES`（16，VLM2Vec 每片段帧数）、`SYNTH_WORKERS`（8）、
`WITH_EVAL`（预处理后是否接评测）、`EVAL_WORKERS`（评测并发段数）、`SKIP_LLM_JUDGE`、
`SMOKE`、`WORLDMM_MEDVIDBENCH_ROOT/SCRATCH/OUTPUT`。

## 注意

- 本地 judge 的 `DVC/VS/RC_llm` 分数与官方 GPT-4.1 不直接可比（videospy 同样如此），
  报告中已标注 judge 模型。
- 全部 mp4/caption/记忆产物可断点续跑：重跑 `run_all.sh` 自动跳过已完成 artifact。
