# WorldMM × LVBench 预处理 + 推理接线

对 LVBench 测试集（videospy 划分：**20 个视频 / 315 题**，`/myworkspace/projects/videospy/experiments/data_splits/main/lvbench/test.json`）构建 WorldMM 全部多模态记忆，产物可直接用 `eval/eval.py` 推理。

多尺度 caption 粒度为 **10s / 30s / 3min / 10min**（10s 为基础 caption，其余为 LLM 分层摘要），**纯视觉 caption，不做语音转写（ASR）**。全部 LLM 步骤使用 **Qwen3.5-4B（LMDeploy 服务）**。

## 一条命令跑完（推荐）

```bash
bash /myworkspace/projects/WorldMM/script/lvbench/launch.sh
```

前台启动（适合 k8s/集群作业的控制台直接粘贴）：自动创建日志目录（避免外层重定向 `No such file or directory`）、实时显示全部输出（含 `PHASE x/5` 分界横幅和 tqdm 进度），同时完整落盘一份日志；启动时会打印 `tail -f` 路径，供别的终端跟看。所有旋钮照常透传：

```bash
GPU_LIST=auto bash /myworkspace/projects/WorldMM/script/lvbench/launch.sh
```

脚本自动完成：QA 转换 → 拉起 LMDeploy（**与 videospy 完全相同的配置**）→ 10s caption（分片并行）→ 30s/3min/10min 多尺度摘要 → episodic 三元组（NER+OpenIE）→ semantic 抽取+consolidation → visual 嵌入（VLM2Vec）→（可选）eval。全程**断点续跑**：中断/失败后重新执行同一条命令即可跳过已完成部分。

### 实时进度

启动后控制台实时显示全部输出；此外 run_all 每 60 秒（`PROGRESS_INTERVAL` 可调）自动打印一块**纯文本进度看板**（无 ANSI 转义符，平台网页日志可正常渲染），当前阶段的分组下有 total 和各分片的【完成数/总数】条，并标注分片正在处理的视频及段级进度：

```
====== captions ======
  total  【  7/ 20】 35% ███████░░░░░░░░░░░░░
  shard0 【  2/  5】 40% ████████░░░░░░░░░░░░  进行中 16Z-XQh9jhk 312/503 段
====== multiscale ======
  total  【  0/ 20】  0% ░░░░░░░░░░░░░░░░░░░░
...
```

```bash
# 方式一：跟随主日志（看板 + 各阶段横幅都在里面）
tail -f /workspace/worldmm/lvbench/logs/run_all_<时间戳>.log

# 方式二：手动查五阶段完成度
bash /myworkspace/projects/WorldMM/script/lvbench/check_progress.sh
```

## 模型部署配置（与 videospy 保持一致）

每个 GPU 启动一个实例（`conda activate llm_deploy`，lmdeploy==0.14.0），端口从 23333 递增：

```bash
CUDA_VISIBLE_DEVICES=<gpu> lmdeploy serve api_server /myworkspace/models/Qwen/Qwen3.5-4B \
  --backend pytorch --tp 1 --server-name 0.0.0.0 --server-port <port> \
  --model-name Qwen3.5-4B --max-batch-size 8 --cache-max-entry-count 0.8 \
  --trust-remote-code --reasoning-parser default --tool-call-parser qwen3coder \
  --log-level REQUEST
```

不设置 `--session-len` / `--max-prefill-token-num`（用默认值），与 `videospy/experiments/aicloud_submit.sh` 和 `videospy/docs/scripts/lmdeploy.md` 一致。

## 常用旋钮（环境变量）

| 变量 | 默认 | 说明 |
|---|---|---|
| `GPU_LIST` | `0,1` | 使用的 GPU；`auto` 自动探测全部卡。正式推理建议至少双卡；`GPU_LIST=0` 仅作为可能 OOM 的兼容入口 |
| `CACHE_RATIO` | `0.8` | LLM 独占卡时的 KV cache 比例（videospy 同款） |
| `COLOCATED_CACHE_RATIO` | `0.35` | 单卡模式下 semantic/eval 阶段 LLM 服务与 embedding 模型共享显存时的 KV cache 比例（自动以小缓存重启服务） |
| `SAMPLE_FPS` | `1.0` | caption 抽帧率；`0.5` 可显著提速（约省一半时间） |
| `MAX_FRAME_EDGE` | `0` | caption 帧分辨率上限（最长边，像素）。**默认 0 = 原生分辨率，与原版源代码行为一致**；设 `1280` 可把视觉 token 量降到约 1/2.3，caption 阶段耗时约减半（论文未规定帧分辨率，属可选提速项） |
| `NUM_FRAMES` | `16` | visual memory 每个 10s 片段编码的帧数（Video-MME 脚本用 10） |
| `NUM_SHARDS` | 自动=2×卡数 | caption/multiscale/episodic 的客户端分片数 |
| `CAPTION_WORKERS` | `16` | 每个分片内并发请求的段数 |
| `WITH_EVAL` | `0` | `1` = 预处理完成后自动对 315 题跑 eval（eval 的 episodic 索引本身就需大量 LLM 调用，首次很耗时） |
| `SMOKE` | `0` | `1` = 冒烟模式：剪 3 分钟片段 + 3 题，产物只写 `/workspace` |
| `BASE_PORT` | `23333` | LMDeploy 起始端口 |
| `MODEL` / `MODEL_PATH` | `Qwen3.5-4B` / `/myworkspace/models/Qwen/Qwen3.5-4B` | 模型名与路径 |

路径默认值：数据 `/myworkspace/data/LVBench/...`、输出 `/myworkspace/projects/worldmm_needed/lvbench`、中间产物/日志 `/workspace/worldmm/lvbench`、eval 结果 `/myworkspace/projects/output/worldmm/lvbench`，均可用对应环境变量覆盖（见 `run_all.sh` 头部）。

## 产物布局（推理直接可用）

```
/myworkspace/projects/worldmm_needed/lvbench/
├── qa/lvbench_test.json          # 315 题（Video-MME eval 格式）
├── qa/test_videos.json           # 20 个测试视频 key
├── caption/{videoID}/10sec.json  # 基础 caption（start/end_time 为 HHMMSScc 字符串, date="DAY1", video_path 绝对路径）
├── caption/{videoID}/{30sec,3min,10min}.json
├── episodic_memory/{videoID}/{openie,episodic_triple}_results_Qwen3.5-4B.json
├── semantic_memory/{videoID}/{semantic_extraction,semantic_consolidation}_results_Qwen3.5-4B.json
└── visual_memory/{videoID}/visual_embeddings.pkl   # key = 每段的 start_time
```

注意：semantic/episodic 产物文件名带模型后缀，eval 按 `--retriever-model` 名字查找，因此**预处理与推理必须用同一个模型名**（默认都是 `Qwen3.5-4B`）。

## 单独跑 eval（预处理完成后）

```bash
cd /myworkspace/projects/WorldMM && GPU_LIST=0,1 EVAL_ATTEMPTS=2 WORLDMM_LVBENCH_MAX_ROUNDS=5 bash script/lvbench/4_eval.sh
```

这是双卡正式推理的无人值守前台命令。启动前会只读检查 315 题引用的 20 个视频是否都具备
四档 caption、episodic、semantic 和 visual memory；不会重建离线产物。推理期间每题立即追加到
`/myworkspace/projects/output/worldmm/lvbench/records.jsonl`，作业重启时跳过成功题、只重试失败或缺失题。
默认最多执行两轮；两轮后仍不完整则以非零状态退出，不生成误导性的“完成”报告。

双卡分工：GPU 0 独占 Qwen3.5-4B LMDeploy，GPU 1 加载 Qwen3-Embedding-4B 与 VLM2Vec
检索模型。LMDeploy 参数逐项对齐 VideoSpy：PyTorch、TP=1、batch=8、KV cache=0.8、
`reasoning-parser=default`、`tool-call-parser=qwen3coder`，且不设置 session length。

原始结果输出到 `/myworkspace/projects/output/worldmm/lvbench/Qwen3.5_4B_Qwen3.5_4B/lvbench_eval.json`，同时自动生成 **videospy 风格报告**（`/myworkspace/projects/output/worldmm/lvbench/report/`）：

- `report.md` — 汇总表（Overall + 各 question_type 的 Completed / Accuracy / Avg. rounds / Avg. tokens）
- `metrics.json` — overall + by_question_type 准确率 + run_statistics（agent_rounds、token_usage）
- `predictions.json` — uid → 归一化选项字母
- `records.jsonl` — 每题一条记录（状态、原始回答、预测字母、轮次）

## 冒烟验证（改完配置先跑一次）

```bash
SMOKE=1 bash script/lvbench/run_all.sh
```

从第一个测试视频剪 3 分钟 + 3 题，全链路（caption→multiscale→episodic→semantic→visual→eval）只产出在 `/workspace/worldmm/lvbench/smoke/`，不污染正式目录。

## 注意事项

- 保留**单卡**兼容入口 `GPU_LIST=0`，但长视频索引会让 LMDeploy 与 Qwen3-Embedding-4B 同卡争抢显存；实测 1×4090 即使把 KV cache 降到 0.30 仍可能 OOM。正式推理使用 `GPU_LIST=0,1`，让 GPU 0 跑 LMDeploy、GPU 1 跑 embedding/visual retrieval。
- 日志里不再出现 `INFO:httpx:HTTP Request ...` 刷屏（已在 `worldmm/llm/lmdeploy.py` 统一静音）；每个大阶段开始、每个分片日志开头都有 `====` 分界线。
- 若端口 23333 已有可用实例，脚本会直接复用且**不会**替你关掉它；此时请确保最后一卡未被该实例占用（单卡模式下 semantic/eval 阶段该实例最好带 `--cache-max-entry-count 0.35`）。
- 复用 23333 前会核对 `/v1/models` 的 model id；若不是 `Qwen3.5-4B`，脚本会直接失败，避免把结果混入错误模型的 checkpoint。
- caption 段级失败会自动重试 3 次；整视频失败会在下一轮 pass 重试（共 `CAPTION_ATTEMPTS` 轮），重跑命令只补缺失视频。
- 参考耗时（2×4090）：源代码对齐档（原生分辨率帧）约 14–20h，提速档 `SAMPLE_FPS=0.5 NUM_FRAMES=10` 或 `MAX_FRAME_EDGE=1280` 约 7–10h；更多卡近似线性下降，单卡（1×4090）约再翻倍。
