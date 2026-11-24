# 2026-11-23 正式构建巡检：问题与处理方案

> **状态（2026-11-24）：** #1、#2 已在代码中修复，快照 identity 升为 version 4，v2/v3 快照加载时直接报错并提示重建。
> `upgrade_snapshot_identity` 和 `--previous-openie-dir` 已删除，因为迁移会把有问题的旧图标成有效。
> EgoLife 三个后端的 episodic 快照尚未重建，下面的 C–F 步还要执行。

巡检时间 2026-11-23 09:45。对象是 2026-11-22 晚启动的 5 个构建 run，状态目录在
`/myworkspace/projects/worldmm_needed/pipeline_runs/build/<RUN_ID>/`：

| RUN_ID | 命令 | 巡检时阶段 |
|---|---|---|
| lvbench-shared | `WORLDMM_BUILD_BACKENDS=none ... run_variant_build.sh lvbench-shared lvbench` | shared，32/103 视频 |
| videomme-shared | `WORLDMM_BUILD_BACKENDS=none ... run_variant_build.sh videomme-shared videomme_long` | shared，99/300 视频 |
| egolife-freeret | `WORLDMM_BUILD_BACKENDS=freeret ... egolife-freeret egolife` | episodic，337/432 快照 |
| egolife-wemm4b | `WORLDMM_BUILD_BACKENDS=wemm4b ... egolife-wemm4b egolife` | visual，5280/6223 片段 |
| egolife-original | `WORLDMM_BUILD_BACKENDS=original ... egolife-original egolife` | visual，2816/6223 片段 |

## 总览

| # | 问题 | 影响结果？ | 处理 |
|---|---|---|---|
| 1 | HippoRAG 增量建图时反复添加全部边（重复边） | **是**，仅 EgoLife episodic | 改代码后重建 EgoLife episodic |
| 2 | `add_fact_edges` 的已存在判断永远为 False，事实边权重被反复累加 | **是**，仅 EgoLife episodic | 与 #1 一起修 |
| 3 | FreeRet 的同义边天然更密 | 是后端特性，不是 bug | 不修，分析时注意 |
| 4 | shared 阶段慢：OpenIE 并发低，且常被"复读"请求卡住 | 否，只影响速度 | 可选优化 |
| 5 | OpenIE "Structured output truncated" 警告 | 否 | 无需处理 |
| 6 | 语义抽取 / 语义合并偶发解析失败，结果留空 | 极小 | 不处理，记录即可 |
| 7 | `code_commit` / `working_tree_status` 为空 | 否，但缺少来源记录 | 配置 git safe.directory |
| 8 | egolife-original 残留旧的 `exit_code=1` | 否 | 可选：脚本启动时清理 |
| 9 | visual 阶段不打进度，看起来像卡住 | 否 | 可选：加进度输出 |

**#1、#2 只影响 EgoLife。** LVBench 和 Video-MME 只在 `FINAL_TIME` 上调用一次
`index()`，不会触发这两个 bug。shared、语义记忆、visual 的产物都不受影响。

---

## 1. 重复边（必须修）

**位置：** `src/HippoRAG/src/hipporag/HippoRAG.py` 的 `add_new_edges`，以及 `index()` 中的调用链。

**原因：**
- `node_to_node_stats` 在整个构建过程中一直累积；续跑时也会从 `snapshot_runtime.json` 恢复。
- 每次有新 chunk 时，`index()` → `augment_graph()` → `add_new_edges()` 会把字典里**所有**边
  再 `graph.add_edges()` 一遍。
- igraph 允许多重边，所以每次增量更新，已有的边都会多一份。
- EgoLife 要按 432 个 cutoff 依次增量建快照，于是越早加入的边重复次数越多。

**证据**（10min 图，巡检时 build 目录中的最新版本）：

| 后端 | 总边数 | 去重后节点对 | 重复倍数 |
|---|---|---|---|
| FreeRet（未建完） | 4,635,388 | 34,603 | 134× |
| original | 3,764,009 | 26,163 | 144× |
| WeMM | 2,495,288 | 17,537 | 142× |

- WeMM 1h 图有 934,955 条边，但只有 20,417 个不同节点对，同一对最多出现 120 次。
- 每个快照的 `snapshot_runtime.json` 中，`node_stats` 的节点对集合与图里去重后的集合完全一致
  （抽查 WeMM 两个快照的 1h / 10min 图）。这说明多出来的全是重复，没有凭空多出的边。

**影响：**
- 检索：`run_ppr` 使用 `personalized_pagerank(weights='weight')`，多重边的权重会累加。
  越早加入的边被放大的倍数越多，所以 EgoLife 检索会系统性偏向较早的内容。
- 资源：快照越来越大，巡检时 views 已有 FreeRet 642G、original 572G、WeMM 536G；
  最新的单个 FreeRet 快照 5.3G。
- 速度：建图越来越慢，FreeRet episodic 从每小时约 57 个快照降到约 14 个。
- 上游 HippoRAG 的代码本身就是这样，上游 WorldMM 的 EgoLife 增量构建很可能也受影响（未验证）。

**修法**：让图的边始终等于 `node_to_node_stats`，每条只写一次。在 `add_new_edges` 加边之前清空旧边：

```python
# add_new_edges() 中，self.graph.add_edges(valid_edges, ...) 之前
# node_to_node_stats 已包含全部历史边；只按它写一遍，避免增量更新时产生多重边。
self.graph.delete_edges()
```

igraph 1.0.0 已实测：`Graph.delete_edges()` 不带参数时会删除全部边、保留全部节点。

## 2. 事实边权重被反复累加（必须修）

**位置：** `HippoRAG.py` 的 `add_fact_edges`（约第 435 行）。

```python
if "name" in self.graph.vs:            # 现状：判断的是 VertexSeq 成员，永远为 False
    current_graph_nodes = set(self.graph.vs["name"])
```

- 已实测 igraph 中 `"name" in G.vs` 为 False，而 `"name" in G.vs.attribute_names()` 为 True。
- 后果：`current_graph_nodes` 永远是空集，每次 `index()` 都会把**所有**已有 chunk 的三元组
  再计数一遍。事实边权重变成"真实计数 × 此后的更新次数"。快照里能看到权重为 192 的事实边。
- 同一文件里的 `add_passage_edges` 写的是 `attribute_names()`，是正确的。
- 这个错误也写进了 `snapshot_runtime.json` 的 `node_stats`。**所以不能用 `node_stats` 离线修补
  现有快照**，只能重建。

**修法：**

```python
if "name" in self.graph.vs.attribute_names():
```

## 1+2 的配套改动：让旧快照无法再被使用

改完后，旧快照（identity version 2/3）里依然带着错误的图，必须让加载时直接报错：

- `src/worldmm/memory/episodic/memory.py`
  - `_snapshot_identity(..., version=3)` 的默认值改为 `4`。
  - `_load_snapshot` 中的 `if version not in (2, 3)` 改为 `if version != 4`。
  - `upgrade_snapshot_identity` 目前会把 v2 改写成"当前版本"。改完后会把旧图直接标成 v4，
    所以要让它遇到 v2/v3 时直接报错（例如提示"图构建已修复，需要重建快照"）。
- `eval/eval_egolife.py`（约第 760 行）的 `if version not in (2, 3)` 同样改为 `!= 4`。
- 建议加一个离线单测（`tests/`），构造一个小 igraph 和 `node_to_node_stats`，检查：
  - 连续调用两次 `add_new_edges()` 后，边数等于去掉自环后的 `len(node_to_node_stats)`；
  - 图上已有某个 chunk 时，`add_fact_edges` 不会再给它的三元组计数。

**修完之后仍保留的上游行为（可接受，不改）：**
- 同义边每次都对全部实体重算 KNN 并赋值，字典里不会删除旧键。
  所以某对实体后来被挤出 top-k，它的同义边仍会保留。
- 同一键既是事实边又是同义边时，同义分数会覆盖计数。

## 3. FreeRet 的同义边更密（不修）

同义边规则：cos ≥ 0.8（`synonymy_edge_sim_threshold`），每个实体最多约 101 个邻居。
去掉重复之后，每个实体平均的实体间边数（10min 图）：

| FreeRet | original | WeMM |
|---|---|---|
| 7.6 | 3.4 | 1.7 |

- 三个后端的事实边来自同一份 OpenIE，差异来自同义边。
- 原因是 FreeRet 的相似度整体偏高，过阈值的实体对更多。这是 embedding 自身的特性，
  按"上游检索不改"的原则不调阈值。
- 分析 EgoLife / 其他数据集的 episodic 结果时要考虑这一点。如果之后想做敏感性分析，
  可以单独跑一组按后端调过阈值的对照。

## 4. shared 阶段慢（可选优化）

**流程**：对每个视频依次执行以下步骤，全部调用 Qwen3.5：
1. 多尺度摘要：`gen_multiscale` 生成 30sec / 3min / 10min caption。
2. 10sec OpenIE：先做 NER，再按 ≤1200 字符的分段抽三元组。
3. 语义抽取：每 10 条 caption 一组。
4. 粗粒度 OpenIE：对 30sec / 3min / 10min 再做一遍（`prepare_openie`）。

一个 1 小时的 LVBench 视频大约需要 2300 次 LLM 调用。

**耗时分布**（按服务端日志统计，前 13.9 小时）：

| | LVBench | Video-MME |
|---|---|---|
| OpenIE | 12.1h（87%） | 11.7h（84%） |
| 多尺度摘要 | 1.3h | 1.8h |
| 语义抽取 | 0.5h | 0.4h |

**OpenIE 慢的原因**（代码在 `eval/egolife_precompute.py` 的 `prepare_openie`）：
- `ThreadPoolExecutor(max_workers=4)`：只有 4 个并发，而服务 `max_batch_size=8`，GPU 约有一半空闲。
- 每 16 条 caption 一批，整批等最慢的那条完成后才开始下一批。
- 复读失控：LVBench 有 725 次请求生成到 4096 token 上限，平均 52 秒一次，
  合计约 10.5 小时请求时间；正常请求平均约 1 秒。

**可选改法**（不改变抽取的输入和生成参数，因此不改变抽取的分布）：
- `max_workers` 改为 8，与服务批大小一致。
- 去掉"每 16 条等齐一批"：所有缺失的 caption 一次性提交，在 `as_completed` 里逐条写盘。
  注意要保留"失败也先保存已成功 caption"的语义。
- 正在跑的 lvbench-shared / videomme-shared 想用上这个优化，需要先停掉再用同一个 RUN_ID 重启。
  OpenIE 是逐条 caption 落盘的，断点续跑没问题。

**不建议**跑到一半时改 `repetition_penalty`、`max_tokens` 或温度。这会让同一数据集前后的抽取
来源不一致。如果要改，应该在新数据集开始之前改，并写进 provenance。

## 5. "Structured output truncated at max_tokens=4096" 警告（无需处理）

- **来源：** `src/worldmm/memory/episodic/openie.py` 捕获 `StructuredOutputTruncatedError` 后会打印这条警告，
  并标记 `output_truncated`。
- **处理过程：** `eval/egolife_precompute.py` 的 `extract_caption` 看到这个标记后，会把文本对半切开递归重试，
  再合并结果。切到最小仍然截断时会抛错，`prepare_openie` 随即让整个 stage 失败（fail-closed）。
  因此不会静默写入空结果。
- **实际原因：** 模型复读，例如反复输出 `["Arsenal", "has player", "number 20"], ...`，直到写满 4096 token。
- **数量**（巡检时）：LVBench 730 次，Video-MME 459 次。只影响速度，见 #4。

## 6. 语义抽取 / 语义合并的解析失败（不处理）

- **语义抽取**（`SemanticRawOutput` 解析失败，该组记为空）：
  - LVBench 41/1427 组，Video-MME 44/2509 组。
  - 其中 34 组和 37 组是模型回复"输入为空"，说明这些组本身没有情景三元组，没有实际损失。
  - 真正因为 JSON 格式错误而丢的只有十来组。
  - 这些都是 shared 产物，三个后端共用，不会造成后端之间的偏差。
- **语义合并**（`Consolidation failed after retry`，原样保留新三元组、不做合并）：
  - EgoLife original 15 次、FreeRet 14 次、WeMM 10 次，以 original 为例约占 11,473 次调用的 0.1%。
  - 影响可以忽略。

## 7. 代码版本没有记录（下次启动前处理）

`run_variant_build.sh` 用 `git rev-parse HEAD ... || true` 记录版本。仓库属主与运行用户不同，
git 报 "dubious ownership" 失败，错误被 `|| true` 吞掉，所以 5 个 run 的 `code_commit` 都是空的。
巡检时核对过：相关代码文件的修改时间都早于启动时间（最晚是 `run_variant_build.sh`，
11-22 19:32）。

**处理：** 在每台构建机器上执行一次：

```bash
git config --global --add safe.directory /myworkspace/projects/WorldMM
```

可选：改脚本，在 git 失败时打印警告，而不是静默写空文件。

另外要注意：每个 stage 都会启动新的 Python 进程。**构建期间改仓库代码，会改变正在跑的 run
后续阶段的行为。**

## 8. 残留的 exit_code（可选）

egolife-original 第一次启动（20:26）时，编码器加载到了 cuda:0，与 LMDeploy 服务同卡，结果 OOM，
留下了 `exit_code=1`。20:35 重启后一切正常，但旧的 `exit_code` 文件要到这次 run 结束才会被覆盖。

可选：在 `run_variant_build.sh` 拿到 flock 之后执行 `rm -f "$run_root/exit_code"`。

## 9. visual 阶段没有进度输出（可选）

`build_variant.visual_index` 只是每 32 个片段保存一次 `visual_embeddings.pkl` / `frames.json`，
日志里看不到进度。巡检时 egolife-original 的日志 3 小时没有更新，实际一直在推进。

判断是否卡住的办法是看这两个文件的 mtime，以及 `len(frames.json)`。可选：每次保存时打印
`visual {len(samples)}/{len(rows)}`。

---

## 跑完之后的操作清单

### A. 等待与确认

1. 等 egolife-freeret / egolife-wemm4b / egolife-original 三个 run 结束，确认各自的 `exit_code` 为 0，
   `completed_stages` 里有 `verify`。
2. **在此之前不要用 EgoLife 快照跑 eval**，结果会受 #1、#2 影响。
3. lvbench-shared / videomme-shared 不受 #1、#2 影响，可以继续跑。修代码不影响它们：
   这两个 run 只剩 shared 这一个已经在运行的进程，`BACKENDS=none` 不会再启动新的 stage。

### B. 改代码（三个 EgoLife run 全部结束之后）

1. `HippoRAG.py`：按 #1 修改 `add_new_edges`，按 #2 修改 `add_fact_edges`。
2. `memory.py` / `eval_egolife.py`：快照 identity 升到 version 4，拒绝 v2/v3，
   并处理 `upgrade_snapshot_identity`（见"配套改动"）。
3. 补单测，并跑一遍 `pytest tests -q`。pytest 目前没有安装，需要先确认是否安装。
4. 可选：#4 的 OpenIE 并发、#7 的 safe.directory、#8、#9。
5. 提交。

### C. 移走旧的 EgoLife episodic 产物（先改名，不直接删）

对每个后端目录 `B`（`Qwen3.5-4B-FreeRet`、`Qwen3-Embedding-4B+VLM2Vec-V2`、`WeMM-Embedding-4B`），
`E=/myworkspace/projects/worldmm_needed/egolife/$B`：

```bash
mv "$E/views/A1_JAKE/snapshots"             "$E/views/A1_JAKE/snapshots.dupedges-v3"
mv "$E/indexes/A1_JAKE/episodic_memory"     "$E/indexes/A1_JAKE/episodic_memory.dupedges-v3"
mv "$E/reports/A1_JAKE/episodic.json"       "$E/reports/A1_JAKE/episodic.dupedges-v3.json"
mv "$E/reports/A1_JAKE/verify.json"         "$E/reports/A1_JAKE/verify.dupedges-v3.json"
```

- 必须移走 `snapshots`：`prepare_indexes` 看到已有快照会尝试加载或复用，
  `save_snapshot` 也会因为目录已存在而报错。
- `views/A1_JAKE/index_manifest.json` 只记录 embedding 和推理模型，内容不变，可以保留。

### D. 只重跑 EgoLife

每个后端使用一个新的 RUN_ID，GPU 环境变量与原来的启动方式保持一致：

```bash
cd /myworkspace/projects/WorldMM
WORLDMM_BUILD_BACKENDS=freeret  bash script/run_variant_build.sh egolife-freeret-fix  egolife
WORLDMM_BUILD_BACKENDS=wemm4b   bash script/run_variant_build.sh egolife-wemm4b-fix   egolife
WORLDMM_BUILD_BACKENDS=original bash script/run_variant_build.sh egolife-original-fix egolife
```

- 脚本会按顺序重新走一遍各个 stage。manifests / shared / semantic / semantic-index / visual 的产物
  都已存在，预期会很快跳过（例如 consolidation 文件已存在就跳过、embedding store 只插入新的条目、
  visual 已有的记录直接跳过）。**请在日志里确认这些阶段没有重新调用模型。**
- 真正重做的只有 episodic（新的 `build-*` 目录和 432 个新快照）和 verify。
- 修复后图规模应该大幅缩小，episodic 预计比这次快很多，但具体快多少未实测。

### E. 验收

1. 三个 run 的 `exit_code` 为 0，verify 能读回全部 432 个 cutoff。
2. 抽查最新快照的每个粒度：图边数应等于 `snapshot_runtime.json` 中 `node_stats` 去掉自环后的条数，
   并且不存在多于 2 条的平行边（`(a,b)` 和 `(b,a)` 各一条是正常的）。例如：

   ```python
   import glob, json, igraph
   from collections import Counter
   snap = sorted(glob.glob(f"{E}/views/A1_JAKE/snapshots/*/"))[-1]
   for g in ["30sec", "3min", "10min", "1h"]:
       G = igraph.Graph.Read_Pickle(glob.glob(f"{snap}{g}/*/*/graph.pickle")[0])
       stats = json.load(open(f"{snap}{g}/snapshot_runtime.json"))["node_stats"]
       expected = sum(a != b for a, b, _ in stats)
       print(g, G.ecount(), expected, max(Counter(G.get_edgelist()).values()))
   ```

3. 事实边权重不应再出现几十、上百这样的值。可以抽几条事实边，与 OpenIE 中该三元组的真实出现次数对比。
4. views 总大小应从每个后端约 0.5–0.6 TB 大幅下降。

### F. 清理

验收通过后，删除各后端的 `*.dupedges-v3` 目录（合计约 1.7–1.9 TB），然后再开始跑 EgoLife eval。
