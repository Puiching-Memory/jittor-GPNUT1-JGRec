# 性能基准

本文档只保留当前默认模型、已验证工程优化和后续实验门禁。已经失败或被撤回的模型结构实验不再展开细节，避免和当前主链路混淆。

## 当前提交候选

记录日期：2026-06-07

当前工作区已验证提交包：

```text
result/hybrid_submit_v14_d1_quality_v9_d2_quality_stream_v6_seed60/result.zip
```

线上反馈：

```text
1.0715546895407047
```

拼包来源：

| 数据集     | CSV 来源                                                                 |
| ---------- | ------------------------------------------------------------------------ |
| `dataset1` | `result/hybrid_submit_v13_d1_quality_v9_d2_stream_memmap_v2_seed60/csv/dataset1.csv` |
| `dataset2` | `result/hybrid_d2_quality_stream_v6_seed60/csv/dataset2.csv`             |

当前模型链路：

```text
stats + candidate_prior + structure + two_tower + graph + sequence -> Fusion MLP
```

关键设置：

- `max_fit_events=0`，final encoder 使用完整训练历史；
- dataset2 自动画像为 `new_link_cold`，默认 `test_candidate_negative_ratio=0.60`；
- 监督融合训练使用 `max_train_events=50000`、`max_val_events=10000`；
- 质量优先，不通过关闭 `structure`、`two_tower`、`candidate_prior` 解决内存或速度问题；
- 监督特征使用 memmap 和流式 fusion 控制内存。

提交结论：

- 该包是当前工作区可定位、可检查、已得到线上反馈的提交候选。
- 若继续冲分，优先单跑 dataset2 对比 `max_train_events/max_val_events`、`selection_metric=mrr` 和
  `test_candidate_negative_ratio`，再与稳定 dataset1 拼包。
- 不使用 `--limit-rows` 产物提交。

## 2026-06-07 Dataset2 速度诊断

`hybrid_d2_quality_stream_v6_seed60` 的 `memory.log` 显示主要耗时不是神经网络训练，而是推理阶段结构特征：

```text
predict_start: 2026-06-07T17:28:03
predict_done : 2026-06-07T20:40:19
feature-profile rows=150016 structure=11346.1s
```

原因是 future-only 结构索引压缩后，`cooccurs_by_left` 为空，旧逻辑在大历史源节点上退回到
`candidate_count * src_neighbors` 的暴力共现查询。修复后，future-only 模式也使用
`future_cooccur_count_maps` 判断是否可以预聚合，并缓存大历史源节点的候选共现计数。

验证：

```text
python -m compileall -q src scripts tests
ruff check src/jgrec/rankers/hybrid/structure.py tests/test_hybrid_structure.py
WSL/Jittor targeted tests: 26 passed
dataset2 small smoke: result/structure_speed_smoke_v8
```

该修复是等价加速：不改变特征定义、不减少训练事件、不关闭模型塔。

## 基准原则

- 使用固定数据、固定查询数、固定候选数做前后对比。
- 记录 median 时间，避免单次运行波动影响判断。
- 涉及打分逻辑的改动，需要校验 checksum 或输出概率格式，确保性能收益不是来自语义退化。
- 涉及提交文件的改动，需要校验 CSV 行列数、概率范围、每行和、zip 内容。
- 没有稳定收益的改动也要记录，避免后续重复投入。

## 线上冠军对照基线

以下记录第一版完整 GNN 提交的线上冠军对照。阶段 3 之后，当前工作区默认 `hybrid`
训练已启用 mixed hard negatives；该线上基线仍作为后续实验门禁对照。

- `num_negatives=31`
- 当时使用随机负采样
- XSimGCL 三窗口图塔：`gnn_full`、`gnn_recent`、`gnn_short`
- SASRec 序列塔可训练，但由本地验证在 `stats`、`stats_gnn`、`stats_gnn_seq` 中选择最终特征组
- 当时无 Semantic ID、无 mixed hard negatives、无 SVD 谱图分支、无 item-transition 图分支

提交产物：

```text
result/rw32-bs2048-vr0p1-cr0p75-tr20000-va5000-neg31-fit0-ep5-tbs512-lr0p001-wd0-fh64-gnnxsimgcl-ge3-gd128-gl2-gmge0-gmte40000-seqon-se3-sd128-sl64-s42/result.zip
```

| 数据集     | 本地 MRR | 选择的融合特征 |
| ---------- | -------: | -------------- |
| `dataset1` |  0.80293 | `stats_gnn`    |
| `dataset2` |  0.51770 | `stats_gnn`    |

| 版本                | 线上得分 |
| ------------------- | -------: |
| 第一版完整 GNN 提交 |   1.1452 |

结论：

- 第一版端到端链路已经满足比赛提交格式，并且线上成功计分，是当前冠军基线。
- 本地验证 MRR 加和为 \(0.80293 + 0.51770 = 1.32063\)，高于线上总分 \(1.1452\)，说明本地时间切分偏乐观。
- 后续任何模型结构改动，只有在线上反馈或已校准代理验证中超过该基线，才进入默认提交流程。

## 阶段 0 基线复核

复核日期：2026-06-02

复核仓库：

```text
D:\tmp\jittor-GPNUT1-JGRec-latest
1a0f5ea 移除废弃的 third_party 统计/结构特征重排器后端
```

复核环境：

```text
Docker image: jgrec-stage0
Base: docker.m.daocloud.io/library/python:3.10-slim
Python: 3.10.20
Jittor: 1.3.11.0
JittorGeometric: 2.0.0
```

该 Docker 环境使用 pip 安装运行依赖。阶段 0 只验证 CPU stats-only smoke，不运行默认 GNN/SASRec
hybrid 基线。

### 数据接入

数据包：

```text
D:\work\jittor-GPNUT1-JGRec\data\data_A.zip
```

已解压到新 clone：

```text
data/dataset1/train.csv
data/dataset1/test.csv
data/dataset2/train.csv
data/dataset2/test.csv
```

数据不提交，`data/` 和 `*.zip` 已由 `.gitignore` 忽略。

### 已运行检查

数据结构检查：

```bash
python -c "from pathlib import Path; print(sorted(map(str, Path('data').glob('dataset*/train.csv')))); print(sorted(map(str, Path('data').glob('dataset*/test.csv'))))"
```

结果：

```text
['data/dataset1/train.csv', 'data/dataset2/train.csv']
['data/dataset1/test.csv', 'data/dataset2/test.csv']
```

代码编译检查：

```bash
docker run --rm -v D:/tmp/jittor-GPNUT1-JGRec-latest:/workspace -w /workspace jgrec-stage0 \
  python -m compileall -q src scripts
```

结果：通过。

文档构建：

```bash
docker run --rm -v D:/tmp/jittor-GPNUT1-JGRec-latest:/workspace -w /workspace jgrec-stage0 \
  /bin/sh -lc "python -m pip install zensical && python -m zensical build"
```

结果：通过，`Build finished`，`No issues found`。

完整测试：

```bash
docker run --rm -v D:/tmp/jittor-GPNUT1-JGRec-latest:/workspace -w /workspace jgrec-stage0 \
  /bin/sh -lc "python -m pip install rich tyro pytest pytest-cov && PYTHONPATH=src python -m pytest"
```

结果：通过，`24 passed`。

修复前，`pytest` 在收集 `tests/test_cli.py` 时导入 `jgrec.cli`，会过早导入 `jittor_geometric`，
触发自定义算子编译并失败于 `cuda.h: No such file or directory`。当前修复点：

- `rankers/hybrid/config.py` 承载 `TrainingConfig`、GNN/sequence tower 配置和轻量特征名常量；
- `jgrec.cli` 导入 hybrid 配置时不再触发 `hybrid.ranker`、`.gnn` 或 `.sequence`；
- `hybrid/ranker.py` 顶层不再导入 `.gnn` 和 `.sequence`；
- `TrainingConfig.gnn_enabled=False` 时使用零图特征占位，不导入 JittorGeometric；
- `TrainingConfig.seq_enabled=False` 时使用零序列特征占位，不导入 SASRec/JittorGeometric；
- 新增回归测试覆盖 CLI import 和 disabled hybrid ranker 创建路径。

### Stats-only smoke

```bash
docker run --rm -v D:/tmp/jittor-GPNUT1-JGRec-latest:/workspace -w /workspace jgrec-stage0 \
  /bin/sh -lc "python -m pip install rich tyro pytest pytest-cov >/tmp/pip-stage0.log && PYTHONPATH=src python -m jgrec.cli --model hybrid --cpu --disable-gnn --disable-seq --limit-rows 2 --max-fit-events 512 --max-train-events 32 --max-val-events 16 --num-negatives 3 --epochs 1 --fusion-hidden-dim 16 --quiet-ranker"
```

结果：通过，退出码为 0。为避免 Rich 表格截断，记录时追加 `COLUMNS=220` 重新运行过同一命令，语义参数不变。

输出目录：

```text
result/hybrid_sample-2-rows_cpu_seed-42_gnn-off_sequence-off_f15e2a3c
result/hybrid_sample-2-rows_cpu_seed-42_gnn-off_sequence-off_f15e2a3c/result.zip
```

| dataset | train events | val events | AP | MRR | selected fusion | feature count | rows |
| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: |
| `dataset1` | 32 | 16 | 0.58049 | 0.96875 | `stats` | 8 | 2 |
| `dataset2` | 32 | 16 | 0.58814 | 1.00000 | `stats_gnn` | 11 | 2 |

说明：本次命令显式使用 `--disable-gnn --disable-seq`，因此 `stats_gnn` 中的 3 个 GNN 特征是零占位，
并不表示真实 XSimGCL/LightGCN 图塔已参与训练。

输出校验：

| 文件 | 行数 | 每行列数 | 概率范围 | 8 位小数格式 |
| --- | ---: | --- | --- | --- |
| `csv/dataset1.csv` | 2 | `100, 100` | `[0.00950972, 0.01000495]` | 通过 |
| `csv/dataset2.csv` | 2 | `100, 100` | `[0.01000000, 0.01000000]` | 通过 |

`result.zip` 内容：

```text
dataset1.csv
dataset2.csv
```

### 阶段 0 结论

当前最新上游代码在 Docker 中已确认：

- Python 语法编译通过；
- 文档构建通过；
- 完整 pytest 通过；
- stats-only smoke 已跑通，生成 CSV 和 `result.zip`；
- stats-only smoke 全程没有触发 `cuda.h` 编译失败路径；
- 默认 GNN hybrid 基线仍未在该 CPU Docker 中刷新。

默认 hybrid 基线仍需要满足以下条件之一后再刷新：

- 准备包含 CUDA headers 的 Docker 镜像；
- 或验证出 JittorGeometric 在 CPU-only 环境中可编译自定义算子的兼容构建方式。

## 阶段 1：query-time 统计特征截断

复核日期：2026-06-02

目标：修正 `hybrid` 统计特征在 query time 早于训练集末尾时使用未来交互的问题，确保
`time >= query.time` 的交互不会进入 pair、source、destination 统计。

实现范围：

- `TemporalStats.fit()` 额外保留按时间排序的 source、destination、pair 历史数组；
- `TemporalStats.features_for_queries()` 在整批 query time 都晚于 `max_train_time` 时保留原聚合快路径；
- 只要批内存在 `query.time <= max_train_time`，改走 cutoff-aware 路径；
- cutoff 使用 `np.searchsorted(..., side="left")`，严格只统计 `time < query.time`；
- 最新上游已移除 `third_party` ranker/indexes 后端，因此原阶段 1 中 `third_party/indexes.py` 的同类改造不适用。

新增测试：

```text
tests/test_temporal_stats.py
```

覆盖点：

- 同一 `(src, dst)` 在 query time 前后各有交互时，只使用 query time 前的交互；
- `time == query.time` 的交互被排除；
- query time 晚于训练集末尾时，batch fast path 与单条聚合路径一致。

验证命令：

```bash
docker run --rm -v D:/tmp/jittor-GPNUT1-JGRec-latest:/workspace -w /workspace jgrec-stage0 \
  /bin/sh -lc "python -m pip install rich tyro pytest pytest-cov >/tmp/pip-stage1.log && PYTHONPATH=src python -m pytest tests/test_temporal_stats.py"
```

结果：通过，`3 passed`。

完整测试：

```bash
docker run --rm -v D:/tmp/jittor-GPNUT1-JGRec-latest:/workspace -w /workspace jgrec-stage0 \
  /bin/sh -lc "python -m pip install rich tyro pytest pytest-cov >/tmp/pip-stage1.log && PYTHONPATH=src python -m pytest"
```

结果：通过，`27 passed`。

语法与文档：

```bash
docker run --rm -v D:/tmp/jittor-GPNUT1-JGRec-latest:/workspace -w /workspace jgrec-stage0 \
  python -m compileall -q src scripts

docker run --rm -v D:/tmp/jittor-GPNUT1-JGRec-latest:/workspace -w /workspace jgrec-stage0 \
  /bin/sh -lc "python -m pip install zensical >/tmp/pip-stage1.log && python -m zensical build"
```

结果：均通过，文档构建输出 `Build finished` 和 `No issues found`。

### 阶段 1 stats-only smoke

```bash
docker run --rm -v D:/tmp/jittor-GPNUT1-JGRec-latest:/workspace -w /workspace jgrec-stage0 \
  /bin/sh -lc "python -m pip install rich tyro pytest pytest-cov >/tmp/pip-stage1.log && COLUMNS=220 PYTHONPATH=src python -m jgrec.cli --model hybrid --cpu --disable-gnn --disable-seq --limit-rows 2 --max-fit-events 512 --max-train-events 32 --max-val-events 16 --num-negatives 3 --epochs 1 --fusion-hidden-dim 16 --quiet-ranker"
```

结果：通过，退出码为 0。输出路径：

```text
result/hybrid_sample-2-rows_cpu_seed-42_gnn-off_sequence-off_f15e2a3c/result.zip
```

| dataset | train events | val events | AP | MRR | selected fusion | feature count | rows |
| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: |
| `dataset1` | 32 | 16 | 0.66250 | 0.96875 | `stats_gnn` | 11 | 2 |
| `dataset2` | 32 | 16 | 0.25000 | 1.00000 | `stats_gnn_seq` | 12 | 2 |

说明：本次仍显式使用 `--disable-gnn --disable-seq`，因此 `stats_gnn` 和 `stats_gnn_seq`
中的 GNN/sequence 特征为零占位；指标只用于阶段 1 stats-only smoke 记录，不代表默认 GNN/SASRec
hybrid 基线。

输出校验：

| 文件 | 行数 | 每行列数 | 概率范围 | 8 位小数格式 |
| --- | ---: | --- | --- | --- |
| `csv/dataset1.csv` | 2 | `100, 100` | `[0.00942996, 0.01000576]` | 通过 |
| `csv/dataset2.csv` | 2 | `100, 100` | `[0.01000000, 0.01000000]` | 通过 |

`result.zip` 内容：

```text
dataset1.csv
dataset2.csv
```

阶段 1 结论：`hybrid` 统计特征已满足 query-time 因果截断约束；阶段 0 stats-only smoke
闭环仍成立，且仍未触发 `cuda.h` 编译失败路径。

## 阶段 2：结构统计特征合并到 hybrid

复核日期：2026-06-02

目标：把旧 `third_party` 后端中更丰富的结构统计信号合入默认 `hybrid` 特征流，作为 fusion
可选特征组，而不是依赖单独后端。

实现范围：

- 新增 `src/jgrec/rankers/common/temporal_index.py`，提供 source/destination 历史、reverse pair、
  邻居集合、cooccur 和 transition 的时序索引；
- 新增 `src/jgrec/rankers/hybrid/structure.py`，输出 11 个结构特征：
  `pair_decay_short`、`pair_decay_medium`、`pair_decay_long`、`dst_unique_src`、`dst_pop_rank`、
  `reverse_log_count`、`reverse_recency`、`common_neighbors`、`jaccard`、`cooccur_score`、
  `transition_score`；
- `HybridFeatureEncoder.feature_names` 扩展为 `stats + structure + graph + sequence`；
- `_feature_masks()` 扩展为 `stats`、`stats_structure`、`stats_structure_gnn`、
  `stats_structure_gnn_seq`；
- 最新上游已移除 `third_party` ranker/indexes 后端，因此“保留 third_party ablation”和
  `third_party` 测试项不适用。

新增测试：

```text
tests/test_hybrid_structure.py
```

覆盖点：

- 结构特征使用 query-time cutoff，未来 cooccur/transition 不泄漏；
- query time 晚于训练集末尾时使用完整历史；
- hybrid feature masks 包含 `stats_structure*` 组，且 `stats_structure` 长度等于
  `len(stats) + len(structure)`。

验证命令：

```bash
docker run --rm -v D:/tmp/jittor-GPNUT1-JGRec-latest:/workspace -w /workspace jgrec-stage0 \
  /bin/sh -lc "python -m pip install rich tyro pytest pytest-cov >/tmp/pip-stage2.log && PYTHONPATH=src python -m pytest tests/test_hybrid_structure.py tests/test_temporal_stats.py"
```

结果：通过，`6 passed`。

完整测试与静态检查：

```bash
docker run --rm -v D:/tmp/jittor-GPNUT1-JGRec-latest:/workspace -w /workspace jgrec-stage0 \
  /bin/sh -lc "python -m pip install rich tyro pytest pytest-cov >/tmp/pip-stage2.log && PYTHONPATH=src python -m pytest"

docker run --rm -v D:/tmp/jittor-GPNUT1-JGRec-latest:/workspace -w /workspace jgrec-stage0 \
  /bin/sh -lc "python -m pip install ruff >/tmp/pip-stage2.log && python -m ruff check ."

docker run --rm -v D:/tmp/jittor-GPNUT1-JGRec-latest:/workspace -w /workspace jgrec-stage0 \
  python -m compileall -q src scripts
```

结果：`pytest` 通过，`30 passed`；`ruff check .` 通过；`compileall` 通过。

### 阶段 2 stats-only smoke

```bash
docker run --rm -v D:/tmp/jittor-GPNUT1-JGRec-latest:/workspace -w /workspace jgrec-stage0 \
  /bin/sh -lc "python -m pip install rich tyro pytest pytest-cov >/tmp/pip-stage2.log && COLUMNS=220 PYTHONPATH=src python -m jgrec.cli --model hybrid --cpu --disable-gnn --disable-seq --limit-rows 2 --max-fit-events 512 --max-train-events 32 --max-val-events 16 --num-negatives 3 --epochs 1 --fusion-hidden-dim 16 --quiet-ranker"
```

结果：通过，退出码为 0。输出路径：

```text
result/hybrid_sample-2-rows_cpu_seed-42_gnn-off_sequence-off_f15e2a3c/result.zip
```

| dataset | train events | val events | AP | MRR | selected fusion | feature count | rows |
| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: |
| `dataset1` | 32 | 16 | 0.25712 | 0.45312 | `stats_structure` | 19 | 2 |
| `dataset2` | 32 | 16 | 0.25000 | 1.00000 | `stats_structure_gnn_seq` | 23 | 2 |

说明：本次仍显式使用 `--disable-gnn --disable-seq`，因此 `stats_structure_gnn_seq`
中的 GNN/sequence 特征为零占位；新增的真实信号来自 `structure` 特征组。`hybrid`
报告 feature count 已从阶段 1 的 stats-only 8/12 增加到 19/23，且 selected fusion
可以选择 `stats_structure*`。

输出校验：

| 文件 | 行数 | 每行列数 | 概率范围 | 8 位小数格式 |
| --- | ---: | --- | --- | --- |
| `csv/dataset1.csv` | 2 | `100, 100` | `[0.00999285, 0.01070793]` | 通过 |
| `csv/dataset2.csv` | 2 | `100, 100` | `[0.01000000, 0.01000000]` | 通过 |

`result.zip` 内容：

```text
dataset1.csv
dataset2.csv
```

阶段 2 结论：结构统计特征已合并到 `hybrid` 默认特征流，fusion mask 和训练报告都能看到
`stats_structure*` 组；stats-only smoke 闭环仍成立，且仍未触发 `cuda.h` 编译失败路径。

## 阶段 3：hybrid 混合负采样

复核日期：2026-06-03

目标：改进 fusion 训练阶段的负样本构造，让训练候选比纯随机 destination 更接近测试候选难度。
该阶段只影响监督训练 query 构造和训练特征学习，不改变 `predict_batch()` 推理路径。

实现范围：

- `TrainingConfig` 新增 `hard_negative_ratio=0.5` 和 `popular_negative_ratio=0.25`；
- CLI 新增 `--hard-negative-ratio` 和 `--popular-negative-ratio`；
- `_build_supervised_queries()` 把正样本事件时间传入 `_sample_negatives()`，负采样使用 query-time cutoff；
- 默认 `hard_negative_ratio=0.5` 拆为两类 hard negatives：
  - 约 25% source recent hard negatives，允许使用源节点近期历史，但仍排除当前 positive；
  - 约 25% cooccur / transition hard negatives，复用阶段 2 的 `TemporalInteractionIndex`；
- 默认 `popular_negative_ratio=0.25` 使用 query time 前可见的热门 destination；
- 剩余配额走 random destination，非 hard 桶优先排除 source 已见历史；
- 只有候选池不足时才进入 fallback，最终允许用 positive 补齐固定候选数。

新增测试：

```text
tests/test_hybrid_negatives.py
```

覆盖点：

- 小图中有 recent history、cooccur 和 transition 时，hard 桶优先产出这些候选；
- 固定 seed 下 `_sample_negatives()` 结果可复现；
- 候选池充足时不返回 positive dst，只有候选池耗尽时才用 positive fallback。

验证命令：

```bash
docker run --rm -v D:/tmp/jittor-GPNUT1-JGRec-latest:/workspace -w /workspace jgrec-stage0 \
  /bin/sh -lc "python -m pip install rich tyro pytest pytest-cov >/tmp/pip-stage3.log && PYTHONPATH=src python -m pytest tests/test_hybrid_negatives.py"
```

结果：通过，`3 passed`。

完整测试与静态检查：

```bash
docker run --rm -v D:/tmp/jittor-GPNUT1-JGRec-latest:/workspace -w /workspace jgrec-stage0 \
  /bin/sh -lc "python -m pip install rich tyro pytest pytest-cov >/tmp/pip-stage3.log && PYTHONPATH=src python -m pytest"

docker run --rm -v D:/tmp/jittor-GPNUT1-JGRec-latest:/workspace -w /workspace jgrec-stage0 \
  python -m compileall -q src scripts

docker run --rm -v D:/tmp/jittor-GPNUT1-JGRec-latest:/workspace -w /workspace jgrec-stage0 \
  /bin/sh -lc "python -m pip install ruff >/tmp/pip-stage3.log && python -m ruff check ."

docker run --rm -v D:/tmp/jittor-GPNUT1-JGRec-latest:/workspace -w /workspace jgrec-stage0 \
  /bin/sh -lc "python -m pip install zensical >/tmp/pip-stage3.log && python -m zensical build"
```

结果：`pytest` 通过，`33 passed`；`compileall` 通过；`ruff check .` 通过；文档构建通过，
输出 `Build finished` 和 `No issues found`。

### 阶段 3 stats-only smoke

```bash
docker run --rm -v D:/tmp/jittor-GPNUT1-JGRec-latest:/workspace -w /workspace jgrec-stage0 \
  /bin/sh -lc "python -m pip install rich tyro pytest pytest-cov >/tmp/pip-stage3.log && COLUMNS=220 PYTHONPATH=src python -m jgrec.cli --model hybrid --cpu --disable-gnn --disable-seq --limit-rows 2 --max-fit-events 512 --max-train-events 32 --max-val-events 16 --num-negatives 3 --epochs 1 --fusion-hidden-dim 16 --quiet-ranker"
```

结果：通过，退出码为 0。输出路径：

```text
result/hybrid_sample-2-rows_cpu_seed-42_gnn-off_sequence-off_190218e6/result.zip
```

| dataset | train events | val events | AP | MRR | selected fusion | feature count | rows |
| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: |
| `dataset1` | 32 | 16 | 0.71401 | 0.87500 | `stats_structure_gnn` | 22 | 2 |
| `dataset2` | 32 | 16 | 0.25000 | 1.00000 | `stats_structure_gnn_seq` | 23 | 2 |

说明：本次仍显式使用 `--disable-gnn --disable-seq`，因此 `stats_structure_gnn_seq`
中的 GNN/sequence 特征为零占位；阶段 3 的差异来自训练阶段负样本构造，不代表默认 GNN/SASRec
hybrid 基线已在 CPU Docker 中刷新。

输出校验：

| 文件 | 行数 | 每行列数 | 概率范围 | 8 位小数格式 |
| --- | ---: | --- | --- | --- |
| `csv/dataset1.csv` | 2 | `100, 100` | `[0.00899201, 0.01001018]` | 通过 |
| `csv/dataset2.csv` | 2 | `100, 100` | `[0.01000000, 0.01000000]` | 通过 |

`result.zip` 内容：

```text
dataset1.csv
dataset2.csv
```

阶段 3 结论：`hybrid` 默认训练已从纯随机负采样升级为 recent / structural hard /
popular / random 混合负采样；stats-only smoke 闭环仍成立，且推理路径和输出格式未改变。

## 阶段 4：图塔重复边压缩和时间衰减采样

复核日期：2026-06-03

目标：让图塔在构图阶段表达重复交互强度和近期边重要性，同时在 `max_graph_edges` 生效时降低
重复边带来的训练成本。

实现范围：

- `GraphTowerConfig` 新增 `edge_weighting` 和 `time_decay_ratio`；
- `TrainingConfig` 和 CLI 新增 `--gnn-edge-weighting none|repeat|time_decay`
  以及 `--gnn-time-decay-ratio`；
- run name 和运行面板新增 edge weighting 信息，例如
  `gnn-xsimgcl_edges-repeat_sequence-off`；
- `none` 保持旧语义：按时间窗口取原始边，超过 `max_graph_edges` 时保留最新边；
- `repeat` 对窗口内 `(src_id, dst_id)` 压缩，权重为 `log1p(count)`；
- `time_decay` 对窗口内 `(src_id, dst_id)` 压缩，权重为
  `log1p(count) * exp(-(train_end_time - last_time) / tau)`；
- 当前 JittorGeometric 模型接口不接收 edge weight，因此 `repeat/time_decay` 的权重用于
  `max_graph_edges` 截断时的无放回 weighted sampling；
- CPU smoke 使用小图 dense LightGCN 风格 fallback，避免当前 CPU Docker 导入 JittorGeometric
  自定义 CUDA/cuSPARSE 算子；大图或 CUDA 路径仍走 JittorGeometric。

新增测试：

```text
tests/test_hybrid_gnn_edges.py
```

覆盖点：

- `none` 保持原始边最新截断；
- `repeat` 压缩重复边并按重复次数赋权；
- `time_decay` 对近期重复边给更高权重；
- weighted sampling 固定 seed 可复现并保留时间顺序；
- 未知 `edge_weighting` 明确报错；
- CPU dense fallback 的二部图归一化邻接矩阵为对称结构。

真实数据边预处理检查：

```bash
docker run --rm -v D:/tmp/jittor-GPNUT1-JGRec-latest:/workspace -w /workspace jgrec-stage0 \
  /bin/sh -lc "python -m pip install rich >/tmp/pip-stage4.log && PYTHONPATH=src python - <<'PY'
import itertools
import numpy as np
from pathlib import Path
from jgrec.core.io import read_interactions
from jgrec.idmap import NodeIdMap
from jgrec.rankers.hybrid.config import GraphTowerConfig
from jgrec.rankers.hybrid.gnn import _graph_window_edges, _mapped_edges, _weighted_mapped_edges

interactions = list(itertools.islice(read_interactions(Path('data/dataset1/train.csv')), 512))
id_map = NodeIdMap.from_interactions(interactions)
mapped = _mapped_edges(interactions, id_map)
print('raw_mapped_edges', len(mapped))
for mode in ['none', 'repeat', 'time_decay']:
    config = GraphTowerConfig(edge_weighting=mode, max_graph_edges=256, time_decay_ratio=0.05)
    edges = _graph_window_edges(mapped, config, np.random.default_rng(42))
    print(mode, 'window_edges', edges.shape[1])
    if mode != 'none':
        compressed, weights = _weighted_mapped_edges(mapped, mode, 0.05)
        print(mode, 'compressed_edges', compressed.shape[1], 'weight_range', (round(float(weights.min()), 8), round(float(weights.max()), 8)))
PY"
```

结果：

```text
raw_mapped_edges 512
none window_edges 256
repeat window_edges 256
repeat compressed_edges 326 weight_range (0.69314718, 2.48490665)
time_decay window_edges 256
time_decay compressed_edges 326 weight_range (0.0, 1.7931264)
```

完整测试与静态检查：

```bash
docker run --rm -v D:/tmp/jittor-GPNUT1-JGRec-latest:/workspace -w /workspace jgrec-stage0 \
  /bin/sh -lc "python -m pip install rich tyro pytest pytest-cov >/tmp/pip-stage4.log && PYTHONPATH=src python -m pytest"

docker run --rm -v D:/tmp/jittor-GPNUT1-JGRec-latest:/workspace -w /workspace jgrec-stage0 \
  python -m compileall -q src scripts

docker run --rm -v D:/tmp/jittor-GPNUT1-JGRec-latest:/workspace -w /workspace jgrec-stage0 \
  /bin/sh -lc "python -m pip install ruff >/tmp/pip-stage4.log && python -m ruff check ."

docker run --rm -v D:/tmp/jittor-GPNUT1-JGRec-latest:/workspace -w /workspace jgrec-stage0 \
  /bin/sh -lc "python -m pip install zensical >/tmp/pip-stage4.log && python -m zensical build"
```

结果：`pytest` 通过，`40 passed`；`compileall` 通过；`ruff check .` 通过；文档构建通过，
输出 `Build finished` 和 `No issues found`。

### 阶段 4 repeat GNN CPU smoke

```bash
docker run --rm -v D:/tmp/jittor-GPNUT1-JGRec-latest:/workspace -w /workspace jgrec-stage0 \
  /bin/sh -lc "python -m pip install rich tyro pytest pytest-cov >/tmp/pip-stage4.log && COLUMNS=220 PYTHONPATH=src python -m jgrec.cli --model hybrid --cpu --disable-seq --limit-rows 2 --max-fit-events 512 --max-train-events 32 --max-val-events 16 --num-negatives 3 --epochs 1 --fusion-hidden-dim 16 --gnn-edge-weighting repeat --gnn-epochs 1 --gnn-embedding-dim 16 --gnn-layers 1 --gnn-max-graph-edges 256 --gnn-max-train-edges 128 --quiet-ranker"
```

输出路径：

```text
result/hybrid_sample-2-rows_cpu_seed-42_gnn-xsimgcl_edges-repeat_sequence-off_90b8745a/result.zip
```

| dataset | train events | val events | AP | MRR | selected fusion | feature count | rows |
| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: |
| `dataset1` | 32 | 16 | 0.60312 | 0.90625 | `stats` | 8 | 2 |
| `dataset2` | 32 | 16 | 0.30769 | 0.90625 | `stats_structure_gnn_seq` | 23 | 2 |

输出校验：

| 文件 | 行数 | 每行列数 | 概率范围 | 8 位小数格式 |
| --- | ---: | --- | --- | --- |
| `csv/dataset1.csv` | 2 | `100, 100` | `[0.00967837, 0.01000325]` | 通过 |
| `csv/dataset2.csv` | 2 | `100, 100` | `[0.01000000, 0.01000000]` | 通过 |

### 阶段 4 time_decay GNN CPU smoke

```bash
docker run --rm -v D:/tmp/jittor-GPNUT1-JGRec-latest:/workspace -w /workspace jgrec-stage0 \
  /bin/sh -lc "python -m pip install rich tyro pytest pytest-cov >/tmp/pip-stage4.log && COLUMNS=220 PYTHONPATH=src python -m jgrec.cli --model hybrid --cpu --disable-seq --limit-rows 2 --max-fit-events 512 --max-train-events 32 --max-val-events 16 --num-negatives 3 --epochs 1 --fusion-hidden-dim 16 --gnn-edge-weighting time_decay --gnn-epochs 1 --gnn-embedding-dim 16 --gnn-layers 1 --gnn-max-graph-edges 256 --gnn-max-train-edges 128 --quiet-ranker"
```

输出路径：

```text
result/hybrid_sample-2-rows_cpu_seed-42_gnn-xsimgcl_edges-time-decay_sequence-off_a1aeecee/result.zip
```

| dataset | train events | val events | AP | MRR | selected fusion | feature count | rows |
| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: |
| `dataset1` | 32 | 16 | 0.83162 | 0.90625 | `stats_structure` | 19 | 2 |
| `dataset2` | 32 | 16 | 0.29091 | 0.81250 | `stats_structure_gnn_seq` | 23 | 2 |

输出校验：

| 文件 | 行数 | 每行列数 | 概率范围 | 8 位小数格式 |
| --- | ---: | --- | --- | --- |
| `csv/dataset1.csv` | 2 | `100, 100` | `[0.00802917, 0.01001991]` | 通过 |
| `csv/dataset2.csv` | 2 | `100, 100` | `[0.01000000, 0.01000000]` | 通过 |

`repeat` 和 `time_decay` 的 `result.zip` 内容均为：

```text
dataset1.csv
dataset2.csv
```

阶段 4 结论：图塔已支持 `none`、`repeat`、`time_decay` 三种边构建模式。当前 CPU Docker
可以用小图 dense CPU fallback 跑通 repeat/time_decay smoke；正式 CUDA/JittorGeometric 路径仍需要
包含完整 CUDA/cuSPARSE 外部算子的运行环境。曾尝试仅补 `nvidia-cuda-runtime-cu12`，可解决
`cuda.h`，但随后失败于 `jittor.compile_extern` 缺少 `cusparse_ops`，因此默认完整 GNN 基线仍需单独镜像刷新。
阶段 4 之后新 run name 会显式包含 `edges-*`；阶段 0-3 的历史路径保持原记录，不回写重命名。

## 阶段 5：Two-Tower 候选特征

复核日期：2026-06-03

目标：在 `hybrid` 内新增可关闭的 two-tower candidate feature，不替换已有 stats、structure、GNN 和 sequence
特征。阶段 5 仍以 CPU Docker smoke 为验收主线，显式关闭 GNN/SASRec，避免触发当前缺 CUDA/cuSPARSE 外部算子的路径。

实现范围：

- 新增 `TwoTower`，输出 `two_tower_dot` 和 `two_tower_cosine` 两个候选级特征；
- source tower 使用 source id、source activity bucket、source recency bucket、time bucket；
- destination tower 使用 destination id、destination popularity bucket、destination recency bucket、time bucket；
- two-tower 训练使用 sampled softmax，每个正样本配 `num_negatives` 个负样本；
- 负样本复用阶段 3 的 mixed negative 逻辑，抽到 `hybrid/sampling.py` 供 fusion 和 two-tower 共用；
- `HybridFeatureEncoder` 特征顺序为 `stats + structure + two_tower + graph + sequence`；
- feature mask 增加 `stats_structure_tower`、`stats_structure_tower_gnn`、
  `stats_structure_tower_gnn_seq`；
- CLI 增加 `--disable-two-tower`、`--two-tower-embedding-dim`、`--two-tower-hidden-dim`、
  `--two-tower-epochs`、`--two-tower-batch-size`、`--two-tower-max-samples`；
- run name 和运行面板增加 `tower-on/off`。

验证命令：

```bash
docker run --rm -v D:/tmp/jittor-GPNUT1-JGRec-latest:/workspace -w /workspace jgrec-stage0 \
  /bin/sh -lc "python -m pip install rich tyro pytest pytest-cov >/tmp/pip-stage5.log && PYTHONPATH=src python -m pytest"

docker run --rm -v D:/tmp/jittor-GPNUT1-JGRec-latest:/workspace -w /workspace jgrec-stage0 \
  python -m compileall -q src scripts

docker run --rm -v D:/tmp/jittor-GPNUT1-JGRec-latest:/workspace -w /workspace jgrec-stage0 \
  /bin/sh -lc "python -m pip install ruff >/tmp/pip-stage5.log && python -m ruff check ."

docker run --rm -v D:/tmp/jittor-GPNUT1-JGRec-latest:/workspace -w /workspace jgrec-stage0 \
  /bin/sh -lc "python -m pip install zensical >/tmp/pip-stage5.log && python -m zensical build"
```

结果：`pytest` 通过，`45 passed`；`compileall` 通过；`ruff check .` 通过；文档构建通过，
输出 `Build finished` 和 `No issues found`。

### 阶段 5 Two-Tower CPU smoke

```bash
docker run --rm -v D:/tmp/jittor-GPNUT1-JGRec-latest:/workspace -w /workspace jgrec-stage0 \
  /bin/sh -lc "python -m pip install rich tyro pytest pytest-cov >/tmp/pip-stage5.log && COLUMNS=220 PYTHONPATH=src python -m jgrec.cli --model hybrid --cpu --disable-gnn --disable-seq --limit-rows 2 --max-fit-events 512 --max-train-events 32 --max-val-events 16 --num-negatives 3 --epochs 1 --fusion-hidden-dim 16 --two-tower-epochs 1 --two-tower-embedding-dim 16 --two-tower-hidden-dim 16 --quiet-ranker"
```

输出路径：

```text
result/hybrid_sample-2-rows_cpu_seed-42_gnn-off_edges-off_tower-on_sequence-off_bb7e6027/result.zip
```

| dataset | train events | val events | AP | MRR | selected fusion | feature count | rows |
| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: |
| `dataset1` | 32 | 16 | 0.53152 | 0.79688 | `stats_structure_tower` | 21 | 2 |
| `dataset2` | 32 | 16 | 0.34556 | 0.58333 | `stats_structure_tower_gnn_seq` | 25 | 2 |

说明：本次 smoke 显式使用 `--disable-gnn --disable-seq`，因此 `stats_structure_tower_gnn_seq`
中的 GNN/sequence 特征仍为零占位；新增真实信号来自 two-tower。阶段 5 smoke 未证明 two-tower
相对阶段 4 有稳定收益，当前先保留为默认开启但可通过 `--disable-two-tower` 关闭的实验特征。

输出校验：

| 文件 | 行数 | 每行列数 | 概率范围 | 8 位小数格式 |
| --- | ---: | --- | --- | --- |
| `csv/dataset1.csv` | 2 | `100, 100` | `[0.00999803, 0.01019490]` | 通过 |
| `csv/dataset2.csv` | 2 | `100, 100` | `[0.01000000, 0.01000000]` | 通过 |

`result.zip` 内容：

```text
dataset1.csv
dataset2.csv
```

阶段 5 结论：two-tower 已作为 `hybrid` 的候选级特征接入，并可在 CPU Docker 下与 stats/structure
闭环跑通。默认完整 GNN/SASRec hybrid 基线仍未在该 CPU Docker 中刷新，后续若要评估完整默认链路，
仍需要包含 CUDA headers 和 JittorGeometric cuSPARSE 外部算子的镜像。

## 实验门禁

模型实验必须记录以下字段：

| 字段     | 要求                                                       |
| -------- | ---------------------------------------------------------- |
| 实验状态 | `keep`、`reject`、`archive` 三选一                         |
| 代码状态 | 说明是否进入默认 CLI；未进入默认链路的实验代码应删除或隔离 |
| 对照基线 | 默认使用第一版完整 GNN 提交，线上分 `1.1452`               |
| 协议     | 数据集、split、负采样、seed、训练事件数、验证事件数、epoch |
| 本地结果 | 分 dataset AP/MRR、选择的融合特征、关键耗时                |
| 线上结果 | 提交产物路径、线上总分；未提交要写明原因                   |
| 最终决策 | 明确保留、拒绝或仅归档，不能只列数字                       |

当前门禁：

- 本地 AP 用于对齐官方 baseline 的早停/选择口径；本地 MRR 只能作为诊断信号，不能单独决定模型进入默认链路。
- 如果线上结果低于第一版冠军基线，默认链路必须恢复到第一版模型。
- 失败实验可以保留简短归档记录，但实验代码不能污染默认命令和运行手册。
- 如果没有可信代理验证，继续比赛应采用线上 A/B：一次只改一个数据集或一个模块，用线上总分反推增量。

## 已归档模型实验

| 实验                              | 状态      | 决策原因                                                                           |
| --------------------------------- | --------- | ---------------------------------------------------------------------------------- |
| Semantic ID 聚类塔                | `reject`  | dataset2 中样本在 seed 42 有局部提升，但 seed 7 不复现；不满足稳定性要求。         |
| TMS-GNN item-transition 图分支    | `reject`  | 本地 `neg99/mixed` 很高，但线上仅 `0.40792831454706824`，显著低于第一版 `1.1452`。 |
| 官方 split + test-candidate proxy | `archive` | 代理验证压低了 transition 虚高，但仍无法复现线上排序；不能用于模型选择。           |

归档结论：

- 旧版未校准 mixed hard negatives、transition 图和未校准 proxy 不能直接作为保留依据；阶段 3 已作为
  独立、可测的默认训练改动重新落地。
- 后续重新尝试 SVD、transition 或 graph-hard negatives 时，必须作为隔离实验重新进入评估，不能继承旧本地 MRR 作为保留依据。

## 当前基准环境

候选打分与写出基准使用：

- 数据集：`dataset2`
- 测试查询：前 8192 行
- 候选总数：819200 个
- 指标：多次运行 median 时间

统计索引构建基准使用：

- 数据文件：`dataset2/train.csv`
- 训练交互数：2261283 行
- 指标：多次运行 median 时间

## 已验证改进

### CSV 批量写出

提交 CSV 原实现逐行调用 `csv.writer`。改为按数据块使用 `np.savetxt` 后，减少 Python 层循环和格式化调用开销。

|   指标 |  改动前 |  改动后 |  收益 |
| -----: | ------: | ------: | ----: |
| median | 0.7484s | 0.2420s | 3.09x |

结论：有效。CSV 写出属于纯工程瓶颈，batch 写出收益明确，保留该实现。

### Per-source pair 索引

候选特征原实现使用全局 `(src, dst)` tuple key 查询重复交互和近因时间。改为每个源节点维护独立的目标统计索引后，减少 tuple 构造和全局 dict 查询。

候选打分基准：

|   指标 |  改动前 |  改动后 |  收益 |
| -----: | ------: | ------: | ----: |
| median | 2.7651s | 2.0327s | 1.36x |

统计索引构建基准：

|   指标 |  改动前 |  改动后 |  收益 |
| -----: | ------: | ------: | ----: |
| median | 7.3741s | 4.1678s | 1.77x |

结论：有效。该改动同时降低训练统计构建耗时和候选特征查询耗时。

### Dense 目标节点特征

目标热度和目标最近交互时间原实现依赖 dict 查询。对节点 ID 范围可控的数据集，改为 dense array 查询：`dst_popularity_dense` 和 `dst_recent_time_dense`。

|   指标 |  改动前 |  改动后 |  收益 |
| -----: | ------: | ------: | ----: |
| median | 2.0327s | 1.3122s | 1.55x |

结论：有效。当前实现保留 `DENSE_NODE_LIMIT`，避免异常大的节点 ID 造成不可控内存占用。

### 批量目标节点特征填充

`features_for_queries` 原实现每个 query 单独调用 `_fill_dst_features`，每行都重复创建候选数组并计算 dense array 有效掩码。改为先构造 batch 级候选矩阵，再一次性填充 `dst_popularity` 和 `dst_recency`，pair 相关特征仍保留原来的逐候选 dict 查询。

基准命令：

```bash
uv run python scripts/bench_stats_features.py --repeats 5 --warmups 1
```

| 指标                   |  改动前 |  改动后 |  收益 |
| ---------------------- | ------: | ------: | ----: |
| `features_cold_median` | 0.7591s | 0.2930s | 2.59x |
| `features_warm_median` | 0.7865s | 0.2776s | 2.83x |
| `fit_median`           | 2.9293s | 2.6004s | 1.13x |

输出校验：

| 项目                    | 结果               |
| ----------------------- | ------------------ |
| `feature_shape`         | `(8192, 100, 8)`   |
| `feature_checksum`      | `1625183.75162584` |
| `cold_feature_checksum` | `1625183.75162584` |

结论：有效。该改动在 cold 口径下仍有稳定收益，并且 checksum 完全一致。

## 无明显收益改进

### 特征矩阵预分配

候选特征构造从 list 收集后 `stack` 改为预分配 numpy array 后逐行填充。

|   指标 |  改动前 |  改动后 |  收益 |
| -----: | ------: | ------: | ----: |
| median | 2.6447s | 2.6369s | 1.00x |

结论：单独看这个改动没有明显收益。保留预分配实现的原因是它让后续向量化改造更直接，但不能把它计入主要性能收益。

### Per-source 排序索引批量查 pair 特征

尝试为每个源节点按需构建排序后的 `dst`、pair 计数、pair 最近时间和 recent rank 数组，再用 `np.searchsorted` 批量查询 `pair_strength`、`repeat_rate`、`pair_recency`、`recent_hit`。

| 指标                   |  改动前 |  改动后 | 结论               |
| ---------------------- | ------: | ------: | ------------------ |
| `features_cold_median` | 0.7489s | 0.9468s | 变慢               |
| `features_warm_median` | 0.7489s | 0.4915s | 首次推理不代表收益 |

结论：不保留。warm 口径收益不能代表正式提交路径的首次推理成本。

## 后续优化方向

优先级最高的是继续减少候选级 Python 循环：

- 将可 dense 化的节点特征统一转为数组查询。
- 对测试候选批次做批量特征构造，减少逐候选函数调用。
- 缓存训练/验证阶段反复使用的候选特征。
- 对负采样后的训练样本提前构造特征块，避免每个 epoch 重复查统计。
- 如果引入图编码器，需要单独记录训练耗时、推理耗时、MRR 变化和线上反馈。

## 复测命令

基础正确性检查：

```bash
uv run python -m compileall -q src scripts
uv run jgrec-build --limit-rows 2 --max-fit-events 512 --max-train-events 32 --max-val-events 16 --num-negatives 3 --epochs 1 --gnn-epochs 1 --gnn-embedding-dim 16 --gnn-layers 1 --gnn-max-graph-edges 256 --gnn-max-train-edges 128 --seq-epochs 1 --seq-max-samples 128 --seq-max-len 16 --seq-hidden-size 16 --fusion-hidden-dim 16 --quiet-ranker
uv lock --check
```

统计特征性能基准：

```bash
uv run python scripts/bench_stats_features.py --repeats 5 --warmups 1
```

文档检查：

```bash
uv run zensical build
```
