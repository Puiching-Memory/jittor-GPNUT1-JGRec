# 性能基准

本文档只保留当前默认模型、已验证工程优化和后续实验门禁。已经失败或被撤回的模型结构实验不再展开细节，避免和当前主链路混淆。

## 基准原则

- 使用固定数据、固定查询数、固定候选数做前后对比。
- 记录 median 时间，避免单次运行波动影响判断。
- 涉及打分逻辑的改动，需要校验 checksum 或输出概率格式，确保性能收益不是来自语义退化。
- 涉及提交文件的改动，需要校验 CSV 行列数、概率范围、每行和、zip 内容。
- 没有稳定收益的改动也要记录，避免后续重复投入。

## 当前冠军基线

当前默认提交模型已恢复为第一版完整 GNN 提交对应的代码路径：

- `num_negatives=31`
- 随机负采样
- XSimGCL 三窗口图塔：`gnn_full`、`gnn_recent`、`gnn_short`
- SASRec 序列塔可训练，但由本地验证在 `stats`、`stats_gnn`、`stats_gnn_seq` 中选择最终特征组
- 无 Semantic ID、无 mixed hard negatives、无 SVD 谱图分支、无 item-transition 图分支

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

- mixed hard negatives、transition 图和未校准 proxy 都不能作为当前优化目标。
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
