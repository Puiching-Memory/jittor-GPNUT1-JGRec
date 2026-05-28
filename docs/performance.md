# 性能基准

本文档记录当前工程中已经验证过的性能改进。后续优化需要继续沿用同一规则：先记录改动前基准，再实现改动，最后用相同输入复测，并确认输出语义没有变化。

## 基准原则

- 使用固定数据、固定查询数、固定候选数做前后对比。
- 记录 median 时间，避免单次运行波动影响判断。
- 涉及打分逻辑的改动，需要校验 checksum 或输出概率格式，确保性能收益不是来自语义退化。
- 涉及提交文件的改动，需要校验 CSV 行列数、概率范围、每行和、zip 内容。
- 没有稳定收益的改动也要记录，避免后续重复投入。

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

## 线上提交记录

### 第一版完整提交

提交时间：2026-05-27

提交产物：

```text
result/rw32-bs2048-vr0p1-cr0p75-tr20000-va5000-neg31-fit0-ep5-tbs512-lr0p001-wd0-fh64-gnnxsimgcl-ge3-gd128-gl2-gmge0-gmte40000-seqon-se3-sd128-sl64-s42/result.zip
```

本地时间切分验证：

| 数据集     | 本地 MRR | 选择的融合特征 |
| ---------- | -------: | -------------- |
| `dataset1` |  0.80293 | `stats_gnn`    |
| `dataset2` |  0.51770 | `stats_gnn`    |

线上总分：

| 版本                | 线上得分 |
| ------------------- | -------: |
| 第一版完整 GNN 提交 |   1.1452 |

提交前结构校验：

| 文件           |   行数 | 每行列数 | 概率范围 | 行和       |
| -------------- | -----: | -------: | -------- | ---------- |
| `dataset1.csv` |  61051 |      100 | 合法     | 约等于 `1` |
| `dataset2.csv` | 153420 |      100 | 合法     | 约等于 `1` |

结论：

- 第一版端到端链路已经满足比赛提交格式，并且线上成功计分，可作为后续优化的正式基线。
- 本地验证 MRR 加和为 `1.32063`，高于线上总分 `1.1452`，说明当前本地时间切分偏乐观，不能只按本地 MRR 判断改动是否有效。
- 线上只反馈总分，暂时无法拆分 `dataset1` 和 `dataset2` 的真实贡献；后续需要保留每次提交的本地分项 MRR、线上总分、运行参数和产物路径。
- `dataset1` 本地 MRR 已经较高，短期主要风险是过拟合历史重复边；`dataset2` 本地 MRR 较低且规模更大，后续提分优先级更高。
- 当前默认模型在两个数据集上都选择了 `stats_gnn`，说明图塔特征在本地验证中有收益；序列塔没有进入最终选择，后续要么改进序列建模，要么降低它的默认训练成本。

## 已验证改进

### CSV 批量写出

提交 CSV 原实现逐行调用 `csv.writer`。改为按数据块使用 `np.savetxt` 后，减少了 Python 层循环和格式化调用开销。

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

结论：有效。该改动同时降低训练统计构建耗时和候选特征查询耗时，是当前收益最高的结构性优化之一。

### Dense 目标节点特征

目标热度和目标最近交互时间原实现依赖 dict 查询。对节点 ID 范围可控的数据集，改为 dense array 查询：`dst_popularity_dense` 和 `dst_recent_time_dense`。

|   指标 |  改动前 |  改动后 |  收益 |
| -----: | ------: | ------: | ----: |
| median | 2.0327s | 1.3122s | 1.55x |

结论：有效。该改动直接减少候选级 Python dict 查询，适合作为后续特征工程的默认方向。当前实现保留 `DENSE_NODE_LIMIT`，避免异常大的节点 ID 造成不可控内存占用。

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

结论：有效。该改动在 cold 口径下仍有稳定收益，并且 checksum 完全一致，保留实现。

## 无明显收益改进

### 小样本 GNN 融合消融

基准配置：

- `max_fit_events=2048`
- `max_train_events=256`
- `max_val_events=128`
- `num_negatives=7`
- `epochs=2`
- CPU smoke，输出每个数据集前 2 行，仅用于验证模型链路和相对趋势。
- 表中 GNN 消融来自接入验证选择层前后的早期小样本测试；后续正式性能结论需要用当前因果清零逻辑重新跑。

| 模型组                                           | dataset1 MRR | dataset2 MRR | 结论               |
| ------------------------------------------------ | -----------: | -----------: | ------------------ |
| `stats + MLP`                                    |      0.85033 |      0.81547 | 当前小样本最稳     |
| `LightGCN + SASRec + stats + MLP`                |      0.71310 |      0.71352 | 小样本未体现收益   |
| `XSimGCL + SASRec + stats + MLP`，无特征选择     |      0.53315 |      0.47481 | 小样本明显拖累     |
| `XSimGCL + SASRec + stats + MLP`，带验证特征选择 |      0.80432 |      0.79357 | 选择层有效降低风险 |

结论：图塔和序列塔已经接入主链路，但小样本、低 epoch 配置下不能证明 GNN 特征稳定增益。当前默认保留图塔训练，同时通过验证 MRR 在 `stats`、`stats_gnn`、`stats_gnn_seq` 中选择最终融合头；如果验证选择 `stats`，最终全量拟合会跳过图塔和序列塔，避免无效耗时。

### 特征矩阵预分配

候选特征构造从 list 收集后 `stack` 改为预分配 numpy array 后逐行填充。

|   指标 |  改动前 |  改动后 |  收益 |
| -----: | ------: | ------: | ----: |
| median | 2.6447s | 2.6369s | 1.00x |

结论：单独看这个改动没有明显收益。保留预分配实现的原因是它让后续向量化改造更直接，但不能把它计入主要性能收益。

### Per-source 排序索引批量查 pair 特征

尝试为每个源节点按需构建排序后的 `dst`、pair 计数、pair 最近时间和 recent rank 数组，再用 `np.searchsorted` 批量查询 `pair_strength`、`repeat_rate`、`pair_recency`、`recent_hit`。

| 指标                   |  改动前 |  改动后 | 结论       |
| ---------------------- | ------: | ------: | ---------- |
| `features_cold_median` | 0.7489s | 0.9468s | 变慢       |
| `features_warm_median` | 0.7489s | 0.4915s | 缓存后变快 |

结论：不保留。测试 batch 中重复 `src` 的历史较长，首次为这些源节点排序建索引的成本高于节省的 dict 查询成本；warm 口径收益不能代表正式提交路径的首次推理成本。

## 当前结论

已经确认有效的方向：

- 减少候选级 tuple 构造。
- 减少候选级 Python dict 查询。
- 对高频标量特征使用 dense array。
- 对提交文件使用 batch 写出。

已经确认收益不足的方向：

- 只把 list/stack 改成预分配数组，但仍逐候选执行 Python 查询。

## 后续优化方向

优先级最高的是继续减少候选级 Python 循环：

- 将可 dense 化的节点特征统一转为数组查询。
- 对测试候选批次做批量特征构造，减少逐候选函数调用。
- 缓存训练/验证阶段反复使用的候选特征。
- 对负采样后的训练样本提前构造特征块，避免每个 epoch 重复查统计。
- 如果引入图编码器，需要单独记录训练耗时、推理耗时、MRR 变化，不能只看模型表达能力。

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

正式性能改动完成后，需要补充该文档中的前后对比表，并说明是否采纳。
