# 模型方案

## 当前定位

当前实现已经从统计线性模型切换为激进的混合图推荐模型：

```text
TemporalHybridRanker =
  因果时间切分
  + JittorGeometric XSimGCL/LightGCN 图塔
  + JittorGeometric SASRec 序列塔
  + 时序统计特征
  + Jittor MLP 候选重排序
```

目标不是做通用 link prediction，而是直接优化赛题的 100 个候选节点重排序。

## 模块划分

实现位置：

```text
src/jgrec/
├── idmap.py
└── rankers/hybrid/
    ├── stats.py      # 时序统计特征
    ├── gnn.py        # XSimGCL/LightGCN 图塔
    ├── sequence.py   # SASRec 序列塔
    ├── fusion.py     # 融合 MLP
    └── ranker.py     # TemporalHybridRanker 对外接口
```

对外接口保持稳定：

```python
ranker = TemporalHybridRanker(recent_window=32)
report = ranker.fit(interactions, training_config=config)
probs = ranker.predict_batch(queries)
```

## 因果训练流程

每个数据集单独训练：

```text
完整 train.csv 时间排序
        │
        ├── context events
        │       ├── 训练图塔
        │       ├── 训练序列塔
        │       └── 构建统计索引
        │
        ├── supervised train events
        │       └── 正样本 + 负采样候选
        │
        └── validation tail
                ├── 本地 AP
                └── 本地 MRR 诊断
```

监督样本第一个候选固定为真实目标：

```text
[positive_dst, negative_1, negative_2, ...]
```

融合层使用候选集 softmax cross entropy：

```python
loss = -log_softmax(logits, dim=1)[:, 0].mean()
```

默认验证选择指标是 AP，对齐官方 CRAFT baseline 的 early stopping 口径；融合 MLP 的 early stop patience 默认为 10。MRR 继续保留为比赛指标诊断：

```text
AP = average_precision_score(flat_labels, flat_scores)
```

MRR 计算方式：

```text
rank = 1 + count(score_negative > score_positive)
MRR = mean(1 / rank)
```

训练完成后，模型会用完整训练历史重新训练图塔、序列塔和统计索引，再对正式 `test.csv` 输出概率。

## 图塔

图塔使用 `third_party/JittorGeometric`：

- 默认模型：`XSimGCL`
- 可选模型：`LightGCN`
- 图结构：二部图 `src <-> dst`
- 训练目标：BPR，XSimGCL 额外使用对比学习扰动

当前训练三个时间窗口：

| 特征名       | 边窗口      | 目的         |
| ------------ | ----------- | ------------ |
| `gnn_full`   | 全量历史边  | 长期协同过滤 |
| `gnn_recent` | 最近 35% 边 | 近期偏好     |
| `gnn_short`  | 最近 10% 边 | 短期趋势     |

每个窗口输出：

```text
dot(src_embedding, dst_embedding)
```

## 序列塔

序列塔使用 JittorGeometric 的 `SASRec` 实现。每个 `src` 的历史目标节点序列作为行为序列：

```text
src: dst_1, dst_2, dst_3, ...
```

训练时用历史前缀预测下一个 `dst`，使用 BPR 损失。预测时对每个候选输出：

```text
sasrec_score = dot(sequence_embedding(src_history), candidate_embedding)
```

序列塔用于补图塔的短板：LightGCN/XSimGCL 更偏静态协同过滤，SASRec 负责顺序兴趣变化。

## 统计特征

统计特征继续保留，作为强规则信号和冷启动兜底：

| 特征             | 含义                                             |
| ---------------- | ------------------------------------------------ |
| `pair_strength`  | `log1p(pair_count)`，源目标历史强度              |
| `repeat_rate`    | `pair_count / src_total`，源节点历史中该目标占比 |
| `pair_recency`   | 源目标最近交互相对查询时间的衰减                 |
| `dst_popularity` | 目标节点全局热度                                 |
| `dst_recency`    | 目标节点最近被交互的时间衰减                     |
| `recent_hit`     | 目标是否命中源节点最近交互序列                   |
| `src_activity`   | 源节点历史活跃度                                 |
| `src_recency`    | 源节点最近活跃度                                 |

## 融合层

最终候选特征为：

```text
8 个统计特征
+ 3 个图塔分数
+ 1 个序列塔分数
= 12 维候选特征
```

融合层是 Jittor MLP：

```text
feature_dim -> hidden_dim -> hidden_dim/2 -> 1
```

输出 logits 后在每行 100 个候选内做 softmax，得到提交概率。

融合训练会同时比较以下候选特征组：

- `stats`
- `stats_gnn`
- `stats_gnn_seq`

最终默认使用本地验证 AP 最高的一组。若验证选择 `stats`，最终全量拟合阶段会跳过图塔和序列塔训练，避免未收敛图特征拖慢并拖垮提交结果。

## 关键参数

| 参数                    | 默认值    | 说明                         |
| ----------------------- | --------- | ---------------------------- |
| `--gnn-model`           | `xsimgcl` | 图塔模型，可选 `lightgcn`    |
| `--gnn-embedding-dim`   | `128`     | 图 embedding 维度            |
| `--gnn-layers`          | `2`       | 图传播层数                   |
| `--gnn-epochs`          | `3`       | 每个图窗口训练轮数           |
| `--gnn-max-graph-edges` | `0`       | 每个图窗口最多建图边数       |
| `--gnn-max-train-edges` | `40000`   | 每轮图训练采样边数           |
| `--seq-epochs`          | `3`       | SASRec 训练轮数              |
| `--seq-max-samples`     | `50000`   | SASRec 最多训练样本          |
| `--seq-max-len`         | `64`      | 源节点历史序列长度           |
| `--seq-hidden-size`     | `128`     | SASRec hidden size           |
| `--fusion-hidden-dim`   | `64`      | 最终 MLP hidden width        |
| `--max-fit-events`      | `0`       | 训练历史尾部截断，0 表示全量 |

## 当前取舍

- 选择 XSimGCL/LightGCN 作为主图模型，因为它直接服务推荐排序，工程风险低于 TGN/DyGFormer。
- 保留 SASRec，因为测试查询带时间，源节点近期行为顺序很可能有收益。
- 保留统计特征，因为重复交互、目标热度和近因信号在该赛题中非常强。
- 使用 MLP 融合而不是直接加权，避免手工调不同模型分数尺度。
- TGN/DyGFormer 暂不作为主线；它们更适合事件级动态 link prediction，直接迁移到 100 候选 MRR 风险更高。

## 验证策略

每次模型改动至少记录：

- `stats + MLP`
- `LightGCN + SASRec + stats + MLP`
- `XSimGCL + SASRec + stats + MLP`

需要同时记录本地 AP、本地 MRR、训练耗时、推理耗时和输出校验结果。性能与质量数据统一写入 [性能基准](performance.md)。
