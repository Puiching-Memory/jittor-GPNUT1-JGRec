# 模型方案

## 当前定位

当前模型是工程 MVP，不是最终竞赛上限方案。目标是：

- 保证端到端可运行。
- 满足 Jittor 框架使用要求。
- 输出符合比赛提交格式。
- 为后续可训练模型留下稳定接口。

## Baseline 类

实现位置：

```text
src/jgrec/model.py
```

核心类：

```python
HeuristicJittorRanker
```

使用方式：

```python
ranker = HeuristicJittorRanker(recent_window=32)
ranker.fit(interactions)
probs = ranker.predict_batch(queries)
```

## 统计索引

`fit()` 会按时间排序训练交互，并构建以下索引：

| 索引               | 说明                                                   |
| ------------------ | ------------------------------------------------------ |
| `src_histories`    | 每个源节点的交互次数、最近时间、目标频次、最近目标序列 |
| `dst_counts`       | 每个目标节点作为目标出现的次数                         |
| `dst_recent_time`  | 每个目标节点最近被交互的时间                           |
| `pair_counts`      | `(src, dst)` 历史交互次数                              |
| `pair_recent_time` | `(src, dst)` 最近交互时间                              |

## 候选特征

对每个测试查询 `(src, time, c1...c100)`，每个候选目标节点会生成 6 个特征：

| 特征             | 含义                                             |
| ---------------- | ------------------------------------------------ |
| `pair_strength`  | `log1p(pair_count)`，源目标历史强度              |
| `repeat_rate`    | `pair_count / src_total`，源节点历史中该目标占比 |
| `pair_recency`   | 源目标最近交互相对查询时间的衰减                 |
| `dst_popularity` | 目标节点全局热度                                 |
| `dst_recency`    | 目标节点最近被交互的时间衰减                     |
| `recent_hit`     | 目标是否命中源节点最近交互序列，越近分值越高     |

## Jittor 打分

特征组装为：

```text
batch_size x 100 x 6
```

随后通过 Jittor 张量进行线性融合：

```python
logits = (features * weights).sum(dim=2)
probs = softmax(logits, dim=1)
```

输出 `batch_size x 100` 的概率矩阵。

## 设计取舍

| 取舍       | 当前选择                | 原因                                 |
| ---------- | ----------------------- | ------------------------------------ |
| CSV 读取   | 标准库 `csv`            | 低依赖、流式、内存可控               |
| 特征计算   | Python 字典和 `Counter` | MVP 简单可靠，便于调试               |
| 打分       | Jittor batch 张量       | 满足框架要求，减少逐行开销           |
| 训练       | 无监督启发式            | 没有公开标签文件，先保证可提交       |
| 模型持久化 | 不保存                  | 当前统计可快速重建，避免状态文件管理 |

## 升级路线

优先级建议：

1. **本地验证切分**：从训练集尾部按时间切出验证集，用 MRR 选择权重。
2. **权重学习**：把当前 6 个特征作为输入，用 Jittor 训练 logistic/BPR/listwise reranker。
3. **负采样策略**：对每个正样本采样同时间附近或高热度负样本，减少训练评估偏差。
4. **序列模型**：按源节点构建目标序列，引入 GRU/SASRec 类模型。
5. **图模型**：用 JittorGeometric 的 TGN/JODIE/GraphMixer 思路替换统计 ranker。
6. **大规模优化**：对 B 榜千万级边，改造为分块读取、压缩索引或磁盘缓存。

## 接口稳定性

后续模型应优先兼容当前接口：

```python
fit(interactions: list[Interaction]) -> None
predict_batch(queries: list[TestQuery]) -> np.ndarray
```

`predict_batch()` 必须返回：

- shape: `(len(queries), 100)`
- dtype: 可转为 float
- value range: `[0, 1]`
- 每行对应测试集候选顺序
