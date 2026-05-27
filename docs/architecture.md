# 系统架构

## 包结构

```text
src/jgrec/
├── __init__.py
├── cli.py
├── data.py
├── model.py
└── submission.py
```

## 模块职责

| 模块               | 职责                                                 |
| ------------------ | ---------------------------------------------------- |
| `jgrec.cli`        | 命令行参数、Jittor CUDA 开关、数据集遍历、主流程编排 |
| `jgrec.data`       | 数据集发现、训练交互读取、测试查询读取、CSV 行数统计 |
| `jgrec.model`      | 构建历史图统计、候选特征、Jittor 批量打分            |
| `jgrec.submission` | 单数据集输出、CSV 格式化、ZIP 打包、结果校验         |

## 数据流

```text
data/dataset*/train.csv
        │
        ▼
read_interactions()
        │
        ▼
HeuristicJittorRanker.fit()
        │
        ├── src histories
        ├── pair counts / pair recency
        └── dst popularity / dst recency

data/dataset*/test.csv
        │
        ▼
read_test_queries()
        │
        ▼
predict_batch()
        │
        ▼
result/<dataset>.csv
        │
        ▼
result.zip
```

## 运行边界

当前 MVP 没有持久化模型文件。每次运行都会：

1. 读取一个数据集的完整训练交互。
2. 在内存中构建统计索引。
3. 流式读取测试集并按 batch 生成预测。
4. 写出该数据集 CSV。
5. 释放局部对象并进入下一个数据集。

这种设计避免跨数据集状态污染，便于后续为不同场景引入不同模型或参数。

## 性能策略

- 训练集读取使用标准库 `csv`，避免额外依赖和高内存 DataFrame。
- 测试集按行流式读取，输出按行写入。
- 候选特征按 batch 组装为 `batch_size x 100 x feature_dim` 的 `numpy` 数组。
- Jittor 负责 batch 内张量融合和 softmax。
- 默认 `batch_size=2048`，在当前数据规模下内存占用可控。

## 工程约束

- 不使用比赛外部数据。
- 输出文件无表头。
- 每行必须对应测试集同一行的 100 个候选节点顺序。
- 每个概率必须保留 8 位小数。
- `data/`、`result/`、`result.zip` 不提交到仓库。

## 可替换点

后续如果引入可训练模型，优先替换 `jgrec.model` 内部实现，保持以下接口稳定：

```python
ranker = Ranker(...)
ranker.fit(interactions)
probs = ranker.predict_batch(queries)
```

只要 `predict_batch()` 返回形状为 `(batch, 100)` 的概率数组，`submission.py` 和 `cli.py` 不需要变更。
