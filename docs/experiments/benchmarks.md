# 性能基准

本文档记录当前默认模型 `temporal-graph` 的工程验证命令和后续实验门禁。已经移除的统计融合与多塔路线不再作为默认链路记录。

## 基准原则

- 使用固定数据、固定查询数、固定候选数做前后对比。
- 记录数据集、split、负采样、seed、训练事件数、验证事件数和 epoch。
- 涉及提交文件的改动，需要校验 CSV 行列数、概率范围、每行和、zip 内容。
- 本地 AP/MRR 只作为诊断，不能替代线上分数。

## 当前默认模型

默认后端：

```bash
uv run jgrec-build --model temporal-graph
```

核心结构：

- JittorGeometric `TemporalData`
- temporal neighbor sampler
- compact global node id，`0` 作为 padding
- src history tokens
- candidate history tokens
- CRAFT-style cross-attention
- candidate-set softmax loss

默认训练候选数对齐测试候选尺度：

```text
1 positive + 99 negatives
```

## 冒烟基准

用于验证环境、Jittor/JittorGeometric 初始化、端到端训练、CSV 写出、校验和 ZIP 打包：

```bash
uv run jgrec-build --cpu --limit-rows 2 --max-fit-events 512 --max-train-events 32 --max-val-events 16 --num-negatives 3 --epochs 1 --history-len 8 --candidate-history-len 4 --hidden-size 32 --layers 1 --heads 2 --no-refit-full --quiet-ranker
```

最近一次验证：

| 数据集     | train | val | rows | 结果 |
| ---------- | ----: | --: | ---: | ---- |
| `dataset1` |    32 |  16 |    2 | 通过 |
| `dataset2` |    32 |  16 |    2 | 通过 |

输出示例：

```text
result/temporal-graph_sample-2-rows_cpu_seed-42_hist-8_candhist-4_dim-32_<hash>/result.zip
```

## 实验门禁

模型实验必须记录以下字段：

| 字段     | 要求                                                       |
| -------- | ---------------------------------------------------------- |
| 实验状态 | `keep`、`reject`、`archive` 三选一                         |
| 代码状态 | 说明是否进入默认 CLI；未进入默认链路的实验代码应删除或隔离 |
| 协议     | 数据集、split、负采样、seed、训练事件数、验证事件数、epoch |
| 本地结果 | 分 dataset AP/MRR、关键耗时和输出校验结果                  |
| 线上结果 | 提交产物路径、线上总分；未提交要写明原因                   |
| 最终决策 | 明确保留、拒绝或仅归档，不能只列数字                       |

当前门禁：

- 默认后端只能是 `temporal-graph` 或显式指定的 `craft`。
- 如果实验代码依赖旧融合/多塔接口，应迁移为 `temporal_graph` 消融或删除。
- 结构性改动至少跑冒烟命令、单元测试和 Ruff。

## 复测命令

基础正确性检查：

```bash
uv run ruff check .
uv run python -m compileall -q src
uv run --group dev pytest
```

完整提交链路冒烟：

```bash
uv run jgrec-build --cpu --limit-rows 2 --max-fit-events 512 --max-train-events 32 --max-val-events 16 --num-negatives 3 --epochs 1 --history-len 8 --candidate-history-len 4 --hidden-size 32 --layers 1 --heads 2 --no-refit-full --quiet-ranker
```

文档检查：

```bash
uv run zensical build
```
