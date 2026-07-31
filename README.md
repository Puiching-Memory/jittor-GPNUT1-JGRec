# jittor-GPNUT1-JGRec

**第六届计图人工智能挑战赛 · 赛道一：基于图学习的动态推荐任务**

- 战队名：GPNUT1
- GitLink 仓库：https://gitlink.org.cn/Puiching-Memory/jittor-GPNUT1-JGRec
- GitHub 仓库：https://github.com/Puiching-Memory/jittor-GPNUT1-JGRec
- 开源协议：MIT

项目目标是在给定历史时序交互 `(src, dst, time)` 和测试候选集合
`(src, time, c1...c100)` 的条件下，为每个候选目标输出可提交的交互概率分布。

当前主线模型是 `hybrid`：以时间因果切分训练融合器，将强统计记忆、候选先验、结构共现、
Two-Tower 表示、图协同过滤和序列偏好组合为候选级特征，再用 Jittor MLP 在每行 100 个候选内
做 softmax 重排序。

## 当前提交候选

当前工作区已验证的 A 榜提交包：

```text
result/hybrid_perfcheck_d1d2_50k20k_mrr_r100_ch32_seed60/result.zip
```

线上反馈分数：

```text
1.2044345219596662
```

该包为全量 `dataset1` + `dataset2` 重新生成结果，不使用 `--dataset` 或 `--limit-rows`：

```text
dataset1 -> result/hybrid_perfcheck_d1d2_50k20k_mrr_r100_ch32_seed60/csv/dataset1.csv
dataset2 -> result/hybrid_perfcheck_d1d2_50k20k_mrr_r100_ch32_seed60/csv/dataset2.csv
```

提交时只上传 `result.zip`。不要上传 `data/`、`result/` 目录或中间日志。

## 数据目录

程序按数据集目录发现输入文件：

```text
data/
  dataset1/
    train.csv
    test.csv
  dataset2/
    train.csv
    test.csv
```

`train.csv` 需要包含 `src,dst,time`。`test.csv` 需要包含 `src,time,c1,...,c100`。

## 快速运行

安装依赖：

```bash
uv sync
```

生成完整提交：

```bash
uv run jgrec-build
```

快速冒烟：

```bash
uv run jgrec-build --limit-rows 2 --max-fit-events 512 --max-train-events 32 --max-val-events 16 --num-negatives 3 --epochs 1 --disable-gnn --disable-seq
```

输出结构：

```text
result/<run_id>/
  csv/
    dataset1.csv
    dataset2.csv
  result.zip
  memory.log
```

`result.zip` 根目录直接包含 `dataset1.csv`、`dataset2.csv`，不包含额外目录层级。

## 训练后保存 checkpoint

为完整 Hybrid 训练增加 `--save-checkpoint`，每个数据集的 `fit()` 结束后会立即写入模型状态：

```bash
uv run jgrec-build --save-checkpoint checkpoints/checkpoint1.pkl
```

也可以沿用分数据集冲分流程。第一次运行会保留 `checkpoint1.pkl.tmp`，第二个数据集训练完成后自动发布正式文件：

```bash
uv run jgrec-build --dataset dataset2 --save-checkpoint checkpoints/checkpoint1.pkl
uv run jgrec-build --dataset dataset1 --save-checkpoint checkpoints/checkpoint1.pkl
```

正式 checkpoint 同时包含 `dataset1` 和 `dataset2` 状态。`.tmp` 表示仍缺少数据集，不能提交。已有正式文件不会被覆盖，新的模型应使用新的 checkpoint 路径。

只加载 checkpoint 推理，不再读取训练集或调用 `fit()`：

```bash
uv run jgrec-build --load-checkpoint checkpoints/checkpoint1.pkl --run-name checkpoint1_verify
```

checkpoint 和 `result.zip` 是两个独立产物；比赛材料要求 checkpoint 时单独提交 `.pkl`。

## 冲分运行建议

当前本地 8G 显存、24G 内存环境下，建议优先单独重跑 `dataset2`，确认改动收益后再与稳定的
`dataset1.csv` 拼包。完整命令和提交检查表见
[提交说明](docs/operations/submission.md)。

核心思路：

- `--max-fit-events 0` 保持最终 encoder 使用完整训练历史。
- 提高 `--max-train-events`、`--max-val-events` 可以让融合器看到更多监督事件，但会增加训练时间。
- 默认 `--selection-metric mrr` 与线上 query-level MRR 对齐；需要复现实验时仍可显式切回 `ap`。
- 默认 `--fusion-mode ensemble --expert-blend-mode rrf`，避免 MLP 概率与 LGBM LambdaRank 分数量纲直接相加。
- 基础 FusionMLP 默认以 `--fusion-context-transform-version 1` 同时消费 raw、减 query 行均值、减 query 行最大值三组通道；LGBM 仍消费 raw。
- 调试时可用 `--no-refit-full` 复用验证口径 encoder；正式提交默认仍做全量 refit。
- `--test-candidate-negative-ratio` 用于校准冷启动/新链接数据的负采样分布。
- `--supervised-feature-cache-dir .cache/supervised-features` 可在只调整融合器参数时复用 train/val 监督特征。
- 不建议为了提速直接关闭 `structure`、`two_tower`、`candidate_prior`，这些是当前 dataset2 提升的关键特征。

## 模型摘要

`hybrid` 特征顺序为：

```text
stats + candidate_prior + structure + two_tower + graph + sequence
```

主要模块：

- `stats.py`：历史重复、近期命中、目标热度、源节点活跃度等因果统计特征。
- `candidate_prior.py`：只使用 `test.csv` 候选出现频次和行内 rank，不使用标签，用于补足未见目标节点信号。
- `structure.py`：共同邻居、Jaccard、共现、转移等结构特征。
- `two_tower.py`：Jittor 双塔候选表示，输出 dot/cosine 特征。
- `gnn.py`：XSimGCL/LightGCN 图协同过滤塔。
- `sequence.py`：SASRec/GRU 序列偏好塔。
- `fusion.py`：Jittor MLP 融合器，按验证集在多个特征 mask 中选择最终特征组。

自动策略不会读取数据集名称，而是根据训练 holdout 和测试候选分布将数据画像为
`repeat_memory`、`balanced` 或 `new_link_cold`，再自动调整 test-candidate 负采样比例。

## 开发检查

```bash
uv run python -m compileall -q src scripts tests
uv run --group dev pytest
uv run --group dev ruff check .
uv run zensical build
```

WSL/Jittor 本地环境可以直接用项目源码运行：

```bash
export PYTHONPATH=src
/mnt/d/work/jittor-local/env/bin/python -m jgrec.cli --help
```

## 文档入口

- [提交说明](docs/operations/submission.md)
- [运行手册](docs/operations/runbook.md)
- [模型设计](docs/system/modeling.md)
- [系统架构](docs/system/architecture.md)
- [数据契约](docs/task/data-contract.md)
- [实验与基准](docs/experiments/benchmarks.md)
