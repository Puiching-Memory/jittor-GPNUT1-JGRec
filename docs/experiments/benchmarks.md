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

## Optuna 调参

自动调参脚本：

```bash
uv run jgrec-tune-temporal-graph --n-trials 32 --n-jobs 1 --gpu-id 0 --quiet
```

等价脚本入口为 `uv run python scripts/tune_temporal_graph.py ...`。

默认优化两个数据集的平均 MRR，不做测试集推理和 ZIP 写出；trial 结果写入同一个 study 目录：

```text
result/optuna/temporal_graph_mrr/
├── study.db
├── trials.jsonl
└── best.json
```

推荐多 GPU 方式是多个 Python worker 共享同一个 SQLite study，每个进程绑定一张卡：

```bash
for gpu in 0 1 2 7; do
  uv run jgrec-tune-temporal-graph \
    --study-name temporal_graph_mrr_v1 \
    --n-trials 8 \
    --n-jobs 1 \
    --gpu-id "$gpu" \
    --max-train-events 20000 \
    --max-val-events 5000 \
    --epochs-max 6 \
    --quiet &
done
wait
```

`--n-jobs` 只表示单个 Python 进程内的 Optuna 并发。Jittor 的 CUDA 状态是进程级全局设置，GPU 实验保持 `--n-jobs 1`，通过多进程扩展到多卡。
`--gpu-id` 会同时作为默认 worker id，并让每个 worker 的 TPE sampler seed 自动错开；需要手工复现实验时可显式设置 `--worker-id` 和 `--sampler-seed`。

最近一次链路冒烟：

```bash
uv run jgrec-tune-temporal-graph --datasets dataset1 --n-trials 1 --n-jobs 1 --cpu --max-fit-events 512 --max-train-events 32 --max-val-events 16 --epochs-max 2 --study-name temporal_graph_optuna_smoke --quiet
```

结果：通过。生成 `study.db`、`trials.jsonl`、`best.json`，trial0 在极小样本验证集上的 MRR 为 `0.05193`。该结果只验证调参链路，不作为模型性能结论。

并发链路冒烟：

```bash
rm -rf result/optuna/temporal_graph_optuna_concurrency_smoke
for worker in 0 1; do
  uv run jgrec-tune-temporal-graph --datasets dataset1 --n-trials 1 --n-jobs 1 --cpu --worker-id "$worker" --max-fit-events 512 --max-train-events 32 --max-val-events 16 --epochs-max 2 --study-name temporal_graph_optuna_concurrency_smoke --quiet &
done
wait
```

结果：通过。全新 SQLite study 首次多进程初始化使用 `.study-init.lock` 避免表结构竞态；2 个 worker 共享同一 study，最终得到 2 个 COMPLETE trial，`trials.jsonl` 为 2 行。

CUDA 入口冒烟：

```bash
uv run jgrec-tune-temporal-graph --datasets dataset1 --n-trials 1 --n-jobs 1 --gpu-id 1 --max-fit-events 256 --max-train-events 16 --max-val-events 8 --epochs-max 2 --study-name temporal_graph_gpu_entry_smoke --quiet
```

结果：通过。GPU 绑定、Jittor CUDA 编译、`best_config` 输出和 `study.db` 写入正常；极小样本 MRR 为 `0.04676`，不作为模型性能结论。

当前搜索空间覆盖：

- `num_negatives`: 15/31/63/99
- `history_len`: 16/32/64/96
- `candidate_history_len`: 8/16/32/48
- `hidden_size`: 64/96/128/192
- `layers`: 1..4
- `heads_h{hidden_size}`: 与 hidden size 整除的 2/3/4/6/8；复跑时优先看 `best.json` 里的完整 `config.heads`
- `dropout`: 0.05..0.45
- `lr`: 2e-4..3e-3
- `weight_decay`: 1e-7..3e-3
- `selection_metric`: AP 或 MRR

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
