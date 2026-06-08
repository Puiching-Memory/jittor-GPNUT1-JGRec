# 模型优化

本文档记录当前默认模型 `temporal-graph` 的调参、提交、模型消融和实验门禁。训练、推理、数据读取、批构造和邻居采样相关的工程优化记录见 [架构优化](architecture-optimization.md)。

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

| 数据集     | train |  val | rows | 结果 |
| ---------- | ----: | ---: | ---: | ---- |
| `dataset1` |    32 |   16 |    2 | 通过 |
| `dataset2` |    32 |   16 |    2 | 通过 |

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

等价脚本入口为 `uv run python scripts/tune_temporal_graph.py ...`。调参默认固定 `--num-negatives 99`，
并使用 `--validation-candidates test_like` 从真实 `test.csv` 候选分布构造验证负样本。

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
    --num-negatives 99 \
    --validation-candidates test_like \
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

当前搜索空间覆盖。`num_negatives` 属于评估协议，固定为 99，不参与搜索：

- `history_len`: 16/32/64/96
- `candidate_history_len`: 8/16/32/48
- `hidden_size`: 64/96/128/192
- `layers`: 1..4
- `heads_h{hidden_size}`: 与 hidden size 整除的 2/3/4/6/8；复跑时优先看 `best.json` 里的完整 `config.heads`
- `dropout`: 0.05..0.45
- `lr`: 2e-4..3e-3
- `weight_decay`: 1e-7..3e-3
- `selection_metric`: AP 或 MRR

## 线上提交记录

### hybrid perfcheck 50k/20k MRR r100 seed60

实验日期：2026-06-08。

实验状态：`keep`，当前线上最好提交。

协议：

- 提交产物：`result/hybrid_perfcheck_d1d2_50k20k_mrr_r100_ch32_seed60/result.zip`
- 全量输出 `dataset1` 和 `dataset2`，不使用 `--dataset` 或 `--limit-rows`
- Seed：60
- 训练事件历史上限：`max_fit_events=240000`
- 融合器监督训练事件数：`max_train_events=50000`
- 验证事件数：`max_val_events=20000`
- 候选数：`1 positive + 63 negatives`
- `selection_metric=mrr`
- `test_candidate_negative_ratio=1.00`
- `structure_cooccur_history_limit=32`

关键参数：

- `epochs=3`
- `train_batch_size=512`
- `fusion_hidden_dim=64`
- `gnn_model=xsimgcl`
- `gnn_edge_weighting=none`
- `gnn_epochs=1`
- `gnn_max_graph_edges=120000`
- `gnn_max_train_edges=60000`
- `seq_epochs=1`
- `two_tower_epochs=1`
- `supervised_feature_memmap=True`
- `supervised_feature_batch_size=256`

结果：

| 数据集     |      AP |     MRR | fusion                        | auto            |
| ---------- | ------: | ------: | ----------------------------- | --------------- |
| `dataset1` | 0.75512 | 0.77363 | `stats_prior_structure_tower` | `repeat_memory` |
| `dataset2` | 0.32691 | 0.55251 | `stats_prior_structure_tower_gnn` | `new_link_cold` |

| 指标            | 值                                                                 |
| --------------- | ------------------------------------------------------------------ |
| 线上总分        | `1.2044345219596662`                                               |
| 运行耗时        | `55m57s`                                                           |
| 运行时间        | `2026-06-08T05:22:19+00:00` 至 `2026-06-08T06:18:16+00:00`         |
| zip sha256      | `c4a5a16a9a1e65b0d7ac1dec5b23de4dca88f37d28c2bbe30231e42e7aa28b12` |
| `dataset1` 行数 | `61051`                                                            |
| `dataset2` 行数 | `153420`                                                           |

结论：性能优化后按同一主线配置全量复跑，线上反馈从上一版 `1.1983` 提升到
`1.2044345219596662`。保留为当前线上冠军基线。

### hybrid full 50k/20k MRR r100 seed60

实验日期：2026-06-08。

实验状态：`archive`，上一版线上最好提交。

协议：

- 提交产物：`result/hybrid_full_d1d2_50k20k_mrr_r100_ch32_seed60/result.zip`
- 全量输出 `dataset1` 和 `dataset2`，不使用 `--dataset` 或 `--limit-rows`
- Seed：60
- 训练事件历史上限：`max_fit_events=240000`
- 融合器监督训练事件数：`max_train_events=50000`
- 验证事件数：`max_val_events=20000`
- 候选数：`1 positive + 63 negatives`
- `selection_metric=mrr`
- `test_candidate_negative_ratio=1.00`
- `structure_cooccur_history_limit=32`

关键参数：

- `epochs=3`
- `train_batch_size=512`
- `fusion_hidden_dim=64`
- `gnn_model=xsimgcl`
- `gnn_edge_weighting=none`
- `gnn_epochs=1`
- `gnn_max_graph_edges=120000`
- `gnn_max_train_edges=60000`
- `seq_epochs=1`
- `two_tower_epochs=1`
- `supervised_feature_memmap=True`
- `supervised_feature_batch_size=256`

结果：

| 数据集     |      AP |     MRR | fusion                        | auto            |
| ---------- | ------: | ------: | ----------------------------- | --------------- |
| `dataset1` | 0.75681 | 0.77466 | `stats_prior_structure_tower` | `repeat_memory` |
| `dataset2` | 0.33042 | 0.54808 | `stats_prior_structure_tower` | `new_link_cold` |

| 指标            | 值                                                                 |
| --------------- | ------------------------------------------------------------------ |
| 线上总分        | `1.1983`                                                           |
| zip sha256      | `bd6ae23521528e6b7d92e4073eff25ac6d7a0922be2ceac3523bd1be19307736` |
| `dataset1` 行数 | `61051`                                                            |
| `dataset2` 行数 | `153420`                                                           |

结论：保留为历史线上基线。后续冲分需要同时记录本地 AP/MRR、全量 zip 路径和线上反馈；若只重跑
`dataset2`，需要明确拼包来源并重新提交确认。

### temporal-graph Optuna best v1

实验日期：2026-06-02。

实验状态：`archive`，历史 temporal-graph 基线。

协议：

- Study：`result/optuna/temporal_graph_search_20260602_testlike_mrr_v1/`
- Trial：18
- 验证协议：`validation_candidates=test_like`
- 候选数：`1 positive + 99 negatives`
- 训练事件数：`max_train_events=20000`
- 验证事件数：`max_val_events=5000`
- Seed：60
- 提交产物：`result/temporal-graph_full_cuda_seed-60_hist-64_candhist-32_dim-128_f25b2f55/result.zip`

关键参数：

- `epochs=5`
- `train_batch_size=128`
- `lr=0.00044010925741869584`
- `weight_decay=0.0005923122960393677`
- `selection_metric=ap`
- `early_stop=5`
- `history_len=64`
- `candidate_history_len=32`
- `hidden_size=128`
- `layers=1`
- `heads=4`
- `dropout=0.17350208779836748`

结果：

| 指标                    | 值                   |
| ----------------------- | -------------------- |
| 本地 test-like 平均 MRR | `0.5890914935816703` |
| 本地 `dataset1` MRR     | `0.7326604298928556` |
| 本地 `dataset2` MRR     | `0.4455225572704851` |
| 线上总分                | `0.9406339921073574` |

结论：保留为历史 temporal-graph 提交基线。后续若重启该路线，优先做该配置附近的窄范围搜索、不同 seed 复测，
以及检查线上分数与本地 test-like 验证的相关性。

## 模型实验记录

### 时间残差流 v0

实验日期：2026-06-02。

实验状态：`reject`，不进入默认模型。

动机：本地错题显示模型容易把近期热门候选排到长尾正例前面。尝试在 `_pair_stats()` 外增加短期/长期时间流特征，并通过一个初始为 0 的 residual head 学习候选级 logit 校正：

```text
final_logit = graph_logit + flow_residual(flow_features)
```

特征包括：`src_fast_flow`、`src_slow_flow`、`candidate_fast_flow`、`candidate_slow_flow`、`pair_fast_flow`、`pair_slow_flow`、候选/配对加速度，以及 pair 相对 candidate popularity 的残差。

协议：使用当前线上 best v1 的完整超参，只替换模型结构；`validation_candidates=test_like`，`max_train_events=20000`，`max_val_events=5000`，`seed=60`，`refit_full=False`。

结果：

| 配置                        | `dataset1` MRR | `dataset2` MRR | 平均 MRR |
| --------------------------- | -------------: | -------------: | -------: |
| 当前 best v1                |       0.732660 |       0.445523 | 0.589091 |
| 时间残差流 v0，两数据集复测 |       0.731884 |       0.442893 | 0.587388 |

补充诊断：单独复测 `dataset2` 时有一轮达到 `report_mrr=0.449493`，但固定候选后处理网格只从 `0.447333` 提升到 `0.447557`，最佳为 `score - 0.5 * candidate_fast_flow`。说明短期热门流有弱信号，但当前训练负例没有稳定教会 residual head 正确扣热门候选。

结论：这个 v0 结构不保留。下一步若继续走该方向，应避免直接叠可学习 residual head；优先考虑离线挖掘固定 hard-candidate replay、按错题类型拆分的重排规则，或更贴近线上候选生成机制的训练候选协议。

### Recent-popular hard negatives v0

实验日期：2026-06-02。

实验状态：`reject`，不进入默认训练协议。

动机：时间残差流 v0 暴露出“近期热门候选压过长尾正例”的错题模式，因此尝试在训练负样本中混入近期热门目标节点。实现方式是在每个训练事件的时间窗口内按近期出现过的 `dst` 抽取部分负样本，其余负样本仍从全局 `dst_pool` 随机抽取；验证仍使用 `validation_candidates=test_like`。

协议：使用当前线上 best v1 的完整超参，只改训练负样本协议；`recent_negative_ratio=0.25`，`popular_negative_window=2592000`，`max_train_events=20000`，`max_val_events=5000`，`seed=60`，`refit_full=False`。

结果：

| 配置                           | `dataset1` MRR | `dataset2` MRR | 平均 MRR | 备注                               |
| ------------------------------ | -------------: | -------------: | -------: | ---------------------------------- |
| 同代码 random baseline，复测 A |       0.730777 |       0.448057 | 0.589417 | 完整两数据集                       |
| 同代码 random baseline，复测 B |       0.731186 |       0.445349 | 0.588268 | 完整两数据集                       |
| recent-popular v0，早期实现    |       0.728256 |             NA |       NA | `dataset1` 已掉分，耗时 `140.4s`   |
| recent-popular v0，轻量采样后  |       0.725278 |             NA |       NA | `dataset1` 继续掉分，耗时 `138.7s` |

补充诊断：`dataset2` 单独复测在超过 baseline 常规耗时数倍后仍未产出结果，进程被停止。GPU 利用率很低，主要耗时落在 Python 侧候选构造而不是模型训练。即使把 `np.random.choice` 大窗口抽样换成轻量索引抽样，`dataset1` MRR 仍从同轮 baseline 的 `0.731186` 降到 `0.725278`，并且耗时约 `2.7x`。

结论：这个训练候选协议不保留。简单混入“近期热门”负样本会让模型更关注热门项区分，但没有改善首位排序，反而把 MRR 拉低；同时 batch 构造明显变慢。若继续做 hard negatives，应换成离线挖掘的固定 hard-candidate replay，避免每个 batch 在 Python 里动态扫时间窗口。

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
