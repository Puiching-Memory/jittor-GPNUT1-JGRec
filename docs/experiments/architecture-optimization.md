# 架构优化

本文档记录训练、推理、数据读取、批构造和邻居采样相关的工程架构优化。模型结构、调参、线上提交和模型消融记录见 [模型优化](model-optimization.md)。

## 基准原则

- 使用固定数据、固定查询数、固定候选数做前后对比。
- 记录数据集、split、batch size、候选数、训练事件数、验证事件数和关键超参。
- 性能改动同时记录耗时、内存、正确性校验和是否进入默认链路。
- 涉及提交文件的改动，需要校验 CSV 行列数、概率范围、每行和、zip 内容。

## 加速实验记录

### CSV 读取与交互数组改造

实验日期：2026-06-02。

实验内容：训练交互数据从 `csv.DictReader -> Interaction dataclass list` 全面切换为 `np.loadtxt -> np.ndarray[int32]`。`read_interactions()` 现在直接返回 shape 为 `(n, 3)` 的数组，列顺序固定为 `src, dst, time`；训练、调参、CRAFT baseline 和 temporal-graph 默认链路不再保留旧对象兼容层。

基准协议：

- 数据：全量 `data/dataset1/train.csv` 与 `data/dataset2/train.csv`。
- 旧路径：`csv.DictReader` 逐行构造 frozen dataclass 对象列表。
- 新路径：`np.loadtxt(..., usecols=(src,dst,time), dtype=np.int32, ndmin=2)`。
- 计时口径：每个数据集重复 3 次，记录均值；内存口径为 `tracemalloc` Python 侧峰值。

结果：

| 数据集     |      行数 | 旧版耗时 | 新版耗时 | 加速比 | 旧版峰值 | 新版峰值 |
| ---------- | --------: | -------: | -------: | -----: | -------: | -------: |
| `dataset1` |   690,848 |   7.092s |   0.089s | 79.78x | 127.7MiB |  13.0MiB |
| `dataset2` | 2,261,283 |  23.669s |   0.326s | 72.52x | 419.0MiB |  31.3MiB |

结论：CSV 读取和训练入口对象构造是明确瓶颈。数组化后，数据读入耗时降到亚秒级，并显著降低 Python 对象内存压力；该改造进入默认链路，后续训练批构造统一基于列数组切片。

下游训练准备阶段也有收益。以下 benchmark 固定使用已经读入内存的数据，对比旧对象列表和当前数组路径，不包含 CSV 读取时间：

| 阶段                | `dataset1` 旧版 | `dataset1` 新版 | 加速比 | `dataset2` 旧版 | `dataset2` 新版 | 加速比 |
| ------------------- | --------------: | --------------: | -----: | --------------: | --------------: | -----: |
| 按 `time` 排序      |         0.0349s |         0.0105s |  3.32x |         0.1335s |         0.0379s |  3.52x |
| 提取 unique src/dst |         0.0582s |         0.0248s |  2.35x |         0.1801s |         0.1079s |  1.67x |
| tail slice          |         0.0030s |       0.000008s |   377x |         0.0036s |       0.000013s |   273x |
| sample events       |         0.0026s |         0.0007s |  3.94x |         0.0036s |         0.0008s |  4.64x |
| batch slicing       |         0.0001s |       0.000021s |  5.38x |         0.0002s |       0.000032s |  5.05x |
| batch id extract    |         0.0104s |         0.0093s |  1.12x |         0.0124s |         0.0096s |  1.28x |

结论补充：数组化的收益不只来自 IO。训练启动阶段的排序、unique、切片和采样都从 Python 对象访问转为数组操作，因此有稳定收益；当前 batch 内 `raw id -> compact id` 仍使用 Python dict/list comprehension，收益有限，是后续若继续优化 batch 构造时更值得盯的点。

### TestQueryArray 与批量映射改造

实验日期：2026-06-02。

实验内容：测试集查询从逐行 `TestQuery` 对象改成 `TestQueryArray`，并同步改 `Ranker.predict_batch` 协议、runner 批切片、CRAFT/temporal-graph 预测入口。随后补上 `TemporalNodeMap` 批量 `searchsorted` 映射，避免数组化查询在下游 batch 构造中退回 Python 逐元素 dict 映射。

基准协议：

- 数据：全量 `data/dataset1/test.csv`、`data/dataset2/test.csv`，以及对应 train.csv 建出的 `TemporalNodeMap`。
- 旧路径：`csv.reader` 逐行构造 `LegacyQuery`，batch 内逐行映射候选。
- 新路径：`np.loadtxt -> TestQueryArray`，runner 切片传入 ranker，`TemporalNodeMap.src_ids()/dst_ids()` 批量映射。
- 计时口径：重复 3 次取均值；`dataset2` 全量候选索引因耗时较长重复 2 次；内存为 `tracemalloc` Python 侧峰值。

测试查询读取和 runner 写出前批处理：

| 阶段                     | `dataset1` 旧版 | `dataset1` 新版 | 加速比 | `dataset2` 旧版 | `dataset2` 新版 | 加速比 |
| ------------------------ | --------------: | --------------: | -----: | --------------: | --------------: | -----: |
| 全量 `read_test_queries` |          9.178s |          0.273s | 33.68x |         22.251s |          0.652s | 34.15x |
| runner 全量 batch 累积   |          8.791s |          0.275s | 31.99x |         21.681s |          0.669s | 32.42x |
| `limit_rows=2` 读取/切片 |        0.00108s |        0.00136s |  0.80x |        0.00109s |        0.00191s |  0.57x |

内存补充：全量读取峰值从 `223.6MiB -> 30.8MiB`（dataset1）和 `559.7MiB -> 69.4MiB`（dataset2）。runner 旧路径流式累积时峰值更低，但耗时超过 20s；新路径选择一次性数组读取，以推理吞吐优先。`limit_rows=2` 是亚毫秒级小回退，保留全量主路径收益。

下游映射和训练启动路径：

| 阶段                                | `dataset1` 旧版 | `dataset1` 新版 | 加速比 | `dataset2` 旧版 | `dataset2` 新版 | 加速比 |
| ----------------------------------- | --------------: | --------------: | -----: | --------------: | --------------: | -----: |
| 全量 train event id 映射            |          1.028s |          0.153s |  6.71x |          3.456s |          0.506s |  6.84x |
| `queries_to_prediction_batch(2048)` |          0.170s |          0.026s |  6.50x |          0.176s |          0.032s |  5.46x |
| `TestCandidateIndex.from_queries`   |          5.344s |          1.504s |  3.55x |         14.227s |          4.160s |  3.42x |
| `dst_pool` 构建                     |          0.534s |          0.091s |  5.88x |          1.778s |          0.344s |  5.16x |

候选索引曾测试过一次性映射整张候选矩阵：`dataset1=1.333s`、`dataset2=3.798s`，但峰值内存升到 `291.1MiB` 和 `731.6MiB`。最终保留 4096 行 chunk 映射，速度仍有 3.4x 以上收益，峰值内存基本回到旧路径水平（`57.4MiB -> 59.2MiB`，`138.6MiB -> 139.4MiB`）。

结论：该组改动保留。收益不只来自 CSV IO，也来自预测 batch、test-like 候选索引、TemporalData 构建和训练启动候选池构建；其中 `TestCandidateIndex` 采用 chunk 版而不是最快的一次性矩阵版，是为了避免高峰值内存。

### 训练 batch 构造热点与负采样原型

实验日期：2026-06-02。

实验内容：在真实 JittorGeometric neighbor sampler 下拆解 `build_training_batch()`，确认训练吞吐瓶颈后再决定是否改负采样。

基准协议：

- 数据：两个数据集按时间排序后的尾部窗口。
- batch：`batch_size=256`，`num_negatives=99`，`history_len=64`，`candidate_history_len=32`。
- 计时对象：仅 batch 构造，不包含模型 forward/backward。

热点拆解：

| 阶段               | `dataset1` 均值 | `dataset2` 均值 | 结论                       |
| ------------------ | --------------: | --------------: | -------------------------- |
| id 映射            |        0.00023s |        0.00024s | 批量映射后已不是瓶颈       |
| src 邻居采样       |        0.00145s |        0.00148s | 很小                       |
| 负采样             |        0.01125s |        0.01135s | 可优化但占比约 8%          |
| candidate 邻居采样 |        0.12422s |        0.12363s | 主瓶颈，约占 90%           |
| 完整 batch 构造    |        0.13496s |        0.13794s | 后续应优先研究候选邻居采样 |

负采样向量化原型：

| 数据集     | 当前负采样 | 原型负采样 | 当前完整构造 | 原型完整构造 | 决策   |
| ---------- | ---------: | ---------: | -----------: | -----------: | ------ |
| `dataset1` |   0.01057s |   0.00798s |     0.12873s |     0.12864s | 不保留 |
| `dataset2` |   0.01128s |   0.00832s |     0.13121s |     0.13163s | 不保留 |

结论：负采样局部快约 25%，但完整 batch 构造没有可兑现收益，`dataset2` 还轻微回退。该原型不进入源码；后续训练吞吐应先针对 `candidate_neighbors` 的大矩阵邻居采样做 profiling。

### Recent 邻居采样批量 CSR 改造

实验日期：2026-06-02。

实验内容：默认 temporal graph 训练使用 JittorGeometric 的 `get_neighbor_sampler(..., "recent")`。旧实现对每个 `(node_id, time)` 逐行 `np.searchsorted` 并切片；候选侧每个 batch 要查 `batch_size * (1 + num_negatives)` 个候选节点，成为 batch 构造主瓶颈。本实验在 `SafeTemporalNeighborSampler` 内为 `"recent"` 策略构建 CSR 扁平索引，用 numpy 批量二分查找每行历史截止位置，再一次性 gather 最近 `candidate_history_len` 个邻居。非 `"recent"` 策略仍委托原 sampler。

基准协议：

- 数据：两个数据集按时间排序后的尾部窗口。
- batch：`batch_size=256`，`num_negatives=99`，`history_len=64`，`candidate_history_len=32`。
- 对照：脚本内复现旧版 `SafeTemporalNeighborSampler` 行循环路径；新路径为当前源码 CSR recent sampler。
- 正确性：对 `neighbor_ids`、`edge_ids`、`neighbor_times` 做逐元素一致性校验，并补充大时间戳边界单测，避免 `float32` 精度导致等于当前时间的边被误采。

原型记录：

| 原型                       | `dataset1` 结果           | `dataset2` 结果           | 决策   |
| -------------------------- | ------------------------- | ------------------------- | ------ |
| unique node 分组批量查询   | `0.124s -> 0.405s`，0.31x | `0.133s -> 0.508s`，0.26x | 不保留 |
| CSR + 向量化 binary search | `0.121s -> 0.021s`，5.69x | `0.130s -> 0.024s`，5.34x | 保留   |

落地后复测：

| 阶段                          | `dataset1` 旧版 | `dataset1` 新版 | 加速比 | `dataset2` 旧版 | `dataset2` 新版 | 加速比 |
| ----------------------------- | --------------: | --------------: | -----: | --------------: | --------------: | -----: |
| candidate 邻居采样            |          0.129s |          0.021s |  6.07x |          0.134s |          0.024s |  5.63x |
| 完整 `build_training_batch()` |          0.142s |          0.037s |  3.89x |          0.149s |          0.039s |  3.86x |

补充：加入 `node_interact_times=None` 兼容分支后，用最终代码状态再次复测完整 `build_training_batch()`，`dataset1` 为 `0.150s -> 0.037s`（4.10x），`dataset2` 为 `0.151s -> 0.038s`（4.00x），checksum 一致。

结论：保留。该改造直接命中上一节识别出的主瓶颈，且完整 batch 构造有接近 4x 的真实收益。实现中特意保留原始 timestamp dtype 做二分；不能把时间戳先转成 `float32`，否则在 `1e8` 量级会丢精度并改变“严格早于当前时间”的边界语义。

### 数组化扩展候选清单

标记日期：2026-06-02。

原则：只标记，不直接扩大改动面。每一项进入实现前都要先做局部 benchmark，确认收益超过复杂度和回归风险。

| 优先级 | 位置                                                                     | 当前形态                                                                  | 数组化方向                                                                                                           | 预期收益                                                                 | 风险                                                                                      |
| ------ | ------------------------------------------------------------------------ | ------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------- |
| P0     | `core.io.read_test_queries()` / `core.runner.build_dataset_submission()` | 测试集逐行生成 `TestQuery`，runner 维护 `list[TestQuery]` batch           | 新增 `TestQueryArray`，一次读成 `src: int32[n]`、`time: int32[n]`、`candidates: int32[n,100]`，runner 按数组切片推理 | 推理阶段减少 CSV 解析、对象构造、batch append 和候选 tuple 访问          | 需要同步改 `Ranker.predict_batch` 协议；提交写出和 `limit_rows` 语义要保持一致            |
| P0     | `TemporalNodeMap.src_id()` / `dst_id()` / `dst_ids()` 与 batch 构造      | 每个 batch 用 Python dict/list comprehension 做 raw id 到 compact id 映射 | 构建 sorted raw id 数组和 compact id 数组，用 `np.searchsorted` 或 dense lookup table 批量映射                       | 直接针对当前 benchmark 中收益最小的 `batch id extract`，训练和推理都受益 | 原始 ID 稀疏程度未知；dense lookup 可能浪费内存，`searchsorted` 要处理 missing id/padding |
| P0     | `temporal_data_from_interactions()`                                      | 全量 train 事件用 list comprehension 映射 src/dst                         | 复用批量映射函数，一次性映射整列 `src/dst`                                                                           | 训练启动阶段再降 Python 循环开销                                         | 需要保证 test-only 节点仍能映射，missing id 仍为 padding                                  |
| P0     | `queries_to_prediction_batch()`                                          | 推理 batch 从 `list[TestQuery]` 抽 `src/time/candidates`                  | 接收 `TestQueryArray` 切片后直接批量映射候选矩阵                                                                     | 完整推理吞吐收益，尤其全量 test.csv                                      | 需要修改 `craft` 和所有 ranker 的预测协议                                                 |
| P1     | `_sample_candidate_ids()`                                                | 每行维护 Python `set`，循环抽负样本                                       | 先批量 `rng.choice(dst_pool, size=(batch, k))`，再用 numpy mask 去除 positive/forbidden，不足再局部补齐              | 训练 batch 构造可能受益，候选数 99 时更明显                              | 去重和 forbidden 逻辑复杂；可能改变负采样分布，需要固定 seed 对照                         |
| P1     | `_sample_test_like_candidate_ids()`                                      | test-like 验证负样本按行查 dict、逐候选去重                               | 将 `by_src` 存成候选矩阵或 ragged offsets；常见 src 用矩阵抽行，fallback 用批量全局池                                | 验证阶段更快，Optuna 多 trial 受益                                       | ragged 数据结构复杂；候选去重仍需保底路径                                                 |
| P1     | `TestCandidateIndex.from_queries()`                                      | 已用 numpy chunk，但输入仍是 `TestQuery` 迭代器                           | 从 `TestQueryArray` 直接构建 `global_candidates` 和 src 分组 offsets                                                 | 进一步降低 test-like 构建时间和内存                                      | 依赖 P0 测试集数组协议                                                                    |
| P1     | `TemporalTrainingBatch` / `_batch_to_jittor()`                           | 每个 batch 用 dataclass 包多个 numpy 数组，再逐项 `jt.array`              | 可评估返回 tuple 或预分配/复用 numpy buffer，减少临时对象                                                            | 小幅降低 batch 构造开销                                                  | 可读性下降；Jittor `jt.array` 转换可能才是主成本，要先测                                  |
| P1     | `CRAFTBaselineRanker.predict_batch()`                                    | 仍从 `list[TestQuery]` 抽数组                                             | 接入 `TestQueryArray`，候选矩阵直接传入                                                                              | craft 推理路径与默认路径一致受益                                         | CRAFT 依赖原始 ID 和 `dst_min_idx`，要单独 smoke                                          |
| P2     | `rankers/temporal_graph/index.scan_test_nodes_csv()`                     | CSV 逐行扫 test 节点集合                                                  | 若仍需要该函数，改用 `TestQueryArray` 或 `np.loadtxt`                                                                | 低频工具函数收益                                                         | 当前默认路径几乎不用，优先级低                                                            |
| P2     | `scripts/analyze_data_profile.py`                                        | 独立 `Event` / `Query` dataclass 和大量 Python Counter/set                | 把基础统计、候选矩阵、时间切分改为 numpy 数组，保留必要 Counter                                                      | 数据画像脚本运行更快，便于反复分析                                       | 脚本逻辑多、研究输出多，容易改坏统计口径                                                  |
| P2     | `submission.validate_submission_file()`                                  | CSV 逐行校验概率                                                          | 可用 `np.loadtxt` 读取并整体校验 shape/range                                                                         | 提交校验更快                                                             | 输出文件可能很大，整体读取增加瞬时内存                                                    |

建议顺序：

1. 先实现并 benchmark `TestQueryArray`，因为它同时影响推理、test-like 验证和候选索引构建。
2. 再实现 `TemporalNodeMap` 批量映射，重点复测 `batch id extract`、`temporal_data_from_interactions()` 和端到端 smoke。
3. 最后再碰负采样向量化，因为收益可能大，但分布和边界更容易被改坏。

### 训练 loop 同步改动

实验日期：2026-06-02。

实验内容：在 `--quiet`/`quiet-ranker` 场景下，尝试跳过每个训练 batch 后的 `loss.item()` 和 `jt.sync_all()`，只在 epoch 末尾显式同步，目标是减少 GPU/CPU 同步开销。

基准协议：

- 设备：CUDA。
- 数据：`dataset1`。
- 候选数：`1 positive + 99 negatives`。
- 计时方式：先 warmup 编译和对应 shape；正式计时不包含 Jittor operator 编译；每轮末尾显式 `jt.sync_all()`，避免异步执行导致虚假加速。

结果：

| 场景                        | 旧版均值 | 新版均值 |  收益 |
| --------------------------- | -------: | -------: | ----: |
| 小模型，预构 batch          |  0.1898s |  0.1997s | -5.2% |
| 小模型，含 batch 构造       |  1.5573s |  1.5945s | -2.4% |
| 接近默认形状，预构 batch    |  0.1931s |  0.1757s | +9.0% |
| 接近默认形状，含 batch 构造 |  0.9069s |  0.8618s | +5.0% |

结论：收益不稳定，完整训练口径只有低个位数收益，并且小模型下出现回退。该改动不作为默认链路保留，训练 loop 恢复为每个 batch 后记录 loss 并同步。后续同步/日志类优化必须先用同样协议做前后对照，再决定是否进入默认路径。

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
