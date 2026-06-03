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

结论补充：数组化的收益不只来自 IO。训练启动阶段的排序、unique、切片和采样都从 Python 对象访问转为数组操作，因此有稳定收益。本次实验结束时，batch 内 `raw id -> compact id` 仍使用 Python dict/list comprehension，因此 `batch id extract` 收益有限；该问题已在下一节 `TemporalNodeMap` 批量 `searchsorted` 映射中解决，当前主链路不再把这一步作为 batch 构造热点。

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

### 剩余 batch 构造与工具路径收敛

实验日期：2026-06-03。

实验内容：继续推进剩余优化候选，并补做前后对照 benchmark，避免只凭代码形态判断收益。对照项包括：`TestCandidateIndex` ragged/offset 原型、`_sample_test_like_candidate_ids()` 批量 src 命中原型、`_sample_test_like_candidate_ids()` 真实 test 候选行快路径、`_batch_to_jittor()` 的 `jt.Var(array.astype(np.int32, copy=False))` 转换，以及 `submission.validate_submission_file()` 的 `np.loadtxt` 整体验证。

基准协议：

- 数据：全量 `data/dataset1/test.csv`、`data/dataset2/test.csv`，以及对应 train.csv 建出的 `TemporalNodeMap`。
- `TestCandidateIndex.from_queries()`：旧实现为 `dict[int, tuple[np.ndarray, ...]]`；原型为 `src_raw_ids + row_offsets + candidate_offsets + candidate_values` 扁平 ragged 结构；重复 3 次取均值。
- `_sample_test_like_candidate_ids()`：取 train tail `batch_size=256`，`num_negatives=99`；使用同一 seed 重建 RNG；重复 7 次取均值。
- test-like 候选行快路径：当选中的真实 test 候选行过滤掉 positive 和 padding 后，前 `num_negatives` 个候选已足量且唯一时，直接按原顺序填充；否则回到原 Python 去重和 fallback 逻辑。用 5000 条 validation events、20 个 batch，对照旧逻辑候选矩阵是否完全一致。
- `_batch_to_jittor()`：合成 `batch_size=256`、`candidate_count=100`、`history_len=64`、`candidate_history_len=32` 的 batch；CPU；warmup 后重复 50 次取均值。补充测试 `int32/int64/float32/float64/bool` dtype，确认转换语义。
- `validate_submission_file()`：合成 `20000 x 100` 概率 CSV；旧实现逐行 `float` 转换，新实现首行 CSV 检查 + `np.loadtxt` 整体验证；重复 3 次取均值。

结果：

| 阶段                                   | `dataset1` 旧版 | `dataset1` 原型/新版 | 加速比 | `dataset2` 旧版 | `dataset2` 原型/新版 | 加速比 | 决策          |
| -------------------------------------- | --------------: | -------------------: | -----: | --------------: | -------------------: | -----: | ------------- |
| `TestCandidateIndex.from_queries()`    |          0.897s |               0.919s |  0.98x |          2.630s |               2.724s |  0.97x | ragged 不保留 |
| `_sample_test_like_candidate_ids(256)` |         0.0065s |              0.0058s |  1.12x |         0.0057s |              0.0054s |  1.07x | ragged 不保留 |

test-like 快路径补充结果：

| 阶段                                                 | `dataset1` 旧版 | `dataset1` 新版 | 加速比 | `dataset2` 旧版 | `dataset2` 新版 | 加速比 | 输出校验 |
| ---------------------------------------------------- | --------------: | --------------: | -----: | --------------: | --------------: | -----: | -------- |
| `_sample_test_like_candidate_ids()`，5000 val events |          0.117s |          0.073s |  1.60x |          0.112s |          0.064s |  1.74x | 20/20 batch 候选矩阵一致 |

快路径覆盖：`dataset1` 5000 条 validation events 中有 4196 行具备结构性快路径条件，`dataset2` 为 4626 行。进一步拆完整 `build_evaluation_batch(test_like)` + `_batch_to_jittor()` 后，`dataset1` 从约 `0.566s` 降到 `0.527s`（1.08x），`dataset2` 从约 `0.415s` 降到 `0.363s`（1.14x）。快路径保留，但它已经不是新的主瓶颈；快路径后 candidate neighbor gather 仍占 batch 构造加转换总耗时的约 68-76%。

补充结果：

| 阶段                                   | 旧版均值 | 新版均值 | 加速比 | 决策 |
| -------------------------------------- | -------: | -------: | -----: | ---- |
| `_batch_to_jittor()` 合成 batch 转换   | 0.00119s | 0.00063s |  1.90x | 保留 |
| `validate_submission_file(20000 rows)` | 0.36184s | 0.14141s |  2.56x | 保留 |

dtype 补充：不能把所有输入直接替换为 `jt.Var(array)`，因为 `float32` 会保持 `float32`、`float64` 会变 `float32`、`bool` 会保持 `bool`，不满足模型输入统一 `int32` 的旧语义。最终实现使用 `jt.Var(array.astype(np.int32, copy=False))`：`int32` 输入零拷贝快路径，非 `int32` 输入显式 cast 后再建 Var。非 `int32` 场景与旧 `jt.array(..., dtype=jt.int32)` 基本持平，`int32` 场景保持约 2x 局部转换收益。

内存补充：ragged `TestCandidateIndex` 的纯数组字节数没有下降，`dataset1` 为 `46.6MiB -> 47.2MiB`，`dataset2` 为 `117.1MiB -> 118.3MiB`，且未计入旧实现 Python dict/tuple 对象头。考虑默认 `max_val_events=5000`、`train_batch_size=256` 时 test-like 验证约 20 个 batch，采样局部节省仅毫秒级，无法抵消索引构建回退和复杂度。

结论：保留有明确局部收益且风险较低的 `_batch_to_jittor()`、test-like 候选行快路径和提交校验改造；ragged/offset 候选索引原型不进入源码。后续若要继续优化 batch 构造，需要重新做端到端 profiling，重点确认 candidate neighbor gather 是否仍是主耗时。

### 提前 Jittor 原生化数据容器评估

实验日期：2026-06-03。

实验动机：评估是否应在项目更早阶段把 `np.ndarray` 数据容器替换成 `jt.Var`，并在 Jittor 上做排序、切片、列提取、unique、ID 映射前处理等下游变换。该方向的潜在收益是减少模型边界的 numpy -> Jittor 转换；风险是当前 CSV IO、`TemporalNodeMap`、CSR neighbor sampler 和 JittorGeometric sampler 仍大量依赖 numpy，提前转 Var 可能引入额外 `.numpy()` 桥接和同步。

基准协议：

- 数据：全量 `dataset1/dataset2` train/test。
- 设备：CPU 路径；Jittor op 后显式 `jt.sync_all()`。
- 对比项：全量 `np.ndarray -> jt.Var` 与 `jt.Var -> np.ndarray`、Var 切片/列提取、Var unique/argsort、当前 numpy ID 映射 vs Var 切片后转回 numpy 再映射、当前 `build_training_batch()` vs “tail 事件先转 Var，每 batch `.numpy()` 后构造”。
- 稳定性：Jittor row gather `v[jt_order]` 单独隔离进程测试，避免 native crash 影响其他 benchmark。

基础转换和桥接结果：

| 阶段                                | `dataset1` | `dataset2` | 结论                                          |
| ----------------------------------- | ---------: | ---------: | --------------------------------------------- |
| 全量 train `np -> jt.Var`           |    0.0029s |    0.0061s | 转换本身不贵                                  |
| 全量 train `jt.Var -> np`           |    0.0024s |    0.0056s | 转回也不贵，但频繁桥接会累积                  |
| 全量 test candidates `np -> jt.Var` |    0.0084s |    0.0194s | 大矩阵转换仍可接受                            |
| 全量 test candidates `jt.Var -> np` |    0.0051s |    0.0156s | 仍是额外成本                                  |
| Var tail slice 20k                  |    0.0025s |    0.0028s | 明显慢于 numpy 视图切片                       |
| Var column extraction               |    0.0028s |    0.0044s | 明显慢于 numpy 列视图                         |
| Var column extraction 后 `.numpy()` |    0.0004s |    0.0014s | 若下游要 numpy，还要再付桥接成本              |
| query batch Var slice 2048          |    0.0026s |    0.0032s | 当前 `TestQueryArray.rows()` 是微秒级视图切片 |
| query batch Var slice 后 `.numpy()` |   0.00012s |   0.00013s | 对当前预测 batch 构造是纯额外成本             |

排序、unique 和 batch 构造结果：

| 阶段                                                 | `dataset1` numpy | `dataset1` Jittor/桥接 | `dataset2` numpy | `dataset2` Jittor/桥接 | 结论                                       |
| ---------------------------------------------------- | ---------------: | ---------------------: | ---------------: | ---------------------: | ------------------------------------------ |
| `argsort(time)` order                                |          0.0026s |                0.0178s |          0.0090s |                0.0393s | Jittor 慢 4-7x，且排序稳定性语义需额外确认 |
| `unique(src/dst)`                                    |          0.0248s |                0.1996s |          0.1079s |                0.5962s | Jittor 慢 5-8x                             |
| batch raw id 映射                                    |         0.00018s |               0.00247s |         0.00020s |               0.00283s | Var 切片后转回 numpy 再映射慢约 13-14x     |
| `build_training_batch(256)` 当前 numpy               |          0.0355s |                      - |          0.0393s |                      - | 当前 CSR sampler 后的完整构造              |
| `build_training_batch(256)` tail Var -> numpy bridge |                - |                0.0350s |                - |                0.0392s | 无可兑现收益，只是把桥接藏进 batch 前      |

稳定性补充：`v[jt_order]` 形式的二维 Var row gather 在本地 Jittor 1.3.11.0 / CUDA 初始化环境下触发 native `getitem` segfault（buffer overflow）。即使不考虑速度，这也使“在 Jittor 上完成全量排序后 row gather”不适合作为默认数据准备路径。

结论：不保留“项目尽早替换为 Jittor 原生类型”的方向。当前架构应继续以 numpy 作为 IO、ID 映射、候选构造和 neighbor sampler 前处理的数据容器，只在模型执行边界用 `_batch_to_jittor()` 转为 `jt.Var`。后续若要进一步 Jittor 原生化，必须先满足两个条件：一是 JittorGeometric neighbor sampler 和本项目 CSR sampler 的输入输出也能稳定停留在 Var 上；二是排序、unique、gather 等基础变换在真实数据上有明确性能和稳定性优势。

### 数据画像脚本局部优化

实验日期：2026-06-03。

实验动机：`scripts/analyze_data_profile.py` 是低频研究脚本，但全量数据画像仍需几十秒。评估是否可以在不改变统计口径的前提下，收敛部分明显的 Python 循环开销。

实现内容：

- `read_train()` / `read_test()` 改为 `np.loadtxt(..., dtype=np.int64, ndmin=2)` 批量读取，再构造现有 `Event` / `Query` 对象；`read_train()` 保持 stable time sort。
- `time_drift()` 和 `sequence_behavior()` 使用 `recent_hit_rank()` / `SourceState.recent_ranks`，避免每条事件反复 `list -> set`。
- `test_candidate_distribution()` 去掉内层 `Counter` 字符串 key 累加，改用局部整数计数器和缓存局部引用。
- `unseen_dst_analysis()` 使用 `extend` 和缓存 `dst_set`，减少候选循环中的重复属性访问。

基准协议：

- 数据：全量 `data/dataset1` / `data/dataset2`。
- 对照：用 `git show HEAD:scripts/analyze_data_profile.py` 加载旧版函数，与当前新版同进程逐阶段计时。
- 校验：对 `read_train` / `read_test` 做输入 digest；对 `basic_stats`、`time_drift`、`test_candidate_distribution`、`unseen_dst_analysis`、`sequence_behavior` 做 JSON digest，确保输出一致。
- 补充测试：新增小样本 CSV 和手工状态单元测试，覆盖 header 列顺序、单行 test、行宽错误、recent hit 口径、候选统计和 unseen 分析。

结果：

| 阶段                            | `dataset1` 旧版 | `dataset1` 新版 | 加速比 | `dataset2` 旧版 | `dataset2` 新版 | 加速比 |
| ------------------------------- | --------------: | --------------: | -----: | --------------: | --------------: | -----: |
| `read_train()`                  |          1.484s |          1.210s |  1.23x |          6.305s |          4.769s |  1.32x |
| `read_test()`                   |          1.138s |          1.093s |  1.04x |          2.804s |          2.295s |  1.22x |
| `time_drift()`                  |          3.475s |          2.178s |  1.60x |         11.640s |          7.470s |  1.56x |
| `test_candidate_distribution()` |          6.307s |          3.977s |  1.59x |         11.824s |          8.036s |  1.47x |
| `unseen_dst_analysis()`         |          0.782s |          0.607s |  1.29x |          3.113s |          3.013s |  1.03x |
| `sequence_behavior()`           |          1.085s |          0.741s |  1.46x |          2.946s |          1.878s |  1.57x |

输出校验：两个数据集的 `events`、`queries`、`basic_stats`、`time_drift`、`test_candidate_distribution`、`unseen_dst_analysis`、`sequence_behavior` digest 均一致。

结论：保留本轮局部优化。它们有真实数据前后对照和输出一致性校验，且不改变脚本的数据结构边界。更激进的“把整个画像脚本改成数组状态机”暂不推进；该脚本仍以统计可读性为优先，除非未来画像脚本再次成为日常迭代瓶颈。

### 剩余优化候选清单

标记日期：2026-06-02。
更新日期：2026-06-03。

原则：本清单只保留尚未完成、仍可能继续评估的优化项；已落地项和已评估不保留项只保留在上方实验记录中。每一项进入实现前仍需先做局部 benchmark，确认收益超过复杂度和回归风险。

当前没有保留中的未完成优化项。上一版清单中的 P0/P1/P2 项已分别落地、拒绝或归档到上方实验记录中。

当前建议顺序：

1. 若继续优化训练 batch 构造，先基于当前 CSR recent sampler 重新拆解完整 `build_training_batch()`，不要沿用 CSR 改造前的热点排序。
2. 优先确认剩余 candidate 邻居 gather、Jittor 转换、以及负采样在当前实现中的实际占比。负采样向量化曾评估不保留，除非新 profiling 显示占比显著上升，否则不重新列入候选清单。
3. `analyze_data_profile.py` 已完成低风险局部优化；除非运行时间再次影响日常迭代，否则不继续做大规模结构重写。

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
