# Hai TDD: Dataset2 Setwise 时间窗口多样性

## Target Behavior

在不复制 200k full-100 缓存的情况下取最近 50k / 100k / 200k 连续尾部窗口；为 Setwise query loss 提供随 shuffle 同步的可选行权重；生成固定半衰期、均值为 1 的时间衰减权重；只用前两时间片从四个专家的 15 个非空均匀子集中确定唯一方案。

## RED

- **Test added**：
  - `tests/test_hybrid_window_diversity.py`：窗口尾部视图、衰减权重、15 子集前缀选择、末片隔离、平分规则、固定外层融合；
  - `tests/test_hybrid_fusion_listwise.py`：加权 listwise loss 及训练 shuffle 后的 query-weight 对齐。
- **Behavior asserted**：
  - 50k / 100k 是 200k 缓存的共享内存尾部视图；
  - 衰减权重正、单调递增、均值为 1，并满足冻结半衰期；
  - 选择器允许 forward 行为 NaN，证明不读取第三片；
  - query loss 是按权重归一化的逐 query 交叉熵；
  - batch 权重严格按同一个 `batch_idx` 重排。
- **Local command**：
  `uv run --no-sync pytest tests/test_hybrid_window_diversity.py -q`
- **Observed local failure**：
  收集阶段报 `ModuleNotFoundError: jgrec.rankers.hybrid.window_diversity`。
- **Remote command**：
  `.deps/uv/bin/uv run pytest tests/_codex_window_red_fusion_listwise.py -q -k query_weights`
- **Observed remote failure**：
  两个加权 loss 测试均报
  `TypeError: _listwise_positive_loss() takes 1 positional argument but 2 were given`。
- **Failure is correct because**：
  窗口多样性模块与 Setwise query 权重入口均尚不存在。本机 Jittor 的 Windows C++ 编译失败属于环境故障，不计作 RED 证据。

## GREEN

- **Minimal implementation**：
  - 新增 `window_diversity.py`，提供只读尾部窗口、固定指数时间衰减、前缀子集选择和锁定子集融合；
  - `_listwise_positive_loss` 增加可选 query 权重；
  - fixed / streaming listwise trainer 增加可选 `train_row_weights`，并用训练 shuffle 的同一 `batch_idx` 取权重；
  - `train_row_weights=None` 保留原均值 loss 路径。
- **Local command**：
  `uv run --no-sync pytest tests/test_hybrid_window_diversity.py tests/test_hybrid_fusion_analysis.py -q`
- **Observed local pass**：`19 passed`。
- **Linux/Jittor command**：
  `.deps/uv/bin/uv run pytest tests/test_hybrid_fusion_listwise.py tests/test_hybrid_window_diversity.py -q`
- **Observed Linux/Jittor pass**：`16 passed`。

## REFACTOR

- **Refactor done**：yes
- **Change**：
  将训练/选择与第三片门禁设计为同一脚本的两个独立子命令；`train-select` 只持久化前缀 MRR 并为报告写 SHA-256 锁，`gate` 必须先验证锁和全部输入哈希才计算 full / slice 指标。
- **Command after refactor**：
  `uv run --no-sync ruff check scripts/run_dataset2_setwise_window_diversity.py src/jgrec/rankers/hybrid/window_diversity.py tests/test_hybrid_window_diversity.py tests/test_hybrid_fusion_listwise.py src/jgrec/rankers/hybrid/fusion.py`
- **Observed result**：`All checks passed!`

## Next Behavior

已完成运行。前缀锁定 `recent100k + recent200k + recent200k_decay100k`；full MRR 相对冠军 `+0.0013961154`，slice0 / slice2 分别 `+0.0020468235` / `+0.0024259775`，但 slice1 为 `-0.0002843004`。冻结门禁因此拒绝候选，没有生成包。
