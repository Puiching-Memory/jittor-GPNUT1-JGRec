# Hai TDD: Dataset2 时间窗口保守融合

## Target Behavior

- 按 `champion + alpha × (window - champion)` 生成保守分数；
- alpha 必须在 `[0,1]` 且输入 shape/有限性一致；
- selection 只读取两个可见时间片，隐藏 forward 行可以是 NaN；
- 任一可见时间片下降或 prefix 增益不足的 alpha 不得锁定；
- independent gate 要求 forward 非退化、三片非退化和 full 增益达标。

## RED

- **Test added**:
  - `tests/test_hybrid_conservative_window_blend.py`，4 个纯 NumPy contract；
  - `tests/test_hybrid_checkpoint.py::test_hybrid_snapshot_round_trips_predictions`
    增加 conservative window checkpoint/prediction 约束。
- **Behavior asserted**:
  - residual 公式、端点与 alpha/shape 校验；
  - selection 不读取 forward 行，逐可见片非退化且 prefix 增益达标；
  - gate 要求第三片与 full 同时达标；
  - 附加窗口模型必须参与生产预测并经 snapshot/hydrate 保留。
- **Command**:
  - 本地：
    `uv run --no-sync pytest tests/test_hybrid_conservative_window_blend.py -q`
  - Linux：
    `uv run --no-sync pytest tests/test_hybrid_checkpoint.py::test_hybrid_snapshot_round_trips_predictions -q`
- **Observed failure**:
  - 首个 RED：
    `ModuleNotFoundError: No module named 'jgrec.rankers.hybrid.conservative_window_blend'`；
  - checkpoint RED：ranker 仍返回原 champion，和期望 conservative residual
    最大差约 `0.0005742`。
- **Failure is correct because**: 两次失败分别来自目标 contract 缺失和生产
  ranker 尚未应用附加窗口专家，不是 fixture、环境或随机误差。

## GREEN

- **Minimal implementation**:
  - 纯 NumPy conservative score、prefix selection 与 gate dataclass/API；
  - ranker 新增两个附加 Setwise expert 的 snapshot/hydrate/predict 路径；
  - 生产 builder 只在 gate 授权后注入 recent100k 与
    recent200k_decay100k，并保持主 recent200k/LGBM 不变。
- **Command**:
  - 本地：
    `uv run --no-sync pytest tests/test_hybrid_conservative_window_blend.py tests/test_hybrid_window_diversity.py tests/test_hybrid_temporal_robust_selection.py -q`
  - Linux：同上，加 checkpoint round-trip focused test。
- **Observed pass**: 本地最终相关回归 `17 passed, 4 skipped`；Linux 合并
  回归 `12 passed`（含 checkpoint round-trip）。

## REFACTOR

- **Refactor done**: yes
- **Change**:
  - alpha 0/1 返回精确 champion/window endpoint；
  - checkpoint hydrate 对 expert state/result/hidden-dim/config 做完整性校验；
  - selection 与 gate 拆成独立运行阶段并用 SHA sidecar 锁定；
  - 生产 smoke 改为在同一 causal validation cache 上比较 checkpoint 内嵌模型
    与 gate artifact，避免拿历史 causal cache 和 full-trained production
    encoder 错误比较。
- **Command after refactor**:
  - 本地与 Linux `ruff check` 覆盖 core、ranker、tests、selection/gate runner
    和 production builder；
  - production checkpoint reload、六行跨片预测、CSV validation、
    `unzip -t` 与 SHA-256 复核。
- **Observed result**: Ruff 全通过；缓存预测最大绝对误差
  `1.9908e-7` 且排序完全相同；zip 两个成员均 `OK`。

## Next Behavior

done。候选包已生成，等待是否提交线上。
