# Hai TDD: Dataset1 Rolling-Origin 多时间折

## Target Behavior

- 生成 4 个 sliding-origin folds，每折只用 origin 前 100k 行训练并评分
  随后 25k 行。
- 四个 score ranges 不重叠，覆盖 cache 最后 100k 行。
- 前三折选择时 API 不接收、也不计算第四折 metric。
- 配置必须在三个 selection folds 均非退化且 mean delta 达标。
- 独立 gate 要求第四折非退化且四折 mean delta 达标。

## RED

- **Test added**: `tests/test_hybrid_rolling_origin.py`，共 5 个测试。
- **Behavior asserted**:
  - 精确的 4 折 sliding-origin layout 与最后 100k 行单次覆盖；
  - score range 不重叠、训练历史充足且不读取未来行；
  - query time 单调并允许 origin 两侧 timestamp 相等；
  - selection 任一折回退即拒绝，且没有 forward metric 输入；
  - gate 同时要求 forward delta 非负和四折 mean delta 达标。
- **Command**:
  `uv run --no-sync pytest tests/test_hybrid_rolling_origin.py -q`
- **Observed failure**:
  `ModuleNotFoundError: No module named 'jgrec.rankers.hybrid.rolling_origin'`
- **Failure is correct because**: contract 测试先于目标模块建立，失败来自缺失实现，
  不是 fixture、环境或断言错误。

## GREEN

- **Minimal implementation**:
  `src/jgrec/rankers/hybrid/rolling_origin.py` 中的 fold、time boundary、
  selection 与 gate 纯 NumPy contract。
- **Command**:
  `uv run --no-sync pytest tests/test_hybrid_rolling_origin.py -q`
- **Observed pass**: `5 passed`。

## REFACTOR

- **Refactor done**: 是。
- **Change**:
  - 用 frozen dataclass 显式区分 fold、trial、selection、gate result；
  - 抽出 `_finite_vector`，统一长度和非有限值校验；
  - selection API 在类型和参数层面不接受 gate fold metric。
- **Command after refactor**:
  - 本地：
    `uv run --no-sync pytest tests/test_hybrid_rolling_origin.py tests/test_hybrid_time_ramp.py -q`
  - Linux：
    `uv run --no-sync pytest tests/test_hybrid_rolling_origin.py tests/test_hybrid_time_ramp.py tests/test_hybrid_fusion_listwise.py -q`
  - Ruff：对 core、tests、manifest/selection/gate 三个 runner 执行
    `uv run --no-sync ruff check`。
- **Observed result**: 本地 `12 passed`；Linux `23 passed`；Ruff
  `All checks passed!`。

## Next Behavior

Level-1 selection 已执行。所有 gamma 都至少在一个 selection 折回退，因此
没有 locked candidate，独立 fold3 gate 按 contract 保持未读取、未运行。
