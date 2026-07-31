# Hai TDD: Dataset1 Time-Ramped Setwise Blend

## Target Behavior

- 时间进度和 ramp 权重位于 `[0,1]`、随 query time 单调不下降。
- 最早/最晚端点分别 exact champion/expert；constant-time exact champion。
- candidate 轴置换只置换输出 candidate 轴。
- prefix selector 要求 slice0/slice1 分别非退化，且不读取 forward metrics。
- 无合格 ramp 时 exact champion fallback。
- final gate 要求 full 最小增益和三个 slice 全部非退化。

## RED

- **Test added**: `tests/test_hybrid_time_ramp.py`
- **Behavior asserted**: 上述 ramp、selector、fallback 和 gate 契约。
- **Command**:
  `uv run --no-sync pytest tests/test_hybrid_time_ramp.py -q`
- **Observed failure**:
  `ModuleNotFoundError: No module named 'jgrec.rankers.hybrid.time_ramp'`。
- **Failure is correct because**: 目标模块/API 尚不存在；测试在导入目标行为时失败，
  不是数值断言、fixture 或环境错误。
- **Deployment RED added**: 固定 test 全局 `minimum_time/maximum_time` 后，
  任意 source 重排和 batch 切分必须产生与一次性推理相同的权重。
- **Deployment RED command**:
  `uv run --no-sync pytest tests/test_hybrid_time_ramp.py -q`
- **Deployment RED observed**:
  `TypeError: time_ramp_weights() got an unexpected keyword argument 'minimum_time'`。
- **Deployment failure is correct because**: 初版 API 只能按当前 batch
  归一化，无法安全部署到会重排/分批的 submission runner。

## GREEN

- **Minimal implementation**:
  - 新增纯 NumPy `time_ramp.py`，实现进度、权重、融合、prefix selection
    和 independent gate；
  - 权重 API 支持成对提供的全局时间边界；
  - `TemporalHybridRanker` 支持可选 recent-100k Setwise
    secondary expert，并把模型、`gamma`、test 全局时间边界持久化到
    checkpoint；
  - 生产脚本只接受 hash 锁定且通过独立门禁的选择报告。
- **Command**:
  `uv run --no-sync pytest tests/test_hybrid_time_ramp.py tests/test_hybrid_temporal_robust_selection.py -q`
- **Observed pass**: Windows 与 Linux 均 `9 passed`；目标文件 Ruff
  `All checks passed!`。

## REFACTOR

- **Refactor done**:
  - 用 `itertools.pairwise` 统一 slice boundary 遍历；
  - ranker 内共享一次 Setwise context transform，避免 primary/secondary
    expert 重复构造；
  - 生产 checkpoint 保存固定 test horizon，消除 batch-local normalization。
- **Operational correction**: 首次拼包在 Python 3.12
  `pickle.PickleBuffer` 哈希 sink 上失败，未进入预测、未生成包；改为
  `memoryview(...).nbytes` 后从干净输出重跑成功。
- **Command after refactor**:
  - Windows/Linux focused pytest；
  - Ruff；
  - checkpoint reload；
  - submission row validator；
  - Dataset2 CSV/state hash 对比；
  - `unzip -t result.zip`。
- **Observed result**:
  - full delta `+0.0023190998`；
  - 三个 chronological slice delta 均非负；
  - Dataset2 CSV SHA 与 checkpoint state SHA 均保持；
  - ZIP 完整性通过。

## Next Behavior

只把该包视为离线通过的线上候选；提交后由 leaderboard 决定是否替换冠军。
