# Hai TDD: Dataset2 多专家置信路由 v2

## Target Behavior

给定顺序冻结的多组 query×candidate 专家分数，生成不依赖候选列位置且支持并列值的 score-only descriptors；给定每个替代专家的预测 lift，超过阈值时选择最大 lift 专家，平分时遵循冻结顺序，否则逐值返回 current gate fallback。

## RED

- **Test added**: `tests/test_hybrid_multi_expert_gate.py`
- **Behavior asserted**:
  - descriptor schema、shape 与关键数值固定；
  - 同时重排所有专家的候选列不改变 query descriptor；
  - top1 ties 使用集合 Jaccard，不借候选位置打破；
  - 路由选择最大预测 lift；
  - 最大 lift 未达 threshold 时 exact fallback；
  - lift 平分时使用冻结专家顺序。
- **Command**: `uv run --no-sync pytest tests/test_hybrid_multi_expert_gate.py -q`
- **Observed failure**: 测试收集时报 `ModuleNotFoundError: No module named 'jgrec.rankers.hybrid.multi_expert_gate'`。
- **Failure is correct because**: 多专家 descriptor/route 模块尚未实现；失败发生在目标 API 缺失处，不是测试语法、fixture 或环境错误。

## GREEN

- **Minimal implementation**:
  - `multi_expert_score_descriptors` 生成每专家 margin/entropy/top-k mass 与每对专家的 tie-neutral top-k Jaccard、交叉 top 偏好、交叉 rank、分数差；
  - `route_multi_expert` 在 threshold 之上按冻结顺序选择最大 lift，否则逐值 fallback；
  - 每个替代专家用一个浅层 `DecisionTreeRegressor` 预测相对 fallback 的 RR reward；
  - forward selector 先切片再校验/训练，因此完全不读取 gate 行。
- **Command**:
  - `uv run --no-sync pytest tests/test_hybrid_multi_expert_gate.py -q`
  - `uv run --no-sync pytest tests/test_hybrid_multi_expert_gate.py tests/test_hybrid_multi_interest_gate.py tests/test_hybrid_temporal_robust_selection.py -q`
- **Observed pass**:
  - 目标测试 `5 passed`；
  - 新旧 gate 与前缀选择组合回归 `14 passed`。

## REFACTOR

- **Refactor done**: yes
- **Change**:
  - descriptor 名称与计算拆成稳定的纯函数；
  - 用 tie-neutral descending competition rank 取代 query×candidate×candidate 广播，避免 20k×100 输入产生 GB 级临时张量；
  - 归一化、entropy 和 masked mean 使用 float64 聚合，保证候选列重排后 descriptor 在 float32 输出上稳定。
- **Command after refactor**:
  - `uv run --no-sync pytest tests/test_hybrid_multi_expert_gate.py -q`
  - `uv run --no-sync ruff check scripts/run_dataset2_multi_expert_confidence_v2.py src/jgrec/rankers/hybrid/multi_expert_gate.py tests/test_hybrid_multi_expert_gate.py`
  - `uv run --no-sync python -m py_compile scripts/run_dataset2_multi_expert_confidence_v2.py src/jgrec/rankers/hybrid/multi_expert_gate.py`
- **Observed result**: `5 passed`；Ruff `All checks passed!`；编译通过。

## Next Behavior

已完成远端四专家精确重放和 slice0→slice1 前向选型。27 个配置均未达到 `+0.001`；coverage≤25% 的最佳 delta 为 `+0.0001826203`。selection status 为 `no_eligible_candidate`，协议在读取 slice2 前停止，未生成包。
