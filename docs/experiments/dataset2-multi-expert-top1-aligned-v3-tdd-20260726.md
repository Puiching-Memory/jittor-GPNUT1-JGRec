# TDD Evidence: Dataset2 多专家 top1 对齐路由 v3

## Target Behavior

- 新 API 接收多位专家的 query×candidate score 和对齐的 query×candidate×feature tensor。
- fallback 与每个替代专家的 top1 都是完整并列集合，不使用 `argmax` 的首列偏置。
- 每个输出 descriptor 是：

  `mean(feature on alternative top1 set) - mean(feature on fallback top1 set)`

- 同步置换 score 与 feature 的 candidate 轴后，descriptor 名称和值不变。
- feature 名称缺失、重复，或 score/feature shape 错位时立即拒绝。
- 未达到路由阈值的行仍由既有 `route_multi_expert` 逐值 fallback。

## RED

- **Test file**: `tests/test_hybrid_multi_expert_gate.py`
- **New tests**:
  - `test_expert_top1_feature_deltas_are_tie_neutral_and_permutation_invariant`
  - `test_expert_top1_feature_deltas_reject_invalid_feature_schema`
- **Command**:

  ```bash
  uv run --no-sync pytest tests/test_hybrid_multi_expert_gate.py -q
  ```

- **Observed failure**:

  ```text
  ImportError: cannot import name 'expert_top1_feature_deltas'
  ```

- **Why this was the right failure**: 测试在目标 API 尚未实现时停止，没有被环境、fixture 或无关断言抢先击中。

## GREEN

- **Production surface**: `src/jgrec/rankers/hybrid/multi_expert_gate.py`
- **Minimal implementation**:
  - 新增 `expert_top1_feature_deltas`；
  - 复用现有 expert score shape/finite/positive-mass 契约；
  - `_top_k_mask(..., top_k=1)` 保留所有并列 top1；
  - `_top1_feature_means` 逐 feature 计算集合均值，避免复制完整 3D tensor；
  - 输出顺序冻结为 alternative order，再按 selected feature order；
  - 输出为 `float32`。
- **GREEN command**:

  ```bash
  uv run --no-sync pytest tests/test_hybrid_multi_expert_gate.py -q
  ```

- **Observed result**: `7 passed`

## Refactor Decision

- 保留独立通用 API，不把 raw/proxy 逻辑写死在实验脚本中；同一函数可对 63 维 base schema 的白名单和九维 proxy schema 分别调用。
- 不改现有 74 维 `multi_expert_score_descriptors`，v2 结果仍可精确重放。
- 不把完整 63 维 tensor 转成选中特征的 3D 副本；按通道扫描降低峰值内存。
- 不为这轮引入新的模型或依赖。

## Experiment Integration

- **Script**: `scripts/run_dataset2_multi_expert_top1_aligned_v3.py`
- **Launcher**: `scripts/run_dataset2_multi_expert_top1_aligned_v3_20260726.sh`
- 复用 v2 r3 已 SHA 锁定的四组 validation expert score。
- descriptor 固定为：
  - score-only：74 维；
  - 3 位替代专家 × 12 个 raw top1 delta：36 维；
  - 3 位替代专家 × 9 个 multi-interest top1 delta：27 维；
  - 合计：137 维。
- r1 preflight 因 proxy 报告把 schema 放在 `frozen_config.proxy_feature_names` 而非顶层而停止；没有生成 descriptor 或进行选型。
- 修正报告读取路径后以不可覆盖的新 tag `r2` 重跑。

## Verification

### Local

```bash
uv run --no-sync pytest \
  tests/test_hybrid_multi_expert_gate.py \
  tests/test_hybrid_multi_interest_gate.py \
  tests/test_hybrid_temporal_robust_selection.py -q
```

Observed: `16 passed`

```bash
uv run --no-sync ruff check \
  src/jgrec/rankers/hybrid/multi_expert_gate.py \
  tests/test_hybrid_multi_expert_gate.py \
  scripts/run_dataset2_multi_expert_top1_aligned_v3.py
```

Observed: `All checks passed!`

### Linux target

同一组测试与 Ruff 在目标 Linux 环境通过：

```text
16 passed in 1.85s
All checks passed!
```

### Forward-selection evidence

- Result: `result/dataset2_multi_expert_top1_aligned_v3_r2_20260726/selection-report.json`
- Selection SHA-256: `ca22b016109d4a3499679b403474d8768316ad44717b0f4abdc1a2f35022dfb7`
- Sidecar matches: true
- Status: `no_eligible_candidate`
- `evaluation-report.json`: absent
- `selection-router.pkl`: absent

## TDD Judgment

实现契约通过；实验价值门禁未通过。功能代码可保留作为后续受控 descriptor 组件，但本轮不得进入 Dataset2 生产包。
