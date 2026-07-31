# Hai TDD: Dataset2 部分混合 Listwise 专家

## Target Behavior

在不改当前冠军主干的前提下，以固定一维权重网格混入一个 listwise 辅助专家；
只允许 slice0 选择权重、slice1 前向确认，并要求 selection lock 匹配后才能由
唯一胜者进入 slice2/full 门禁。即使所有权重均失败，也必须留下完整扫描证据。

## RED

- **Test added**:
  `tests/test_hybrid_partial_listwise_blend.py` 的首批 6 个行为测试。
- **Behavior asserted**:
  - 固定 residual 公式和权重白名单；
  - shape/finite 校验；
  - Two-Tower descending midrank tie-neutral 变换；
  - slice0 选择只按 MRR，精确并列取更小权重；
  - slice1 不能改变锁定权重；
  - final gate 必须验证 selection-lock SHA 且三片全部不下降。
- **Command**:
  `uv run --no-sync pytest tests/test_hybrid_partial_listwise_blend.py -q`
- **Observed failure**:
  collection 以
  `ModuleNotFoundError: jgrec.rankers.hybrid.partial_listwise_blend`
  失败。
- **Failure is correct because**:
  仓库此前没有部分 listwise 混合、前向选择或 lock 门禁模块。

## GREEN

- **Minimal implementation**:
  新增 `partial_listwise_blend.py`，实现固定混合公式、midrank 概率变换、
  slice0 selector、slice1 gate、winner lock 与 final gate。
- **Command**:
  `uv run --no-sync pytest tests/test_hybrid_partial_listwise_blend.py -q`
- **Observed pass**:
  首批 6 个测试在 Windows 与 Linux 均通过。

## REFACTOR

- **Refactor done**: yes
- **Change**:
  第二个 RED 测试要求“所有权重都失败时仍保留完整 trials”。将扫描从 selector
  中抽成 `scan_auxiliary_weights()`；selector 继续负责只为合格权重生成 lock，
  evaluator 则无条件持久化两条支路的完整 slice0 扫描。
- **Command after refactor**:
  - `uv run --no-sync pytest tests/test_hybrid_partial_listwise_blend.py -q`
  - `uv run --no-sync pytest tests/test_hybrid_partial_listwise_blend.py
    tests/test_hybrid_time_ramp.py
    tests/test_hybrid_conservative_window_blend.py -q`
  - `uv run --no-sync ruff check
    src/jgrec/rankers/hybrid/partial_listwise_blend.py
    tests/test_hybrid_partial_listwise_blend.py
    scripts/score_dataset2_partial_listwise_experts.py
    scripts/evaluate_dataset2_partial_listwise_blends.py`
- **Observed result**:
  focused 7 passed，相关 blend 回归 17 passed，Linux focused 7 passed，
  Ruff 全绿。

## Next Behavior

Done for this goal。实际前向门禁拒绝了两个专家，因此 checkpoint hydrate/
score-equivalence 行为没有进入 RED；按冻结 stop condition 跳过 Phase 4，而不是
为失败候选增加生产接线。

## Integration Gate Evidence

评分 runner 还留下了两次在任何权重指标产生前失败的集成约束：

1. 首版错误地把 short-none overlay 同时喂给 LightGBM，冠军 full MRR 变为
   `0.5481883879`；精确复现门禁阻止继续。修正为 A1 原口径：overlay 只喂
   Setwise，LightGBM 使用原 validation cache。
2. Two-Tower 首次恢复把 embedding 的 OOV 预留行误当作 id-map 漂移；
   状态 shape 实际合同为 `num_ids + 1`。修正后成功恢复模型并完成统一 sidecar
   打分。

最终 runner 精确复现冠军 full/slice 指标，随后才允许执行冻结的权重选择。
