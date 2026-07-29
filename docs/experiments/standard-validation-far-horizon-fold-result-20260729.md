# 标准验证远视界折升级结果

## Verdict

协议基础设施已升级；successor v2 双候选 plan 已在任何新指标之前冻结。

新的 time-local 路径已经能在同一 selection lock 中联合近视界三折、三档
gapped 折和可选 `zero-short` 诊断臂。当前本地没有真实 gapped 分数，因此结论
是“候选和判决已预注册、selection 尚未授权”，不是“v2 已通过”。

## Frozen Dataset2 Profile

| Item | Frozen value |
|---|---:|
| Short window | `17,038,080s` / `197.2d` |
| Online all-zero short rows | `0.39971972363446745` |
| Near deployment weight | `0.6002802763655326` |
| Gapped P75 | `251d` / `21,686,400s` |
| Gapped P90 | `308d` / `26,611,200s` |
| Gapped P100 | `349d` / `30,153,600s` |
| External calibration discount | `19.5x` |

三档 gap 均不小于短窗。分位点、gap、collapse fraction 和 selection order
进入 plan lock 并分别由 SHA-256 绑定，不能在看到候选结果后调整。

## Decision Rule

非时间局部候选继续使用原有标准协议。只有显式
`temporal_scope.kind=time_local` 的候选走新路径：

1. 至少三折近视界内部等权；
2. successor v2 duel 要求每个 near 折的 MRR/NDCG@10 均不下降；
3. 每个 gapped 折要求 MRR 严格改善、NDCG@10 不下降；
4. deployment mixture 继续报告，但不能抵消任一视界的失败；
5. 两个候选都过门时，按平均 gapped MRR、最差 gapped MRR、平均 near MRR、
   预登记 tie-break 排序。

旧 deployment-mixture 模式为已有计划保留兼容；本次 duel 使用更严格的双视界
合取门。

## Counterfactual and External Semantics

- `counterfactual_arms.zero_short` 可在近折复用已训练模型评分，但固定
  `participates_in_selection=false`，只用于交叉验证机制方向。
- time-local external 仍用 raw delta 执行安全门。
- external report 固定
  `decision_role=safety_gate_only`、
  `effect_size_estimation_authorized=false`，并记录
  `calibrated_effect_size_proxy=raw_delta/19.5`。
- 该 proxy 是保守 transport 标记，不构成线上因果效应量估计。

## Promotion Evidence Amendment

v1 的 accepted/promoted 事实保持不变。指定的 replay report、promoted manifest
和 canonical status 只新增 successor 前向状态：

- `full_only_v2` 与 `gap_aware_v2` 的 plan 已冻结；
- `gapped_fold_materialization_complete=false`；
- `successor_selection_authorized=false`；
- 三份机器产物的下游 SHA-256 已按
  replay report → promoted manifest → status 顺序重算。

当前哈希：

| Artifact | SHA-256 |
|---|---|
| Successor plan | `34e28bf6128458056960f2defba760436e77b6d2a6059eb4f188d415c3385673` |
| Successor plan lock | `32aa123c30108e746e50af936aa7908249c45bcaf1023cb366c74091f8e3bead` |
| Replay report | `a089c83766e49951888c6d1a752a5cba60a2ebce67d46869818be8ecbef014c7` |
| Promoted manifest | `95473458aee80e3cde3c4fabe4237e458b692ad005a9444ad2cea7d2b5eb7976` |

## Verification

- 标准协议：`17 passed`。
- Ruff：`All checks passed!`。
- time-local example plan freeze：通过同一测试入口。
- successor preflight：`candidate_count=2`，双视界 eligibility rule 与 baseline
  均有独立 SHA-256；selection/reserved/external 指标均未读取，未创建 selection
  lock 或 package。

## Next Required Move

按冻结 plan 在远端物化两个候选的三折 near、三档 gapped 和可选 zero-short
分数；只有 manifest 完整且双视界硬门通过，才允许创建 selection lock 和打开
external。
