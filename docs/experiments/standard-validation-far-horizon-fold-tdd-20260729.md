# Hai TDD: 标准验证远视界折

## Target Behavior

时间局部候选必须先冻结短窗、gapped 部署视界、collapse fraction 和 external
解释规则。selector 应允许候选在近视界三折小幅退化、但在 gapped 折改善且部署
混合为正时晋级；短于 `w` 的 gapped fold 必须拒绝；可选 `zero-short` 只报告不
选参；external 只作安全门并记录 `19.5x` 折扣。

## RED

- **Test added**:
  - `test_time_local_plan_requires_preregistered_far_horizon_policy`
  - `test_far_horizon_deployment_mixture_can_accept_near_horizon_regression`
  - `test_far_horizon_fold_gap_must_cover_short_window`
  - `test_time_local_external_is_safety_gate_not_effect_size_estimate`
- **Behavior asserted**: time-local plan 不可绕过远视界策略；三折近视界各有一行
  从 rank 2 退到 rank 3，而三档 gapped 折从 rank 2 升到 rank 1 时，按冻结的
  `60.0280% / 39.9720%` 混合后候选仍可晋级；即使 `zero-short` 臂对该候选
  明显为负，选择仍不改变，证明该臂只报告不选参；external raw delta 不获准
  解释为线上效应量。
- **Command**:
  `uv run --no-sync pytest tests/test_standard_validation_protocol.py -q`
- **Observed failure**: `4 failed, 6 passed`。旧实现不拒绝缺失远视界策略的
  time-local plan，不校验 short-window gap，仍把目标候选判为 `rejected`，且
  external report 没有 `effect_size_interpretation`。
- **Failure is correct because**: 四个失败都到达现有协议并精确暴露缺失行为；最初
  一次 `uv sync` 被 Windows `pymetis` 的 `sys/resource.h` 构建问题拦截，没有
  计作 RED。

## GREEN

- **Minimal implementation**:
  - plan lock 新增 `temporal_scope`、`far_horizon_validation` 及独立 SHA-256；
  - manifest 新增严格绑定 plan 的 `role=gapped` 折和独立
    `counterfactual_arms`；
  - gapped gap 同时校验 `>= short_window_seconds` 和预登记分位点下限；
  - 近折与 gapped 折内部等权，再按 collapse fraction 形成 deployment mixture；
  - gapped 逐折安全门与 deployment-mixture 全指标门共同决定 eligibility；
  - `zero-short` 固定 `participates_in_selection=false`；
  - time-local external 输出 `safety_gate_only`、禁用效应量估计和
    `raw_delta / 19.5` proxy。
- **Command**:
  `uv run --no-sync pytest tests/test_standard_validation_protocol.py -q`
- **Observed pass**: 初次 GREEN `10 passed`；加入正式 time-local example freeze
  smoke 后为 `11 passed`。

## REFACTOR

- **Refactor done**: yes
- **Change**: 抽取统一的 fold artifact 校验、baseline 加载和候选 fold-group
  评分函数，让近折、gapped 折与 counterfactual 臂共享同一数值路径；legacy
  非时间局部 plan 保持原有 gate 和 selection key。
- **Command after refactor**:
  - `uv tool run ruff==0.15.16 check
    src/jgrec/standard_validation_protocol.py
    tests/test_standard_validation_protocol.py
    scripts/preflight_standard_validation_local.py`
  - `uv run --no-sync pytest tests/test_standard_validation_protocol.py -q`
- **Observed result**: `All checks passed!`；`11 passed`。

## Next Behavior

通用协议实现已完成。下一步不是预注册 v2，而是在远端先按冻结的
P75/P90/P100（251/308/349 天）物化真实 gapped folds；当前 preflight 明确为
`far_horizon_folds_required_before_preregistration`。
