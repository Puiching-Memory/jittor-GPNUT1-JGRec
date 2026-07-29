# Goal Document: Dataset2 Confidence-Routed Top-k ID Correction

## Go / No-Go

- **Judgment**: Go
- **Reason**: absolute bounded ID residual 已把外部风险压到近乎零，但全量作用于
  100 candidates 的可迁移收益仅约 `+0.000008`。下一步应只在 router 能识别的
  高置信行、且只在 frozen base 的 top-k 内纠错，直接检验 ID 信号能否被稀疏提纯。

## Target Outcome

冻结纯 Jittor CST logits，只训练 candidate-ID correction 与纯 Jittor confidence
router。推理时 top-k 外候选逐元素不变，未路由行逐元素不变，并将路由行数硬限制在
固定预算内。用 Fold0/1 选择 top-k/预算组合，Fold2 做不可见门禁；只有门禁通过才允许
一次外部 20k 评估。

## Goal Definition

- **Type**: technical / learning / quality
- **Boundary**:
  - Dataset2 现有 `200k × 100 × 63` cache；
  - 复用已经逐元素重放验证的三折 frozen A/CST logits；
  - absolute residual cap 固定为 `0.10`；
  - 候选范围/行预算固定扫描：
    - `top5 × 5% rows`
    - `top10 × 5% rows`
    - `top10 × 10% rows`
  - correction head 与 router 均为 `jt.nn.Module`；
  - NumPy 仅计算固定 top-k、support、router 特征、指标和确定性配额。
- **Non-goals**:
  - 不微调 CST；
  - 不允许 top-k 外候选改变；
  - 不使用 source sequence、LightGBM 或 sklearn；
  - 不按外部 20k 调 top-k、预算、cap 或阈值；
  - 不生成未过外部门禁的提交。
- **Deferred work**:
  - 直接以当前线上冠军分数为 rolling-origin 主干；
  - pairwise hard-negative LambdaMRR；
  - 多 seed 或更大 router。
- **Verification rule**:
  - 自动测试证明稀疏掩码、绝对 cap、候选置换和硬行预算；
  - router 只用 label-free inference features；
  - router 标签只来自每折训练尾部的时间留出段；
  - Fold0/1 selection lock 先于 Fold2；
  - external 20k 只在 gate pass 后读取一次。
- **Evidence source**:
  - Linux/Jittor RED-GREEN tests；
  - frozen-base replay hashes；
  - 9 个 fold reports；
  - selection lock、gate report、external report；
  - checkpoint 与稀疏性审计。
- **Pass criteria**:
  - `abs(residual) <= 0.10`；
  - top-k 外和未路由行严格等于 frozen base；
  - 实际 route rate 不超过固定预算；
  - Fold0/1 每折不退化且 mean delta 至少 `+0.0001`；
  - Fold2 delta 非负，三折 mean delta 至少 `+0.0001`；
  - activity 最差 delta 不低于 `-0.0005`；
  - 外部 full 相对冠军至少 `+0.0002` 且三个时间片均不退化才可提交。
- **Confidence note**:
  - rolling-origin 与时间留出 router labels 能约束历史迁移；
  - external 仍只是线上代理，不能保证 leaderboard 一致；
  - 当前 frozen CST 本身落后冠军约 `0.0035`，因此本轮主要验证稀疏纠错机制，
    只有实际越过冠军才进入提交。
- **Judgment owner**:
  - 自动测试判断结构契约；
  - frozen selection/gate 规则判断实验晋级；
  - external report 判断是否提交。

## Current State

- absolute-cap v2 已完成，cap 0.10 三折 mean delta `+0.00033817`；
- v2 对 full frozen CST 的 external delta 仅约 `+0.00000777`；
- v2 对所有行和全部 candidates 注入 residual，缺少 query-level 置信选择；
- 三折 train/score base logits、candidates、features、times 和 dst 已存在；
- candidate ID 信号在无约束模型中很强，但已证明存在明显 context overfit。

## Priority Rationale

- 先证明“不路由/非 top-k 不变”，防止所谓稀疏路由实际仍污染全行；
- router 先在时间留出 correction outcomes 上学习，再全量重训 correction，
  避免直接拿训练行 oracle 改善标签过拟合；
- 固定三种很小的结构组合，比扫描大量阈值更能回答“稀疏是否有效”。

## Assumptions and Open Decisions

| Item | Status | Impact | Owner / Next step |
|---|---|---|---|
| absolute cap | confirmed | 固定 `0.10`，不再扫描 | 自动测试 |
| correction candidates | confirmed | 只允许 frozen-base top5/top10 | 稀疏审计 |
| row route budget | confirmed | 最多 5%/10% | 配额审计 |
| router supervision | confirmed | 训练尾部时间留出段上，纠错 MRR 是否改善 | fold runner |
| router features | assumed | base margin/entropy、correction pressure/rank-change、ID support；不含 label | 特征测试 |
| router threshold | confirmed | inference batch 内按 probability 取固定 top quota，同时要求 `p>=0.5` | 配额测试 |
| router holdout | assumed | 每折训练尾部 20k，按 timestamp 左边界切分 | cache/time audit |
| 单类别 router holdout | confirmed | 无改善样本时使用纯 Jittor 常量低概率 router，安全退回零路由 | 单元测试 |

## Phases

### Phase 1: 稀疏纠错与路由契约

- **Purpose**: 从结构上保证 correction 不可能污染未授权位置。
- **Entry condition**: 目标文档已落盘。
- **Phase rules**:
  - 先写 RED tests；
  - production module 不得在 RED 前创建；
  - 所有 mask 由 frozen base 和 router probability 决定。
- **Todos**:
  - [ ] 测试 top-k 外逐元素不变
    - **Surface**: test/model
    - **Proof**: adversarial raw correction 仍无法改变 top-k 外
    - **Depends on**: none
  - [ ] 测试未路由行逐元素不变和 route rate 硬预算
    - **Surface**: test/router
    - **Proof**: route mask 数量与逐行相等断言
    - **Depends on**: none
  - [ ] 测试绝对 cap 与 candidate permutation equivariance
    - **Surface**: test/model
    - **Proof**: RED 后 GREEN
    - **Depends on**: none
  - [ ] 测试 router 训练/checkpoint 重放
    - **Surface**: test/checkpoint
    - **Proof**: reload probabilities/logits 一致
    - **Depends on**: router GREEN
- **Exit proof**: 目标测试全部通过。
- **Stop condition**: 任一未授权 candidate/row 改变，或 route rate 超预算。

### Phase 2: 时间留出 router 与 Fold0/1 selection

- **Purpose**: 用历史时间留出结果训练 router，并在两个未来折选择稀疏组合。
- **Entry condition**: Phase1 GREEN。
- **Phase rules**:
  - correction prefix 不读取 router holdout labels；
  - router feature 不包含正例位置或 MRR label；
  - 三个候选共用 cap、训练预算和 base cache。
- **Todos**:
  - [ ] 为每折构建 correction-prefix/router-holdout split
    - **Surface**: runner/report
    - **Proof**: timestamp 严格分离
    - **Depends on**: base cache
  - [ ] 训练 prefix correction，生成 holdout improvement labels
    - **Surface**: Jittor checkpoint/cache
    - **Proof**: label 分布与无泄漏审计
    - **Depends on**: split
  - [ ] 训练 router 并全量重训 correction
    - **Surface**: Jittor model
    - **Proof**: checkpoint/hash
    - **Depends on**: holdout labels
  - [ ] 跑 Fold0/1 三个候选并锁定
    - **Surface**: result
    - **Proof**: 6 份 fold report + selection lock
    - **Depends on**: models
- **Exit proof**: 一个候选被锁定，或全部未通过 selection。
- **Stop condition**: router 使用 label 特征、时间泄漏、稀疏审计失败。

### Phase 3: Fold2 gate 与 external

- **Purpose**: 验证更晚时间和外部 context 的可迁移性。
- **Entry condition**: selection lock 存在且 `gate_metrics_read=false`。
- **Phase rules**:
  - 先跑锁定组合 Fold2；
  - 其他组合只能作 lock 后诊断；
  - gate 未通过不读取 external。
- **Todos**:
  - [ ] 完成三个候选 Fold2
    - **Surface**: result
    - **Proof**: 3 份 fold report
    - **Depends on**: selection lock
  - [ ] 计算 full/time/activity/sparsity gate
    - **Surface**: gate report
    - **Proof**: frozen thresholds
    - **Depends on**: Fold2
  - [ ] 若 gate 通过，full 重训并评估 external 一次
    - **Surface**: checkpoint/evaluation
    - **Proof**: external report
    - **Depends on**: gate pass
- **Exit proof**: 明确 passed/rejected，且只有 external pass 才生成提交。
- **Stop condition**: gate 失败、边界/稀疏审计失败或 checkpoint provenance 不纯。

## Dry-Run Findings

- 仅按 raw residual 大小路由无法判断纠错方向；router 需要学习“proposal 是否改善”
  的标签，但 inference features 必须完全 label-free。
- 若 router 与 correction 在同一行上共同拟合，router target 会过于乐观；因此先用
  prefix correction 在最后 20k 时间留出段制造监督，再全量重训 correction。
- 只限制 candidates 不够；全局作用仍可能复现 v2 的平均化，因此同时设置硬行预算。
- hard quota 是 batch-level deterministic policy，checkpoint 必须同时保存 budget，
  不能只保存一个受分布漂移影响的 probability threshold。
- 若某个时间留出段没有任何改善标签，训练分类器没有意义；正式行为是保存一个
  低于阈值的纯 Jittor 常量 router，使该折精确 fallback 到 frozen base。

## Final Validation

```bash
.venv/bin/ruff check <confidence-routed top-k files>
.venv/bin/python -m pytest -q <confidence-routed top-k tests>
```

并核验：

- 9 份 fold report；
- 100% top-k 外元素不变；
- 100% 未路由行不变；
- route rate 不超过 frozen budget；
- selection lock 早于 Fold2；
- trainable frameworks 只有 Jittor；
- external report 最多一份，未通过则没有提交。

## First Execution Step

新增 top-k sparse correction、hard route quota 和 router checkpoint 的 RED tests，
确认因 API 缺失而失败后再创建 production module。
