# Goal Document: Dataset2 Frozen CST + Bounded ID Residual

## Go / No-Go

- **Judgment**: Go
- **Reason**: A/B/C/D 已证明 candidate ID 在三个时间折均有强信号，但无约束
  `candidate_id_scale` 从 `0.1` 漂到 `1.407`，导致外部 20k 严重退化。
  将 ID 改为冻结 CST 之上的有界 residual，是直接针对已确认失效机制的最小实验。

## Target Outcome

冻结纯 Jittor CST 主干，只训练 candidate-ID residual，在任何 query/candidate
上严格限制 residual 幅度，并通过 rolling-origin 选择
`0.02 / 0.05 / 0.10` 三个固定上限中的一个。只有锁定上限通过第三折门禁，
才允许全量重训和一次外部 20k 评估。

## Goal Definition

- **Type**: technical / learning / quality
- **Boundary**:
  - 使用 Dataset2 现有 `200k × 100 × 63` cache；
  - 每折复用已完成 A 变体的冻结纯 Jittor checkpoint；
  - 只训练 item embedding + 小型 ID residual head；
  - residual 上限扫描固定为 `0.02 / 0.05 / 0.10`；
  - 使用相同三折 timestamp-aligned rolling-origin；
  - 最终模型仍只含 Jittor 可训练模块。
- **Non-goals**:
  - 不联合微调 CST 主干；
  - 不加入 source sequence；
  - 不使用 LightGBM/sklearn；
  - 不扫描外部 20k 融合权重；
  - 不生成未过门禁的提交。
- **Deferred work**:
  - bounded source residual；
  - C/D 动态路由；
  - 多 seed ensemble。
- **Verification rule**:
  - 数学边界、零残差等价、置换等价和 checkpoint 重载由自动测试证明；
  - 上限选择只看 Fold 0/1；
  - Fold 2 只对锁定上限做正式门禁，其余上限仅在锁定后做诊断；
  - 外部 20k 只在门禁通过后读取一次。
- **Evidence source**:
  - Linux/Jittor tests；
  - 9 份 residual fold report；
  - selection lock、gate report、external report；
  - 模型/checkpoint/hash。
- **Pass criteria**:
  - 每个 candidate 的绝对 logit 残差
    `abs(residual) <= cap`；
  - cap=0 或 residual head 输出 0 时，最终 logits 精确等于 frozen base；
  - Fold 0/1 锁定候选相对 A 均不退化，平均增益至少 `+0.0002`；
  - Fold 2 不退化，三折平均增益至少 `+0.0002`；
  - activity 最差分段 delta 不低于 `-0.001`；
  - 外部 full 相对冠军至少 `+0.0002` 且三个时间片均不退化，才允许提交。
- **Confidence note**:
  - rolling-origin 是时间迁移的直接证据；
  - 外部 20k 仍是最终代理指标，不能保证线上完全一致。
- **Judgment owner**:
  - 自动测试负责结构正确性；
  - 冻结 selection/gate 规则负责模型晋级；
  - 外部指标负责是否生成提交。

## Current State

- A/B/C/D 三折已全部完成。
- B 相对 A 三折平均约 `+0.04852`，证明 candidate ID 有信号。
- D 全量外部 MRR `0.474471`，低于纯 Jittor CST 和冠军。
- D 全量 `candidate_id_scale=1.407`，初值仅 `0.1`。
- 三折 A checkpoint、score logits、严格因果 folds 和完整 candidate sidecar
  已存在，可直接复用。

## Priority Rationale

- 先证明数学边界和 base 等价，避免再次出现 residual 偷换主干的问题。
- 再用已冻结 A checkpoint 训练极小 residual，隔离 candidate-ID 的真实增量。
- 先两折选择、后第三折门禁，最后才打开外部 20k。

## Assumptions and Open Decisions

| Item | Status | Impact | Owner / Next step |
|---|---|---|---|
| residual 上限的单位 | confirmed | 绝对 logit 幅度；`0.10` 就是硬上限 `0.10` | 自动测试 |
| residual 公式 | confirmed | `base + cap × tanh(centered_id_logit)` | 自动测试 |
| 主干训练状态 | confirmed | 预计算 logits，训练 residual 时无主干梯度 | checkpoint/hash |
| residual epoch | assumed | 所有 cap 固定 3 epoch，不按 cap 单独 early-stop | rolling metrics |
| embedding regularization | assumed | dim=32、ID dropout=0.10、weight decay=1e-3 | frozen config |
| full CST 主干 | unresolved until external phase | 优先复用已验证纯 Jittor full CST；契约不匹配则同协议重训 | preflight |

## Phases

### Phase 1: 数学与 checkpoint 契约

- **Purpose**: 保证 bounded residual 不可能覆盖 frozen base。
- **Entry condition**: 目标文档已落盘。
- **Phase rules**:
  - 先写 RED 测试；
  - production code 不得在 RED 前创建；
  - base logits 只作为常量输入。
- **Todos**:
  - [ ] 测试 cap=0 和 zero head 精确复现 base logits
    - **Surface**: tests/model
    - **Proof**: RED 后 GREEN
    - **Depends on**: none
  - [ ] 测试 residual 数学上界
    - **Surface**: tests/model
    - **Proof**: adversarial large raw residual 仍不越界
    - **Depends on**: none
  - [ ] 测试 candidate permutation equivariance
    - **Surface**: tests/model
    - **Proof**: 同步置换 input 后 logits 同步置换
    - **Depends on**: none
  - [ ] 测试训练、保存和重载
    - **Surface**: tests/checkpoint
    - **Proof**: reload logits tolerance
    - **Depends on**: model GREEN
- **Exit proof**: 目标测试全部通过。
- **Stop condition**: 任一情况下 residual 可越界或改变 zero-residual base。

### Phase 2: Fold base-logit cache 与 residual 训练

- **Purpose**: 在不重训/微调主干的情况下运行三个 cap。
- **Entry condition**: Phase 1 GREEN。
- **Phase rules**:
  - A checkpoint/hash 必须与前一实验 report 一致；
  - score base logits 必须逐元素复现已保存 A score logits；
  - cap、epoch、regularization 对三个候选完全相同。
- **Todos**:
  - [ ] 每折生成 frozen A train logits
    - **Surface**: cache
    - **Proof**: shape/hash 与 score replay
    - **Depends on**: A checkpoints
  - [ ] 运行 Fold 0/1 × 三个 cap
    - **Surface**: result
    - **Proof**: 6 份 report
    - **Depends on**: base-logit cache
  - [ ] 写入不可变 selection lock
    - **Surface**: result
    - **Proof**: lock 明确 `gate_metrics_read=false`
    - **Depends on**: 两个选择折完成
- **Exit proof**: 一个 cap 被锁定，或所有 cap 均不满足非退化规则。
- **Stop condition**: base replay 不一致、NaN、边界审计失败。

### Phase 3: Gate 与外部验证

- **Purpose**: 验证 bounded ID residual 能否迁移到更晚时间和外部 context。
- **Entry condition**: selection lock 已存在。
- **Phase rules**:
  - 先运行锁定 cap 的 Fold 2 门禁；
  - 其他 cap 的 Fold 2 只能在锁定后作为诊断，不能改变选择；
  - 外部 20k 只在 gate pass 后读取一次。
- **Todos**:
  - [ ] 完成三个 cap 的 Fold 2
    - **Surface**: result
    - **Proof**: 3 份 report
    - **Depends on**: selection lock
  - [ ] 计算锁定 cap gate
    - **Surface**: gate report
    - **Proof**: full/time/activity deltas
    - **Depends on**: Fold 2
  - [ ] 若通过，训练 full residual 并评估外部 20k 一次
    - **Surface**: checkpoint/evaluation
    - **Proof**: external report
    - **Depends on**: gate pass
- **Exit proof**: 明确 passed/rejected；仅 passed 可生成提交。
- **Stop condition**: gate 失败立即停止外部读取。

## Dry-Run Findings

- v1 dry-run 曾把 cap 定义为每行 base logit 标准差的倍数；外部审计显示
  `cap=0.10` 时实际 residual 可达 `0.484`，违反“最大幅度 0.02–0.10”
  的字面边界。因此正式 v2 改为绝对 logit 上限，并用独立结果目录重跑。
- 若 residual 不做 `tanh`，参数或 embedding 仍可绕过 gate，因此边界必须写入
  前向公式而不是依赖正则。
- 训练 residual 时无需主干梯度；预计算 logits 能从结构上证明“冻结”。
- Fold 2 可以把三个 cap 都跑完，但 selection lock 必须先落盘，且不能事后换 cap。

## Final Validation

```bash
.venv/bin/ruff check <bounded residual files>
.venv/bin/python -m pytest -q <bounded residual tests>
```

并核验：

- 9 份 fold report；
- selection lock 早于 gate；
- residual bound audit 全为 true；
- external report 最多一份；
- `trainable_frameworks=["jittor"]`；
- `non_jittor_trainable_models=[]`。

## First Execution Step

先新增 zero-residual、硬边界、候选置换和 checkpoint 重载 RED 测试，并在远端
Linux/Jittor 上确认因缺少 bounded residual API 而失败。
