# Goal Document: Dataset2 默认 Short 的纯 Jittor Bounded Top-k Router

## Go / No-Go

- **Judgment**: Go
- **Reason**: 已有最后 40,196 行同时覆盖 short/medium/long 的严格 OOF
  residual，足以把“是否切换”训练成独立的小模型，并用更晚时间段做一次
  不可见门禁；不需要读取外部验证或改动线上冠军。

## Target Outcome

训练一个纯 `jt.nn.Module` 路由器。默认输出始终是 short corrected logits；
只有模型预测 medium/long 相对 short 的收益同时满足高置信阈值时，才允许
切换 residual。切换后只修改 short 当前 top-k 内的候选，行内改变量和为
零且最大绝对幅度受第二层 hard cap 约束。

## Goal Definition

- **Type**: technical / learning / quality
- **Boundary**:
  - 使用多时间跨度 OOF 产物最后共同覆盖的 `[159804, 200000)`；
  - 按 timestamp 切成 60% train、20% selection、20% unseen gate；
  - 扫描 `top_k ∈ {5,10,20}`、`switch_cap ∈ {0.01,0.02}`；
  - 路由器只用无标签、候选置换不变的 logits/residual/gap summary；
  - 训练目标是 medium/long bounded top-k 路径相对 short 的逐行 MRR
    reward；
  - selection 只选择 variant 与高置信 coverage，gate 只读一次。
- **Non-goals**:
  - 不允许路由器直接输出 100 个候选分数；
  - 不允许 candidate ID embedding、source ID embedding；
  - 不在本轮读取外部验证标签或生成提交；
  - 不改变 short/medium/long 现有 checkpoint 和 residual。
- **Deferred work**:
  - gate 通过后，再生成完整训练尾部对应的外部多 origin residual；
  - 468 天外推单独门禁。
- **Verification rule**:
  行为测试证明 exact fallback/top-k/cap/permutation 契约；selection 冻结选择
  后，最终 gate 必须相对 default short 非负且切换率不超过 5%。
- **Evidence source**:
  单元测试、Jittor checkpoint 重放、selection lock、unseen gate report。
- **Pass criteria**:
  - 低于阈值或选择 short 时，输出与 short 逐元素完全相同；
  - top-k 外改变量严格为零；
  - 最大绝对改变量不超过所选 `switch_cap + 2e-6`；
  - 特征与路由对候选列置换保持不变；
  - selection delta > 0，coverage `<= 5%`；
  - unseen gate delta `>= 0`、coverage `<= 5%`，三个连续时间子片最差
    delta `>= -0.0001`；
  - 可训练框架严格为 `["jittor"]`，非 Jittor 可训练模型为空。
- **Confidence note**:
  gate 是真正晚于训练和 selection 的时间段，但总共只有约 8k 行，能证明
  短期稳定性，不能替代 468 天外部验证。
- **Judgment owner**:
  自动化行为审计与冻结后的 unseen gate report。

## Current State

- 多 horizon OOF residual 已生成：
  `result/dataset2_bounded_source_multi_horizon_oof_20260727`
- 共同覆盖区间 40,196 行，short/medium/long delta 分别约
  `+0.002972/+0.002618/+0.001802`（各自相对自己的 frozen base）。
- short/long residual 相关系数约 0.866，有结构性差异但不适合无约束平均。
- 外部时间跨度约 468 天，显著超出 OOF 的 104 天上限。

## Priority Rationale

- 先证明 medium/long 在“以 short 为共同 anchor、只改 short top-k”时仍有
  oracle opportunity；如果没有，训练路由器没有意义。
- 先锁死 exact fallback 和 hard bound，再谈模型收益，避免再次出现自由
  residual 放大。
- selection 负责所有选择，gate 只做一次最终判决，防止反复看末段过拟合。

## Assumptions and Open Decisions

| Item | Status | Impact | Owner / Next step |
|---|---|---|---|
| short corrected 是永久默认输出 | confirmed | 失败时能精确回退现有路径 | 行为测试 |
| 切换只用 medium/long residual，不使用它们较旧的 base | confirmed | 隔离“时间 residual”而非路由整个旧模型 | score builder 审计 |
| 共同 anchor 为 short base + selected residual | confirmed | 三路比较口径一致 | oracle preflight |
| 最大线上式切换率 5% | assumed | 强制“高置信、少量纠错” | selection lock |
| gate 三个时间子片允许最多 -0.0001 噪声 | assumed | 约 2.7k 行/片，避免把微小方差当系统性退化 | gate report |

## Phases

### Phase 1: Oracle 与行为契约

- **Purpose**: 证明 bounded top-k medium/long 路径有可路由机会，并锁住安全边界。
- **Entry condition**: 三路 OOF 数组与 mask/hash 审计通过。
- **Phase rules**:
  - 只读 OOF 数组；
  - 不写生产实现前先看到 RED；
  - oracle 只用于判断上限，不能作为真实路由结果。
- **Todos**:
  - [x] 写 exact fallback、top-k 外零改动、cap、置换不变测试
    - **Surface**: `tests/test_hybrid_high_confidence_topk_router.py`
    - **Proof**: 缺少模块导致正确 RED
    - **Depends on**: none
  - [x] 计算六个 variant 的 train/selection/gate oracle opportunity
    - **Surface**: 训练报告
    - **Proof**: medium/long 至少一个 variant 在 selection 与 gate 都存在正
      oracle gain
    - **Depends on**: OOF artifact
- **Exit proof**: 行为测试 GREEN，oracle 不是零机会。
- **Stop condition**: 所有 bounded top-k alternative 在 selection 或 gate
  都不存在正 oracle opportunity。

### Phase 2: 纯 Jittor 高置信路由训练

- **Purpose**: 学习 medium/long 相对 short 的预期 reward，只在预测优势大
  且两路之间有 margin 时切换。
- **Entry condition**: Phase 1 通过。
- **Phase rules**:
  - `jt.nn.Module` 是唯一可训练模块；
  - fixed epochs，只使用 train；
  - selection 扫 variant 和 coverage；gate 不参与选择；
  - 不允许 candidate ID 或 candidate0/正例位置进入特征。
- **Todos**:
  - [x] 实现 permutation-invariant summary 与 Jittor reward MLP
    - **Surface**: router module
    - **Proof**: 置换测试、checkpoint replay
    - **Depends on**: Phase 1
  - [x] 训练 6 个固定 variant
    - **Surface**: remote CUDA artifacts
    - **Proof**: 每个 checkpoint 的框架审计和 selection metrics
    - **Depends on**: feature/target builder
  - [x] 冻结唯一 selection lock
    - **Surface**: `selection-lock.json`
    - **Proof**: selection delta > 0、coverage <= 5%
    - **Depends on**: 6 个 variant
- **Exit proof**: selection lock 在读取 gate 前落盘并带输入/checkpoint hash。
- **Stop condition**: 没有满足 selection 正收益和 coverage 约束的 variant。

### Phase 3: Unseen Gate 与交付

- **Purpose**: 判断高置信切换是否跨到更晚时间段仍稳定。
- **Entry condition**: selection lock 已冻结。
- **Phase rules**:
  - gate 只评估 lock 指定的一组参数；
  - gate 失败仍保存 rejected model，但不生成外部候选或提交；
  - 结果文档必须报告 oracle、实际覆盖、逐片收益和失败原因。
- **Todos**:
  - [x] 运行 unseen gate
    - **Surface**: `gate-report.json`
    - **Proof**: gate delta、coverage、三个时间子片
    - **Depends on**: selection lock
  - [x] 重载 checkpoint 并复放 logits
    - **Surface**: replay audit
    - **Proof**: 最大误差 `<= 1e-6`
    - **Depends on**: selected checkpoint
  - [x] 写结果与 TDD 证据
    - **Surface**: docs
    - **Proof**: 命令、指标、产物路径完整
    - **Depends on**: gate report
- **Exit proof**: accepted/rejected 结论明确，线上冠军和提交状态明确。
- **Stop condition**: gate 数组/时间边界与 frozen lock 不一致。

## Dry-Run Findings

- medium/long corrected logits 带有更旧 CST base，直接整路切换会混入 base
  老化；本轮改为只把 alternative residual 应用到 short base。
- 只有最后 40,196 行三路同时有效，因此不能再用原 fold2 整体做 gate；
  必须在该片内部按 timestamp 做 train/selection/gate。
- 高置信不能只解释为 softmax 大；冻结为“预测正 advantage + alternative
  margin + selection coverage threshold”三重条件。
- top-k 使用 default short scores 决定，避免 alternative 自己选择可修改范围。

## Final Validation

```bash
.venv/bin/python -m ruff check \
  src/jgrec/rankers/hybrid/high_confidence_topk_router.py \
  scripts/train_dataset2_high_confidence_topk_router.py \
  tests/test_hybrid_high_confidence_topk_router.py

.venv/bin/python -m pytest \
  tests/test_hybrid_high_confidence_topk_router.py \
  tests/test_hybrid_multi_horizon_oof.py \
  tests/test_hybrid_bounded_source_decoder.py -q
```

最终以 `gate-report.json` 和 checkpoint replay 为完成证据。

## First Execution Step

为 pure functions 写 RED：构造 medium/long alternative、生成无标签置换不变
summary、执行阈值 hard route，并验证 exact short fallback、top-k/cap。
