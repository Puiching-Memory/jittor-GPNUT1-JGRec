# Goal Document: Dataset2 short_none 50/40k Checkpoint Integration

## Go / No-Go

- **Judgment**: Go
- **Reason**: `short_none 50 epochs / 40000 train edges` 在匹配的
  200k-train/20k-validation 口径上达到 fixed-blend MRR `0.5484923183`，
  比冠军高 `+0.0015744999`；200k-edge 扩容已被受控实验否定，因此已有唯一候选。

## Target Outcome

生成一个独立、可加载、可回放的双数据集 checkpoint：Dataset1 完全继承当前冠军，
Dataset2 的 `gnn_short` 使用胜出的 50/40k 无权重短窗表示，并由同分布训练特征重训
融合头；加载该 checkpoint 能稳定生成完整 Dataset2 预测，且不覆盖冠军文件。

## Goal Definition

- **Type**: technical / quality / delivery
- **Boundary**:
  - 保留已经是 50/40k 的 Dataset2 encoder，只安装依赖该列训练出的融合状态。
  - 复用已验证的 seed-60、200k 训练缓存、20k 验证缓存、完整 100 候选协议。
  - 将 Dataset2 候选状态写入独立 checkpoint，再与原冠军 Dataset1 组合。
  - 用 checkpoint load/replay 生成并验证完整 Dataset2 CSV。
- **Non-goals**:
  - 不增加 GNN 边数，不改网络结构、边权、损失或其他塔。
  - 不接入 listwise GNN；其 validation-only 与 OOF 融合均已回归。
  - 不同时修复 full-refit 分布漂移，不修改全局默认值。
  - 不提交比赛平台。
- **Deferred work**:
  - 基于 checkpoint 的部分混合权重扫描。
  - Dataset1+Dataset2 最终拼包与线上提交。
  - full-refit 后融合头再校准。
- **Verification rule**:
  1. 在发布 checkpoint 前，持久化融合头必须在同一 50/40k 分数与缓存口径上
     重新通过 fixed-blend full/三切片 gate。
  2. checkpoint 合同测试必须证明只替换 Dataset2 目标状态，Dataset1 继承不变。
  3. 从新 checkpoint 加载生成的 Dataset2 CSV 必须通过行数、列数、有限值、范围、
     排名确定性和重复回放一致性检查。
- **Evidence source**: RED/GREEN 测试、缓存/模型/checkpoint SHA-256、MRR 报告、
  checkpoint state diff、加载回放日志和 CSV 验证报告。
- **Pass criteria**:
  - 持久化融合头相对原基线 full MRR 至少 `+0.001`，三个切片均不回归。
  - 新 checkpoint 可由标准 loader 读取，包含 dataset1/dataset2 完整状态。
  - Dataset1 状态来源保持当前冠军；Dataset2 encoder 和非融合状态不发生变化。
  - Dataset2 完整 CSV 行数为 153420、每行 100 列、全部分数有限且位于 `[0,1]`。
  - 两次加载回放候选排序一致；任何一项失败则不生成候选包。
- **Confidence note**: 本地验证证明的是同一离线切分上的融合价值；最终全量 encoder
  与训练期 encoder 仍可能存在分布漂移，因此本目标只交付可审计 checkpoint，
  不授权线上提交。
- **Judgment owner**: 自动 checkpoint/CSV 合同决定工程完成；离线 MRR gate 决定候选
  是否允许发布；线上效果不在本目标内声明。

## Current State

- 当前冠军 checkpoint：
  `checkpoints/d1_champion_d2_lgbm_lr003_pseudob_seed60_20260722.pkl`。
- 胜出对照由 targeted-GNN 实验生成：
  `short_none 50/40k`，fixed-blend full MRR `0.5484923183`。
- 现有实验保存了 train/validation `gnn_short` score arrays 和哈希；源 checkpoint
  的最终 GraphTower 已经是对应的 50/40k 服务状态。
- 200k-edge 候选 full MRR `0.5475115740`，四项匹配指标全部低于 40k，已拒绝。
- 工作树包含大量用户实验改动；只允许触碰本目标的新文件及明确的 checkpoint
  组装边界。

## Plan Rewrite Notes

| Existing item | Decision | Reason |
|---|---|---|
| 继续扩大 GNN 容量 | remove | 200k edges 已在全部匹配指标上回归 |
| `short_none 50/40k` 离线结果 | keep | 是当前唯一通过收益证据的候选 |
| 直接把验证列替换进旧融合头 | remove | 会制造训练/服务表示不一致 |
| 同分布重训融合 | keep | 是正式接入的必要条件 |
| 覆盖原冠军 checkpoint | remove | 必须保留可回滚基线 |
| 生成提交并上线 | defer | checkpoint 回放与漂移风险尚未完成外部 gate |

## Drift Diagnosis

- **Goal drift**: 容量研究已经结束；本轮目标是可加载交付物，不再做超参搜索。
- **Phase drift**: 不能先拼 checkpoint 后验证，必须先证明状态替换合同和缓存复现。
- **Validation drift**: 文件存在不等于集成成功，必须加载并生成完整 CSV。
- **Compatibility drift**: Dataset1 与 Dataset2 非目标状态必须继承，不能引入双路径。
- **Cleanup drift**: 默认值、旧实验脚本和 unrelated dirty files 不纳入本轮。

## Priority Rationale

- 最危险的不是训练失败，而是 checkpoint 看似可加载却混入错误 encoder/fusion 状态；
  因此先做状态边界审计与合同测试。
- 先复现离线胜出指标，再付出全量 encoder/CSV 回放成本。
- 组合双数据集 checkpoint 放在 Dataset2 候选验证之后，确保失败不会污染冠军。

## Assumptions and Open Decisions

| Item | Status | Impact | Owner / Next step |
|---|---|---|---|
| checkpoint 的 GraphTower 已经是 50/40k | confirmed | encoder 无需改动 | Phase 1 state/hash 审计 |
| 50/40k train/val score arrays 仍在远端且哈希匹配 | confirmed | 可避免重复训练图模型 | Phase 1 SHA-256 校验 |
| 现有 checkpoint 组装器可继承 Dataset1 | confirmed | 支持安全双数据集发布 | 使用 `compose_checkpoint_datasets` 或等价标准 writer |
| 最终 test-time `gnn_short` 必须全量重训 | rejected | 会制造无必要漂移 | 保留源 checkpoint encoder |
| 完整 Dataset2 回放资源充足 | assumed | 决定能否完成交付 | 远端 preflight 后确认 |

## Phases

### Phase 1: 冻结 checkpoint 状态边界

- **Purpose**: 确定 Dataset2 checkpoint 中 encoder、GraphTower、fusion、LGBM 和元数据
  的实际 schema，以及最小合法替换集。
- **Entry condition**: 本目标文档已写入。
- **Phase rules**:
  - 只读检查 checkpoint 与已有脚本，不修改生产状态。
  - 不通过 pickle 私有细节猜测；以标准 save/load round-trip 为准。
  - 发现 `gnn_short` 无法独立替换时，必须明确扩大到“完整 Dataset2 encoder +
    匹配融合”，不能伪装成单窗口替换。
- **Todos**:
  - [ ] 审计冠军 Dataset2 state schema 和 GraphTower 序列化边界。
    - **Surface**: checkpoint / ranker state / graph state
    - **Proof**: 字段清单、目标/非目标状态分类和最小替换方案。
    - **Depends on**: none
  - [ ] 校验 50/40k score arrays、源缓存和报告哈希。
    - **Surface**: remote artifacts
    - **Proof**: 哈希与原报告一致。
    - **Depends on**: none
- **Exit proof**: 一份无歧义的 checkpoint 变更合同和可执行构建命令。
- **Stop condition**: 胜出 artifact 缺失/哈希漂移，或无法隔离 Dataset2 状态。

### Phase 2: TDD 固化候选状态合同

- **Purpose**: 在真实构建前证明目标状态能安全组装、非目标状态不会漂移。
- **Entry condition**: Phase 1 已确定 schema。
- **Phase rules**:
  - RED 必须先因缺少候选组装行为失败。
  - GREEN 只实现最小状态替换与验证 API。
  - 不启动长训练，直到 focused tests 与既有 checkpoint tests 通过。
- **Todos**:
  - [ ] 增加 Dataset2 GNN/fusion 候选状态的 RED 合同测试。
    - **Surface**: tests / checkpoint helper
    - **Proof**: 目标缺失导致的明确 RED。
    - **Depends on**: Phase 1
  - [ ] 实现最小组装与 state-diff 校验。
    - **Surface**: checkpoint integration helper
    - **Proof**: focused GREEN + checkpoint regression tests。
    - **Depends on**: RED
- **Exit proof**: loader round-trip、Dataset1 继承和 Dataset2 允许变更集测试通过。
- **Stop condition**: 合法 checkpoint 必须修改未授权的其他塔或公共格式。

### Phase 3: 构建匹配 Dataset2 候选

- **Purpose**: 复用 50/40k 分数重训并持久化同分布融合头，保持服务 encoder 不变。
- **Entry condition**: Phase 2 全绿，远端 preflight 通过。
- **Phase rules**:
  - seed、split、候选、其他特征和融合口径必须与胜出实验一致。
  - 训练日志与最终解析配置必须落盘。
  - 持久化融合头未重新通过 full/三切片 gate 前不得写最终候选 checkpoint。
- **Todos**:
  - [ ] 复现/复用 train/validation 50/40k 特征并重训融合。
    - **Surface**: remote cache / fusion model
    - **Proof**: score SHA 一致，full 增益至少 `+0.001` 且三切片不回归。
    - **Depends on**: Phase 2
  - [ ] 保留最终 test-time `gnn_short` 并构建 Dataset2 state。
    - **Surface**: graph state / dataset checkpoint
    - **Proof**: 模型哈希、状态 diff、标准 loader round-trip。
    - **Depends on**: MRR reproduction
- **Exit proof**: 独立 Dataset2 候选 checkpoint 可加载且状态合同通过。
- **Stop condition**: 指标不复现、非目标状态漂移、OOM/非有限值或模型状态不完整。

### Phase 4: 组合与加载回放

- **Purpose**: 交付安全的双数据集 checkpoint 和完整 Dataset2 预测证据。
- **Entry condition**: Phase 3 Dataset2 候选通过。
- **Phase rules**:
  - 输出到新路径，禁止覆盖冠军。
  - Dataset1 从冠军原样继承。
  - 不提交线上。
- **Todos**:
  - [ ] 组合 Dataset1 冠军 + Dataset2 候选 checkpoint。
    - **Surface**: final candidate checkpoint
    - **Proof**: metadata/dataset 完整性、SHA-256、state 来源报告。
    - **Depends on**: Phase 3
  - [ ] 两次加载回放完整 Dataset2 CSV 并验证。
    - **Surface**: result CSV / replay logs
    - **Proof**: 153420×100、有限范围、排序一致、无并列异常增长。
    - **Depends on**: combined checkpoint
- **Exit proof**: checkpoint、回放 CSV、验证报告和回滚路径全部存在。
- **Stop condition**: 两次回放排序不一致、CSV 合同失败或 checkpoint 加载失败。

## Dry-Run Findings

- 现有 50/40k targeted 实验只缺少融合头权重；最终服务 GraphTower 已在源
  checkpoint 中，正式接入不得重建 encoder。
- full-refit 可能改变特征分布；因此融合重训与最终服务状态必须在报告中明确区分，
  若无法同口径则停止而不是静默发布。
- 双数据集组合已有标准 helper，应复用它而不是手写 pickle。
- 线上提交不是 checkpoint 工程完成的必要条件，且当前没有授权。

## Final Validation

Focused RED/GREEN 测试、既有 checkpoint 回归测试、远端 MRR gate、state-diff
报告、标准 loader round-trip、双数据集 metadata 校验、两次完整 Dataset2 CSV 回放
一致性和所有产物 SHA-256。

## First Execution Step

只读审计冠军 Dataset2 checkpoint state、HybridRanker 的 snapshot/load 边界，以及
50/40k 远端 score artifacts，确定最小合法替换集。

## Execution Update

- 源 Dataset2 checkpoint 的服务 encoder 已经是
  `short_none / 50 epochs / 40000 max_train_edges`；正式接入不需要、也不允许
  重训或替换 encoder。
- 原实验只保存了训练/验证分数，没有保存进程内 Setwise 权重。正式重训因此以
  “同一分数 SHA + 重新通过 full/三切片 gate”为发布条件，不再假定能逐位复原
  已丢失的旧权重。
- 持久化后的 Setwise + LightGBM `0.80/0.20` 融合达到 full MRR
  `0.5485470649`，相对原基线为 `+0.0016292464`，三个切片均为正增益。
- 候选 checkpoint 从当前 `gamma=0.5` Dataset1 checkpoint 继承 Dataset1；
  Dataset2 只允许变化 `lgbm_result`、`setwise_fusion_state`、
  `setwise_fusion_result`、`setwise_hidden_dim`。
- 输出 checkpoint 已通过标准 loader/hydrate；完整 Dataset2 双回放仍是最终
  交付 gate。
