# Goal Document: Tower Optimization and In-Batch Negatives

## Go / No-Go

- **Judgment**: Go
- **Reason**: 四个学习型塔当前都使用固定学习率；虽然 Adam 已接收
  `weight_decay`，但生产配置把它们与融合头绑定在同一个默认值 `0.0`。双塔只有逐行
  显式负样本，没有 batch 内检索约束。两项改动都能先以关闭默认值实现并独立消融，
  不需要改动当前冠军。

## Target Outcome

为 GNN、GRU sequence、two-tower、source-profile item2vec 提供可审计的逐 epoch
学习率调度和塔级 weight decay；为 two-tower 提供防重复正例误杀的 in-batch
negative 辅助目标。完成本地行为验证后，在固定 Dataset2 时间切分上做 2×2
消融；只有非 MRR 单指标、且跨折稳定的候选才允许进入 external 或提交包。

## Goal Definition

- **Type**: technical / learning / quality
- **Boundary**:
  - 四个学习型 hybrid 塔的 optimizer 配置与训练循环；
  - two-tower 训练期 in-batch 辅助损失；
  - CLI、checkpoint 配置兼容、单元测试和冻结实验 runner；
  - 低成本固定切分筛选，必要时三折最终集成验证。
- **Non-goals**:
  - 不修改 FusionMLP、LGBM、Setwise 的优化器；
  - 不把新参数直接写入当前冠军 checkpoint；
  - 不使用 external/线上成绩反扫 scheduler、weight decay 或 in-batch 权重；
  - 不同时更改塔结构、embedding 维度、边数或训练窗口。
- **Deferred work**:
  - 更复杂的 warmup、plateau scheduler 或逐层 weight decay；
  - memory-bank negatives、跨 batch negatives；
  - 未通过 rolling gate 时的提交包。
- **Verification rule**:
  - 自动测试证明 schedule、旧配置兼容、多正例 in-batch mask 和训练开关；
  - 冻结 2×2 候选只在同一时间数据、seed、结构和负采样下比较；
  - 单切分仅淘汰，晋级必须通过最终集成后的 rolling-origin 多折硬门禁。
- **Evidence source**:
  - pytest RED/GREEN 记录；
  - frozen config、训练日志、逐塔诊断和多指标 ranking report；
  - rolling-origin selection report。
- **Pass criteria**:
  - 本地测试与 Ruff 通过；
  - 旧 checkpoint 缺少新字段时解析为 `constant / 0 / disabled`；
  - in-batch 对同一 destination 的重复正例使用 multi-positive softmax，不能互相当负例；
  - Stage 1 候选至少不降低 MRR、Hit@1/3/10、NDCG@10，不恶化平均排名，
    且 improved queries 多于 worsened，才可进入多折；
  - rolling 每折 MRR/NDCG@10 非负，六项均值门禁与 query movement 通过。
- **Confidence note**: 单元测试只能证明训练契约；真实收益必须由时间外推验证。
  单切分结果不具备晋级权。
- **Judgment owner**: 自动测试决定实现完成；冻结的多指标 rolling gate 决定实验晋级。

## Current State

- `GraphTower`、`GRUSequenceTower`、`TwoTower`、`SourceProfileTower` 都使用
  `jt.nn.Adam` 和固定 `lr`。
- 四个塔配置已有 `weight_decay` 字段，但 `TrainingConfig` 把它们绑定到融合头的
  全局 `weight_decay=0.0`，不能独立实验。
- GNN 有独立 `gnn_lr`；其余三个塔复用融合头 `lr`。
- TwoTower 使用显式混合负样本，支持 BCE/listwise，但没有 batch 内负样本。
- 当前工作树包含既有实验改动；本任务只能做局部增量，不能清理或覆盖无关文件。

## Priority Rationale

- 先把兼容和损失语义用测试锁死，再接训练循环，避免远端长跑后才发现 false
  negatives 或旧 checkpoint 回放破坏。
- 先做 2×2 因子消融，避免把 scheduler/weight decay 与 in-batch 收益混为一谈。
- Stage 1 只淘汰；把昂贵的最终集成多折留给最多一个预登记候选。

## Assumptions and Open Decisions

| Item | Status | Impact | Owner / Next step |
|---|---|---|---|
| scheduler 候选使用 cosine，末端 LR 比例 0.1 | assumed | 控制候选空间，避免扫描 | frozen config |
| 塔级 weight decay 候选使用 `1e-4` | assumed | 常见的小幅正则；不改冠军默认值 | Stage 1 淘汰 |
| in-batch 辅助权重为 1.0、温度为 1.0 | assumed | 只保留一个预登记强度 | Stage 1 淘汰 |
| in-batch destination 使用 id-only/零上下文表示 | assumed | 避免把另一事件的未来上下文借给当前 query | 行为测试与报告 |
| 重复 destination 是多正例，不互相作为负例 | confirmed | 防 false negative | 单元测试 |
| 四塔联合优化是否值得跑完整多折 | unresolved | 取决于 Stage 1 全指标 | rolling gate |

## Phases

### Phase 1: Freeze optimizer contracts

- **Purpose**: 建立共享、可复现、旧 checkpoint 安全的塔级优化配置。
- **Entry condition**: 当前四个训练循环和配置映射已审计。
- **Phase rules**:
  - 先写失败测试；
  - 旧实例缺少字段时必须保持 constant LR、weight decay 0；
  - 新策略默认关闭，未验证前不改变冠军复现。
- **Todos**:
  - [ ] 定义 constant/cosine epoch LR 语义。
    - **Surface**: common optimization helper + tests
    - **Proof**: 首尾 LR 和非法参数测试
    - **Depends on**: none
  - [ ] 增加四塔独立 LR、schedule、minimum ratio、weight decay 配置。
    - **Surface**: hybrid config、CLI、config tests
    - **Proof**: CLI/config mapping + legacy fallback tests
    - **Depends on**: scheduler contract
  - [ ] 四个训练循环逐 epoch 应用并记录 LR。
    - **Surface**: GNN/sequence/two-tower/source-profile
    - **Proof**: optimizer helper tests + training smoke
    - **Depends on**: config mapping
- **Exit proof**: 相关配置和优化 helper 测试通过。
- **Stop condition**: Jittor optimizer 无法安全更新 epoch LR，或旧 checkpoint 行为改变。

### Phase 2: Add safe two-tower in-batch objective

- **Purpose**: 在不增加采样存储的情况下扩大双塔检索对比集合。
- **Entry condition**: Phase 1 green。
- **Phase rules**:
  - 先写 multi-positive RED；
  - 只在训练损失中加入辅助项，验证 MRR 仍基于原始完整候选组；
  - 同 destination 重复行不得互相作为负例；
  - 关闭开关时数值路径与旧实现一致。
- **Todos**:
  - [ ] 定义 multi-positive in-batch softmax。
    - **Surface**: two_tower loss + tests
    - **Proof**: NumPy 参考、重复 destination、单行边界
    - **Depends on**: none
  - [ ] 接入 id-only destination 表示和配置开关。
    - **Surface**: two_tower trainer/config/CLI
    - **Proof**: disabled parity、enabled gradient smoke
    - **Depends on**: loss contract
- **Exit proof**: TwoTower 精确单测及小数据训练通过。
- **Stop condition**: 辅助损失产生 NaN、单行 batch 无合法负例、或显著扩大显存不可控。

### Phase 3: Freeze and run the 2×2 screen

- **Purpose**: 独立判断优化策略与 in-batch 是否有继续投入价值。
- **Entry condition**: 全部本地回归 green，远端 GPU 无其他计算进程。
- **Phase rules**:
  - 固定 control、optimizer-only、inbatch-only、combined；
  - 结构、seed、训练行、负样本和最终评分路径一致；
  - 单切分只淘汰，不生成提交包；
  - 使用低优先级和资源碰撞 watchdog。
- **Todos**:
  - [ ] 生成 frozen config 和运行前哈希。
    - **Surface**: experiment runner/artifacts
    - **Proof**: preflight report
    - **Depends on**: Phase 2
  - [ ] 运行固定时间切分并报告完整指标。
    - **Surface**: remote CUDA run
    - **Proof**: MRR、Hit@1/3/10、NDCG@10、mean rank、query movements
    - **Depends on**: idle resource gate
- **Exit proof**: 四臂报告完整，唯一候选晋级或全部停止。
- **Stop condition**: 外部 GPU 任务出现、候选配置漂移、或 control 无法复现。

### Phase 4: Exact integrated rolling gate

- **Purpose**: 判断 Stage 1 唯一候选能否进入真正生产路径。
- **Entry condition**: 恰好一个候选通过 Stage 1 全指标门槛。
- **Phase rules**:
  - 至少三折 rolling-origin 等权平均；
  - 每折使用最终集成后的精确分数；
  - 不读取 reserved fold/external，直到 selection lock 存在。
- **Todos**:
  - [ ] 运行三折最终集成 control/candidate。
    - **Surface**: standard validation protocol
    - **Proof**: selection report + lock or rejection
    - **Depends on**: Phase 3 pass
- **Exit proof**: 标准 selector 接受或拒绝。
- **Stop condition**: 任一折 MRR/NDCG@10 为负或多指标均值门禁失败。

## Dry-Run Findings

- 直接把全局 `weight_decay` 改成非零会同时改变融合头，无法归因；必须拆成塔级字段。
- TwoTower 的 destination context 与事件时间绑定，直接用其他行完整 destination
  context 做 in-batch negatives 会引入时间语义混杂；辅助目标改用 destination ID 与
  零上下文表示。
- batch 中可能有重复正例 destination；普通 diagonal cross-entropy 会制造 false
  negatives，因此必须用 multi-positive log-softmax。
- 四塔联合多折成本高，先做 factorized screen 可以在不牺牲选择纪律的前提下止损。

## Final Validation

- `uv run pytest` 的相关 optimizer/config/tower/CLI/checkpoint 测试；
- `uv run ruff check` 覆盖所有改动文件；
- CUDA 小数据训练 smoke；
- Stage 1 frozen report；
- 若晋级，标准三折 rolling selection report。

## First Execution Step

为共享 epoch LR 计算和 four-tower config legacy fallback 写 RED 测试。
