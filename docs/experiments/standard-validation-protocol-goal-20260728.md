# Goal Document: 标准化 rolling-origin 选择与长跨度 external gate

## Go / No-Go

- **Judgment**: Go
- **Reason**: 已有 rolling-origin、external-20k 和候选级 gate 的可复用基础；本轮可在不读取 external 标签、不运行远端训练、不生成提交包的前提下，先完成标准协议、调用契约、测试和本地 preflight。

## Target Outcome

项目形成一套候选无关、默认可复用的验证协议：特征组和 ensemble 权重必须由多个 causal rolling-origin 选择折的聚合表现与跨折稳定性共同决定；权重锁定后，长跨度 external holdout 只允许打开一次，并作为生成提交候选的硬门禁。所有选择、锁定、external 打开和授权产物都可追溯且能阻止单折选优、external 反向调参和结果串用。

## Goal Definition

- **Type**: quality / operational
- **Boundary**: 本轮包括标准协议的数据契约、聚合和硬门禁逻辑、配置/CLI 接线、一次性 external 状态机、产物哈希绑定、无标签本地 preflight、单元/集成测试和运行手册；不包括远端模型训练、读取 external 标签、线上提交。
- **Non-goals**:
  - 不为某一个候选扫描最优阈值或权重。
  - 不修改模型结构、候选生成或特征定义。
  - 不用单次 external 或 leaderboard 结果反向改变已锁定配置。
- **Deferred work**:
  - 服务器恢复后的正式 rolling-origin 训练和评分。
  - 权重锁定后的 external-20k 一次性评估。
  - external gate 通过后的 checkpoint、replay 和提交包生成。
- **Verification rule**: 自动化测试必须证明多折聚合参与选择、稳定性为硬门禁、gate 折不参与选择、external 在锁定后且仅能打开一次、external 产物与选择锁哈希绑定；本地 preflight 只能检查元数据和文件谱系，不能打开 external 特征或标签数组。
- **Evidence source**: RED/GREEN 测试记录、pytest、Ruff、CLI smoke、preflight JSON、运行手册。
- **Pass criteria**: 目标测试和相关回归测试全绿；Ruff 通过；本地 preflight 状态为 `ready_for_remote_rolling`；没有 external open receipt；没有获准提交包；标准 runner 不允许单折选择。
- **Confidence note**: 自动化测试可高置信验证协议和防泄漏行为；真实泛化效果仍必须由服务器端 rolling-origin 与一次性 external holdout 给出，不能由本地工程完成代替。
- **Judgment owner**: 本地工程完成由测试和 preflight 判定；正式候选是否可打包由 external gate 产物判定；是否提交线上由用户决定。

## Current State

- 已有候选专用 `robust_weight_selection`、Dataset1 Setwise-context rolling/external runner、一次性 external receipt 和哈希绑定，可作为标准化输入。
- 当前新增链路针对单一候选，尚未证明所有特征组和 ensemble 权重选择都会强制走同一协议。
- 历史 rolling 折跨度约 31–36 天，而部署跨度约 468 天；只看短窗或单切分可能偏爱会快速过期的静态信号。
- 服务器暂时不可用，因此本轮只能完成不依赖远端训练和 external 标签的工程与验证。

## Plan Rewrite Notes

| Existing item | Decision | Reason |
|---------------|----------|--------|
| rolling-origin 作为事后压测 | rewrite | 升级为选择阶段的必经输入，禁止单折直接锁权重 |
| 单切分选择特征组和 ensemble 权重 | remove | 与目标冲突，且无法证明跨时间稳定性 |
| external-20k 长跨度验证 | keep | 升级为锁权重后的一次性标准 gate |
| 候选专用 Setwise-context gate | merge | 保留精确候选 runner，但复用统一协议核心和产物 schema |
| leaderboard 后扫描混合权重 | remove | 构成 holdout/leaderboard 过拟合 |

## Drift Diagnosis

- **Goal drift**: 过去把“某折 MRR 最高”当作候选选择目标，偏离“跨时间稳定且可部署”的目标。
- **Phase drift**: rolling-origin 被放在候选选定之后，无法约束特征组和融合权重的选择。
- **Validation drift**: 只记录单折或单指标提升，缺少 Hit@K、NDCG、平均排名、改善/恶化 query 数和跨折硬门禁。
- **Compatibility drift**: 候选专用 manifest 与通用选择逻辑边界不清，容易出现绕过标准 gate 的平行路径。
- **Cleanup drift**: 本轮不清理历史实验脚本，只约束新的标准入口和现有正式候选链路。

## Priority Rationale

- 先定义标准产物和失败模式，再接 runner，避免把候选专用实现直接复制成另一套协议。
- 优先封死数据泄漏、单折选优和 external 反向调参；这些风险比增加一个指标更可能造成“本地赢、线上输”。

## Assumptions and Open Decisions

| Item | Status | Impact | Owner / Next step |
|------|--------|--------|-------------------|
| 选择折至少 3 个，另保留不参与选择的 gate 折 | confirmed | 确保平均值和稳定性有最低样本基础 | 由协议测试固化 |
| 指标集合为 MRR、Hit@1/3/10、NDCG@10、平均排名、改善/恶化 query 数 | confirmed | 避免只按 MRR 过拟合 | 由标准 schema 固化 |
| 权重/特征候选必须在跑分前声明 | confirmed | 防止看结果后扩展搜索空间 | 由 candidate-space 哈希绑定 |
| external gate 的具体非劣阈值可按实验预注册 | assumed | 不同数据集可能需要不同容忍度 | schema 要求阈值在 external 打开前写入锁文件 |
| 现有历史脚本是否全部迁移 | deferred | 不影响新协议本地完成 | 后续按正式实验逐个迁移 |

## Phases

### Phase 1: 冻结标准协议与产物边界

- **Purpose**: 明确选择折、gate 折、指标、硬门禁、一次性 external 和哈希谱系。
- **Entry condition**: 用户已确认升级方向和本地完成范围。
- **Phase rules**:
  - 只写目标、测试契约和 schema 约束，不读取 external 数组。
  - 标准协议不得内置某个模型或某个权重。
  - 每个可调决定必须在 external 打开前冻结并可哈希。
- **Todos**:
  - [x] 审计现有 rolling/external 核心和调用点。
    - **Surface**: source / scripts / tests / docs
    - **Proof**: 调用图和差距记录
    - **Depends on**: none
  - [x] 用 RED 测试定义标准选择锁和 external 状态机。
    - **Surface**: tests
    - **Proof**: 因缺少目标 API/约束而失败
    - **Depends on**: 审计
- **Exit proof**: RED 失败原因准确指向缺少标准协议行为。
- **Stop condition**: 如果现有 external runner 在选择锁前已读取 external 标签，先停止并修复泄漏边界。

### Phase 2: 实现多折选择与稳定性硬门禁

- **Purpose**: 让多个 causal selection folds 的聚合和稳定性成为唯一锁定候选的标准入口。
- **Entry condition**: Phase 1 RED 已确认。
- **Phase rules**:
  - 至少 3 个 selection folds。
  - 保留 fold 不参与调参，只参与后续 gate。
  - 不允许只按 MRR；标准指标必须完整。
  - 候选空间、fold manifest、指标协议和阈值全部哈希绑定。
- **Todos**:
  - [x] 实现候选无关的 rolling selection manifest 与 lock。
    - **Surface**: source
    - **Proof**: 单元测试
    - **Depends on**: Phase 1
  - [x] 把正式候选训练与选择入口接到标准协议。
    - **Surface**: scripts
    - **Proof**: 集成测试 / CLI smoke
    - **Depends on**: 标准入口
- **Exit proof**: 单折胜但跨折不稳的候选被拒；跨折稳定候选才能生成不可变选择锁。
- **Stop condition**: 任一正式路径仍能绕过标准选择锁直接进入 external。

### Phase 3: 实现长跨度一次性 external gate

- **Purpose**: 权重锁定后只打开一次 external，且结果不能被用于再次扫描。
- **Entry condition**: 标准选择锁已生成并含完整哈希。
- **Phase rules**:
  - external 路径、时间边界和 gate 阈值在打开前冻结。
  - 先原子写 open receipt，再读取 external 数据。
  - 同一锁、同一 external lineage 只允许一次评估。
  - external 结果不提供重新选择候选的 API。
- **Todos**:
  - [x] 实现标准 external gate request/receipt/result。
    - **Surface**: source / tests
    - **Proof**: 重开、错锁、错 lineage 均失败
    - **Depends on**: Phase 2
  - [x] 接入冻结候选 runner 配置和 package authorization。
    - **Surface**: scripts
    - **Proof**: 未通过 gate 时授权为 false
    - **Depends on**: external gate
- **Exit proof**: 测试证明打开顺序、单次性和全链路绑定。
- **Stop condition**: 发现 external 数据可在 receipt 前被加载或哈希后补写。

### Phase 4: 本地 preflight、文档与回归

- **Purpose**: 在服务器不可用时证明本地协议已可执行且没有提前消费 holdout。
- **Entry condition**: Phase 2–3 全绿。
- **Phase rules**:
  - preflight 只能检查 manifest、时间边界、文件大小/路径和预登记哈希；不得加载 external 特征或标签内容。
  - 不生成 authorized checkpoint 或 submission package。
- **Todos**:
  - [x] 运行目标测试、相关回归、Ruff 和 CLI smoke。
    - **Surface**: test / lint / CLI
    - **Proof**: 命令输出
    - **Depends on**: Phase 3
  - [x] 输出 preflight JSON、运行手册和结果文档。
    - **Surface**: artifacts / docs
    - **Proof**: `ready_for_remote_rolling`、`external_opened=false`、`package_authorized=false`
    - **Depends on**: 全部验证
- **Exit proof**: 本地证据齐全，服务器恢复后可按固定命令继续。
- **Stop condition**: preflight 意外打开 external 数组、发现时间穿越或产物哈希不一致。

## Dry-Run Findings

- 现有 Setwise-context runner 已包含三折选择、第四折 gate、external open receipt 和哈希绑定，适合作为首个迁移方，但不能直接把候选名写进标准核心。
- “多折平均”不能只实现为平均 MRR；必须保留逐折全指标，稳定性规则需在选择前冻结。
- 为避免 external 反向扫描，选择锁需绑定候选空间，而不仅是最终权重；否则可以在 external 后声称是“新候选”继续试。
- 本地 preflight 必须沿用不打开 external 数组的设计；内容哈希应来自预先存在的 lineage metadata，而不是本轮重新读取数据计算。

## Final Validation

- 目标协议和回归 pytest 全绿。
- Ruff 对新增/修改 Python 文件全绿。
- 标准 CLI `--help` smoke 全绿。
- 本地 preflight JSON 明确报告：多折结构有效、长跨度边界有效、external 未打开、未授权打包。
- TDD 证据文档包含每个行为切片真实的 RED、GREEN、REFACTOR 命令和结果。

## First Execution Step

审计现有 `robust_weight_selection`、rolling/external runner 与测试，识别可提升为候选无关标准核心的接口及所有可能绕过点。

## Execution Result

- **Local judgment**: complete；真实 rolling/external 仍为 remote pending。
- 已新增候选无关的标准 plan / selection / external 协议和三条执行 CLI。
- Hybrid 已能冻结单个 feature mask 与 ensemble 权重，正式 rolling 不再需要折内单切分扫描。
- 多折选择使用等权平均 MRR 作为第一排序条件，跨折稳定性是进入排序前的硬门禁。
- external 长跨度按 `score_time_max - training_time_max` 校验；本地元数据确认 Dataset2 为精确 468 天。
- 相关回归 `69 passed`，Ruff 和 CLI smoke 通过。
- 本地 preflight 未读取 external 数组，未创建真实 selection lock、external receipt 或提交包；`package_authorized=false`。
