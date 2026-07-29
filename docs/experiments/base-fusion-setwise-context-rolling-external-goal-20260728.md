# Goal Document: 基础融合 Setwise Context 的 Rolling-Origin 与 External Gate

## Go / No-Go

- **Judgment**: Go
- **Reason**: 基础三通道融合已完成训练、推理、服务与 checkpoint 接入；
  Dataset1 已有 rolling-origin 基础设施和缓存，当前只缺“最终集成路径 v0 对照
  vs v1 候选”的严格时间验证。候选参数可以在读取指标前完全冻结。

## Target Outcome

判断基础 `FusionMLP` 的 v1 三通道输入是否能稳定替换 Dataset1 当前 raw v0
基础头，同时保持现有 time-ramp Setwise `γ=0.5` 集成语义不变。先执行至少
三个 rolling-origin 折；只有跨折硬门禁通过，才锁定唯一候选并一次性评估
长跨度 external holdout。rolling 和 external 均通过后才生成提交包，否则停止。

## Goal Definition

- **Type**: learning / quality / delivery
- **Boundary**:
  - Dataset1 最终集成路径的 raw v0 对照与 context v1 候选。
  - 每折重新训练基础头；time-ramp Setwise 专家、`γ=0.5`、seed、特征布局、
    训练规模和候选顺序保持一致。
  - 报告 MRR、Hit@1/3/10、NDCG@10、平均排名及 improved/worsened/tied query。
  - rolling selection lock、external one-shot receipt，以及通过后提交包。
  - Dataset2 预测保持当前冠军，不参与本轮选择。
- **Non-goals**:
  - 不扫描 context transform v2、`γ`、hidden dim、epoch、seed 或训练窗口。
  - 不重训 GNN、Two-Tower、Setwise 专家或修改 Dataset2。
  - 不根据 external 或 leaderboard 结果回调参数。
  - 不把 standalone 基础头指标冒充最终 time-ramp 集成指标。
- **Deferred work**:
  - context v1 与独立 Setwise 的新融合权重搜索。
  - Dataset2 基础头替换；当前 Dataset2 由 Setwise champion 主导，基础头不是
    本轮高杠杆路径。
- **Verification rule**:
  1. 至少三个时间递进、训练严格早于评分区间的 rolling 折。
  2. 每折 v0/v1 只允许 context transform 不同；最终分数均经过冻结的
     Dataset1 time-ramp `γ=0.5` 路径。
  3. 每折 MRR 与 NDCG@10 均不得下降；pooled Hit@1/3/10 不得下降、平均排名
     不得变差、improved query 必须多于 worsened query。
  4. rolling 全过后写入唯一 selection lock；未过则不读取 external。
  5. external 必须绑定 lock、相同 candidate fingerprint 与唯一候选，且只能
     开启一次；MRR、NDCG@10、Hit@1/3/10、平均排名均不得下降，improved 必须
     多于 worsened。
  6. 只有 external 通过才允许组合当前 Dataset2 冠军并生成提交包。
- **Evidence source**: rolling manifest、每折完整分数矩阵、多指标报告、
  selection lock SHA-256、external receipt/report、提交包 manifest/hash、
  focused tests、Ruff。
- **Pass criteria**: 上述六条全部满足；任一硬门禁失败即拒绝候选且不生成包。
- **Confidence note**: rolling 折提供局部时间稳定性，长跨度 external 提供部署
  距离证据；历史 external 已被其他候选使用，因此它是工程 gate 而非统计上
  从未暴露的盲集。线上效果仍只能由用户提交后确认。
- **Judgment owner**: rolling/external 状态机机械决定是否通过；用户决定是否
  将生成的包提交 leaderboard。

## Current State

- 基础 FusionMLP v1 已实现为 raw、减行均值、减行最大值三通道，默认启用。
- 旧 checkpoint 缺新字段时保持 v0，允许精确构造对照。
- Dataset1 当前冠军通过 time-ramp 以 `γ=0.5` 接入 Setwise；本轮必须在这条
  最终路径上比较，不能只比较 standalone 基础头。
- 当前冠军基础主干实际为 `FusionMLP + LGBM`；rolling 必须为每折重训一个
  共享 LGBM，external 必须复用冠军 LGBM。只比较 MLP 会遗漏真实排序交互。
- Dataset1 已有 recent-200k 缓存、rolling manifest、训练与 gate 脚本，但原
  实验针对 raw-vs-Setwise，不等价于 v0-vs-v1 基础头。
- 当前工作树包含大量既有实验资产；本轮只新增/修改本目标直接需要的 runner、
  tests、result artifacts 和文档。
- CUDA 远端此前无法返回 SSH banner；本轮先做资产 preflight，若真实训练只能
  在远端执行，则恢复连接后继续，不用假数据替代。
- 本地可完成部分已经封口：真实 manifest/cache-report/champion ZIP preflight
  通过；head schema、package authorization、checkpoint/replay/package runner
  和恢复 runbook 均已落盘。

## Plan Rewrite Notes

| Existing item | Decision | Reason |
|---|---|---|
| rolling-origin 多折 | keep | 作为选择阶段唯一调参/晋级依据 |
| long-span external | keep and isolate | 只在 lock 后一次开启 |
| 多指标面板 | keep | 防止只追单一 MRR |
| 跨折稳定硬门禁 | strengthen | 每折 MRR/NDCG 非退化，并约束 pooled 指标 |
| Dataset1 Setwise rolling 脚本 | reuse infrastructure, rewrite candidate | 折和缓存可复用，但比较对象改为最终集成 v0/v1 |
| 扫描 `γ` 或 v1 权重 | remove | 本轮只验证结构性输入替换 |
| 直接生成包 | reorder last | rolling 与 external 未过前禁止 |

## Drift Diagnosis

- **Goal drift**: 只比较基础头 raw logits 会偏离线上最终 time-ramp 路径。
- **Phase drift**: 在 rolling 前读取 external 会把 gate 变成选择折。
- **Validation drift**: 单折 MRR 上升不足以证明跨时间稳定；多折和长跨度是核心。
- **Compatibility drift**: v0 对照必须显式冻结，不能因新默认 v1 而静默改变。
- **Cleanup drift**: 不顺手改其他失败实验、默认参数或 Dataset2 专家。

## Priority Rationale

- 先验证真实缓存、折边界和现有专家资产是否足以复现最终路径，避免写完 runner
  才发现只能做 standalone proxy。
- 再用测试冻结 v0/v1 唯一差异、指标和 external 隔离，然后才训练与读取结果。
- 提交包放在最后，确保失败实验不会消耗线上机会。

## Assumptions and Open Decisions

| Item | Status | Impact | Owner / Next step |
|---|---|---|---|
| Dataset1 rolling cache 与 manifest 可用 | unresolved | 缺失则无法立即训练三折 | Phase 1 preflight |
| 每折可重放同一个 time-ramp Setwise 专家语义 | confirmed | 每折因果重训共享专家，v0/v1 共用 | runner 已冻结 |
| external 使用哪一个长跨度资产 | confirmed | Dataset1 官方 20k validation cache，固定 SHA-256 | one-shot runner |
| `γ=0.5` 固定 | confirmed | 隔离基础三通道的净效果 | 不扫描 |
| Dataset2 保持当前冠军 | confirmed | 本轮包只改变 Dataset1 | package manifest 校验 |
| rolling 硬门槛不因结果放宽 | confirmed | 防止单折过拟合 | 状态机执行 |

## Phases

### Phase 1: 真实资产 preflight 与协议冻结

- **Purpose**: 确认三折最终集成与长跨度 external 能被真实重放。
- **Entry condition**: 本目标文档已写盘。
- **Phase rules**:
  - 只读资产元数据、shape、时间边界与 fingerprint，不读取 external 指标。
  - 不训练、不写 selection lock、不生成提交包。
- **Todos**:
  - [x] 审计 Dataset1 rolling manifest、raw 特征缓存和折边界。
    - **Surface**: result/cache/manifest
    - **Proof**: 三折均满足 train_end < score_start，shape/fingerprint 完整。
    - **Depends on**: none
  - [x] 审计 time-ramp Setwise 专家与 v0/v1 基础头的精确重放路径。
    - **Surface**: scripts/checkpoints
    - **Proof**: 候选唯一差异表只包含 context transform。
    - **Depends on**: rolling assets
  - [x] 冻结 external 路径、最小时间跨度和 one-shot 状态目录。
    - **Surface**: external manifest metadata
    - **Proof**: 写入 protocol，未读取分数。
    - **Depends on**: candidate fingerprint
- **Exit proof**: preflight report 标记 `ready=true`，或明确列出不可绕过的真实资产缺口。
- **Stop condition**: 无法生成最终 time-ramp 分数，只能得到 standalone proxy。

### Phase 2: RED/GREEN 精确 runner

- **Purpose**: 把最终集成、全指标和隔离门禁变为可测试行为。
- **Entry condition**: Phase 1 ready，且现有脚本不能直接满足全部契约。
- **Phase rules**:
  - 先 RED 后实现。
  - selection runner 不接受 external score path。
  - v0/v1 配置除 transform version 外必须完全相同。
- **Todos**:
  - [x] 为候选唯一差异、time-ramp 最终分数和指标面板写 RED。
    - **Surface**: tests
    - **Proof**: 以缺失目标行为失败。
    - **Depends on**: Phase 1
  - [x] 实现/扩展三折训练与最终集成评分 runner。
    - **Surface**: scripts/src
    - **Proof**: focused tests GREEN。
    - **Depends on**: RED
  - [x] 接入 selection lock 与 external one-shot 状态。
    - **Surface**: robust selection infrastructure
    - **Proof**: lock 前 external 失败、第二次开启失败。
    - **Depends on**: score runner
- **Exit proof**: focused tests、相关回归和 Ruff 全绿。
- **Stop condition**: runner 需要扫描 `γ`、seed 或其他参数才能运行。

### Phase 2B: 本地可恢复交付链

- **Purpose**: 在服务器不可用期间消除 gate 通过后拿错 head/checkpoint/package
  的工程风险。
- **Entry condition**: Phase 2 GREEN；真实指标仍未读取。
- **Phase rules**:
  - 只审计 metadata、manifest 和现有冠军包，不打开 external 数组。
  - 不生成候选提交包，不伪造 rolling/external 指标。
- **Todos**:
  - [x] 审计四个 rolling folds、cache lineage、external 时间边界和冠军 ZIP。
    - **Surface**: local preflight report
    - **Proof**: `status=ready_for_remote_rolling`。
    - **Depends on**: Phase 2
  - [x] 定义候选 head schema 与 package authorization 哈希绑定。
    - **Surface**: src/tests
    - **Proof**: 正确 RED 后 GREEN。
    - **Depends on**: exact candidate protocol
  - [x] 实现条件 checkpoint、双 replay 和最终 package runner。
    - **Surface**: scripts/runbook
    - **Proof**: Ruff、CLI help smoke、相关回归。
    - **Depends on**: package authorization
- **Exit proof**: 本地 preflight、15 个核心测试、84 个相关回归和 Ruff 全通过。
- **Stop condition**: 任何实现需要读取 external 指标或绕过 selection lock。

### Phase 3: Rolling-origin 真实执行

- **Purpose**: 用未读取 external 的三折最终集成分数决定候选是否晋级。
- **Entry condition**: Phase 2 GREEN。
- **Phase rules**:
  - v0/v1 同折、同 seed、同训练规模顺序执行。
  - 不因中间折结果修改配置。
  - 任一硬门禁失败即停止。
- **Todos**:
  - [ ] 训练并评分至少三个 rolling folds。
    - **Surface**: model/score artifacts
    - **Proof**: 每折完整矩阵、hash、时间审计。
    - **Depends on**: Phase 2
  - [ ] 生成多指标与 movement 报告并执行硬门禁。
    - **Surface**: rolling report
    - **Proof**: per-fold + pooled 决策。
    - **Depends on**: fold scores
  - [ ] 仅在通过时创建 selection lock。
    - **Surface**: lock artifact
    - **Proof**: candidate/config/fingerprint SHA 绑定。
    - **Depends on**: gate pass
- **Exit proof**: `rolling_pass=true` 与不可覆盖 lock，或拒绝报告。
- **Stop condition**: 任一折 MRR/NDCG 下降或 pooled 二级指标不满足。

### Phase 4: External one-shot 与提交决策

- **Purpose**: 用长跨度证据接受/拒绝锁定候选，并在通过时生产可提交 artifact。
- **Entry condition**: Phase 3 selection lock 存在。
- **Phase rules**:
  - external 只开启一次，失败不回扫。
  - 打包只允许读取锁定候选和当前 Dataset2 冠军。
- **Todos**:
  - [ ] 执行一次 external 并写 receipt/report。
    - **Surface**: external state
    - **Proof**: lock SHA、fingerprint、跨度与全指标齐全。
    - **Depends on**: selection lock
  - [ ] 根据冻结门槛接受或拒绝。
    - **Surface**: gate report
    - **Proof**: mechanical pass/fail，无参数反馈。
    - **Depends on**: external report
  - [ ] 仅在通过时生成并校验提交包。
    - **Surface**: checkpoint/result.zip/manifest
    - **Proof**: Dataset1=v1 locked、Dataset2=current champion、行数/hash/并列检查通过。
    - **Depends on**: external pass
- **Exit proof**: 拒绝报告，或一份可提交且有完整 manifest/hash 的包。
- **Stop condition**: external 失败、lock 漂移或 Dataset2 预测不一致。

## Dry-Run Findings

- Dataset2 当前由独立 Setwise champion 主导，基础头三通道的净收益主要应从
  Dataset1 最终 time-ramp 路径验证；把两数据集一起重训会增加噪声。
- 旧 Dataset1 rolling 实验比较 raw 与独立 Setwise，不能直接回答 v0-vs-v1；
  但 manifest、缓存和折定义可复用。
- external 资产此前已被其他候选使用，只能作为一次性工程门，报告必须披露。
- 如果每折缺少可重放的 time-ramp Setwise 专家，必须停止，不能用 standalone
  基础头通过后直接生成提交包。

## Final Validation

- focused runner/metric/isolation tests。
- rolling 三折最终集成报告与 selection lock 校验。
- external one-shot receipt；第二次调用确定性失败。
- Ruff、compileall、相关 Linux/Jittor 回归。
- 通过时校验 `result.zip` 行数、候选顺序、有限值、确定性去并列与 SHA-256。

## First Execution Step

服务器恢复后上传已验证代码，按 runbook 执行三个 rolling selection folds；
selection lock 不存在时禁止运行 external。
