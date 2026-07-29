# Goal Document: Hybrid 特征掩码留一法远端验证

## Go / No-Go

- **Judgment**: Go after read-only capacity audit
- **Reason**: 本地候选生成、冻结别名和预检已经完成；远端只在确认不会争抢现有进程后启动。

## Target Outcome

在不终止、不抢占、不显著拖慢服务器现有进程的前提下，对“完整基础融合候选 + 8 个逐组
留一候选”执行同配置、同预算的 rolling-origin 多折验证；按标准稳定性门禁锁定一个候选
或得出全部不通过。只有锁定后，才允许打开一次长跨度 external holdout。

## Goal Definition

- **Type**: operational / learning
- **Boundary**: 只包含服务器容量审计、代码同步、候选预注册、rolling-origin 运行、结果汇总，
  以及满足门禁后的单次 external gate。
- **Non-goals**:
  - 不终止、降优先级或修改其他用户的进程。
  - 不在 rolling 结果后反向增加候选、扫描新特征组合或修改训练预算。
  - 不根据 external 或线上结果回扫候选。
  - 本目标不自动生成提交包。
- **Deferred work**:
  - 通过 external gate 后的正式 checkpoint/refit 与提交包。
- **Verification rule**: 所有预注册候选必须在相同 rolling folds、随机种子与资源预算下完成；
  标准选择器必须给出唯一通过者或明确 No-Go；external 仅对锁定候选运行一次。
- **Evidence source**: 远端进程/GPU审计、冻结计划 JSON、逐折指标 JSON、标准选择报告、
  external gate 报告和运行日志。
- **Pass criteria**:
  - 启动前及运行中未发现资源冲突或被影响进程。
  - 9 个精确候选的预注册清单保持不变。
  - 每个候选完成相同 rolling-origin 多折。
  - 同时报 MRR、Hit@1/3/10、NDCG@10、平均排名、改善/恶化 query 数。
  - 以跨折稳定性硬门禁决定唯一候选；无稳定候选则停止。
  - external 只对锁定候选打开一次，且不用于反向调参。
- **Confidence note**: rolling-origin 是部署前的主要选择证据；external 是锁定后的长跨度审计，
  不是调参集。线上效果仍只有正式提交后才能确认。
- **Judgment owner**: 标准验证脚本及其冻结计划；资源安全由启动前审计和持续监控共同判定。

## Current State

- 本地 63 维基础融合特征已形成 1 个完整候选和 8 个留一别名。
- 默认训练候选从 12 个递进候选扩为 19 个唯一索引掩码，但正式验证不会让单折自由选择。
- 本地预检状态为 `ready_for_remote_rolling`，未读取 external，未生成提交包。
- 服务器刚恢复；当前负载、GPU 占用、远端代码/数据状态未知。
- 用户要求不得影响其他进程，因此容量不足时必须等待，而不是抢占。

## Plan Rewrite Notes

| Existing item | Decision | Reason |
|---|---|---|
| 默认 19 候选单切分搜索 | remove | 会把留一法重新变成单折过拟合 |
| 完整候选 + 8 留一别名 | keep | 是可解释且已冻结的精确候选集 |
| rolling-origin 多折 | keep, move first | 是候选选择门禁 |
| external holdout | keep, defer | 只能在候选锁定后打开一次 |
| 提交包 | defer | external 通过前无授权生成 |
| 并行跑满 GPU | rewrite to sequential low-impact | 避免影响其他进程 |

## Drift Diagnosis

- **Goal drift**: 不允许顺手测试新权重、特征组合或训练超参。
- **Phase drift**: external 不能与 rolling 并行，也不能提前查看。
- **Validation drift**: 不能以单折 MRR 或“任务跑完”替代跨折稳定性。
- **Compatibility drift**: 必须使用 `loo_without_<group>` 冻结别名记录精确候选。
- **Cleanup drift**: 不在远端实验期间清缓存、删文件或修改其他实验目录。

## Priority Rationale

- 先做只读资源审计，因为“不影响其他进程”优先于实验速度。
- 候选预注册早于任何训练，防止看到结果后改变候选集合。
- rolling 早于 external，保持长跨度留出的独立性。

## Assumptions and Open Decisions

| Item | Status | Impact | Owner / Next step |
|---|---|---|---|
| 远端地址和工程目录可从现有脚本发现 | assumed | 决定连接和同步方式 | 本地只读检查脚本 |
| 服务器存在至少一个安全可用的 GPU 时段 | unresolved | 无空闲资源则不能启动 | 远端只读审计 |
| 现有 rolling/external 脚本可直接承载 9 个候选 | unresolved | 可能需要最小编排脚本 | 审计后验证 `--help`/dry-run |
| 当前 external 尚未为本候选集打开 | assumed | 保持门禁有效 | 检查冻结计划和结果目录 |

## Phases

### Phase 1: 只读容量与状态审计

- **Purpose**: 确认连接、工程、数据、GPU、CPU、内存、磁盘及现有任务状态。
- **Entry condition**: 本目标文档已落盘。
- **Phase rules**:
  - 只允许读命令；不得 kill、renice、清缓存或改远端文件。
  - 不在命令行、日志或产物中保存密码。
- **Todos**:
  - [ ] 检查现有连接脚本和远端目标。
    - **Surface**: 本地脚本/文档
    - **Proof**: 明确主机、用户、工程目录且不打印凭据
    - **Depends on**: none
  - [ ] 检查远端 `nvidia-smi`、进程、负载、内存、磁盘和实验目录。
    - **Surface**: 远端只读状态
    - **Proof**: 容量快照
    - **Depends on**: 连接成功
- **Exit proof**: 找到不会争抢资源的执行窗口和资源上限。
- **Stop condition**: GPU 已被计算进程占用、内存/磁盘不足、远端工程状态不明或连接不稳定。

### Phase 2: 冻结计划与低干扰试跑

- **Purpose**: 在不读取 external 的情况下确认 9 个候选、folds、数据和命令均可执行。
- **Entry condition**: Phase 1 容量审计通过。
- **Phase rules**:
  - 候选列表一经写入计划即不得修改。
  - 使用单任务串行、低 CPU/IO 优先级和显式 GPU 绑定。
  - 先运行 `--help`、preflight 或最小 dry-run，不直接启动全量。
- **Todos**:
  - [ ] 同步必要代码并验证工作树目标版本。
    - **Surface**: 远端项目目录
    - **Proof**: 文件哈希/语法/聚焦测试
    - **Depends on**: Phase 1
  - [ ] 生成候选和 folds 冻结计划。
    - **Surface**: 实验计划 JSON
    - **Proof**: 1 full + 8 LOO，external 未打开
    - **Depends on**: 代码可用
  - [ ] 执行一个不落指标的低成本预检。
    - **Surface**: 远端运行环境
    - **Proof**: preflight 通过
    - **Depends on**: 冻结计划
- **Exit proof**: 计划、代码、数据和运行命令全部可复现。
- **Stop condition**: 需要改变候选/折定义、发现 external 已被读取，或试跑干扰现有负载。

### Phase 3: Rolling-origin 串行验证

- **Purpose**: 取得所有候选的同预算多折证据。
- **Entry condition**: Phase 2 通过且资源仍安全。
- **Phase rules**:
  - 一次只运行一个训练任务。
  - 每个候选/折结束后检查资源与其他进程；异常即暂停。
  - 不基于中途结果提前新增、删除或调参。
- **Todos**:
  - [ ] 串行完成 9 个候选的全部 folds。
    - **Surface**: rolling 指标与日志
    - **Proof**: 完整结果矩阵、无失败/缺折
    - **Depends on**: Phase 2
  - [ ] 运行标准候选选择器。
    - **Surface**: 选择报告
    - **Proof**: 唯一稳定通过者或明确 No-Go
    - **Depends on**: 结果矩阵完整
- **Exit proof**: 锁定候选且计划哈希未改变，或 No-Go。
- **Stop condition**: 资源冲突、折间配置不一致、数据/指标异常或无候选通过稳定性门禁。

### Phase 4: 单次 External Gate

- **Purpose**: 对锁定候选做长跨度独立审计。
- **Entry condition**: Phase 3 有唯一锁定候选且 external 尚未打开。
- **Phase rules**:
  - 只运行锁定候选一次。
  - 结果不用于回扫留一候选。
  - 本阶段不生成提交包。
- **Todos**:
  - [ ] 运行标准 external gate 并保存报告。
    - **Surface**: external 报告
    - **Proof**: gate 结果及计划哈希一致
    - **Depends on**: 锁定候选
- **Exit proof**: external 明确 Pass/No-Go。
- **Stop condition**: 候选未锁定、计划哈希变化或 external 已被非预期读取。

## Dry-Run Findings

- 最大未知量不是模型，而是服务器现有负载；必须先审计再决定是否立即启动。
- 默认掩码搜索会在单折内部自由选择，因此正式运行必须逐个冻结候选。
- 9 候选乘多折可能耗时较长；低干扰要求优先采用串行和可恢复日志，而不是并发。
- 现有脚本能否直接编排该矩阵尚待只读检查；若不能，只允许添加最小编排层。

## Final Validation

- 容量与干扰审计记录完整。
- 冻结计划包含且仅包含 9 个候选。
- rolling 结果矩阵无缺折并通过标准多指标稳定性选择。
- external 至多读取一次且仅针对锁定候选。
- 没有提交包，没有终止或修改其他进程。

## First Execution Step

只读检查本地远端连接脚本和既有验证 runbook，随后连接服务器获取资源快照；在快照通过前
不启动任何训练。
