# Goal Document: 标准验证远视界折与时间局部特征解释规则

## Go / No-Go

- **Judgment**: Go
- **Reason**: 现有标准协议已经冻结候选空间、近视界 rolling 选择和一次性
  external gate；本轮只需扩展同一证据链，不需要读取新的 external 标签或启动
  `full-only v2`。

## Target Outcome

标准验证能够对时间局部特征族同时评估近视界和预登记的 gapped 远视界折：
gapped 折的训练端到评分端间隔不小于短窗，并覆盖多个部署视界分位点；候选由近/
远视界按预登记部署占比形成的混合指标裁决，不再被近视界单折提前误杀。近视界可
附带 `zero-short` 反事实诊断臂。external 对该特征族只承担安全门职责，报告明确
禁止把 raw delta 当线上效应量，并记录 `19.5x` 校准折扣。

## Goal Definition

- **Type**: quality / operational
- **Boundary**: 升级标准 plan、rolling manifest、selector、external report、
  示例 plan、运行手册、preflight、测试和本次 cooccur-lift promotion 的前向协议
  声明；保留非时间局部候选的 v1 行为。
- **Non-goals**:
  - 不预注册、训练或选择 `full-only v2`。
  - 不修改已上线 v1 的权重、模型、公式、checkpoint 或 promotion 结论。
  - 不用 `zero-short` 臂回扫新权重或新候选。
- **Deferred work**:
  - 在远端按新协议物化真实 gapped folds。
  - 新 plan 锁定后的 `full-only v2` 正式验证。
- **Verification rule**: 自动化测试证明时间局部 plan 缺少远视界策略时拒绝；
  gapped gap 小于 `w` 时拒绝；近折为负但部署混合与远折过门的候选可被选择；
  `zero-short` 只报告不参与选择；external 输出 safety-only 与 `19.5x` 折扣。
- **Evidence source**: pytest RED/GREEN、Ruff、示例 plan freeze、文档和 JSON 哈希
  一致性检查。
- **Pass criteria**: 定向测试和相关回归全绿；所有新字段被 plan lock 哈希绑定；
  promotion 三份机器产物互相引用的 SHA-256 一致；没有 v2 selection lock、external
  receipt 或 package。
- **Confidence note**: 测试可证明协议不会沿旧的近视界误判路径运行；真实 v2
  效果仍必须由后续远端 gapped 评分证明。
- **Judgment owner**: 本轮工程完成由测试和哈希审计判定；未来 v2 是否晋升由新
  plan 下的 deployment-mixture selector 与 external safety gate 判定。

## Current State

- 标准 selector 只读取至少三折 `role=selection` 的近视界分数，并要求候选每折
  MRR/NDCG 非退化。
- cooccur-lift 审计显示 strict external 的 short 一层能量占 `36.18%`，线上
  `39.9720%` 行的 short 全零；因此近视界单折硬门会系统性偏向保留 short。
- 当前 external 结果仍可被误读为效应量；审计观察到 external/online 比率
  `19.50x`。

## Plan Rewrite Notes

| Existing item | Decision | Reason |
|---|---|---|
| 三折近视界 rolling | keep | 仍测量 short 有效时的收益和代价 |
| 近折逐折非退化硬门 | rewrite for time-local | 否则 full-only 候选在看到塌缩状态前已被淘汰 |
| 预留 gate 折 | keep | 保持选择与后续 gate 隔离 |
| external 全指标 gate | keep as safety gate | 保留方向性安全约束，不再宣称线上效应量 |
| 单一 equal-weight 选择口径 | rewrite for time-local | 改为近/远视界内部等权、视界之间按部署占比混合 |

## Drift Diagnosis

- **Goal drift**: 旧协议验证“近期局部行上谁更好”，没有验证“部署塌缩状态下谁
  更好”。
- **Phase drift**: 若先预注册 v2 再补远折，候选空间会在错误证据协议下被冻结。
- **Validation drift**: 长跨度 external 覆盖终点 horizon，但其历史评分行没有
  覆盖 post-history short collapse。
- **Compatibility drift**: 直接改变所有候选的近折门禁会破坏非时间局部实验；新
  行为必须由显式 temporal scope 开启。
- **Cleanup drift**: 本轮不迁移全部历史实验，只升级标准入口和指定 promotion
  证据。

## Priority Rationale

- 先用 RED 固化误判样例，再实现部署混合口径，确保新增 gapped 字段真正参与裁决。
- 在 v2 plan 之前冻结视界分位点、部署占比与 external 解释角色，防止看到 v2
  结果后调整。

## Assumptions and Open Decisions

| Item | Status | Impact | Owner / Next step |
|---|---|---|---|
| 时间局部候选必须显式声明 `short_window_seconds` | confirmed | 使 gap 下限可验证 | schema/test |
| 近/远视界内部使用等权折均值 | confirmed | 不让行数或某个分位点暗中放大 | selector/test |
| 视界之间使用预登记 collapse fraction 混合 | confirmed | 覆盖线上 `39.9720%` 状态 | plan lock/test |
| gapped 分位点和 gap 秒数由每个实验预登记 | confirmed | 基础设施保持通用 | plan schema |
| `zero-short` 不参与选择 | confirmed | 防止零成本诊断变成额外调参臂 | contract/test |
| `19.5x` 是保守校准而非可识别因果比例 | confirmed | external raw delta 只过门 | report wording |

## Phases

### Phase 1: 协议 RED

- **Purpose**: 锁定时间局部候选的错误判决样例和证据边界。
- **Entry condition**: 现有 v1 测试全绿。
- **Phase rules**:
  - 只改测试、fixture 和目标文档。
  - RED 必须因缺少 far-horizon 行为失败。
- **Todos**:
  - [x] 增加 far-horizon plan/manifest、gap、部署混合和 zero-short 测试。
    - **Surface**: `tests/test_standard_validation_protocol.py`
    - **Proof**: 定向 pytest 正确失败
    - **Depends on**: none
  - [x] 增加 external safety-only/discount 测试。
    - **Surface**: 同上
    - **Proof**: 缺少目标字段而失败
    - **Depends on**: none
- **Exit proof**: RED 精确落在缺少 gapped contract 与解释规则。
- **Stop condition**: 若 fixture 依赖 external 标签或 v2 实际结果，停止并缩回纯
  协议测试。

### Phase 2: 最小 GREEN

- **Purpose**: 在不破坏 legacy plan 的前提下实现时间局部路径。
- **Entry condition**: Phase 1 RED 成立。
- **Phase rules**:
  - temporal scope 缺省时保持现有行为。
  - gapped 折 gap 必须同时满足 `>= w` 和预登记分位点下限。
  - near 证据仍完整报告，但 time-local 候选使用 deployment mixture gate/order。
  - zero-short 只能写诊断和哈希，不能进入 gate/order。
- **Todos**:
  - [x] 扩展 plan freeze、rolling validation、selector 和 selection lock。
    - **Surface**: `src/jgrec/standard_validation_protocol.py`
    - **Proof**: RED 转 GREEN
    - **Depends on**: Phase 1
  - [x] 扩展 external report 的解释规则。
    - **Surface**: 同上
    - **Proof**: safety-only/discount 测试通过
    - **Depends on**: plan lock
- **Exit proof**: 定向协议测试全绿。
- **Stop condition**: legacy fixture 输出或选择结果发生非预期变化。

### Phase 3: 基础设施与证据同步

- **Purpose**: 让后续 v2 有可执行模板，并让历史 promotion 明确新的前置条件。
- **Entry condition**: Phase 2 全绿。
- **Phase rules**:
  - 示例使用审计已确认的 `w=17,038,080s`、collapse fraction
    `0.39971972363446745` 和多档部署 gap。
  - promotion 状态仍是 v1 accepted/promoted，只新增
    `required_before_successor_v2` 声明。
  - 修改机器产物后重算全部下游 SHA-256。
- **Todos**:
  - [x] 更新示例 plan、runbook、结果与 preflight。
    - **Surface**: docs / scripts
    - **Proof**: 示例 freeze 与 CLI smoke
    - **Depends on**: Phase 2
  - [x] 更新指定三份 promotion JSON 和结果文档。
    - **Surface**: docs / result
    - **Proof**: 哈希引用审计
    - **Depends on**: protocol wording
- **Exit proof**: 文档和机器产物形成一致的前向协议链。
- **Stop condition**: 任何改动会重写 v1 的模型/线上判决事实。

### Phase 4: 回归与收口

- **Purpose**: 证明升级通用且没有破坏旧协议。
- **Entry condition**: Phase 3 完成。
- **Phase rules**:
  - 使用项目规定的 `uv run`。
  - 只检查本轮相关 Python 文件的 Ruff，再跑相关协议/晋升回归。
- **Todos**:
  - [x] 运行定向、相关回归、Ruff 和 JSON 解析/哈希审计。
    - **Surface**: test / lint / artifacts
    - **Proof**: 命令输出
    - **Depends on**: Phase 3
- **Exit proof**: 全部验证通过并记录 RED/GREEN/REFACTOR。
- **Stop condition**: unrelated dirty-worktree failure 与本轮改动冲突时停止扩大范围。

## Dry-Run Findings

- 只新增 `gapped_folds` 仍会被旧 `all_folds_*` 近折硬门误杀，因此必须同步改
  time-local 的 gate 和 selection key。
- gapped 分位点不能从评分结果反推；必须写入 plan lock，manifest 只证明实际
  gap 达标。
- `zero-short` 若复用 `candidates` 主字段会不小心进入选择；需要独立
  `counterfactual_arms` 命名空间和明确的 `participates_in_selection=false`。
- 修改 canonical replay report 会改变 promoted manifest 和 status 中的哈希，
  更新顺序必须是 replay report → promoted manifest → status。

## Final Validation

- `uv run --group dev pytest tests/test_standard_validation_protocol.py -q`
- `uv run --group dev pytest tests/test_cooccur_lift_promotion.py -q`
- `uv run ruff check src/jgrec/standard_validation_protocol.py
  tests/test_standard_validation_protocol.py
  scripts/preflight_standard_validation_local.py`
- 冻结更新后的 example plan，并验证 preflight 不读取 external。
- 独立脚本核对 replay report、promoted manifest、status 的 SHA-256 引用。

## First Execution Step

先添加“近折负、gapped 折正、部署混合为正时仍可晋级”的最小失败测试。

## Execution Result

- **Judgment**: protocol complete; real gapped materialization pending.
- time-local plan、gapped manifest、deployment-mixture selector、
  `zero-short` diagnostic 和 external safety-only 解释规则均已实现并哈希绑定。
- preflight 明确阻断 v2：`current_gapped_fold_count=0` 且
  `ready_for_time_local_candidate_preregistration=false`。
- 相关回归 `24 passed`，Ruff 全绿，JSON 解析与 promotion 两级哈希链通过。
