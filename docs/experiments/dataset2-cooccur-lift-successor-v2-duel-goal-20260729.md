# Goal Document: Dataset2 Cooccur-Lift Successor V2 双候选预注册

## Go / No-Go

- **Judgment**: Go
- **Reason**: v1 新冠军、远视界协议、部署塌缩比例和三档 gapped gap 均已有
  冻结证据；本轮只冻结两个候选形态及判决规则，不读取任何新指标。

## Target Outcome

在 `cooccur_lift_aux_expert_v1` 新冠军之上，预注册且哈希锁定两个、仅两个
successor 候选：

1. `cooccur_lift_full_only_v2`：删除 short 信号及其 setwise context 通道；
2. `cooccur_lift_gap_aware_v2`：保留 full/short，并增加冻结定义的
   `short_window_supported` 指示特征。

只有逐个近视界折不下降且逐个 gapped 折改善的候选才可生成 selection lock 并
打开一次 external。v1 的集成权重固定 `0.50`，禁止任何权重回扫。

## Goal Definition

- **Type**: learning / quality / operational
- **Boundary**: 包括双视界资格门的协议表达、两个精确候选配置、validation plan、
  plan lock、preflight、TDD 和结果文档；不包括候选实现、gapped 分数物化、训练、
  selection、external 或线上提交。
- **Non-goals**:
  - 不增加第三个候选、权重、窗口、seed、epoch 或辅助头容量扫描。
  - 不修改或重新选择 v1 新冠军。
  - 不把 external raw delta 当效应量。
  - 不声称候选流行度边际审计已排除 source-conditioned 或
    source-candidate joint shift。
- **Deferred work**:
  - 实现 full-only 与 gap-aware materializer/head。
  - 远端生成三折 near、三折 gapped 和可选 zero-short 分数。
  - selector 通过后的 one-shot external。
- **Verification rule**: 协议测试必须证明任一 near fold 下降或任一 gapped fold
  未严格改善都会拒绝候选；两个精确候选配置的 SHA-256 必须进入 plan lock；
  freeze preflight 必须显示 metrics/external 均未读取、无 selection lock、无
  package authority。
- **Evidence source**: RED/GREEN pytest、Ruff、候选配置 SHA-256、validation
  plan lock/preflight JSON 和独立哈希审计。
- **Pass criteria**:
  - candidate space 恰为两个 ID；
  - baseline 绑定 promoted v1 checkpoint
    `796d8d21a0c706ad11f244385b314d471d522c3b807748a54fe4ac78722f5880`；
  - 所有 near folds 的 MRR/NDCG@10 delta `>= 0`；
  - 所有 gapped folds 的 MRR delta `> 0` 且 NDCG@10 delta `>= 0`；
  - selection order 为平均 gapped MRR、最差 gapped MRR、平均 near MRR、
    预登记 tie-break；
  - `selected_weight=0.50` 且所有 rescan authority 为 false；
  - freeze 后没有读取任何 fold/external 指标。
- **Confidence note**: plan lock 能高置信防止候选、权重和门禁事后漂移；候选效果
  只能由后续真实 folds 判断。source-conditioned/joint transport 风险仍未被当前
  审计识别或排除。
- **Judgment owner**: plan freeze/哈希由标准协议判定；external 打开资格由未来
  selector 的双视界硬门判定；最终是否上线由用户决定。

## Current State

- 当前冠军已经包含 `cooccur_lift_aux_expert_v1`，锁定集成权重 `0.50`。
- 审计显示线上一层机制以 full 为主（`76.54%`），short 是唯一已证实发生强
  transport attenuation 的通道。
- 测试行 `39.9720%` short 全零；标准协议已预留 P75/P90/P100 的
  251/308/349 天 gapped 折。
- 现有 time-local selector 只支持 deployment-mixture 资格模式，尚不能表达本轮
  “near 每折不降 + gapped 每折严格改善”的资格门。

## Plan Rewrite Notes

| Existing item | Decision | Reason |
|---|---|---|
| v1 新冠军作为 baseline | keep | 两个 successor 都必须证明相对当前线上状态的增益 |
| full-only v2 | keep and freeze | 移除唯一已证实漂移源，保留 full 主机制 |
| gap-aware v2 | keep and specify | 支持度特征必须依赖 gapped training 才非死特征 |
| deployment-mixture 作为唯一资格门 | rewrite | 本轮用户明确要求 near 不降且 far 改善的合取门 |
| v1 权重空间 | remove | successor 只允许固定 `0.50`，禁止回扫 |
| external 效应量解释 | remove | time-local external 仅作 safety gate，使用 19.5x 校准标记 |

## Drift Diagnosis

- **Goal drift**: 若继续只按 deployment mixture，候选可以用 far 增益抵消 near
  退化，不符合本次明确判决。
- **Phase drift**: gap-aware 若在 gapped infrastructure 前训练，支持度恒为 1，
  形同死特征。
- **Validation drift**: external 只能判方向安全，不能为 v2 估算线上收益幅度。
- **Compatibility drift**: v1 的历史权重扫描不能作为 successor 合法搜索空间继续
  存在。
- **Cleanup drift**: 本轮不实现候选模型，也不重写 v1 历史产物。

## Priority Rationale

- 先补协议的双视界资格模式，再生成 plan lock，避免锁文件声称了实现尚不能执行
  的规则。
- 候选配置在 plan 之前单独哈希，使 feature schema、训练视界和固定权重都进入
  candidate-space identity。

## Assumptions and Open Decisions

| Item | Status | Impact | Owner / Next step |
|---|---|---|---|
| “近视界折不降”解释为每折 MRR/NDCG@10 均 `>=0` | confirmed by literal freeze | 防止均值掩盖某一 near 折退化 | protocol test |
| “远视界折改善”解释为每折 MRR `>0`、NDCG@10 `>=0` | confirmed by literal freeze | 每个部署分位点都必须有正向主指标 | protocol test |
| 两者都过时按 far 表现优先 | assumed and frozen | 解决两个候选同时 eligible 的唯一性 | plan selection order |
| full-only 为最终 tie-break 优先 | assumed and frozen | 同分时选更少漂移源、更简单的形态 | tie priority 0 |
| `short_window_supported = 1[query_time - training_time_max < w]` | assumed from “窗口是否非空” | near 恒 1、gapped 恒 0，可确定复现 | candidate config |
| 两候选使用同一 near/gapped 因果训练集合 | confirmed for fairness | 避免把训练数据差异误判成特征差异 | candidate configs |
| source-conditioned/joint shift 未排除 | confirmed caveat | 远折过而线上缩小时的下一审计向量 | result stop rule |

## Phases

### Phase 1: 双视界资格门 RED

- **Purpose**: 证明现有 deployment-mixture gate 不能表达本轮判决。
- **Entry condition**: 现有标准协议回归全绿。
- **Phase rules**:
  - 只写测试，不先改生产代码。
  - RED 必须分别覆盖 near 下降和 gapped 未严格改善。
- **Todos**:
  - [x] 增加 eligible、near-rejected、gapped-rejected 三个行为样例。
    - **Surface**: `tests/test_standard_validation_protocol.py`
    - **Proof**: 定向 pytest 因缺少 eligibility mode 正确失败
    - **Depends on**: none
- **Exit proof**: RED 指向协议缺少 horizon-conjunctive gate。
- **Stop condition**: 测试只能通过检查内部字段而无法观察 selection 结果。

### Phase 2: 协议 GREEN

- **Purpose**: 最小实现新的 eligibility mode，同时保持原 deployment-mixture
  模式兼容。
- **Entry condition**: Phase 1 RED 成立。
- **Phase rules**:
  - near/gapped 硬门只使用冻结阈值。
  - deployment mixture 仍报告，但不参与本模式 eligibility。
  - selection key 必须与 plan 中的 order 一致。
- **Todos**:
  - [x] 扩展 far-horizon plan 校验、gates、selection key 和报告。
    - **Surface**: `src/jgrec/standard_validation_protocol.py`
    - **Proof**: RED 转 GREEN，旧测试不变
    - **Depends on**: Phase 1
- **Exit proof**: 三个双视界行为样例和旧模式同时全绿。
- **Stop condition**: legacy time-local example 的行为被改变。

### Phase 3: 精确候选与 plan freeze

- **Purpose**: 在任何指标读取前冻结两个候选及全部规则。
- **Entry condition**: Phase 2 全绿。
- **Phase rules**:
  - candidate configs 先落盘并计算 SHA-256。
  - weight 固定 `0.50`；候选配置显式写入所有 rescan authority=false。
  - candidate space 恰为两个，不允许 baseline 作为第三候选。
- **Todos**:
  - [x] 写 full-only 和 gap-aware 精确配置。
    - **Surface**: docs JSON
    - **Proof**: JSON parse + SHA-256
    - **Depends on**: Phase 2
  - [x] 写 validation plan 并运行 freeze。
    - **Surface**: docs / `result/.../plan`
    - **Proof**: plan lock + preflight
    - **Depends on**: candidate hashes
- **Exit proof**: lock 中 candidate IDs/config hashes、双视界规则和 external policy
  与源文件一致。
- **Stop condition**: 任一候选定义仍需根据 fold 指标决定。

### Phase 4: 审计与收口

- **Purpose**: 证明本轮只完成预注册，没有提前消费验证证据。
- **Entry condition**: Phase 3 freeze 成功。
- **Phase rules**:
  - 不生成 rolling manifest、selection lock、external receipt 或 package。
  - 保留 source-conditioned/joint transport 风险，不把它改写为已排除。
- **Todos**:
  - [x] 运行相关回归、Ruff、JSON 和 hash audit。
    - **Surface**: tests / docs / result
    - **Proof**: 命令输出
    - **Depends on**: Phase 3
  - [x] 输出 TDD 和预注册结果文档。
    - **Surface**: docs
    - **Proof**: RED/GREEN/REFACTOR 与下一步一致
    - **Depends on**: validation
- **Exit proof**: preflight 明确 metrics/external 未读，外部动作均未授权。
- **Stop condition**: 出现任何新的候选指标或 external 内容读取。

## Dry-Run Findings

- 当前 far-horizon schema 必须新增显式 eligibility mode；仅改 plan JSON 会被现有
  selector 继续按 deployment mixture 判决。
- gap-aware 支持度若定义为 `lift_short != 0`，近折并不保证恒 1；为符合用户给出的
  “近折恒 1、gapped 才变化”约束，本轮冻结为训练覆盖与 query 短窗是否重叠的
  row-level indicator。
- gap-aware 必须在同一模型训练证据中看到 support=1/0；只在近折训练后拿去评
  gapped 会让新增特征没有可学习系数。
- 两个候选若都过门必须有唯一 selection order；本轮按 far 视界优先，并以
  full-only 的结构更简作为最终 tie-break。

## Final Validation

- 定向与相关标准协议 pytest 全绿。
- Ruff 对协议、测试和 freeze scripts 全绿。
- 两个 candidate config SHA-256 与 validation plan/plan lock 一致。
- plan preflight：
  `selection_metrics_read=false`、
  `reserved_fold_metrics_read=false`、
  `external_holdout_read=false`、
  `package_authorized=false`。
- 结果目录不存在 rolling manifest、selection lock、external receipt 或 package。

## First Execution Step

先增加一个 deployment mixture 为正、但某个 near fold 下降的候选样例，证明旧
模式会错误放行而新规则必须拒绝。

## Execution Result

- RED 正确捕获旧协议无法表达双视界合取门，以及 rolling/external baseline
  身份未向下游 manifest 传播的问题。
- GREEN 为新 duel 增加严格模式，同时保留旧 deployment-mixture 兼容行为。
- 两个候选 config、validation plan、plan lock 和 preflight 已冻结；未读取任何
  successor 指标。
- 相关回归 `34 passed`，Ruff `All checks passed!`。
- 当前状态是 `candidate_plan_frozen_gapped_materialization_pending`；
  successor selection 和 external 都未授权。
