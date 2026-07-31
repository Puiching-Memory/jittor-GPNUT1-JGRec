# Goal Document: Dataset1 K256 Static Setwise Dual-Horizon

## Go / No-Go

- **Judgment**: Go
- **Reason**: Dataset1 的 recent-cache Setwise 已证实存在正信号，但旧的纯
  Setwise 静态实验只做单 validation，随后引入的 `progress**0.5` time-ramp
  又在 rolling origins 上出现两折回退。现在标准验证已经支持 near + gapped
  双视界，可以直接检验一个更简单、可部署且不依赖 query 时间归一化的静态
  Setwise 层。

## Target Outcome

用 K=256 重建 Dataset1 recent-200k 全候选缓存，在每个验证折内因果重训
基础 MLP、LightGBM 和 Setwise context 头；对预声明的固定 Setwise 权重
`0.05..0.80` 做一次扫描。候选只有在 near 不下降、三个 gapped 折均改善后
才允许开启一次 external safety gate。通过者将成为“移除 time-ramp、改用
固定权重”的唯一待生产候选；本轮不自动提交 leaderboard。

## Goal Definition

- **Type**: learning / quality / operational
- **Boundary**:
  - Dataset1 only；Dataset2 不参与训练、选择或 external。
  - 结构与 source-profile 的预测历史上限都固定为 `K=256`。
  - recent 训练缓存固定为 `200,000 x 100 x 63`；候选矩阵、行序、时间
    sidecar 和特征哈希必须绑定。
  - 每折基础 MLP、LightGBM、Setwise context 头都只使用该折 score 起点前的
    数据训练；同折 time-ramp 基线与静态候选共享全部头和特征。
  - 基线集成固定为当前 Dataset1 冠军结构：
    `base MLP + LightGBM -> Setwise time-ramp gamma=0.5`。
  - 候选集成固定为：
    `candidate=(1-static_weight)*base_backbone + static_weight*setwise`。
  - 静态权重网格精确为
    `{0.05, 0.10, 0.15, ..., 0.80}`，共 16 个；不含 `0`、不含大于
    `0.80` 的权重，不扫描 gamma、seed、训练规模或其他结构。
  - near 至少 3 折；gapped 固定为部署视界 `p75/p90/p100` 三档。具体秒数
    只由 Dataset1 无标签 query-time 元数据计算，向上取整后在任何训练和指标
    读取前写入 plan lock。
  - external 只作一次性安全门，不用 raw delta 估计线上效应量，也不将
    Dataset2 审计得到的 `19.5x` 折扣外推到 Dataset1。
- **Non-goals**:
  - 不重新搜索 time-ramp 曲线、breakpoint、gamma 或最大权重。
  - 不复活 7/28 未完成的 base-context v1 候选。
  - 不用 external 选择权重，不因 external 结果回扫。
  - 不在 internal gate 未通过时训练 full-origin head、打开 external 或生成包。
  - 不修改当前冠军 checkpoint、线上晋升记录或历史实验产物。
- **Deferred work**:
  - internal 与 external 都通过后的 checkpoint serving 接入和提交包。
  - 若静态候选远视界通过但线上缩水，再审计 source-conditioned /
    source-candidate joint structure。
- **Verification rule**:
  1. plan lock 精确绑定 K、缓存规模、16 个权重、折边界、训练配置和基线
     checkpoint/Dataset1 state。
  2. near/gapped 每折训练结束时间严格早于评分起点；gapped 实际 gap 不小于
     预登记部署分位数。
  3. 同折 baseline/candidate 共享 base、LGBM 和 Setwise scores；唯一差异是
     time-ramp 与一个不随 query 改变的 scalar weight。
  4. near 每折 MRR、NDCG@10 均 `>=0` delta；gapped 每折 MRR 严格增加、
     NDCG@10 不下降。
  5. 合格候选按
     `mean gapped MRR -> worst gapped MRR -> mean near MRR -> higher weight`
     锁定。
  6. external 在 selection lock 前不可读取，且只允许一个 receipt。
- **Evidence source**: goal/plan lock、cache reports、rolling manifest、每折
  head/score 哈希、standard validation selection report/lock、external
  receipt/report、RED/GREEN 测试和 Ruff。
- **Pass criteria**:
  - internal：三折 near MRR/NDCG 不退化；三折 gapped MRR 都严格改善且
    NDCG 不退化。
  - external 七门：MRR、Hit@1/3/10、NDCG@10 不下降，mean rank 不变差，
    improved queries 多于 worsened queries；MRR 必须严格增加。
  - 所有 source cache/hash 在运行前后不变。
- **Confidence note**: K=256 是本轮预声明的生产参数，同时用于同折 baseline
  与 candidate，以隔离静态集成规则的净效果。external 与当前线上 D1 精确
  对照时会同时包含 K=256 的最终生产差异，因此只读 pass/fail，不读效应量。
- **Judgment owner**: standard-validation 状态机机械决定 internal 与 external
  pass/fail；用户决定通过后是否生成并提交 leaderboard 包。

## Current State

- 当前 promoted checkpoint 为
  `d1_time_ramp_g050_d2_cooccur_lift_aux_w050_20260729.pkl`；其 Dataset1 CSV
  是 byte-identical 的 time-ramp 冠军成员，SHA-256
  `81285eddfba612b9c05e96075be29a15718d8a94839d143e3083bf5353644369`。
- 后续 cooccur-lift 晋升只改变 Dataset2；没有证据表明 Dataset1 在 7/26
  之后晋升过。
- 旧 full100 实验用 `200000 x 100 x 63` 缓存，静态 Setwise full MRR
  `+0.0014274006`，但低于旧门槛且最早 slice `-0.0000677395`，因此正确
  拒绝。
- time-ramp `gamma=0.5` 在单 validation 上 full `+0.0023190998` 并通过，
  但后来 rolling-origin 的同类时间递增融合在三个 gamma 上均有折回退，说明
  该路由不稳健。
- 现有 D1 rolling runner 训练 raw/Setwise heads，但只扫描 gamma，且没有
  gapped 折、静态 16 臂、K256 contract 或 external safety-only 链。
- 7/28 base-context rolling/external 仍为 `REMOTE_PENDING`，没有 selection
  lock、external receipt 或晋升资格，不是当前基线。

## Plan Rewrite Notes

| Existing item | Decision | Reason |
|---|---|---|
| 200k full-candidate cache | keep, rebuild at K256 | 复用已验证形状，同时把生产 K 绑定进 lineage |
| 折内 Setwise context 头 | keep and strengthen | 与 base/LGBM 一起折内重训，避免监督头泄漏 |
| time-ramp gamma 扫描 | remove | 本轮目标就是替换不稳健的时间路由 |
| 静态权重扫描 | add and freeze | 16 个离散臂足以覆盖低到高 Setwise 容量 |
| 旧单 validation 三片 gate | replace | 改成至少三折 near + 三折 gapped |
| external | keep, reinterpret | 只作方向性安全门，不作效应量估计 |
| 通过后自动打包 | defer | 用户本轮只授权完整验证，不授权 leaderboard 提交 |

## Drift Diagnosis

- **Goal drift**: 若只重跑旧 full100 单 validation，仍无法判断静态层是否能在
  部署距离下替代 time-ramp。
- **Phase drift**: 若在 gapped specs/16 臂锁定前读取指标，静态扫描会退化为
  事后调参。
- **Validation drift**: 若 baseline 直接用 full-origin 现成头而 candidate
  折内重训，会混入监督泄漏和训练规模差异。
- **Compatibility drift**: K256 必须同时施加给同折 baseline 与 candidate；
  只改 candidate 会把 K 效果混进内部结构判断。
- **Cleanup drift**: 不顺手修复或继续 7/28 base-context 候选，也不改
  Dataset2。

## Priority Rationale

1. 先从无标签时间元数据和 current champion state 冻结 plan/candidate
   fingerprints。
2. 再用 RED/GREEN 固化静态权重、折内共享头、gapped chronology 与 external
   隔离。
3. 先物化一次 K256 200k cache，并复用冻结候选矩阵/sidecars，减少重复计算。
4. internal 全过后才 full-origin refit 和 external；失败则立即停止。

## Assumptions and Open Decisions

| Item | Status | Impact | Owner / Next step |
|---|---|---|---|
| 当前 promoted checkpoint 的 Dataset1 state 与 time-ramp 冠军一致 | hash-check pending | 决定 baseline lineage | remote preflight |
| K256 指两个预测历史限制都为 256 | confirmed | 决定 cache 特征 | plan lock |
| gapped quantiles 使用 p75/p90/p100 | confirmed | 决定远视界覆盖 | metadata preflight |
| gapped 秒数可由无标签 query-time 元数据计算 | expected | 不足则不能冻结折 | metadata preflight |
| 每折可训练 exact base MLP/LGBM/Setwise 路径 | code-path confirmed, asset pending | 决定是否为 exact integrated comparison | dry-run |
| external Dataset1 holdout lineage 可一次性绑定 | pending | 决定最终 safety gate | internal pass 后检查 |

## Phases

### Phase 0: Goal and Plan Lock

- **Purpose**: 在任何训练或指标读取前锁定实验。
- **Entry condition**: 当前 D1 谱系和旧失败证据已审计。
- **Phase rules**:
  - 只读取配置、哈希、shape 和时间边界，不读取 external 指标。
  - gapped specs 一经写入不可覆盖。
- **Todos**:
  - [x] 写入本目标文档。
    - **Surface**: `docs/experiments/`
    - **Proof**: 文档包含目标、边界、阶段、门槛和 stop rule。
    - **Depends on**: none
  - [ ] 生成机器可读 preregistration 与 validation plan。
    - **Surface**: JSON plan
    - **Proof**: freeze/preflight 通过，16 臂与 K256 哈希绑定。
    - **Depends on**: metadata preflight
- **Exit proof**: immutable plan lock SHA-256。
- **Stop condition**: current D1 state 无法证明来自 promoted champion。

### Phase 1: RED/GREEN Protocol and Runner

- **Purpose**: 让静态集成、折内训练和双视界隔离成为可测试契约。
- **Entry condition**: Phase 0 plan 字段已冻结。
- **Phase rules**:
  - 先 RED 后最小 GREEN。
  - 纯选择逻辑不导入 Jittor，不接受 external score path。
- **Todos**:
  - [ ] 添加 16 权重、constant blend、tie-break 的 RED。
    - **Surface**: focused tests
    - **Proof**: 缺失目标 API/行为而失败。
    - **Depends on**: Phase 0
  - [ ] 添加 fold head lineage、gapped chronology、external one-shot RED。
    - **Surface**: focused tests
    - **Proof**: 错误共享/越界读取被拒绝。
    - **Depends on**: Phase 0
  - [ ] 实现最小 runner/contract。
    - **Surface**: `src/`, `scripts/`
    - **Proof**: focused GREEN、相关回归和 Ruff。
    - **Depends on**: RED
- **Exit proof**: TDD evidence 文档与全绿命令。
- **Stop condition**: 实现要求增加新权重、seed、gamma 或读取 external。

### Phase 2: K256 Cache and Dual-Horizon Materialization

- **Purpose**: 生成一次冻结的 200k 训练缓存和 near/gapped score assets。
- **Entry condition**: Phase 1 GREEN，plan lock 存在。
- **Phase rules**:
  - candidate count 固定 100，train rows 固定 200k。
  - 候选矩阵、row/time sidecars 先冻结；重算特征不得改变候选。
  - 使用服务器可用 CPU/GPU 并行，但 exact feature parity 必须通过。
- **Todos**:
  - [ ] K256 recent-200k cache。
    - **Surface**: mmap arrays/cache report
    - **Proof**: shape、lineage、hash、RSS/elapsed report。
    - **Depends on**: plan lock
  - [ ] near + p75/p90/p100 gapped assets。
    - **Surface**: rolling manifest/score caches
    - **Proof**: train max time < score min time 且 gap 达标。
    - **Depends on**: cache
- **Exit proof**: materialization complete manifest。
- **Stop condition**: exact candidate parity、cache hash 或 gapped chronology 失败。

### Phase 3: Fold-Local Training and Static Selection

- **Purpose**: 用双视界内部证据锁定一个静态权重或拒绝全家族。
- **Entry condition**: Phase 2 全部 assets/hash 通过。
- **Phase rules**:
  - 每折从头训练 base MLP、LGBM、Setwise context 头。
  - 16 个权重只复用同折已冻结分数，不能额外重训或改变特征。
  - 不读取 external。
- **Todos**:
  - [ ] 三折 near 和三折 gapped 训练/评分。
    - **Surface**: fold heads/scores/reports
    - **Proof**: 每折 head 训练区间和 SHA-256。
    - **Depends on**: Phase 2
  - [ ] 标准 selector 锁定唯一权重。
    - **Surface**: selection report/lock
    - **Proof**: near non-decrease + gapped strict improvement。
    - **Depends on**: fold scores
- **Exit proof**: selected lock 或 no-eligible-candidate report。
- **Stop condition**: 任一必要门禁失败时不进入 external。

### Phase 4: Full-Origin Refit and External Safety Gate

- **Purpose**: 判断锁定静态候选是否相对当前线上 D1 安全。
- **Entry condition**: Phase 3 selection lock。
- **Phase rules**:
  - 使用同一个 K256 200k cache 重训一次 full-origin Setwise 头。
  - selected weight 不可改变。
  - external receipt 只能创建一次；报告只读 pass/fail。
- **Todos**:
  - [ ] full-origin refit 并绑定 model hash。
    - **Surface**: final head/model report
    - **Proof**: 200k rows、K256、selected weight 全匹配。
    - **Depends on**: selection lock
  - [ ] 一次性 external safety gate。
    - **Surface**: receipt/evaluation report
    - **Proof**: 七门、当前 D1 baseline hash、candidate fingerprint。
    - **Depends on**: refit
- **Exit proof**: accepted/rejected external report。
- **Stop condition**: external 已被同一 lock 打开、baseline 漂移或任一安全门失败。

## Dry-Run Findings

- 现有 cache builder 能生成 Dataset1 recent-200k full-100 cache，但 K 来自
  checkpoint config；本轮需先构造只改变两个 history limits 的 K256 派生
  execution state，并将其 hash 写入 plan。
- 现有 D1 rolling script只比较 raw vs Setwise time-ramp，不能直接复用为
  static-vs-ramp exact integrated selector；需要一个新 runner 或受保护扩展。
- 静态扫描在 score arrays 上近乎零成本；主要成本是 K256 feature cache 与
  每折三类监督头训练。
- serving 当前只有 time-ramp overlay，没有独立 static Setwise overlay。
  本轮 external 前可直接用冻结 score arrays；只有 safety gate 通过、准备生产
  checkpoint 时才增加 serving 字段，避免提前扩大改动。

## Final Validation

- `uv run pytest` 的 focused RED/GREEN 与相关 standard-validation 回归。
- `uv run ruff check` 覆盖本轮新增/修改 Python。
- plan/fold/head/score/cache SHA-256 全链校验。
- near/gapped chronology 和 K256 双 limit 校验。
- external receipt one-shot 重入失败测试。
- external 只输出 safety verdict，不输出或消费权重回扫建议。

## First Execution Step

读取 promoted checkpoint 的 Dataset1 state/config 与无标签 query-time 元数据，
冻结 K256 派生 state、p75/p90/p100 gap 秒数和 16 个 candidate fingerprints；
随后添加静态网格与 external 隔离的 RED 测试。
