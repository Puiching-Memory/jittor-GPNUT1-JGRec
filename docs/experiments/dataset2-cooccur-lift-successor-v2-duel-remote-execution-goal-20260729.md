# Goal Document: Dataset2 Cooccur-Lift Successor V2 远端产分与裁决

## Go / No-Go

- **Judgment**: Go after deterministic-execution amendment
- **Reason**: 双候选、baseline、双视界门禁和 plan lock 已在任何 successor
  指标前冻结，三档 gapped cache 也已完整物化。首次 duel 在产生任何 successor
  指标前被既有 `0.10732766809698313` V1 replay gate 拦截；后续审计已证明这是
  CUDA 训练非确定性。必须先冻结 CPU 双跑回放执行合同，不能放宽容差或绕过校验。

## Target Outcome

在 `8.134.210.227:22223` 的既有隔离工作区中，按冻结配置实现并运行
`cooccur_lift_full_only_v2` 与 `cooccur_lift_gap_aware_v2`，物化相同的三折 near、
三档 gapped 和可选 zero-short 分数，生成绑定 v1 baseline 的 rolling manifest，
最后由标准 selector 给出 selected 或 rejected 的唯一机器判决。

## Goal Definition

- **Type**: technical / operational / learning
- **Boundary**: 候选 feature/view/head 实现、自动测试、受控远端同步、远端
  preflight、near/gapped 训练与产分、rolling manifest、标准 selector 和结果回传。
- **Non-goals**:
  - 不打开 external，不生成 external receipt。
  - 不生成线上 checkpoint、submission 或 package。
  - 不改变冻结候选、权重 `0.50`、窗口、seed、容量、fold、门禁或 tie-break。
  - 不把历史 CUDA V1 fold score 当作确定性 CPU 新 V1 的等同性真值；它只保留为
    诊断对照。
  - 不复用其他项目环境，不把密码写入文件或日志。
- **Deferred work**:
  - selector 通过后的 one-shot external。
  - external 通过后的 checkpoint/package/上线。
  - 若远折通过而线上缩水，source-conditioned 与 source/candidate joint audit。
- **Verification rule**: 本地 RED/GREEN 与相关回归通过；执行补充合同必须冻结
  CPU training、双跑、原 `rtol=2e-5`/`atol=2e-6`、旧 CUDA manifest
  `diagnostic_only`；远端代码/plan/config 哈希与本地一致；六个正式 fold 的
  score artifact、candidate fingerprint、baseline hash 和 chronology 均通过
  selector 合约；selector 生成不可覆盖的 selection report。
- **Evidence source**: pytest、Ruff、远端 SHA-256、run contract、rolling
  manifest、selection report/lock 和训练日志。
- **Pass criteria**:
  - 两候选 feature schema 分别为 64→192 与 66→198；
  - gap-aware 的 support 严格按冻结公式生成，并在训练证据中同时出现 0/1；
  - V1、full-only、gap-aware 及远折 prior Setwise head 均在 CPU 上双跑，
    state/loss/probability replay 通过未放宽的原容差；
  - 三个 near 与三个 gapped fold 对两候选使用相同训练/评分行；
  - rolling manifest 的 `plan_lock_sha256` 与 `baseline_sha256` 精确匹配；
  - selector 正常完成，且没有 external/package 产物；
  - 无权重、seed、窗口、容量或候选回扫。
- **Confidence note**: fold/score/hash 合约能证明执行与预注册一致；是否有候选胜出
  由真实内部分数决定，rejected 也是合法完成结果。
- **Judgment owner**: 标准 rolling selector；external 是否打开仍由用户另行授权。

## Current State

- validation plan SHA-256：
  `07f0ac9a244077a3ad8e7e3cd76bd7c95c6b7c00d8a42766601785f069e95efd`。
- plan-v2 lock SHA-256：
  `3519a496a5807b18e4b6f0aefdfd9c92dce34cfdf188c0f719458785f2ed6d98`。
- 远端工作区为 `/home/edu/workspace/jittor-GPNUT1-JGRec`，owner=`edu`，
  mode=`775`。
- 远端根盘约 `1.4T` 可用；RTX 4090 可见；项目 `.venv` 为 Python 3.12.13。
- `gapped-cache-v3-parallel4-dual` 已完成并通过 18 个 artifact hash 校验；
  external 未读取。
- 首次自动 duel 于 `2026-07-29T16:33:45+08:00` 在 `fold-0` V1 replay
  处失败，误差精确为 `0.10732766809698313`；没有生成 successor score、
  rolling manifest、selection lock 或 external 产物。
- 后续真实数据审计证明 CUDA 固定 seed 双跑仍有
  `0.05319197303453227`/`0.048016706738819914` 概率误差，而 CPU 在 2k/20k
  行均为 `0.0`；新 full-data V1 CPU 双跑也为 `0.0`，线上安全门通过。
- 正式 `v5-cpu-replay-wiringfix` 已完成六折 CPU 双跑；原始 manifest 漏写
  冻结 `baseline_sha256`，未覆盖原件，仅生成只增加该字段的审计副本。
- 标准 selector 已选中 `cooccur_lift_gap_aware_v2`；selection report/lock 已
  回传本地，external 保持关闭。

## Plan Rewrite Notes

| Existing item | Decision | Reason |
|---|---|---|
| successor 预注册目标 | keep as immutable input | 候选与判决不能随远端结果修改 |
| 候选实现 | move first | 没有可测试 materializer/head 就不能合法产分 |
| 直接启动训练 | remove | 必须先 RED/GREEN、哈希同步和 dry-run |
| external | defer | one-shot 动作超出本轮隐含授权 |
| v1 权重扫描 | remove | 冻结规则明确禁止 |

## Drift Diagnosis

- **Goal drift**: 远端训练若顺手重扫权重，会把验证变成新实验。
- **Phase drift**: gap-aware 若只用 near 训练，support 恒 1，候选定义失效。
- **Validation drift**: 只看平均指标或 deployment mixture 会绕过逐折硬门。
- **Compatibility drift**: 远端旧代码若不支持 baseline binding，不得生成正式
  manifest。
- **Cleanup drift**: 不更新远端无关源码、环境或历史结果。
- **Control-plane starvation**: “任务内存未 OOM”不等于服务器可运维。以后启动
  重物化前必须为 OS/sshd 保留资源冗余，限制任务 CPU/IO 优先级和并发；若启动
  后 SSH banner 退化，则不得叠加第二个任务。

## Priority Rationale

- 先验证候选特征和 support 状态，再碰远端 GPU，避免昂贵但不合约的训练。
- 同步使用精确文件清单和 SHA-256，不覆盖远端数据、checkpoint 与历史结果。
- selector 是本轮终点；无论 selected/rejected 都停止，不按结果追加实验。

## Assumptions and Open Decisions

| Item | Status | Impact | Owner / Next step |
|---|---|---|---|
| 远端已有 v1 baseline score/cache 与训练输入 | resolved | near 复用；gapped 必须补物化 | 旧 cache 仅 179 天 |
| 六折可从现有 rolling 数据按时间边界重建 | resolved | near 复用，gapped 使用精确历史行 | dry-run 已逐行通过 |
| gap-aware head 可在一个 outer fold 内看到 support 0/1 | required | 否则新增特征不可学习 | 测试与 run contract |
| zero-short 是零成本复评分 | assumed | 不得触发额外模型选择 | 实现审计 |
| selector 通过后自动开 external | rejected | one-shot 风险 | 必须另获用户授权 |
| 重任务可以吃满剩余资源 | rejected | 会饿死 sshd/control plane | 后续预留至少 25% RAM 或 8 GiB（取较大者），并保留单任务并发 |
| 旧 CUDA V1 fold score 是新 V1 replay 真值 | rejected | 已由 `0.10733` 和 CPU/GPU 双跑证据证伪 | 旧值只记录 diagnostic drift |
| 新 V1 与两个 V2 head 的训练设备 | resolved | 防止同一 seed 产生不同模型 | CPU 双跑，原容差不变 |

## Phases

### Phase 1: 远端与资产只读审计

- **Purpose**: 证明运行前提和安全边界。
- **Entry condition**: 用户提供服务器连接信息。
- **Phase rules**:
  - 只读；不上传、不建结果目录、不启动训练。
  - 不输出或保存密码。
- **Todos**:
  - [x] 验证身份、工作区、磁盘、GPU、Python 与相关进程。
    - **Surface**: remote host
    - **Proof**: SSH 探测输出
    - **Depends on**: none
  - [x] 核对 v1 checkpoint、cache、rolling fold 输入及其 SHA-256。
    - **Surface**: remote artifacts
    - **Proof**: 资产清单与 frozen config 对照
    - **Depends on**: SSH
- **Exit proof**: 所有训练输入存在且来源明确，或列出必须补物化的最小资产。
- **Stop condition**: baseline/hash 不一致、已有同名任务运行、磁盘/GPU 不安全。
- **Operational reserve**: 正式重任务还必须满足 `MemAvailable` 在预计峰值之外
  留有 `max(25% RAM, 8 GiB)`，以 `nice`/`ionice` 降低 CPU/IO 优先级，并用
  小批量 smoke 验证 SSH banner 仍可在 10 秒内建立。

### Phase 2: 候选执行路径 TDD

- **Purpose**: 用自动测试证明两个候选严格复现冻结形态。
- **Entry condition**: 远端资产足以支持一个最小本地行为切片。
- **Phase rules**:
  - 每个行为先 RED；只做使其 GREEN 的最小实现。
  - 不添加任何可调候选空间。
- **Todos**:
  - [x] RED/GREEN：full-only 删除 short 及三份 context 通道。
    - **Surface**: cooccur-lift feature view / tests
    - **Proof**: shape、值、不可变性测试
    - **Depends on**: none
  - [x] RED/GREEN：gap-aware support 公式、边界与 0/1 训练覆盖。
    - **Surface**: feature view / fold builder / tests
    - **Proof**: near=1、gap≥w=0、禁止 value-proxy
    - **Depends on**: full-only slice
  - [x] RED/GREEN：runner 固定权重、fold、seed、baseline hash 与 manifest。
    - **Surface**: successor runner / tests
    - **Proof**: contract test
    - **Depends on**: candidate views
- **Exit proof**: 新测试及 cooccur/standard-validation 相关回归全绿，Ruff 通过。
- **Stop condition**: 实现需要改变 frozen config 或从指标决定行为。

### Phase 3: 受控同步与远端 smoke

- **Purpose**: 让远端执行字节级匹配本地已验证实现。
- **Entry condition**: Phase 2 GREEN。
- **Phase rules**:
  - 只同步明确列出的源码、测试、config、plan 和 lock。
  - 同步前后计算 SHA-256；不覆盖远端数据与历史结果。
- **Todos**:
  - [x] 上传文件并验证本地/远端哈希。
    - **Surface**: remote workspace
    - **Proof**: hash manifest
    - **Depends on**: Phase 2
  - [x] 运行无正式产分的 contract smoke。
    - **Surface**: remote `.venv`
    - **Proof**: pytest/preflight
    - **Depends on**: sync
- **Exit proof**: 远端实现、plan、lock、候选配置与本地一致。
- **Stop condition**: 远端工作树冲突覆盖未知改动或 smoke 不通过。

### Phase 4: 六折产分

- **Purpose**: 生成 selector 所需的全部内部证据。
- **Entry condition**: 远端空闲复查、Phase 3 通过。
- **Additional entry condition**: deterministic-execution amendment 已在任何
  successor 指标前冻结并通过 RED/GREEN。
- **Phase rules**:
  - 两候选同折同数据；失败不改参数重跑。
  - 可断点恢复只复用哈希一致的已完成 fold。
  - zero-short 不参与选择。
- **Todos**:
  - [x] 生成三折 near 与 optional zero-short。
    - **Surface**: remote result directory
    - **Proof**: per-fold contracts/scores/hashes
    - **Depends on**: Phase 3
  - [x] 生成 P75/P90/P100 gapped 历史 cache。
    - **Superseded proof**: 顺序 PID `62698` 在 65,536 行处经授权 TERM；
      partial 目录保留。
    - **Runtime proof**: 最终正式目录 `gapped-cache-v3-parallel4-dual`；
      18 个 artifact 完整并通过 hash 复验。
    - **Failed continuation proof**: 旧 watcher 在 cache 后启动 duel，但在任何
      successor score 前被 `0.10732766809698313` V1 replay 拒绝；external
      保持关闭。
    - **Surface**: remote result directory
    - **Proof**: gap、support coverage、scores/hashes
    - **Depends on**: near runner proven
  - [x] 组装 rolling manifest。
    - **Surface**: JSON contract
    - **Proof**: plan/baseline/candidate hashes
    - **Depends on**: all folds
- **Exit proof**: manifest 静态校验通过，所有正式 score artifact 完整。
- **Stop condition**: OOM、输入缺失、support 无 0/1 覆盖、任一哈希漂移。

### Phase 5: 标准裁决与回传

- **Purpose**: 在冻结规则下得到唯一内部判决。
- **Entry condition**: 完整 rolling manifest。
- **Phase rules**:
  - 只调用标准 selector；不人工挑候选。
  - selected 或 rejected 后都停止。
  - 不打开 external。
- **Todos**:
  - [x] 运行 selector 并审计 report/lock。
    - **Surface**: selection artifacts
    - **Proof**: 每折 gates、selection order、hash chain
    - **Depends on**: Phase 4
  - [x] 回传小型 JSON/日志与结果文档。
    - **Surface**: local docs/result
    - **Proof**: SHA-256 与远端一致
    - **Depends on**: selector
- **Exit proof**: 本地保存可复核的 selected/rejected 机器判决。
- **Stop condition**: selector 合约错误或试图读取 external。

## Dry-Run Findings

- 本轮不能直接复用旧 v1 trainer：full-only 的输入维度不同，gap-aware 还要求
  support 0/1 同时进入训练证据。
- plan 声明 baseline 后，rolling manifest 必须新增顶层 `baseline_sha256`；
  远端旧脚本若漏写会被新 selector 拒绝。
- formal result directory 必须在代码与资产哈希确认后才创建，避免半成品冒充正式
  运行。
- external 是不可逆的一次性门；服务器凭据不构成打开 external 的授权。
- 旧 near manifest 的 prior champion score 仍可复用，但其中 CUDA 训练的 V1
  `candidate-w0.5` 只能用于报告 legacy drift，不能再作为 CPU 新 V1 等同性门。
- “删掉 replay 检查”会绕过校验；正确修复是把等同性对象换成同合同、同 seed、
  同 CPU 的第二次独立训练，并让任一 state/loss/probability 不一致继续硬失败。

## Final Validation

- 本地与远端聚焦 pytest、Ruff。
- 候选 config、plan、lock、runner 与远端 SHA-256 一致。
- manifest 合同静态审计。
- selector report 中逐折 near/gapped gates 与 frozen rule 一致。
- 结果目录不存在 external receipt、checkpoint、ZIP 或 submission。

## First Execution Step

先为 deterministic-execution amendment 增加 RED：GPU training、单跑、
容差漂移或把 legacy CUDA score 设为 authoritative 都必须被拒绝。
