# Goal Document: Dataset2 多专家置信路由 v2

## Go / No-Go

- **Initial judgment**: Go
- **Final judgment**: No-Go for slice2/package
- **Reason**: 当前线上最高包证明了 query-level confidence routing 有效，且 v2 已在不重建缓存、不重训大模型的前提下完成实现与 slice1 前向选型；但 27 个冻结配置没有一个达到 slice1 `+0.001`。覆盖率不超过 25% 时的最佳增益只有 `+0.0001826203`，因此按协议停止，不读取 slice2、不生成包。

## Target Outcome

以当前 `1.3521011401636023` 多兴趣置信门控包为精确 fallback，构建一个可部署的多专家路由器。路由器仅在预测提升达到冻结阈值时，逐查询改选 v1、未门控 multi-interest 或窗口专家；否则输出必须与当前门控结果逐值一致。只用 slice0 训练、slice1 选型，SHA 锁定后才读取 slice2。未过门禁不生成包。

## Goal Definition

- **Type**: learning / quality / delivery
- **Boundary**:
  - Dataset2 专家固定为 `current_gate`、`v1_champion`、`multi_interest`、`window_ensemble`；
  - `current_gate` 是 fallback，不作为待预测 lift 的替代专家；
  - 替代专家顺序固定为 `v1_champion`、`multi_interest`、`window_ensemble`；
  - window expert 固定为 `0.80 × mean(recent100k, recent200k, recent200k_decay100k) + 0.20 × LightGBM`；
  - 只使用可由专家分数直接生成的 query-level descriptor；
  - 每个替代专家训练一个浅层 reward regressor，预测其相对 current gate 的 RR lift；
  - 超过阈值时选预测 lift 最大的专家，否则精确 fallback。
- **Non-goals**:
  - 不增加 percentile/robust-z 变体；
  - 不训练新 Setwise、GNN 或随机种子；
  - 不搜索窗口专家的内部权重；
  - 不加入需要生产端重建 base/proxy tensor 的描述量；
  - 本轮不实现连续或离散 soft alpha。
- **Deferred work**:
  - candidate-aligned 原始特征和 multi-interest proxy 支持度；
  - top-k residual Setwise；
  - activity-aware multi-interest centers。
- **Verification rule**:
  - descriptor 在候选列重排下不变；
  - 路由 tie 使用冻结专家顺序；
  - 未达 threshold 的行与 current gate 逐值一致；
  - `train-select` 不读取 slice2，selection report 写 SHA-256；
  - 独立 `gate` 先校验锁与输入哈希，再计算 slice2；
  - 只有 gate 通过才允许生成候选包，Dataset1 必须字节不变。
- **Evidence source**: RED/GREEN 测试、frozen-config、selection-report、evaluation-report、checkpoint/CSV/zip hash、线上分数。
- **Pass criteria**:
  - slice0 `[0,6667)` 训练，slice1 `[6667,13334)` 选择；
  - slice1 相对 current gate MRR 至少 `+0.001`；
  - slice1 路由覆盖率不超过 `25%`；
  - 选型在读取 slice2 前以 SHA-256 锁定；
  - 用 `[0,13334)` 重训锁定配置后，slice2 `[13334,20000)` 相对 current gate MRR 至少 `+0.001`；
  - slice2 路由覆盖率不超过 `25%`；
  - fallback 行逐值相等；
  - Dataset1 CSV SHA-256 保持 `6760fd966fde3a8de4693d06f27ec0f2458e49c44e2ea48fc79e1393434e6e2b`。
- **Confidence note**: slice2 没有用于 v2 选型，但该 20k 验证集已经被历史实验多次查看，因此它不是全局意义上的全新 holdout。即使离线通过，也只生成隔离候选包，最终由一次线上提交决定是否替换 `1.3521011401636023`。
- **Judgment owner**: 独立 gate 脚本授权候选包；线上分数授权冠军替换。

## Current State

- 当前生产 confidence gate validation MRR 为 `0.5495003773364516`，相对旧 v1 为 `+0.0025825588899633933`。
- 当前生产 gate 只使用 `champion_top_margin` 与 `expert_top1_agreement`；validation/test coverage 为 `12.665% / 15.4198%`。
- 未门控 multi-interest validation MRR 为 `0.5509812280402855`，但逐查询同时存在 `3353` 个改善和 `3088` 个恶化。
- multi-interest 相对 v1 的 oracle query gate delta 为 `+0.026229650740186645`，专家选择仍有明显理论空间。
- window ensemble 相对旧 v1 full `+0.0013961154`，slice0/slice2 为正，slice1 `-0.0002843004`；它适合被高置信路由，不适合全局平均。
- 三位底层专家的模型已存在；window validation probability 仍保存在远端结果目录，可直接复用或重算。

## Plan Rewrite Notes

| Existing item | Decision | Reason |
|---|---|---|
| 多兴趣二专家 confidence gate | keep as exact fallback | 它是当前线上最高且有可部署模型 |
| window uniform ensemble | rewrite as optional expert | 全局 slice1 下降，但其他片有互补 |
| candidate-aligned raw/proxy descriptors | defer | 生产端代价和 checkpoint 接口扩大，不属于最低成本 v2 |
| query-conditioned soft alpha | defer | 先验证专家选择本身，避免同时改变 routing 与融合 |
| 通用行内 percentile/robust-z | remove | 已在两片稳定退化 |
| 随机种子集成 | remove | full 已下降 `-0.0016566947` |

## Drift Diagnosis

- **Goal drift**: 不把路由实验扩成新专家训练或特征缓存重建。
- **Phase drift**: 先证明 descriptor 与 fallback 契约，再做昂贵的远端评分。
- **Validation drift**: 不以 in-sample full MRR 授权；slice1 和 slice2 必须分别达到冻结增益。
- **Compatibility drift**: current gate 是逐值 fallback，v2 不修改现有包或 checkpoint。
- **Cleanup drift**: 不顺手重构旧二专家 gate 或窗口训练脚本。

## Priority Rationale

- 最高风险是路由泄漏和错误 fallback；先用纯数组测试冻结行为。
- 第二风险是专家分数/候选列错位；所有输入必须验证 shape、hash 和候选顺序。
- 只有 slice1 达标才值得打开 slice2，更不应提前实现生产打包。

## Assumptions and Open Decisions

| Item | Status | Impact | Owner / Next step |
|---|---|---|---|
| current gate 可由已有 `confidence-gate.pkl` 和 v1/MI 分数精确重放 | assumed | 决定 baseline 是否可信 | 远端 preflight 精确复现报告 |
| window validation probability 文件仍在远端 | assumed | 可避免重新训练 | 资产清单/hash 检查 |
| 专家 CSV/模型可以生成 153,420×100 test scores | confirmed | gate 通过后可部署 | 复用现有打包与窗口模型 |
| score-only descriptor 足以产生增量 | unresolved | 决定 v2 是否值得继续 | slice1 forward selection |
| 专家 tie 顺序为 v1、MI、window | confirmed | 保证确定性 | 单元测试 |

## Phases

### Phase 1: 路由契约 RED

- **Purpose**: 在生产代码前冻结 score-only descriptor、阈值、tie 和 fallback 行为。
- **Entry condition**: 专家顺序和输入 shape 已冻结。
- **Phase rules**:
  - 只增加测试与 TDD 记录；
  - 测试必须包含候选列 permutation；
  - 测试必须证明 threshold 以下逐值 fallback；
  - 不导入 Jittor。
- **Todos**:
  - [x] descriptor 值、名称、shape 与 permutation 测试
    - **Surface**: `tests/test_hybrid_multi_expert_gate.py`
    - **Proof**: 因 API 缺失而 RED
    - **Depends on**: none
  - [x] 多专家最大 lift、threshold、tie/fallback 测试
    - **Surface**: 同上
    - **Proof**: 因路由 API 缺失而 RED
    - **Depends on**: none
- **Exit proof**: 最小测试因目标模块/API 不存在正确失败。
- **Stop condition**: descriptor 需要标签或候选位置才能成立。

### Phase 2: GREEN 与前向选择器

- **Purpose**: 实现部署友好的 score-only 多专家路由核心。
- **Entry condition**: RED 原因正确。
- **Phase rules**:
  - 使用浅层 `DecisionTreeRegressor`，不引入新依赖；
  - 每个替代专家独立预测相对 fallback 的 RR lift；
  - 配置、descriptor schema 与专家顺序进入模型契约；
  - selection helper 禁止读取 forward slice。
- **Todos**:
  - [x] 实现 score-only descriptors
    - **Surface**: `src/jgrec/rankers/hybrid/multi_expert_gate.py`
    - **Proof**: permutation/shape 测试 GREEN
    - **Depends on**: Phase 1
  - [x] 实现 fit/predict/route 与序列化模型
    - **Surface**: 同上
    - **Proof**: exact fallback、threshold、tie 测试 GREEN
    - **Depends on**: descriptor
  - [x] 实现 slice0→slice1 配置选择
    - **Surface**: 同上
    - **Proof**: NaN forward slice 不影响选择；固定 tie-break
    - **Depends on**: fit/predict
- **Exit proof**: 目标测试、相关 gate 回归和 Ruff 通过。
- **Stop condition**: 无法精确重放 current gate 或候选列不对齐。

### Phase 3: 远端评分、slice1 选型与 SHA 锁

- **Purpose**: 只用前两个时间片决定 v2 是否有资格读取 slice2。
- **Entry condition**: Phase 2 GREEN；远端输入哈希匹配。
- **Phase rules**:
  - 先写 frozen-config；
  - slice0 训练，slice1 选择；
  - 配置网格固定为 depth `{1,2,3}`、min leaf `{250,500,1000}`、threshold `{0.0025,0.005,0.01}`；
  - slice1 delta `< +0.001` 或 coverage `>25%` 时停止，不读取 slice2；
  - selection report 不包含 slice2/full 指标。
- **Todos**:
  - [x] 复算 v1、MI、current gate 与 window validation scores
    - **Surface**: 远端 20k cache/models
    - **Proof**: 已知 v1/MI/window/current gate 指标逐值复现
    - **Depends on**: Phase 2
  - [x] 选择配置并锁定
    - **Surface**: `selection-report.json/.sha256`
    - **Proof**: slice1 delta/coverage、输入/model hash、`selection_uses_forward_rows=false`
    - **Depends on**: 专家重放
- **Exit proof**: 合格配置被锁定，或明确 no-candidate 停止。
- **Stop condition**: 任一专家基线指标无法复现。

### Phase 4: 独立 slice2 gate 与条件打包

- **Purpose**: 验证锁定路由在下一时间片的增量并生成隔离候选包。
- **Entry condition**: selection SHA 与全部输入 hash 匹配。
- **Phase rules**:
  - 只评估锁定配置；
  - 用 `[0,13334)` 重训 router 后只报告 slice2 forward delta；
  - slice2 delta `< +0.001`、coverage `>25%` 或 fallback 不精确均拒绝；
  - rejected 不生成包、不修改当前冠军；
  - accepted 包保持 Dataset1 字节不变。
- **Todos**:
  - [x] 运行独立 gate
    - **Surface**: `evaluation-report.json`
    - **Proof**: slice1 无合格候选，协议在 gate 前停止；未读取 slice2，未创建 evaluation report
    - **Depends on**: Phase 3
  - [x] accepted 时生成 test window expert、router model 与 zip
    - **Surface**: isolated checkpoint/CSV/package
    - **Proof**: accepted 条件不成立，明确未生成 checkpoint/CSV/zip
    - **Depends on**: gate passed
- **Exit proof**: accepted 候选包或明确 rejected/no-candidate 报告。
- **Stop condition**: slice2 未达到冻结增益或生产 descriptor 不可复现。

## Dry-Run Findings

- 当前 production gate 本身用全 20k 训练过；把它作为固定 fallback 会使 v2 的 forward 比较偏保守，但不会虚增 v2。
- score-only descriptor 能从验证概率和测试 CSV 同样构建，避免再次出现“最终树用了生产端没有的 feature”。
- window 的新模型只有约 30KB；真正成本是生成 153,420×100 test scores，因此只在 slice2 gate 通过后执行。
- slice2 已被旧实验观察，不是全局盲集；必须保留线上一次性判决，不可凭离线结果直接覆盖冠军。

## Final Validation

- `uv run --no-sync pytest tests/test_hybrid_multi_expert_gate.py tests/test_hybrid_multi_interest_gate.py tests/test_hybrid_temporal_robust_selection.py -q`
- `uv run --no-sync ruff check src/jgrec/rankers/hybrid/multi_expert_gate.py tests/test_hybrid_multi_expert_gate.py scripts/run_dataset2_multi_expert_confidence_v2.py`
- 远端已知 expert 指标精确复现。
- `selection-report.sha256` 在 gate 前匹配。
- accepted 时 checkpoint reload、CSV/zip validator、Dataset1 SHA-256 全部通过。

## First Execution Step

新增 descriptor permutation 与 exact-fallback 的失败测试，确认多专家路由 API 尚不存在并为正确原因 RED。

## Execution Result

- 本地与 Linux 目标/回归测试均为 `14 passed`，Ruff 与 Python compile 通过。
- 四组 validation expert 指标及 current gate coverage/delta 均通过精确复现检查。
- v2 descriptor 数为 `74`，全部仅来自四组专家分数；没有标签、base feature 或 proxy tensor 进入生产 schema。
- slice1 current gate baseline MRR：`0.5510080326704802`。
- 27 个冻结配置中：
  - coverage≤25% 的最佳配置为 depth2 / leaf250，delta `+0.0001826203105353974`，coverage `11.7894%`；
  - 绝对最高 delta 为 `+0.000316930883151878`，但 coverage `57.0721%`；
  - 没有配置达到冻结的 `+0.001`，selection status 为 `no_eligible_candidate`。
- selection report SHA-256：`8aecf8cdd198b08aee12c8d852c9d3ba4a541bc39be73f72a2af7d9512d40b6a`，sidecar 已核验。
- 按协议未运行 slice2 gate，因此没有 `evaluation-report.json`、没有 test expert 推理、没有候选包；当前 `1.3521011401636023` 包保持不变。
