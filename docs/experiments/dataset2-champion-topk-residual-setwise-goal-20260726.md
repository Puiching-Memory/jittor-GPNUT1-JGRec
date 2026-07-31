# Goal Document: Dataset2 Champion top-k residual Setwise

## Go / No-Go

- **Initial judgment**: Go for implementation and slice1 selection.
- **Final judgment**: No-Go for slice2/package.
- **Reason**: 当前多专家路由的瓶颈不是缺少更多全局专家，而是无法可靠识别少量冠军错误。本轮把问题缩成冠军 top-k 内的局部纠错：只学习 positive 相对冠军难负例的 residual，并通过高置信 switch gate 保证默认输出逐值等于冠军。

## Target Outcome

构建一个二阶段 residual Setwise 校正器：

1. 训练时用冠军分数选择 top10/top20 negative，并始终加入 positive；
2. 用冻结冠军 log-score 作为 base logit，只训练候选 residual；
3. 使用 champion-rank `ΔMRR` 加权的 pairwise logistic loss；
4. 推理时只允许重排冠军 top-k，复用冠军原有 top-k 分值，top-k 外顺序与分值不变；
5. 仅当 residual 提议改变冠军 top1 且 switch gain 达到冻结阈值时应用，否则逐值返回冠军；
6. slice0 训练、slice1 选择，过门后才允许用 slice0+1 重训并读取 slice2。

## Goal Definition

- **Type**: learning / quality / delivery
- **Boundary**:
  - baseline 固定为线上分数 `1.3521011401636023` 的 Dataset2 `current_gate` validation score；
  - 训练数据固定为现有 20k validation cache；它作为线上 test 之前的二阶段有标签训练集；
  - 时间切分固定为 slice0 `[0,6667)`、slice1 `[6667,13334)`、slice2 `[13334,20000)`；
  - hard-negative width 只比较 `10` 与 `20`；
  - 每个训练组固定为 candidate 0 positive + champion top-k negatives；
  - feature 固定为 63 维 base + 9 维 multi-interest proxy，再做 Setwise v1 的 raw/row-mean/row-max，共 216 维；
  - loss 固定为 champion-rank `ΔMRR` 加权 pairwise softplus；
  - residual MLP 固定为 hidden 32、4 epochs、batch 256、Adam `lr=5e-4`、weight decay `1e-4`；
  - high-confidence switch threshold 网格固定为 `{0.05, 0.10, 0.20, 0.40}`；
  - 路由只在 residual-adjusted top1 不同于 champion top1 且 gain 达标时生效；
  - 应用 residual 时，只把冠军 top-k 的原 score multiset 按新顺序重新赋给同一 top-k 候选。
- **Why train on the 20k temporal prefix**:
  - 当前 production confidence gate 已把 validation 作为二阶段监督数据；
  - 直接复用已锁定 champion score，避免先重算 200k×100 的当前门控分数；
  - slice0→slice1→slice2 能直接检验 residual 规则的时间迁移；
  - 若双门禁通过，再在全 20k 上重训用于 test，仍不需要更改底层专家。
- **Non-goals**:
  - 不重新训练 full-100 Setwise、LightGBM、GNN 或 multi-interest 专家；
  - 不在训练 loss 中重新学习冠军绝对分数；
  - 不允许 residual 改动 top-k 外候选分值；
  - 不做随机种子集成；
  - 不搜索网络宽度、epoch、学习率、特征子集或 loss 变体；
  - 不与失败的多专家 v2/v3 router 叠加；
  - 不放宽 `+0.001` 或 25% coverage 门槛。
- **Deferred work**:
  - 200k 历史训练集上的 champion hard-negative mining；
  - 动态 LambdaRank 权重；
  - top-k 内 pairwise 蒸馏或 teacher calibration；
  - top1 以外的分层置信门控；
  - 双门禁通过后的 test base/proxy tensor 生产与打包。
- **Verification rule**:
  - hard negatives 必须排除 candidate 0，并按冠军分数稳定选 top-k；
  - LambdaMRR weight 必须等于 positive/negative champion rank 交换后的 reciprocal-rank 差；
  - 未过 threshold 的 query 输出必须逐值等于 champion；
  - routed query 的 score multiset 与 champion 完全一致，且 top-k 外逐值不变；
  - train-select 不使用 slice2 label/metric；
  - selection report 先以 SHA-256 锁定，独立 gate 才能读取 slice2。
- **Evidence source**: RED/GREEN 单元测试、训练历史、模型/输入 hash、selection report、独立 evaluation report、Dataset1/CSV/zip hash。
- **Pass criteria**:
  - slice1 相对 current gate MRR 至少 `+0.001`；
  - slice1 residual coverage 不超过 `25%`；
  - selection config 与模型在读取 slice2 前 SHA 锁定；
  - prefix `[0,13334)` 重训后，slice2 相对 current gate MRR 至少 `+0.001`；
  - slice2 residual coverage 不超过 `25%`；
  - fallback exact、top-k 外 exact、score multiset preserved 全部为 true；
  - Dataset1 CSV SHA-256 保持 `6760fd966fde3a8de4693d06f27ec0f2458e49c44e2ea48fc79e1393434e6e2b`。
- **Confidence note**: slice2 对本轮配置选择不可见，但已被历史实验观察，不是全局盲测；离线通过只授权隔离候选包，线上提交仍是最终判断。
- **Judgment owner**: selection 脚本决定是否解锁 slice2；独立 gate 决定是否授权打包；线上分数决定是否替换冠军。

## Current State

- 当前 Dataset2 线上冠军为 `1.3521011401636023`。
- 20k validation 上已保存四专家 score，其中 `current_gate` 可直接作为逐值 baseline。
- validation base tensor 为 `(20000,100,63)`，multi-interest proxy 为 `(20000,100,9)`，candidate 轴对齐且 positive 固定为 column 0。
- score-only 多专家 v2 在 coverage≤25% 时仅 `+0.0001826203`。
- top1 raw/proxy aligned v3 在相近 coverage 下为 `-0.0003988807`。
- 现有 Setwise MLP 学习全部 100 候选的绝对排序；仓库没有“冻结冠军 + top-k hard-negative residual + exact fallback”实现。

## Plan Rewrite Notes

| User proposal | Decision | Reason |
|---|---|---|
| 冠军 top10/top20 难负例 | keep | 两个宽度足以检验局部难度/覆盖范围，不扩大搜索 |
| positive vs hard negatives | keep | 训练组固定为 1+K，避免重新学习全部 100 候选 |
| pairwise/LambdaMRR 型损失 | rewrite as one frozen loss | 使用静态 champion rank `ΔMRR` 加权 pairwise softplus，避免同时比较两种 loss |
| 输出 residual | keep and make explicit | base log-score 不训练；模型只输出 additive residual |
| 默认保持冠军顺序 | strengthen | fallback query 逐值一致；routed query 也只置换冠军 top-k 的原分值 |
| 只纠正高置信错误 | make testable | 只有 adjusted top1 switch gain 达冻结阈值才路由 |

## Drift Diagnosis

- **Goal drift**: 不把 residual 实验扩成新专家训练、特征工程或多路由叠加。
- **Phase drift**: 先证明 hard-negative/loss/routing 契约，再训练；不边看 slice1 边改阈值。
- **Validation drift**: 不用 full 20k MRR 选 top-k/threshold；slice2 在锁定前不可参与判断。
- **Compatibility drift**: current gate 始终是逐 query exact fallback，旧冠军 checkpoint/CSV 不修改。
- **Cleanup drift**: 不重构现有 `fusion.py` 或旧 Setwise 训练路径。

## Priority Rationale

1. residual 路由若不能保证 exact fallback 和 score multiset preserved，就不具备生产安全性，因此先测试。
2. hard-negative candidate 轴错位会让训练完全失真，必须在 Jittor 训练前用纯 NumPy 锁死。
3. top10/top20 模型训练成本低，只有 slice1 真正过门后才值得实现 test 生产路径。

## Assumptions and Open Decisions

| Item | Status | Impact | Owner / Next step |
|---|---|---|---|
| validation candidate 0 始终为 positive | confirmed by existing cache contract | 决定训练标签 | preflight/hash |
| v2 保存的 current_gate score 与 base/proxy candidate 轴一致 | assumed, hash-verifiable | 决定 hard-negative 正确性 | frozen input contract |
| residual MLP 可用 6,667 条 slice0 学到稳定规则 | unresolved | 决定实验是否过门 | slice1 |
| 固定 switch gain 网格覆盖有效 coverage 区间 | assumed | 阈值过粗可能漏掉候选 | 预先冻结，不按 slice1 扩网格 |
| 双门禁通过后可重建 test base/proxy tensor | deferred | 决定能否打包 | 仅 gate pass 后处理 |

## Phases

### Phase 1: 核心契约 RED

- **Purpose**: 锁定 hard negatives、LambdaMRR weights 和安全 residual route。
- **Entry condition**: top-k、loss、阈值定义已冻结。
- **Phase rules**:
  - 只新增纯 NumPy 测试；
  - hard-negative 测试必须覆盖 positive 排除和稳定排序；
  - route 测试必须覆盖 threshold 下 exact fallback、top-k 外 exact、score multiset preserved；
  - 不导入 Jittor。
- **Todos**:
  - [ ] hard-negative indices 与 LambdaMRR weight RED
    - **Surface**: `tests/test_hybrid_champion_residual.py`
    - **Proof**: 目标模块/API 不存在而 RED
    - **Depends on**: none
  - [ ] high-confidence top-k residual route RED
    - **Surface**: 同上
    - **Proof**: exact fallback/multiset 断言先失败
    - **Depends on**: none
- **Exit proof**: 最小测试因缺少目标实现正确失败。
- **Stop condition**: score positive-column 契约与保存 artifact 不一致。

### Phase 2: 最小 GREEN 与训练器

- **Purpose**: 实现可复用 residual 核心和固定训练协议。
- **Entry condition**: RED 原因正确。
- **Phase rules**:
  - 核心选择/路由纯 NumPy；
  - Jittor 只在训练脚本/懒加载训练路径出现；
  - base champion log-score 不参与参数更新；
  - 不修改旧 Setwise API。
- **Todos**:
  - [ ] 实现 hard-negative、LambdaMRR weight、route
    - **Surface**: `src/jgrec/rankers/hybrid/champion_residual.py`
    - **Proof**: 核心测试 GREEN
    - **Depends on**: Phase 1
  - [ ] 实现固定 residual MLP 训练/推理
    - **Surface**: 同模块或实验脚本
    - **Proof**: synthetic loss 下降、模型 round-trip
    - **Depends on**: 核心 GREEN
- **Exit proof**: 目标测试、相关 Setwise 回归、Ruff 通过。
- **Stop condition**: Jittor 无法稳定计算加权 pairwise loss，或模型无法序列化重放。

### Phase 3: slice0→slice1 前向选型

- **Purpose**: 比较 top10/top20 与四个冻结 threshold。
- **Entry condition**: Linux 测试 GREEN，输入 hash 匹配。
- **Phase rules**:
  - 先写 frozen-config；
  - 两个模型都只训练 slice0；
  - slice1 比较 8 个冻结配置；
  - 不报告 slice2/full 指标；
  - 无配置满足 `+0.001` 且 coverage≤25% 时停止。
- **Todos**:
  - [ ] 训练 top10/top20 residual
    - **Surface**: 远端 validation cache
    - **Proof**: loss/history/model hash
    - **Depends on**: Phase 2
  - [ ] 选 threshold 并 SHA 锁定
    - **Surface**: `selection-report.json/.sha256`
    - **Proof**: slice1 delta/coverage 与 no-forward-use 字段
    - **Depends on**: 两个模型
- **Exit proof**: 合格配置锁定，或 `no_eligible_candidate`。
- **Stop condition**: baseline slice1 MRR 无法复现 `0.5510080326704802`。

### Phase 4: 独立 slice2 gate 与条件打包

- **Purpose**: 检验锁定 residual 在下一时间片的迁移。
- **Entry condition**: selection status 合格且 SHA/输入 hash 全部匹配。
- **Phase rules**:
  - 用 `[0,13334)` 重训锁定 top-k；
  - 只评估锁定 threshold；
  - 未通过 delta/coverage/safety 任一条件即拒绝；
  - accepted 后才在 full20k 重训并构建 test residual。
- **Todos**:
  - [ ] 运行独立 gate
    - **Surface**: `evaluation-report.json`
    - **Proof**: slice2 delta/coverage/safety
    - **Depends on**: Phase 3 pass
  - [ ] 条件生产与打包
    - **Surface**: isolated checkpoint/CSV/zip
    - **Proof**: reload、submission validator、Dataset1 hash
    - **Depends on**: gate pass
- **Exit proof**: accepted 包或明确 rejected/no-candidate。
- **Stop condition**: slice2 未达 `+0.001`、coverage 超限或任何 exactness 检查失败。

## Dry-Run Findings

- 训练仅需 slice0 的约 6,667×(K+1) candidate group，不需要物化完整 20k×100×216 tensor。
- 推理仍需为 20k×100 计算 residual，但可以 batch streaming，峰值内存可控。
- 用冠军 top-k 原分值重新赋值而非直接输出 arbitrary residual score，可以同时保证 tail 不动和 score scale 不漂移。
- 若 slice1 不过门，最贵的 test base/proxy 生产完全跳过。
- 当前冠军 gate 本身在 20k validation 上训练过，因此本轮比较是“相对已部署冠军的增量门禁”，不是独立无泄漏基准。

## Final Validation

- `uv run --no-sync pytest tests/test_hybrid_champion_residual.py tests/test_hybrid_setwise.py -q`
- `uv run --no-sync ruff check src/jgrec/rankers/hybrid/champion_residual.py tests/test_hybrid_champion_residual.py scripts/run_dataset2_champion_topk_residual_setwise.py`
- Linux 重复目标/回归测试。
- frozen input hashes、selection SHA、独立 gate 状态。
- 只有双门禁通过时执行模型 reload、CSV/zip validator 与 Dataset1 SHA 校验。

## First Execution Step

新增 hard-negative selection、static LambdaMRR weight 和 exact-fallback top-k residual route 的失败测试。

## Execution Result

- 两轮 RED 分别因目标模块缺失和 pairwise-loss API 缺失而正确失败。
- 本地与 Linux 目标/回归测试均为 `16 passed`，Ruff 通过。
- top10 loss 从 `0.3646783` 降至 `0.3442800`；top20 从 `0.1889222` 降至 `0.1823755`。
- top10 的四个 threshold 在 slice1 全部下降；最佳仍为 `-0.0009451313`。
- top20 的四个 threshold 全部非负；最佳 top20 / threshold 0.20：
  - baseline MRR `0.5510080326704802`
  - candidate delta `+0.00036119648221266676`
  - coverage `9.374531273436328%`
  - routed rows `625`
- 所有 trial 的 fallback exact、score multiset preserved、outside top-k exact 均为 true。
- 没有配置达到 `+0.001`，selection status 为 `no_eligible_candidate`。
- selection report SHA-256 为 `3432b1b8545c939fcc59421af9a0c2ee2146dccfd003f7073ed5c1f8d963fa4b`，sidecar 与模型 hash 已核验。
- 按协议未运行 slice2 gate，未生成 evaluation report 或候选包；当前线上冠军不变。
