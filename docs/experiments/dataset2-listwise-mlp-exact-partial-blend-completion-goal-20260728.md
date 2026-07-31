# Goal Document: 完成 A2 的 listwise-MLP 精确部分混合

## Go / No-Go

- **Judgment**: Go
- **Reason**: Two-Tower 完整 reranker `w=0.20` 已线上证伪，但 A2 的另一条
  独立正信号——基础 listwise-MLP——尚未对当前冠军做多折精确集成。远端已有
  200k×100 特征、winning `short_none` train/val 分数、当前冠军 checkpoint 和
  空闲 GPU，可以在不读取新 external 指标的前提下完成真实 rolling selection。

## Target Outcome

在当前 Dataset2 冠军结构上完成 listwise-MLP 外层部分混合：

1. 三个 rolling-origin 折分别训练 fold-exact Setwise 基线头和基础
   listwise-MLP 辅助头；
2. 对此前已冻结的六个权重生成最终混合分数并通过跨折稳定性门禁锁定唯一权重；
3. 锁权重后训练全量辅助头，只对 official 20k external 计算一次指标；
4. external 通过则生成一个保留冠军 Dataset1、只改变 Dataset2 分数的
   `result.zip`；失败则关闭 A2，不生成包。

## Goal Definition

- **Type**: technical / learning / quality / delivery
- **Boundary**:
  - 新 integration id：
    `listwise_mlp_exact_current_champion_v1`。
  - 当前冠军结构固定为
    `0.80 * Setwise(short_none 50/40k) + 0.20 * frozen LightGBM`。
  - 辅助专家是 63 维基础特征上的 listwise-MLP，不使用 Setwise 相对化通道。
  - rolling 折固定：
    `[0,79909)->[79909,118816)`、
    `[0,118816)->[118816,159804)`、
    `[0,159804)->[159804,200000)`。
  - Setwise fold 头固定训练 4 epochs；listwise-MLP 固定训练 5 epochs；
    batch 256、lr 0.001、seed 60，不在评分折早停。
  - 权重只允许使用 2026-07-28 线上 Two-Tower 结果出现前已经写盘的
    `0.05/0.10/0.20/0.30/0.40/0.50`；不得添加邻近点。
  - 精确候选公式：
    `candidate_w = (1-w) * fold_champion + w * fold_listwise_mlp`。
- **Non-goals**:
  - 不再扫描或复活 Two-Tower full-reranker v1。
  - 不调 epochs、学习率、隐藏维度、特征列、GNN 边数或 LGBM。
  - 不根据 rolling、external 或 leaderboard 结果增加权重点。
  - 不覆盖当前冠军 checkpoint/package。
  - 不自动提交 leaderboard。
- **Deferred work**:
  - 若线上胜出，再把辅助头与外层权重正式写入 checkpoint runtime。
  - refit 后训练—服务分布漂移修复。
- **Verification rule**:
  - 每折评分行严格晚于训练行；Setwise 与辅助 MLP 只在该折前缀上训练。
  - frozen LightGBM 来自早于首个评分折的既有 50k 训练阶段，并在所有候选中
    保持完全相同。
  - rolling selector 使用
    `exact_integrated_rolling_weight_selection_v1`，报告 MRR、Hit@1/3/10、
    NDCG@10、平均排名和 query movement。
  - 每折 MRR/NDCG@10 不下降，pooled Hit@K/mean-rank 不退化且改善 query 多于
    恶化 query，才有资格锁权重。
  - 通过者按最差折 MRR 增益优先；external 不参与选权重。
  - external receipt 在指标读取前独占创建；第二次调用必须失败。
- **Evidence source**: RED/GREEN tests、frozen config、fold reports、模型与分数
  SHA-256、selection report/lock、external receipt/report、package hashes。
- **Pass criteria**:
  - 三折全部完成、shape/hash/fingerprint/时间关系全部通过；
  - 至少一个权重通过稳定性硬门禁；
  - external 锁定候选 MRR 严格高于冠军，MRR/NDCG/Hit@K/mean-rank 和 query
    movement 门禁通过；
  - 通过时生成的 Dataset1 CSV 与冠军逐字节相同，Dataset2 为锁定公式精确回放，
    ZIP 仅含两个合法成员；
  - 失败时没有提交包。
- **Confidence note**: 200k cache 中除 `gnn_short` 外的 encoder 特征是冻结资产，
  并非每折重新编码；本轮 rolling 主要验证融合头和外层权重的时间稳定性。official
  20k 已被其他实验使用，只能视为对本 integration id 的一次性长跨度 gate，
  不能宣称为全局未污染盲集。
- **Judgment owner**: rolling 稳定性门禁锁权重；external 一次性门禁决定是否
  打包；用户提供的线上成绩决定是否晋升冠军。

## Current State

- 线上冠军：`1.3557002251184347`。
- Two-Tower A2 包：`1.3545061936665996`，已关闭。
- 历史基础 listwise-MLP 独立 MRR 相对旧 pointwise MLP
  `+0.0066838965`；旧固定 `0.07` 融合仅 `+0.0002067896` 且一片下降。
- 当前冠军 official 20k MRR：`0.5485470648527594`。
- 当前冠军是 Setwise listwise 头与 LGBM 的 `0.80/0.20` 混合；旧基础
  listwise-MLP 不在当前服务路径中。
- 远端可用：
  - 当前冠军 checkpoint 约 4.7GB；
  - train cache `200000×100×63`；
  - external cache `20000×100×63`；
  - winning `short_none` train/val scores；
  - 4090 GPU 空闲，磁盘约 1.4TB 可用。

## Plan Rewrite Notes

| Existing item | Decision | Reason |
|---|---|---|
| 只完成稳健 selector 框架 | keep as foundation | 现在接入真实训练与分数资产 |
| Two-Tower 邻近权重扫描 | remove | 已有线上反馈，继续扫描会 leaderboard 过拟合 |
| listwise-MLP 旧 50k/32-candidate 模型 | replace | 对当前 200k/100-candidate/short_none 表示重新训练 |
| 单一 20k validation 选权重 | remove | 改为三个 rolling 折选权重 |
| 旧六权重网格 | keep frozen | 它在 Two-Tower 线上结果前已写盘，不是事后新增 |
| external 直接扫描 | remove | 只评估 rolling 锁定的一个权重 |
| checkpoint runtime 接线 | defer until online win | 本轮用户需要的是可提交包，先验证线上价值 |

## Drift Diagnosis

- **Goal drift**: 继续完善通用框架但不训练真实折，不能完成 A2。
- **Phase drift**: external 前训练全量模型可以，但不得在 selection lock 前计算
  external 指标。
- **Validation drift**: 仅复用旧 listwise-MLP 单折成绩不能证明对当前冠军有互补性。
- **Compatibility drift**: 旧 MLP 用旧 `gnn_short` 表示；本轮必须在 winning
  `short_none` overlay 上训练和服务。
- **Cleanup drift**: 不顺手重构 GNN、LGBM 或 checkpoint 格式。

## Priority Rationale

- 先做一折 smoke，验证 fold-exact 基线/辅助头、模型保存与 score manifest；
  再投入完整三折 GPU 时间。
- rolling 先于 full/external，确保任何权重都不能由 external 反向决定。
- 只有 external 通过才做 test 特征编码和大包生产，避免无效的 4.7GB checkpoint
  推理成本。

## Assumptions and Open Decisions

| Item | Status | Impact | Owner / Next step |
|---|---|---|---|
| frozen LGBM 训练行早于首评分折 | confirmed by cache lineage | 可作为三折共享固定专家 | runner 写入 provenance |
| fold Setwise 固定 4 epochs | confirmed from winning head best epoch | 不读取评分折早停 | frozen config |
| auxiliary MLP 固定 5 epochs | confirmed from历史正信号 | 保留原实验因果变量 | frozen config |
| 旧六权重网格不是线上事后扫描 | confirmed by既有 frozen-config | 允许复用 | 绑定源 config SHA |
| external 通过后能生成 test 基础特征 | assumed, standard encoder path available | 决定是否能打包 | 仅通过后执行 |

## Phases

### Phase 1: 目标冻结与训练 runner RED/GREEN

- **Purpose**: 在昂贵训练前固定 fold、模型和精确候选产物契约。
- **Entry condition**: 本目标文档写盘。
- **Phase rules**:
  - 新生产行为先写 RED。
  - runner 不接受 external cache 参数。
  - 所有权重来自 frozen source config，不接受 CLI 临时扩展。
- **Todos**:
  - [ ] 为 fold 时间关系、固定 epochs、精确公式、manifest 和模型 archive 写测试。
    - **Surface**: tests
    - **Proof**: 目标模块/函数缺失导致 RED。
    - **Depends on**: none
  - [ ] 实现 rolling fold 训练、评分和 manifest 物化。
    - **Surface**: source module + remote runner
    - **Proof**: focused GREEN、Ruff、smoke report。
    - **Depends on**: RED
- **Exit proof**: 小型合成 cache 通过，remote smoke 能产出一个合法 fold。
- **Stop condition**: 无法让 Setwise/auxiliary 仅使用 fold 前缀训练。

### Phase 2: 三折训练与 selection lock

- **Purpose**: 用精确最终候选决定是否存在稳定部分混合权重。
- **Entry condition**: Phase 1 smoke 通过且 frozen config 已写盘。
- **Phase rules**:
  - 三折参数完全相同，仅训练前缀和 seed salt 随折变化。
  - 不读取 official 20k 指标。
  - selector 输出后不得修改任何 fold 分数或权重集合。
- **Todos**:
  - [ ] 顺序完成三折 Setwise 与 listwise-MLP 训练及评分。
    - **Surface**: remote models/scores/fold reports
    - **Proof**: finite scan、SHA、时间边界、候选 fingerprint。
    - **Depends on**: Phase 1
  - [ ] 运行稳健 selector。
    - **Surface**: selection report/lock
    - **Proof**: selected weight 或明确 rolling rejection。
    - **Depends on**: all folds
- **Exit proof**: 唯一 selection lock，或 A2 rolling No-Go。
- **Stop condition**: 任一折失败、无权重过门、或 artifact 漂移。

### Phase 3: 全量辅助头与一次 external

- **Purpose**: 验证锁定权重能否从 rolling 折迁移到当前冠军的长跨度 official
  validation。
- **Entry condition**: Phase 2 selection lock 存在。
- **Phase rules**:
  - 先训练全量 auxiliary MLP，再生成锁定权重的 external 分数。
  - external 只运行一次；失败不改权重。
  - 当前冠军 external 分数必须精确复现 `0.5485470648527594`。
- **Todos**:
  - [ ] 在全部 200k 行上训练固定 5 epochs auxiliary MLP。
    - **Surface**: full model/archive/report
    - **Proof**: epochs/loss/normalizer/model hash。
    - **Depends on**: selection lock
  - [ ] 物化 champion 与唯一候选 external scores，创建 manifest。
    - **Surface**: external score artifacts
    - **Proof**: champion exact replay、formula/hash。
    - **Depends on**: full model
  - [ ] 调用 one-shot external evaluator。
    - **Surface**: receipt/report
    - **Proof**: receipt 唯一；第二次不可运行。
    - **Depends on**: manifest
- **Exit proof**: external accepted 或 A2 external No-Go。
- **Stop condition**: baseline 无法精确复现、lock/hash 不匹配或 external 拒绝。

### Phase 4: 条件打包与交付

- **Purpose**: 仅把 external 通过的锁定候选变成用户可提交 ZIP。
- **Entry condition**: Phase 3 accepted。
- **Phase rules**:
  - Dataset1 从当前线上冠军 ZIP 原样复制。
  - Dataset2 只执行锁定公式，不重新归一化或调权重。
  - 不覆盖冠军包。
- **Todos**:
  - [ ] 用全量 auxiliary 模型评分官方 Dataset2 test 候选。
    - **Surface**: expert test probabilities
    - **Proof**: shape/finite/row-sum/candidate fingerprint。
    - **Depends on**: external accepted
  - [ ] 构建并双重回放 result.zip。
    - **Surface**: result directory/ZIP/report
    - **Proof**: Dataset1 byte identity、公式误差、ZIP members、SHA-256。
    - **Depends on**: test scores
- **Exit proof**: 一个可提交但未自动晋升的本地包。
- **Stop condition**: 任一结构、哈希或回放校验失败。

## Dry-Run Findings

- 当前 Setwise 本身已经是 listwise 头，但基础 MLP 使用不同的 63 维视角，A2
  检验的是其相对化以外的互补性，不是假设“冠军仍是 pointwise MLP”。
- fold baseline 必须重训 Setwise 头；直接用全量冠军 Setwise 模型评分历史折会
  引入未来标签信息。
- LGBM 是共享固定专家，因为其既有训练阶段早于首评分折，且候选/基线完全共用。
- full model 可以在 selection 后训练；只要 external 指标不提前读取，就不违反
  lock-before-external。
- 若 rolling 或 external 拒绝，A2 即以有证据的 No-Go 完成，而不是强行产包。

## Final Validation

- focused RED/GREEN tests 与相关 fusion/selector 回归。
- Ruff check + format check。
- remote frozen-config、三 fold report、selection report/lock。
- external receipt/report，确认 `weight_rescan_authorized=false`。
- 条件成立时：Dataset1 SHA、Dataset2 formula replay、CSV/ZIP validator 和本地
  SHA-256。

## First Execution Step

为 fold 配置与精确分数物化写 RED 测试；随后实现不包含 external 参数的
rolling runner。
