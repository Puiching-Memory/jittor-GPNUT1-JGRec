# Goal Document: A3 全量 refit 后训练—服务分布漂移修复

## Completion Status

- **Completed at**: 2026-07-28
- **Outcome**: rolling No-Go
- **Reason**: 固定完整服务 normalizer 校准在三个 rolling folds 中有两个折 MRR
  下降、两个折 NDCG@10 下降，且 pooled Hit@1/Hit@3 下降。
- **Stop rule executed**: official 20k 未打开，未生成提交包，未扫描部分校准系数；
  校准能力保留为显式 opt-in，默认训练与当前冠军不变。
- **Evidence**:
  `dataset2-full-refit-service-normalizer-calibration-tdd-20260728.md` 与
  `dataset2-full-refit-service-normalizer-calibration-result-20260728.md`。

## Go / No-Go

- **Judgment**: Go
- **Reason**: `TemporalHybridRanker.fit()` 先在 causal
  context/train/validation 特征上训练融合头，随后为服务无条件用全部 interaction
  重建 encoder，但没有同步更新神经融合头的归一化统计量。现有冠军、200k 训练缓存、
  20k 服务口径验证缓存、三个 fold-exact Setwise 头和远端 4090 足以实现并验证一个
  不改模型权重的轻量修复。

## Target Outcome

建立一个可审计的 post-refit service calibration 步骤：

1. full encoder 和最终神经融合头安装完成后，使用无标签服务 query/candidate
   输入流式重算所选特征的 `mean/std`；
2. 模型权重、特征顺序、GNN、LightGBM、融合权重和候选顺序保持不变；
3. 三个 rolling proxy 折上同时报告 MRR、Hit@1/3/10、NDCG@10、平均排名和
   query movement，并以跨折稳定性作为硬门禁；
4. 只有 rolling 通过才打开一次 official 20k；external 通过才对真实 test
   执行同一无标签校准并生成提交包；
5. rolling 或 external 拒绝即以证据充分的 No-Go 完成 A3，不扫描校准强度或
   根据线上结果修改方法。

## Goal Definition

- **Type**: technical / quality / delivery
- **Boundary**:
  - 当前 Dataset2 冠军：
    `0.80 * Setwise(short_none 50/40k) + 0.20 * frozen LightGBM`。
  - 只更新神经融合头 `FusionResult.mean/std`；主 Setwise、基础 MLP、
    time-ramp/window 神经头共享同一显式 API，但本次实验候选只改变 Dataset2
    主 Setwise。
  - 校准统计量只来自 query 的候选特征，不读取标签、正样本列或指标。
  - 使用 float64 流式合并矩，输出 float32；`std < 1e-6` 置为 `1.0`。
  - Integration ID：
    `post_refit_service_normalizer_calibration_v1`。
- **Non-goals**:
  - 不微调神经权重，不重训 encoder/GNN/LGBM。
  - 不修改 `0.80/0.20` 融合权重。
  - 不复活已线上失败的 CST 包；CST 的 `0.283086` 只作为问题证据。
  - 不修复 LightGBM 自身的非归一化尺度漂移。
  - 不根据 rolling、external 或 leaderboard 结果增加 shrink、clip、插值系数。
- **Deferred work**:
  - 若纯统计量校准稳定但不足，再独立研究无标签分位数/robust-rank 变换。
  - full encoder 冻结后、严格无目标边泄漏的带标签 head fine-tune。
  - LGBM 的秩融合或温度标定。
- **Verification rule**:
  - RED/GREEN 证明流式统计等价于直接 NumPy 统计、分批顺序不影响结果、非有限值和
    维度漂移被拒绝。
  - 模型 state hash、LGBM model text、融合权重和 feature indices 在校准前后相同。
  - 校准后服务特征在新 normalizer 下逐列均值接近 0，非退化列标准差接近 1。
  - rolling gate 要求每折 MRR/NDCG@10 不下降；pooled
    Hit@1/3/10 不下降、平均排名不恶化、改善 query 多于恶化 query。
  - external 额外要求 MRR 严格提升，且只能运行一次。
- **Evidence source**:
  RED/GREEN 测试、normalizer drift report、模型/分数 SHA-256、三折指标、
  selection lock、external receipt/report 和条件提交包。
- **Pass criteria**:
  - 流式 normalizer 数值与直接统计在 float32 容差内一致。
  - 校准不改变任何可训练 state、LGBM 或融合权重。
  - 至少一个且实际唯一的固定候选通过 rolling 硬门禁。
  - official 20k baseline 精确复现 `0.5485470648527594`，候选严格提升并通过所有
    多指标门禁。
  - 只有 external 通过时才生成合法 ZIP；Dataset1 与当前冠军逐字节一致。
- **Confidence note**:
  三个 rolling proxy 折的 encoder 特征除 `gnn_short` 外来自冻结 200k cache，
  不能完全重演“每折 encoder full refit”；它们验证校准对时间/候选分布变化的稳定性。
  official 20k 则使用 train-end 服务口径特征，是本次 full-refit 漂移的直接
  external gate。该 20k 已被历史实验使用，不能宣称全局未污染。
- **Judgment owner**:
  自动 rolling/external 多指标门禁决定是否生成包；用户提交后的线上分数决定是否
  晋升冠军。

## Current State

- 线上冠军：`1.3557002251184347`。
- 当前冠军 Dataset2 official 20k MRR：`0.5485470648527594`。
- `ranker.py` 在 `_learn_fusion()` 后始终用全 interaction 拟合 final encoder；
  `fusion_result.mean/std` 沿用早期 supervised train 特征。
- 当前 CLI 的 `refit_full` 只传给 temporal-graph，hybrid 未消费该字段；本目标保留
  full refit，并在它之后显式校准。
- CST 曾量化 causal 与 production chain 最大概率差约 `0.283086`，且线上
  `-0.00210069`。
- 远端保留 A2 三个 fold-exact Setwise 模型、aligned score rows、当前冠军
  checkpoint、200k/20k 特征缓存和 winning `short_none` 分数。

## Plan Rewrite Notes

| Existing item | Decision | Reason |
|---|---|---|
| refit 后只重算 normalizer | keep and formalize | 无标签、低风险，直接针对归一化口径 |
| full encoder 上微调融合头数个 epoch | defer | full encoder 已见目标边，容易制造标签泄漏 |
| 只看 official 20k MRR | replace | 改为 rolling 多指标硬门后的一次 external |
| 同时修 LGBM | remove | 会改变第二个变量，无法隔离 normalizer 修复 |
| 必须生成提交包 | rewrite | 只有两级门禁通过才允许生成 |

## Drift Diagnosis

- **Goal drift**: 把 A3 扩成新模型训练或 CST 复活不能隔离 post-refit 归一化问题。
- **Phase drift**: 先看 official 20k 再决定校准方式会把修复变成 holdout 调参。
- **Validation drift**: 最大概率差下降不等于排序改善，必须同时报告完整排名指标。
- **Compatibility drift**: 旧 checkpoint 没有 calibration metadata 时必须保持原
  `mean/std`；不能静默改变加载行为。
- **Cleanup drift**: `--no-refit-full` 的 hybrid no-op 可补真实接线测试，但不以
  “关闭 refit”代替本次默认 full-refit 修复。

## Priority Rationale

- 先证明 streaming moment 与不可变 state 契约，避免昂贵评分后才发现校准公式错误。
- rolling proxy 在 external 前执行，即使拒绝也能给出“方向无稳定价值”的完整结论。
- 实际 test 特征编码和双遍评分只在 external 通过后执行，节省 4.7GB checkpoint
  推理成本。

## Assumptions and Open Decisions

| Item | Status | Impact | Owner / Next step |
|---|---|---|---|
| service candidate 输入可用于无标签 moment calibration | confirmed | 比赛 test.csv 是推理输入而非标签 | 代码禁止访问正样本列 |
| Setwise normalizer 是当前冠军可修复的主要神经尺度边界 | confirmed | 覆盖冠军 80% 分支 | calibration report 记录头类型 |
| A2 三折可作为 rolling proxy | confirmed with limitation | 不能完全复现 encoder refit，但能阻止单折偶然增益 | 在报告中披露 |
| official 20k 只在 rolling 通过后读取指标 | frozen | 防止反向修改校准方案 | exclusive receipt |
| actual test 校准需要双遍编码 | assumed | 第一遍统计，第二遍预测；成本可接受 | 仅 external 通过后执行 |

## Phases

### Phase 1: 归一化契约 RED/GREEN

- **Purpose**: 在模型和大缓存之前证明校准数学与不可变边界。
- **Entry condition**: 本目标文档已写盘。
- **Phase rules**:
  - 先 RED 后实现。
  - 纯 NumPy 核心不得导入 Jittor。
  - 不添加校准系数、截断或经验阈值。
- **Todos**:
  - [ ] 为分批矩合并、常量列、非有限值和维度变化写 RED。
    - **Surface**: tests
    - **Proof**: 因目标模块/API 缺失而失败。
    - **Depends on**: none
  - [ ] 实现 streaming normalizer 与 drift report。
    - **Surface**: `src/jgrec`
    - **Proof**: focused GREEN + Ruff。
    - **Depends on**: RED
  - [ ] 为 ranker 校准 API 写 state 不变测试。
    - **Surface**: ranker/checkpoint tests
    - **Proof**: state hash 不变、只替换 mean/std。
    - **Depends on**: streaming core
- **Exit proof**: 数学、状态和旧 checkpoint fallback 测试全绿。
- **Stop condition**: 校准需要读取标签或改变模型 state。

### Phase 2: 三折 rolling proxy

- **Purpose**: 在不打开 official 20k 指标前判断固定校准是否跨时间稳定。
- **Entry condition**: Phase 1 全绿，A2 三折资产 hash/shape 对齐。
- **Phase rules**:
  - 每折只用该折无标签 score features 计算 normalizer。
  - baseline 与 candidate 共用相同 Setwise state、LGBM 分数和 `0.80/0.20`。
  - 不根据折结果修改方法。
- **Todos**:
  - [ ] 重算三个 fold Setwise normalizer 并物化精确候选分数。
    - **Surface**: remote scores/reports
    - **Proof**: state/feature/candidate hashes 和标准化审计。
    - **Depends on**: Phase 1
  - [ ] 执行多指标稳定性门禁。
    - **Surface**: selection report/lock
    - **Proof**: 唯一 candidate accepted 或明确 rolling No-Go。
    - **Depends on**: three folds
- **Exit proof**: selection lock 或 A3 rolling No-Go。
- **Stop condition**: 任一资产错位、非有限值或任一硬门失败。

### Phase 3: 一次 official external

- **Purpose**: 直接验证 train-cache normalizer 到 train-end service cache 的修复。
- **Entry condition**: Phase 2 selection lock 存在。
- **Phase rules**:
  - 先写 external receipt，再计算指标。
  - baseline 必须精确复现冠军。
  - 失败后不修改方案或再开 external。
- **Todos**:
  - [ ] 用 20k 无标签服务特征重算冠军 Setwise normalizer并评分。
    - **Surface**: calibrated scores/model result
    - **Proof**: moment/state/hash report。
    - **Depends on**: selection lock
  - [ ] 执行一次多指标 external gate。
    - **Surface**: receipt/report
    - **Proof**: accepted 或 external No-Go。
    - **Depends on**: calibrated scores
- **Exit proof**: external accepted 或拒绝报告。
- **Stop condition**: 冠军复现失败或 lock/hash 漂移。

### Phase 4: 条件 checkpoint 与提交包

- **Purpose**: 仅把已通过的固定修复变成用户可提交资产。
- **Entry condition**: external accepted。
- **Phase rules**:
  - Dataset1 逐字节继承冠军。
  - Dataset2 test 使用实际服务输入的 streaming moments，模型 state/LGBM/权重不变。
  - 不自动晋升线上冠军。
- **Todos**:
  - [ ] 对 Dataset2 test 执行统计遍和预测遍。
    - **Surface**: calibrated checkpoint/CSV
    - **Proof**: state invariance、有限值、行和、候选顺序。
    - **Depends on**: external accepted
  - [ ] 生成并验证 ZIP。
    - **Surface**: result directory
    - **Proof**: Dataset1 byte identity、ZIP members、SHA-256。
    - **Depends on**: test prediction
- **Exit proof**: 一个可提交但未自动晋升的 ZIP。
- **Stop condition**: 任一不变量或提交结构校验失败。

## Dry-Run Findings

- 直接在 full encoder 特征上用历史正样本 fine-tune 会让 learned tower 见过目标边，
  因而不是本轮可接受的轻量修复。
- 只改 base `fusion_result` 不覆盖当前冠军；API 必须能校准已安装的 Setwise 头。
- LightGBM 无 `mean/std`，本轮候选只能修复 80% 神经分支，报告必须明确剩余风险。
- actual test batch calibration 是无标签 transductive adaptation；必须双遍或先缓存特征，
  不能边预测边改变 normalizer 造成顺序依赖。
- A2 rolling cache 不完全等价于逐折 full-refit encoder，因此 external 仍是必要直接证据。

## Final Validation

- Focused RED/GREEN、相关 ranker/checkpoint 回归、Ruff check/format。
- 三折 state/hash/shape/normalizer 审计和完整多指标报告。
- external receipt 唯一性、冠军精确复现和多指标 gate。
- 条件成立时：checkpoint reload、Dataset1 byte identity、Dataset2 formula replay、
  CSV/ZIP validator 和 SHA-256。

## First Execution Step

为 streaming service normalizer 写缺失模块的 RED：分成不同 batch 更新必须得到与
直接 flatten 统计相同的 mean/std，同时拒绝非有限特征和列数漂移。
