# Goal Document: Dataset2 精确集成候选稳健选权重协议

## Go / No-Go

- **Judgment**: Go
- **Reason**: 上一包线上从 `1.3557002251184347` 降到
  `1.3545061936665996`，已证明“在 standalone Two-Tower 上选择 `0.20`，
  再把该权重迁移到完整 reranker”无效。现在有明确的纠偏目标、可机械执行的
  隔离边界和无需再次读取 leaderboard 的验证路径。

## Target Outcome

建立一个可复用、可审计的 Dataset2 权重选择器：它只接收每个权重对应的
**最终集成后的精确分数**，先在至少三个 rolling-origin 折上做多指标评估，
只允许跨折稳定的权重进入候选集；选定并锁定唯一权重后，才允许对长跨度
external holdout 执行一次评估。external 结果只能作接受/拒绝判断，不能触发
新的权重扫描。

## Goal Definition

- **Type**: technical / quality / learning
- **Boundary**:
  - Dataset2 排序候选的离线权重选择协议、机器可读 manifest、selection lock、
    external one-shot receipt 和评估报告。
  - 每个 rolling-origin 折、每个权重都必须提供最终集成路径实际产生的完整
    `[query, candidate]` 分数矩阵；不允许用 standalone 专家分数代替。
  - 报告 MRR、Hit@1/3/10、NDCG@10、平均排名，以及相对基线的改善、恶化、
    不变 query 数。
  - 权重只由 rolling-origin 折决定；external holdout 与 selection 使用不同的
    命令和状态文件。
- **Non-goals**:
  - 不根据 `1.3545061936665996` 回扫 `0.10/0.15/0.25`。
  - 不在本轮训练新 listwise 模型、修改 GNN 边数或改冠军 checkpoint。
  - 不自动生成线上提交包，不自动晋升冠军。
  - 不把多个相关的 rank 指标误称为彼此独立的泛化证据；核心证据仍是时间轴
    和候选集的隔离。
- **Deferred work**:
  - 为 Two-Tower 或 listwise MLP 生成完整的多折精确集成分数。
  - external 通过后的 checkpoint/提交包生产。
  - 训练—服务 refit 漂移修复。
- **Verification rule**:
  1. selection manifest 至少包含三个时间递进、训练区间严格早于评分区间的折。
  2. 每个折的基线和每个权重分数矩阵 shape 一致、有限、候选顺序指纹一致；
     所有候选声明同一个 `integration_id`，证明权重没有跨专家迁移。
  3. 稳定性硬门禁不追求单折峰值：MRR 与 NDCG@10 在每折均不得下降；合并
     rolling 折后 Hit@1/3/10 均不得下降、平均排名不得变差，改善 query 总数
     必须大于恶化 query 总数。
  4. 通过门禁的权重按“最大化最差折 MRR 增益 → 最大化折中位数 MRR 增益
     → 最大化合并 MRR 增益 → 更小权重”排序。
  5. selection lock 写盘后才允许 external；external manifest 必须绑定
     selection lock SHA-256、相同 `integration_id` 和唯一锁定权重。
  6. external receipt 使用原子独占创建；同一状态目录第二次调用必须失败，即使
     第一次结果不理想也不能覆盖或重开。
- **Evidence source**: RED/GREEN 测试、manifest 预检、selection report/lock、
  external receipt/report、SHA-256、`uv run --no-sync pytest` 与 Ruff。
- **Pass criteria**:
  - 测试能捕获跨专家权重迁移、折时间泄漏、单折最优但跨折不稳、指标缺失、
    external 提前读取、lock 漂移和重复开启。
  - 合成 dry-run 能选出跨折稳定而非单折 MRR 最高的权重。
  - selection 阶段的代码路径不读取 external 分数文件。
  - external 首次评估成功、第二次评估确定性失败。
- **Confidence note**: 自动化能高置信证明协议隔离和指标计算正确；在精确多折
  分数尚未生成前，它不能证明 listwise 候选本身会涨分。历史 official validation
  已被多次使用，只能作为“对本候选族一次性开启”的长跨度 gate，不能重新宣称为
  完全未污染的统计盲集。
- **Judgment owner**: 测试和协议状态机判断实现是否合规；rolling 门禁决定是否
  锁权重；external 一次性报告只决定该锁定候选接受或拒绝；用户决定是否提交
  leaderboard。

## Current State

- 当前线上冠军：`1.3557002251184347`。
- 被拒绝的完整 reranker `w=0.20` 候选：`1.3545061936665996`，相对冠军
  `-0.0011940314518351`。
- 该 `two_tower_full_reranker_partial_v1` 候选族已关闭：本协议不能被用来围绕
  它回扫邻近权重；真实运行必须先有一个实质变化且预先登记的新
  `integration_id`。
- 旧选择把 standalone Two-Tower 上的 `w=0.20` 迁移给完整 reranker，违反
  本目标的精确候选原则。
- `jgrec.partial_listwise_submission.ranking_metric_panel()` 已报告 MRR、
  Hit@1/3/5/10 和排名分布，但缺 NDCG@10、统一的 query movement、跨折门禁
  和 external one-shot 状态。
- 仓库已有 rolling-origin manifest 和若干 external 报告，但没有通用的
  “最终集成权重 → 多折选择 → lock → 一次 external”执行边界。
- 当前工作树含大量既有实验改动；本轮只新增独立模块、脚本、测试和文档。

## Plan Rewrite Notes

| Existing item | Decision | Reason |
|---|---|---|
| 对最终集成后的精确候选选权重 | keep and strengthen | manifest 强制每个权重绑定同一 `integration_id` 和候选顺序指纹 |
| rolling-origin 多折 | keep | 从建议提升为至少三折的结构校验与硬门禁 |
| 长跨度 external holdout | keep with disclosure | 与 selection 物理隔离并一次开启；披露历史验证集污染 |
| 多指标面板 | keep and expand | 补 NDCG@10 和 improved/worsened/tied query 数 |
| 跨折稳定性硬门禁 | rewrite | 使用全折 MRR/NDCG 非负和 pooled 二级指标非退化，不以单折峰值选权重 |
| 锁权重后只开一次 external | keep and enforce | 通过独占 receipt、lock SHA 和独立命令机械限制 |
| 根据线上结果回扫邻近权重 | remove | 明确禁止；新一轮必须是新假设、新 protocol ID 和新盲集边界 |

## Drift Diagnosis

- **Goal drift**: 继续围绕 `0.20` 邻域扫描会从“验证 listwise 互补性”漂移为
  leaderboard 拟合。
- **Phase drift**: 在精确多折分数尚未生成前打开 external，会把 gate 变成
  selection fold。
- **Validation drift**: 只增加 Hit/NDCG 但仍复用同一单折，不能解决过拟合；
  时间轴隔离优先于指标数量。
- **Compatibility drift**: standalone 与完整 reranker 都叫 Two-Tower 候选，但
  不是同一个可迁移的评分语义；必须以 `integration_id` 分开。
- **Cleanup drift**: 本轮不顺手改提交打包、checkpoint 或训练代码。

## Priority Rationale

- 先用测试固定“不能偷看 external”和“不能跨专家迁权重”，因为这两点一旦
  失守，后续所有指标都会失去解释力。
- 再实现多指标和跨折门禁，最后才提供 external 命令；执行顺序本身就是防过拟合
  协议的一部分。

## Assumptions and Open Decisions

| Item | Status | Impact | Owner / Next step |
|---|---|---|---|
| 至少可生成三个 rolling-origin 折的最终集成分数 | unresolved | 没有精确分数只能完成框架，不能锁真实权重 | 后续训练/回放任务生成 |
| official validation 可作为本候选族一次性长跨度 gate | assumed with contamination disclosure | 能做工程 gate，统计独立性有限 | 报告中持续披露 |
| 正样本固定在候选列 0 | confirmed by现有缓存协议 | 指标可直接由严格排名计算 | 测试保护 |
| 全折 MRR/NDCG 不退化是可接受的保守硬门禁 | assumed | 可能拒绝小而有噪声的真实增益，但显著降低单折过拟合 | 本轮冻结，不因结果放宽 |
| external 失败后的下一步 | confirmed | 拒绝候选，不回扫权重 | 新假设必须新开目标 |
| 当前 Two-Tower full-reranker v1 是否可重扫 | confirmed: no | 避免用已见线上反馈选择邻近权重 | 只保留为失败证据 |

## Phases

### Phase 1: 冻结协议与 RED

- **Purpose**: 用失败测试把精确候选、多折稳定和 external 隔离写成行为契约。
- **Entry condition**: 本目标文档已写盘。
- **Phase rules**:
  - 生产实现前先运行 RED。
  - selection API 不接受 external path。
  - 每个行为测试只约束一个公开边界。
- **Todos**:
  - [ ] 为完整多指标面板和 query movement 写测试。
    - **Surface**: `tests/test_robust_weight_selection.py`
    - **Proof**: 导入目标模块失败。
    - **Depends on**: none
  - [ ] 为 rolling 时间关系、相同 integration/candidate 指纹和至少三折写测试。
    - **Surface**: tests
    - **Proof**: 无校验实现时失败。
    - **Depends on**: none
  - [ ] 为“单折峰值被拒、跨折稳定候选胜出”写测试。
    - **Surface**: tests
    - **Proof**: selector 不存在时失败。
    - **Depends on**: metric tests
  - [ ] 为 lock 前禁止 external、lock 漂移和第二次开启失败写测试。
    - **Surface**: tests
    - **Proof**: one-shot API 不存在时失败。
    - **Depends on**: selector contract
- **Exit proof**: focused test 因缺少目标模块/行为失败，且不是环境或语法错误。
- **Stop condition**: 无法在 API 层把 selection 与 external 文件读取隔开。

### Phase 2: GREEN 与重构

- **Purpose**: 以最小实现满足协议，再整理为清晰的 selection/external 边界。
- **Entry condition**: Phase 1 RED 正确。
- **Phase rules**:
  - 不实现训练逻辑。
  - 不提供根据 external 报告重选权重的 API。
  - 报告写盘不得覆盖现有文件。
- **Todos**:
  - [ ] 实现指标、manifest 校验、跨折门禁与稳定排序。
    - **Surface**: `src/jgrec/robust_weight_selection.py`
    - **Proof**: metric/selection tests GREEN。
    - **Depends on**: RED
  - [ ] 实现 selection report/lock 和 external one-shot receipt/report。
    - **Surface**: module + CLI
    - **Proof**: isolation/one-shot tests GREEN。
    - **Depends on**: selector
  - [ ] 重构重复的 hash、矩阵加载和 report 序列化。
    - **Surface**: same module
    - **Proof**: focused tests 保持 GREEN，Ruff 通过。
    - **Depends on**: GREEN
- **Exit proof**: focused tests、相关回归和 Ruff 全部通过。
- **Stop condition**: 实现必须读取 external 才能完成 selection。

### Phase 3: 合成 dry-run 与真实资产 preflight

- **Purpose**: 证明执行顺序和状态机可用，并明确真实实验还缺哪些精确分数。
- **Entry condition**: Phase 2 GREEN。
- **Phase rules**:
  - dry-run 使用合成矩阵，不冒充真实模型证据。
  - 只盘点真实资产，不开启任何真实 external 分数。
- **Todos**:
  - [ ] 构造三折合成 manifest，验证稳定权重胜过单折峰值权重。
    - **Surface**: temporary artifacts/report
    - **Proof**: selection lock 内容和多指标齐全。
    - **Depends on**: Phase 2
  - [ ] 首次开启合成 external，再确认第二次被拒。
    - **Surface**: temporary external state
    - **Proof**: receipt 存在且第二次命令非零退出。
    - **Depends on**: selection lock
  - [ ] 输出真实 listwise 运行所需 artifact contract。
    - **Surface**: result doc / manifest example
    - **Proof**: 缺失项逐项列出，不读取真实 external。
    - **Depends on**: preflight
- **Exit proof**: dry-run 报告、focused tests、Ruff 和真实资产缺口清单完整。
- **Stop condition**: 现有 artifact 被误当成多折精确集成分数。

## Dry-Run Findings

- 多指标仍都由同一个正样本 rank 派生，不足以替代独立时间折；因此门禁先看
  跨折一致性，再看 pooled 指标。
- 旧的 `0.20` 不能作为新扫描中心或默认候选；真实权重集合必须在生成精确多折
  分数前冻结。
- 历史 external 报告已暴露，不能作为全新统计盲集；one-shot 约束只能从本协议
  和本候选族开始生效。
- 若无法生成同一 `integration_id` 下的多折最终分数，本目标应停在 framework
  ready，不得产出新的提交包。

## Final Validation

- `uv run --no-sync pytest tests/test_robust_weight_selection.py -q`
- `uv run --no-sync pytest tests/test_partial_listwise_submission.py
  tests/test_hybrid_partial_listwise_blend.py -q`
- `uv run --no-sync ruff check src/jgrec/robust_weight_selection.py
  scripts/select_robust_integrated_weight.py
  scripts/evaluate_locked_weight_external.py
  tests/test_robust_weight_selection.py`
- 合成 selection/external dry-run，确认一次性 receipt 和第二次失败。
- 真实 preflight 不产生 selection lock，不读取 external，不生成提交包。

## First Execution Step

新增 `tests/test_robust_weight_selection.py`，先让多指标、跨折稳定选择和 external
one-shot 三组行为以正确原因进入 RED。

## Execution Result

- **Final judgment**: framework complete；real selection No-Go。
- Phase 1：完成。RED 首先失败于目标模块缺失；external candidate weight
  契约也单独经历 RED。
- Phase 2：完成。核心模块、两条隔离 CLI、selection lock 和 external
  one-shot receipt 已实现。
- Phase 3：协议 dry-run 由 7 个 focused tests 覆盖；相关回归共
  `30 passed`，Ruff 通过。
- 真实 preflight 明确缺少三折最终集成分数；因此未创建真实 selection lock，
  未读取 external，未生成提交包。
- 已拒绝的 `two_tower_full_reranker_partial_v1 / w=0.20` 候选族关闭，不根据
  线上回归反扫邻近权重。
