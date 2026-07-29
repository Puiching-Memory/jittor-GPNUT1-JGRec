# Goal Document: Dataset2 部分混合抢救 Listwise 专家

## Go / No-Go

- **Judgment**: Go
- **Reason**: 两个 listwise 候选都已有独立正证据，失败点集中在最终融合而不是
  表征或目标本身。当前线上冠军又已连续证明“保留主干、以部分权重混入候选”
  比直接替换稳健，因此先做严格对齐的一维收缩混合，是成本最低且最可归因的
  下一步。

## Target Outcome

以当前线上冠军
`1.3557002251184347` 对应的 Dataset2
`short_none 50/40k + 0.80 Setwise + 0.20 LightGBM` 为不可变主干，
分别把 cached-feature listwise MLP 和 200k listwise Two-Tower 作为辅助专家
做一维部分混合。两条支路使用相同的时间前向选择协议；只允许一个在完整门禁
上胜出的候选进入正式 checkpoint、双重回放和本地提交包。

## Goal Definition

- **Type**: technical / learning / quality / delivery
- **Boundary**:
  - Dataset2 only；Dataset1 checkpoint state 与预测 CSV 必须字节不变。
  - 主干固定为当前线上冠军，不重训其 `short_none` GNN、Setwise 或 LightGBM。
  - listwise MLP 与 listwise Two-Tower 分开扫描部分权重，不能在同一轮做二维
    联合搜索。
  - 每条支路只允许权重
    `0.05 / 0.10 / 0.20 / 0.30 / 0.40 / 0.50`；`0.00` 仅作冠军基线。
  - 融合公式固定为
    `candidate = (1 - weight) * champion + weight * expert`。
  - MLP 使用行内 softmax 概率；Two-Tower 在读取任何门禁指标前，必须冻结一个
    对分数量纲不敏感的行内变换，并在验证与服务端保持完全一致。
- **Non-goals**:
  - 不增加 GNN 边数、epoch 或新图特征。
  - 不重训当前 Setwise/LightGBM 主干。
  - 不调 listwise MLP 的 epoch、隐藏层、学习率或特征列。
  - 不调 Two-Tower 的 embedding/hidden dimension、负例数、训练目标或温度。
  - 不把两个辅助专家同时接入一个三专家候选。
  - 不自动提交 leaderboard。
- **Deferred work**:
  - 两个 listwise 专家的二维联合权重或动态 query gate。
  - 温度标定、RRF 与其他异构专家尺度研究。
  - refit 后训练—服务漂移修复和 A3/A4 方向。
- **Verification rule**:
  1. 所有模型必须对同一批 query 和同一顺序的 100 个 candidates 打分；先验证
     sidecar/哈希，后计算任何混合指标。
  2. 每条支路仅在最早的 `slice0` 选择权重：要求相对冠军不下降，最大化
     slice0 MRR，精确并列时取更小权重。
  3. 锁定权重后只前向评估 `slice1`；该片不下降且 `slice0+slice1` 合并增益
     至少 `+0.0003` 才允许读取 `slice2`。
  4. 最终候选必须 full MRR 相对冠军至少 `+0.0005`，三个时间片全部不下降。
  5. 两条支路都通过时，以 `slice0+slice1` 合并 MRR 较高者作为唯一候选；
     `slice2` 只判定该锁定胜者，不反向改变专家或权重。
- **Evidence source**: 对齐哈希、RED/GREEN 测试、冻结配置、完整权重扫描报告、
  slice0 选择锁、slice1 前向报告、一次性 slice2/full 门禁、checkpoint 字段
  审计、标准加载回放、CSV/ZIP SHA-256。
- **Pass criteria**:
  - 当前冠军 full/slice MRR 在新 runner 中以 `<= 1e-12` 误差复现。
  - 每个实际评估的专家与冠军 query/candidate sidecar 完全相同。
  - 唯一锁定胜者达到 full `>= +0.0005` 且三片 delta `>= 0`。
  - checkpoint round-trip 后辅助专家分数、最终混合分数与门禁 artifact
    `allclose(atol=1e-7, rtol=1e-6)`。
  - 两次独立标准 Dataset2 回放逐字节一致；Dataset1 与当前冠军逐字节一致。
- **Confidence note**: 两个专家都曾在这套历史验证数据上被观察，因此 slice2
  只是对“本轮权重未参与选择”的前向门禁，并非真正盲集。它适合淘汰不稳定
  候选；最终晋升仍由一次线上分数高于 `1.3557002251184347` 决定。
- **Judgment owner**: 自动化离线门禁决定是否生成候选包；checkpoint/replay
  验证决定包是否可提交；用户提供的 leaderboard 分数决定是否替换线上冠军。

## Current State

- 当前线上冠军分数：`1.3557002251184347`。
- 当前 Dataset2 离线主干 full MRR：`0.5485470648527594`；slice0/1/2 为
  `0.5882028774417708 / 0.5493313411199712 / 0.5081009093765456`。
- 当前线上包 SHA-256：
  `104f68dc82aed862600be3328f779d80e04746283c0ec75193a3582266438193`。
- cached-feature listwise MLP 独立 full MRR 从 `0.5285863234` 提升至
  `0.5352702199`，`+0.0066838965`；旧固定 `0.07` 外层混合只得到
  `+0.0002067896`，且 slice1 为负。
- listwise Two-Tower 独立 raw MRR 从 `0.0149464592` 提升至
  `0.4641248061`，三片全部通过；原计划的完整集成 Phase 2/3 未完成。
- listwise MLP 本地模型已存在，SHA-256 为
  `552bbc7e0e17b27f9501b3f8fd1f3ae6fa4a625b07f7bd8b00134a21023d53fb`。
- Two-Tower 本地仅有报告；模型路径记录在远端，SHA-256 为
  `8e99cc6354b8576a69538b046212e87c8e7a94fac0724580087552856f7afbbf`。
- 两个旧实验的验证切片边界并不完全相同，且当前冠军更换过 GNN/Setwise
  主干；旧报告数字不能直接用于本轮混合，必须重新生成对齐分数。

## Plan Rewrite Notes

| Existing item | Decision | Reason |
|---|---|---|
| listwise MLP 固定 `0.07` 替换旧 MLP | rewrite | 旧权重是失败点；改为保留当前冠军并扫描有限的一维 residual 权重 |
| listwise MLP 重新训练 | remove | 已有正信号模型；先验证融合，不把训练方差混入因果链 |
| Two-Tower 完整重建全部 reranker | reorder | 先用冻结 standalone 专家验证最终分数互补性，过门后才做服务化重训/持久化 |
| 两条专家同时融合 | remove | 二维搜索无法归因且显著增加复用验证集的过拟合风险 |
| 固定 0.07 外层融合门禁 | replace | 改用与 γ/α 同构的主干收缩混合 |
| full MRR 直接选最佳 | replace | 先 slice0 选权重、slice1 前向确认，再一次性打开 slice2 |

## Drift Diagnosis

- **Goal drift**: 继续训练 listwise 模型会偏离“验证融合是否压制信号”的核心问题。
- **Phase drift**: 在 candidate sidecar 对齐前做权重扫描，会把候选差异误当成
  专家互补。
- **Validation drift**: 只报告独立专家 MRR 或 full 最佳权重，不能证明最终主干
  混合稳定。
- **Compatibility drift**: 只输出 CSV 而不持久化辅助专家，会再次留下无法复现的
  集成债。
- **Cleanup drift**: 分数并列、默认配置和 refit 漂移属于后续独立目标，不混入
  本轮。

## Priority Rationale

- 先做 MLP 支路，因为模型与旧报告完整、成本最低，最适合验证一维收缩协议。
- Two-Tower 第二，因为它需要恢复远端模型、候选 sidecar，并解决稳定的行内
  score transform，成本和服务化风险更高。
- 两支都先完成离线归因，最后只对一个胜者付 checkpoint 接线和双回放成本。

## Assumptions and Open Decisions

| Item | Status | Impact | Owner / Next step |
|---|---|---|---|
| 当前冠军远端 checkpoint 与线上 ZIP 对应 | confirmed by prior replay, online score supplied by user | 冻结主干 | Phase 1 再验 checkpoint/package hashes |
| listwise MLP 可在当前 63 列验证缓存上回放 | assumed | 决定 MLP 支路能否零重训 | Phase 1 以 feature schema、normalizer 和有限性验证 |
| Two-Tower 远端模型仍存在且可恢复 | unresolved | 决定能否完成便宜互补性扫描 | Phase 1 检查路径与 SHA；缺失则停止该支路，不重训后冒充旧模型 |
| 两个专家可与当前冠军对齐到同一 candidate sidecar | unresolved | 是任何混合的硬前提 | Phase 1 重放并比较 query/candidate hashes |
| Two-Tower 行内无尺度变换 | unresolved but must be frozen pre-metric | 决定混合语义和服务一致性 | RED 测试后选定单一 rank-based transform，写入 frozen config |
| `+0.0005` full 门槛足以授权一次提交 | assumed | 控制线上机会成本 | 自动门禁；用户仍拥有提交决定 |

## Phases

### Phase 1: 冻结主干与对齐资产

- **Purpose**: 证明三个打分源能在同一 query/candidate 轴上比较，并冻结本轮
  不可变配置。
- **Entry condition**: 本目标文档完成。
- **Phase rules**:
  - 只做只读 preflight 和 artifact 复制，不训练、不扫描权重。
  - 任一候选缺少精确 sidecar 时，必须通过当前冠军的标准验证构造重放，不能按
    行号猜测对齐。
  - 在输出冻结配置前不得读取新混合指标。
- **Todos**:
  - [ ] 验证当前冠军 checkpoint、线上包、Dataset2 head 和历史报告哈希。
    - **Surface**: checkpoint/result artifacts。
    - **Proof**: machine-readable preflight report。
    - **Depends on**: none。
  - [ ] 恢复并核验两个 listwise 模型与所需缓存/sidecar。
    - **Surface**: local/remote models, feature caches, query/candidate arrays。
    - **Proof**: SHA-256、shape、schema、finite scan。
    - **Depends on**: none。
  - [ ] 写出统一的 20k×100 query/candidate manifest 和冻结选择协议。
    - **Surface**: frozen-config JSON。
    - **Proof**: 三个源 sidecar hash 完全相同。
    - **Depends on**: asset preflight。
- **Exit proof**: 当前冠军可精确复现，至少一个辅助专家对齐成功，冻结配置在
  任何权重指标产生前写盘。
- **Stop condition**: 当前冠军无法复现；或两个专家都无法精确对齐。

### Phase 2: RED/GREEN 一维部分混合合同

- **Purpose**: 用自动测试固定“选权重—前向门禁—锁胜者”的行为，防止看到结果
  后改协议。
- **Entry condition**: Phase 1 的至少一条支路可执行。
- **Phase rules**:
  - 先 RED 后 production implementation。
  - selector 只能读取 slice0；slice1 gate 不得改权重；slice2 evaluator 必须要求
    selection lock 的内容哈希匹配。
  - 同分始终偏向更小辅助权重。
- **Todos**:
  - [ ] 为概率收缩公式、权重白名单、shape/finite 校验写 RED 测试。
    - **Surface**: tests。
    - **Proof**: 失败原因是 partial-listwise blend API 不存在。
    - **Depends on**: Phase 1。
  - [ ] 为 slice0 selector、slice1 gate、slice2 lock 写 RED 测试。
    - **Surface**: tests。
    - **Proof**: 测试能抓住 final-slice 偷看、负 slice、错误 tie-break 和 hash 漂移。
    - **Depends on**: frozen protocol。
  - [ ] 实现最小 blend/selection/gate 模块。
    - **Surface**: `src/jgrec/rankers/hybrid/`。
    - **Proof**: focused tests GREEN，相关回归与 Ruff 通过。
    - **Depends on**: RED。
- **Exit proof**: 所有协议行为由测试锁定，runner 尚未读取 slice1/2。
- **Stop condition**: 无法在 API 层隔离 slice0 selection 与后续门禁。

### Phase 3: 两条专家独立前向选型

- **Purpose**: 判断正信号能否真实转化为当前冠军之上的稳定增益。
- **Entry condition**: Phase 2 GREEN，统一分数 artifact 完整。
- **Phase rules**:
  - MLP 与 Two-Tower 分开生成完整扫描表。
  - 每条支路只在 slice0 选权重，随后锁定。
  - slice1 通过前不运行 slice2 evaluator。
  - 两条都通过 slice1 时，仅按已冻结规则选一个唯一胜者，再打开 slice2。
- **Todos**:
  - [ ] 运行 listwise MLP 六权重扫描、锁定与 slice1 gate。
    - **Surface**: selection report/lock。
    - **Proof**: 完整权重表、chosen weight、prefix delta、pass/fail。
    - **Depends on**: Phase 2。
  - [ ] 运行 listwise Two-Tower 六权重扫描、锁定与 slice1 gate。
    - **Surface**: selection report/lock。
    - **Proof**: 与 MLP 同口径报告。
    - **Depends on**: Phase 2。
  - [ ] 对唯一胜者运行 slice2/full 一次性门禁。
    - **Surface**: evaluation report。
    - **Proof**: full/slice MRR、delta、selection-lock SHA。
    - **Depends on**: 至少一个 slice1 pass。
- **Exit proof**: 一个候选满足 full `+0.0005` 且三片不降，或留下明确 No-Go。
- **Stop condition**: 两条支路都在 slice1 失败；此时不读取 slice2、不接 checkpoint。

### Phase 4: 胜者 checkpoint 服务化与交付

- **Purpose**: 把唯一胜者从实验分数变成可标准加载、可重复推理的正式候选。
- **Entry condition**: Phase 3 最终门禁通过。
- **Phase rules**:
  - 新增状态只能描述辅助专家、固定变换和固定权重；主干 state 不得改变。
  - 若 Two-Tower 胜出，服务化模型必须按冻结训练配置重训到最终服务上下文，
    并重新通过同口径门禁；standalone context 模型不得直接冒充正式模型。
  - 不覆盖当前冠军 checkpoint/package。
- **Todos**:
  - [ ] 先写 checkpoint snapshot/hydrate/score-equivalence RED 测试。
    - **Surface**: checkpoint/ranker tests。
    - **Proof**: 当前 loader 无法恢复辅助 listwise blend。
    - **Depends on**: Phase 3 pass。
  - [ ] 实现最小持久化与推理接线。
    - **Surface**: ranker/checkpoint/fusion module。
    - **Proof**: in-memory 与 reload scores 一致；旧 checkpoint 兼容测试通过。
    - **Depends on**: RED。
  - [ ] 生成双数据集 checkpoint、两次 Dataset2 标准回放和本地 ZIP。
    - **Surface**: checkpoints/result package。
    - **Proof**: replay byte identity、Dataset1 byte identity、ZIP integrity、SHA-256。
    - **Depends on**: GREEN。
- **Exit proof**: 一个未自动提交、可复现、带完整审计报告的本地 `result.zip`。
- **Stop condition**: 正式服务化模型未复现门禁、主干字段漂移、回放不一致或
  Dataset1 字节变化。

## Dry-Run Findings

- 旧 `+0.0067` 与 `+0.449` 都不是相对当前新冠军的最终融合增益，只能作为进入
  本轮的先验，不能直接授权 checkpoint。
- 两个旧实验使用的 slice 边界不同；本轮必须从统一 sidecar 重新切片，不能拼接
  两份报告数字。
- listwise MLP 可以先尝试零重训回放；Two-Tower 若只剩模型权重而缺少 id map /
  temporal state，便宜支路可能被阻断，不能静默重训后宣称复现旧信号。
- Two-Tower 的 raw dot score 与冠军概率量纲不同。目标文档因此要求在看指标前
  冻结一个无尺度行内变换，而不是边看结果边调温度。
- 两条支路均通过时先在 slice0+slice1 决定唯一胜者，避免用 slice2 在专家之间
  二次选择。
- 当前工作树包含大量既有实验改动；本轮只新增独立模块、测试、runner 和
  artifact，不清理或覆盖无关文件。

## Final Validation

- `uv run pytest` 的 focused RED/GREEN、checkpoint 和相关 hybrid 回归。
- `uv run ruff check` 覆盖新增/修改的 Python 文件。
- machine-readable preflight、frozen config、selection lock、evaluation report。
- checkpoint 字段差异审计与标准 hydrate。
- 两次完整 Dataset2 replay 字节一致；Dataset1 CSV 与当前冠军字节一致。
- ZIP 成员、行数、有限性、候选集合和 SHA-256 校验。
- leaderboard 分数只有高于 `1.3557002251184347` 才正式晋升。

## First Execution Step

只读核验远端两个 listwise 模型、当前冠军 checkpoint、统一 full-100 validation
cache 与 query/candidate sidecar；在任何权重扫描前写出冻结 preflight 和 config。

## Execution Result

- **Final judgment**: No-Go；A2 验证完成，不产生 checkpoint 或提交包。
- Phase 1：通过。所有模型/缓存哈希匹配，统一 sidecar 为 `20000 × 100`，
  candidate SHA-256 为
  `dec159209d9c6913825591b585afa0689b7b7323912543204ca6190dad4e4a95`。
- Phase 2：通过。focused 7 tests 在 Windows/Linux 均通过；相关 blend 回归
  17 passed；Ruff 通过。
- Phase 3：
  - listwise MLP 六个 slice0 权重全部下降，最小退化也为
    `-0.0000361286`；
  - listwise Two-Tower 在 slice0 锁定 `0.20`，delta
    `+0.0000790706`；
  - 锁定权重在 slice1 delta 为 `-0.0000349051`，前两片合并 delta
    `+0.0000220828 < +0.0003`，因此被拒。
- Phase 4：按 stop condition 跳过。没有读取候选 slice2 指标，没有修改当前
  冠军，没有生成 checkpoint/ZIP。
- 完整结果见
  `docs/experiments/dataset2-partial-listwise-expert-blend-result-20260728.md`。
