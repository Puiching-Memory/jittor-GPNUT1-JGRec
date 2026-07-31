# Goal Document: Dataset2 Partial-Listwise Online Candidate

## Go / No-Go

- **Judgment**: Go
- **Reason**: 用户明确指出线上分数才是最终裁决，并授权在保留离线风险证据的
  前提下生成提交包。上一轮 MRR 门禁只能阻止“离线晋升冠军”，不能替代一次
  leaderboard 验证。

## Target Outcome

生成一个可直接提交的双数据集 `result.zip`：

- Dataset1 必须逐字节复用当前线上冠军包；
- Dataset2 固定使用当前冠军分数作为 0.80 主干，并以 0.20 混入已经完成
  Phase 2/3 的 listwise Two-Tower 完整直接替换候选概率；
- 包、CSV、模型来源、权重、变换、多指标离线画像和 SHA-256 全部可审计；
- 该产物只标记为 `online_candidate`，用户回传线上分数前不得替换当前冠军。

## Goal Definition

- **Type**: technical / learning / delivery
- **Boundary**:
  - 只改变 Dataset2 的最终分数；Dataset1 CSV 必须与当前冠军 ZIP 成员哈希相同。
  - 辅助权重固定为上一轮在未读取 slice2 时锁定的 `0.20`；辅助分支使用
    本地已验证的完整 Two-Tower reranker 提交包，不重新训练或修改其内部融合。
  - 融合公式固定为
    `candidate = 0.80 * champion + 0.20 * two_tower_full_reranker`。
  - 当前冠军和 Two-Tower 完整 reranker 均以各自提交 CSV 为冻结输入，不重训。
  - 离线评价同时报告 MRR、Hit@1/3/5/10、平均排名和中位排名；这些指标描述
    风险，不再构成拒绝打包的自动门禁。
- **Non-goals**:
  - 不自动提交 leaderboard。
  - 不因查看新指标而改专家、权重、温度或 score transform。
  - 不把探索候选写回当前冠军 checkpoint。
  - 不同时接入 listwise MLP，不做二维专家搜索。
- **Deferred work**:
  - 线上胜出后再做正式 checkpoint snapshot/hydrate 接线和双重标准回放。
  - listwise MLP 的独立线上候选、温度标定、RRF 和动态 gate。
- **Verification rule**:
  1. 测试先固定部分混合公式、CSV shape/range/格式和不可覆盖语义。
  2. Dataset1 ZIP 成员与当前冠军逐字节相同。
  3. 两个 Dataset2 来源都由标准提交管线对同一 `test.csv` 生成；行列位置即
     query/candidate 合同，两个来源都必须严格为 `153420 × 100`。
  4. 输出 Dataset2 每行恰有 100 个有限值且均在 `[0, 1]`。
  5. 随机抽查和全量重载都必须满足保存值等于固定公式在 8 位小数下的结果。
  6. ZIP 根目录只能有 `dataset1.csv`、`dataset2.csv`，行数分别为
     `61051`、`153420`。
- **Evidence source**: RED/GREEN 测试、多指标报告、模型与来源包哈希、测试
  candidate manifest 哈希、CSV/ZIP 完整性报告。
- **Pass criteria**:
  - focused tests、相关提交回归和 Ruff 全部通过。
  - Dataset1 member SHA-256 与当前冠军完全相同。
  - Dataset2 辅助候选 shape 为 `153420 × 100`，有限且在 8 位 CSV 舍入误差内
    逐行归一。
  - Dataset2 候选 CSV 通过标准提交校验。
  - 组合 ZIP 的成员、行数和所有 SHA-256 被 machine-readable report 固定。
- **Confidence note**: 该包的离线 MRR 信号很弱且在 slice1 有轻微回归，因此
  它是一次有明确风险的探索提交；多指标画像只能帮助解释结果，不能预测
  leaderboard 的私有权重与长时间跨度行为。
- **Judgment owner**: 自动化验证只决定文件是否正确；用户提交后得到的线上分数
  决定候选是否淘汰或晋升。晋升阈值为严格高于当前
  `1.3557002251184347`。

## Current State

- 当前线上冠军：`1.3557002251184347`。
- 当前冠军包：
  `result/d1_time_ramp_g050_d2_gnn_short_none_e50_edges40000_setwise_w080_seed60_20260727/result.zip`。
- 当前冠军包 SHA-256：
  `104f68dc82aed862600be3328f779d80e04746283c0ec75193a3582266438193`。
- 上一轮在 slice0 冻结的唯一辅助权重为 listwise Two-Tower `weight=0.20`，
  selection lock 为
  `8ec5afadbfc5a904a074a460d7379845dc8adfc77311223c16b0adf0a32f6eb6`。
- 它在 slice0 MRR 为 `+0.0000790706`，slice1 为 `-0.0000349051`，
  前两片合并为 `+0.0000220828`，未达到原离线晋升门槛。
- 用户已经明确覆盖“因此不打包”的结论，要求生成提交包由线上裁决。
- 进一步资产审计发现 Two-Tower Phase 2/3 并非缺失：完整 reranker 探索包
  `result/d1_champion_d2_twotower200k_exploratory_seed60_20260724/result.zip`
  已存在，SHA-256 为
  `f0b637fab7ff65dfc64b6b1d8175a475cf3e329864776ed547b7687e8fedede7`；
  其 Dataset2 member SHA-256 为
  `d88fe864c472c75083d386c9c823f44207ce791ed5afe4971752312cb4f7dcb9`。

## Plan Rewrite Notes

| Existing item | Decision | Reason |
|---|---|---|
| MRR 门禁决定是否生成候选包 | replace | MRR 仅保留为风险证据，线上提交是最终裁决 |
| Phase 4 因离线 No-Go 跳过 | reopen | 用户明确授权生成探索包 |
| 通过离线 gate 后才固定候选 | replace | 沿用已冻结的唯一候选 Two-Tower 0.20，避免事后挑权重 |
| 直接晋升 checkpoint | defer | 线上胜出前不污染当前冠军或支付正式服务化成本 |
| 单一 MRR 报告 | expand | 增加 Hit@K 与排名分布，但不借此重新选择权重 |
| 重新从 standalone Two-Tower 模型生成测试分数 | replace | 已发现完整 Phase 2/3 直接候选包；直接对最终候选做部分混合更符合“被拒替换方案 → 混入主干” |

## Drift Diagnosis

- **Goal drift**: 把离线 MRR 当成最终目标，偏离了“生成包并获得线上证据”的真实
  目标。
- **Phase drift**: 原 Phase 4 同时绑定“正式 checkpoint 晋升”和“生成提交包”，
  导致探索性线上验证被过强门禁阻断。
- **Validation drift**: 只看 MRR 无法覆盖 top-k 命中和整体排名移动，也无法代表
  leaderboard 的未知聚合方式。
- **Compatibility drift**: 若只手改 CSV 而没有模型、候选顺序和公式校验，线上结果
  无法归因。
- **Cleanup drift**: 本轮不顺带修默认参数、refit 漂移或其他专家，避免改变因果链。

## Priority Rationale

- 固定 Two-Tower 0.20，因为它是上一轮按预注册 slice0 规则产生的唯一合法锁定
  候选；这不是根据新增指标事后挑选。
- 先建立提交混合合同和不变量测试，再运行 153420 行完整测试集推理。
- 多指标只用于风险画像，防止再次让单一离线指标替用户作线上决定。

## Assumptions and Open Decisions

| Item | Status | Impact | Owner / Next step |
|---|---|---|---|
| 当前冠军 ZIP 是线上 `1.3557002251184347` 对应包 | confirmed by user context and prior replay | 决定 Dataset1/主干来源 | 打包前再验 SHA |
| Two-Tower 完整 reranker 提交包可恢复 | confirmed locally | 决定辅助测试分数可复现 | 再验 ZIP/member/report SHA |
| 两个提交 CSV 的行列位置对应同一测试 sidecar | repository contract and standard pipeline reports | 决定分数行对齐 | 全量 shape/finite/range/row-sum 校验 |
| leaderboard 是否偏好该互补信号 | unknown | 决定是否晋升 | 用户提交后回传分数 |

## Phases

### Phase 1: 锁定提交合同

- **Purpose**: 把固定权重、不可变 Dataset1 和 Dataset2 行顺序变成可测试合同。
- **Entry condition**: 本目标文档完成。
- **Phase rules**:
  - 先 RED 后 GREEN。
  - 不修改上一轮 selection lock。
  - 新打包路径拒绝覆盖已有输出。
- **Todos**:
  - [ ] 为固定混合公式、8 位输出、shape/finite/range 写 RED 测试。
  - [ ] 为 Dataset1 字节复制和 ZIP 成员不变量写 RED 测试。
  - [ ] 实现最小 CSV/包生成接口并使 focused tests GREEN。
- **Exit proof**: focused tests 通过，生产 runner 尚未生成正式候选。
- **Stop condition**: 无法证明 Dataset1 字节不变或 Dataset2 行顺序来源唯一。

### Phase 2: 多指标画像与完整候选资产恢复

- **Purpose**: 保存离线风险全貌，并从已验证的 Two-Tower 完整 reranker 包恢复
  官方测试集候选分数。
- **Entry condition**: Phase 1 GREEN。
- **Phase rules**:
  - 多指标不得反向改变 Two-Tower 0.20。
  - 来源 ZIP、Dataset2 member、checkpoint provenance 必须写入报告。
  - 辅助输出必须 finite，并在 8 位舍入误差内 row-normalized。
- **Todos**:
  - [ ] 对验证集 full/slice 计算 MRR、Hit@1/3/5/10、mean/median rank。
  - [ ] 核验历史完整 reranker ZIP/report/checkpoint provenance。
  - [ ] 从其 Dataset2 member 保存辅助概率 `.npy` 和 reproducibility report。
- **Exit proof**: `153420 × 100` 专家分数和完整审计报告。
- **Stop condition**: 模型哈希/shape 不符、candidate 数不为 100 或存在非有限值。

### Phase 3: 组包与交付

- **Purpose**: 生成用户可直接提交的唯一探索包。
- **Entry condition**: Phase 2 通过所有文件正确性检查。
- **Phase rules**:
  - 不覆盖当前冠军包。
  - Dataset1 只做 ZIP member 原字节复制。
  - Dataset2 只应用固定公式，不做任何二次校准。
- **Todos**:
  - [ ] 从冠军 ZIP 读取主干 CSV，生成固定混合 Dataset2 CSV。
  - [ ] 组合双数据集 ZIP 并执行全量标准校验。
  - [ ] 写 candidate report、TDD evidence 和 SHA-256。
- **Exit proof**: 一个 `online_candidate` 状态的 `result.zip`。
- **Stop condition**: 任一不变量或哈希检查失败。

## Dry-Run Findings

- 当前冠军 CSV 已经是 100 维概率，可与逐行归一的 Two-Tower midrank 概率直接
  使用冻结收缩公式。
- 正式 checkpoint 接线不是生成一次探索提交包的必要条件；将其推迟到线上胜出
  后，可以避免复制约 5 GB 状态和污染当前冠军。
- Dataset2 测试集约 153420 行；已发现完整 reranker 提交包，因此无需重复 GPU
  推理。保存 `.npy` 仅用于固定解压后的输入哈希和混合公式。
- 输出使用 8 位小数会引入确定性舍入；验证必须针对保存后的 CSV，而不是只检查
  内存数组。
- 当前工作树有大量既有实验改动；本轮只新增独立提交 runner、测试、文档和结果
  目录，不清理无关内容。

## Final Validation

- `uv run --no-sync pytest` focused + 相关 submission/hybrid 回归。
- `uv run --no-sync ruff check` 覆盖新增/修改文件。
- 验证集多指标风险报告。
- Dataset1 member byte identity。
- Dataset2 shape、finite、range、row sum、固定公式与 8 位 CSV 回读校验。
- ZIP 成员、行数、列数和 SHA-256 校验。
- 线上分数严格高于 `1.3557002251184347` 才允许后续正式 checkpoint 晋升。

## First Execution Step

写出包含两个来源 ZIP/member 哈希和固定 `0.20` 的 immutable delivery lock，
再从历史完整 reranker 包恢复 Dataset2 辅助概率；不得先计算或查看新混合结果。

## Execution Result

- **Final judgment**: 完成一个 `online_candidate`；未晋升当前冠军。
- 固定公式：
  `0.80 * current_champion + 0.20 * two_tower_full_reranker`。
- 提交包：
  `result/d1_time_ramp_g050_d2_short_none50k_setwise_w080_twotower_full_w020_20260728/result.zip`。
- ZIP SHA-256：
  `10fe35d73d7981e29a33a3bab45e8e7737fdc9686f5c48c5a76679e0e263a1c6`。
- Dataset1 与当前线上冠军 member 逐字节相同，SHA-256 为
  `81285eddfba612b9c05e96075be29a15718d8a94839d143e3083bf5353644369`。
- Dataset2 为 `153420 × 100`、全有限，固定公式回算最大误差
  `4.00000022e-09`；CSV SHA-256 为
  `6712c2bc7af810d8adbce6ee7b22082df0ff7b7e45bd2eab8b4d9b7c791c1caa`。
- 相对当前冠军，测试集 top-1 改变 `7370 / 153420`（`4.8038%`），说明
  `0.20` 混合不是空操作。
- focused/相关测试 `18 passed`，Ruff 通过。
- 用户提交后，只有线上分数严格高于 `1.3557002251184347` 才进入正式
  checkpoint 晋升阶段。
