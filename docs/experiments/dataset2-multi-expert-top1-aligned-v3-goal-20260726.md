# Goal Document: Dataset2 多专家 top1 对齐路由 v3

## Go / No-Go

- **Initial judgment**: Go for implementation and slice1 selection.
- **Final judgment**: No-Go for slice2/package.
- **Reason**: score-only v2 在覆盖率不超过 25% 时，slice1 最好只提升 `+0.0001826203`。它能看出专家分歧，却不知道各专家选择的候选为什么不同。本轮只补充各替代专家 top1 相对当前门控 top1 的候选原始特征差和九维多兴趣支持度差，不改变专家、路由目标、时间切分或门槛。

## Target Outcome

以线上分数 `1.3521011401636023` 的 Dataset2 多兴趣置信门控为逐行精确 fallback，为 v2 路由器增加 candidate-aligned descriptor：

1. 取每位专家的 tie-neutral top1 候选集合；
2. 对集合内候选的冻结原始特征和多兴趣 proxy 取均值；
3. 为每个替代专家计算 `alternative_top1_mean - current_gate_top1_mean`；
4. 与原有 74 维 score-only descriptor 拼接；
5. 仅用 slice0 训练、slice1 选择，达到 `+0.001` 且覆盖率不超过 25% 后才允许读取 slice2。

未通过任一门禁时，当前冠军和 Dataset1 文件保持不变，不生成候选包。

## Goal Definition

- **Type**: learning / quality / delivery
- **Boundary**:
  - descriptor 专家顺序固定为 `current_gate`、`v1_champion`、`multi_interest`、`window_ensemble`；
  - 待路由的替代专家固定为 `v1_champion`、`multi_interest`、`window_ensemble`；
  - fallback 固定为 `current_gate`；
  - 保留 v2 的 74 维 score-only descriptor；
  - 原始特征白名单固定为：
    - `dst_popularity`
    - `dst_recency`
    - `candidate_test_freq`
    - `candidate_unseen_test_freq`
    - `candidate_dst_pop_row_rank`
    - `candidate_dst_recency_row_rank`
    - `candidate_test_freq_row_rank`
    - `target_pop_w020`
    - `target_recency_w020`
    - `source_profile_cosine_sum`
    - `source_profile_recent_cosine_sum`
    - `source_profile_recent_item2vec_cosine`
  - 多兴趣支持度使用现有 proxy 的全部九个通道；
  - 每个替代专家增加 `12 + 9 = 21` 个相对 fallback 的 top1 支持度差，合计 63 维；
  - 最终 descriptor 固定为 `74 + 63 = 137` 维；
  - top1 并列必须视作集合，集合内特征取均值，禁止依赖候选列顺序；
  - descriptor 只使用模型分数和候选特征，不读取标签。
- **Why this raw-feature subset**:
  - 保留 candidate-specific 的流行度、近因性、频次、行内秩和 source-profile 相似度；
  - 排除 `src_activity`、`src_recency` 等同一 query 内常量，因为其 top1 差恒为零；
  - 不把全部 63 维扩成 189 个增量，避免在 slice0 只有 6,667 行时无约束扩大搜索空间。
- **Non-goals**:
  - 不训练新 Setwise/GNN/多兴趣专家；
  - 不更换或重新调参四位专家；
  - 不增加随机种子集成；
  - 不搜索原始特征子集；
  - 不放宽 `+0.001` 与 25% 覆盖率门槛；
  - 不在 slice1 未过门时读取 slice2；
  - 不提前实现 153,420×100 test tensor 的昂贵生产推理。
- **Deferred work**:
  - top-k（非 top1）候选支持度残差；
  - 专家专属特征子集；
  - 连续 soft routing；
  - 路由器与专家联合训练。
- **Evidence source**: RED/GREEN 测试、descriptor schema、输入 SHA-256、selection report、独立 slice2 evaluation report、候选包校验。
- **Pass criteria**:
  - 单元测试证明 tie-neutral、候选列同步置换不变、feature schema 错误会拒绝；
  - slice0 `[0,6667)` 训练，slice1 `[6667,13334)` 选择；
  - slice1 相对 current gate MRR 至少 `+0.001`；
  - slice1 替代专家覆盖率不超过 `25%`；
  - selection report 在读取 slice2 前以 SHA-256 锁定；
  - 仅在 selection 通过时，使用 `[0,13334)` 重训并打开 slice2 `[13334,20000)`；
  - slice2 相对 current gate MRR 至少 `+0.001`，覆盖率不超过 `25%`；
  - 所有 fallback 行输出逐值等于 current gate；
  - Dataset1 CSV SHA-256 保持 `6760fd966fde3a8de4693d06f27ec0f2458e49c44e2ea48fc79e1393434e6e2b`。
- **Confidence note**: slice2 曾被历史实验观察，因此它只是不可见于本轮选型的时间门禁，不是全局盲测。即使离线双门禁通过，也只能生成隔离候选包，最终仍由一次线上提交决定是否替换冠军。

## Current State

- v2 已冻结四位专家的 20k validation score，并保存 74 维 score-only descriptor。
- v2 slice1 baseline MRR 为 `0.5510080326704802`。
- v2 在 coverage≤25% 下最好为 `+0.0001826203105353974`，未达到门槛；未读取 slice2、未生成包。
- validation base tensor 为 `(20000, 100, 63)`，multi-interest proxy 为 `(20000, 100, 9)`，与四组专家 score 的 candidate 轴对齐。
- 现有专家 score 可直接复用，因而本轮 slice1 选型无需重新执行 Jittor 推理。

## Plan Rewrite Notes

| Existing item | Decision | Reason |
|---|---|---|
| v2 的 74 维 score-only descriptor | keep | 已验证可部署、可重放，作为控制变量 |
| candidate-aligned descriptors | activate | 用户指定的新增信息，直接解释专家 top1 的候选属性差 |
| 全量 63 维 base feature | shrink | 189 个增量对 6,667 行训练集过宽，且包含行内常量 |
| 九维 multi-interest proxy | keep all | 通道数小，分别表达 temporal/cluster2/cluster4 的 max/top2/coverage |
| 四位专家与 window 配方 | freeze | 防止把“路由增强”混成“专家调参” |
| v2 的树深/叶子/阈值网格 | keep | 控制比较维度，仅检验新增 descriptor 的价值 |
| slice1 / slice2 门禁 | keep | 防止 full validation 反复调参造成虚假提升 |

## Drift Diagnosis

- **Goal drift**: 本轮检验的是 top1 candidate support 是否改善路由，不借机重训专家。
- **Phase drift**: 先通过纯 NumPy 契约测试，再实现实验脚本，最后才运行远端数据。
- **Validation drift**: 不用 full 20k 指标选配置；slice2 在 selection 通过前不可读。
- **Compatibility drift**: 线上 current gate 始终是 exact fallback，旧 v2 产物只读复用。
- **Complexity drift**: 特征白名单和 27 个配置预先冻结，不根据 slice1 结果增删特征。

## Priority Rationale

1. top1 集合和 candidate 轴对齐最容易静默出错，先用置换/并列测试锁死。
2. 输入 schema、shape、finite 检查必须在训练前失败，避免错误特征产生貌似合理的树。
3. slice1 未达到既有门槛时立即停止，避免为无价值方案支付 test 特征构建成本。

## Assumptions and Open Decisions

| Item | Status | Impact | Owner / Next step |
|---|---|---|---|
| v2 保存的四组 validation score 与 base/proxy candidate 顺序一致 | assumed, hash-verifiable | 决定 aligned delta 是否有效 | 新脚本校验 v2 frozen config 与输入 hash |
| 十二维 base 白名单在 63 维 schema 中唯一存在 | to verify | 缺失/重复时必须拒绝 | descriptor API 与 preflight |
| 九维 proxy 名称与生成器常量一致 | to verify | 决定生产重放契约 | 导入冻结名称并写入报告 |
| test 时可从当前 ranker 重建 base/proxy tensor | deferred until both gates pass | 决定最终打包成本 | gate 通过后再实现 |

## Phases

### Phase 1: 契约 RED

- **Purpose**: 冻结 top1 集合均值差的数值、名称、顺序和置换不变性。
- **Entry condition**: 专家顺序、特征白名单与 fallback 已冻结。
- **Rules**:
  - 只新增测试和测试证据；
  - 必须包含并列 top1；
  - 同步置换 score 与 candidate feature 后 descriptor 必须不变；
  - 缺失特征名、重复特征名、shape 错位必须失败。
- **Exit proof**: 测试因目标 API 不存在而 RED。

### Phase 2: 最小 GREEN

- **Purpose**: 实现通用的 `expert_top1_feature_deltas`。
- **Rules**:
  - 纯 NumPy；
  - 对每行 top1 最大值的完整并列集合取 feature 均值；
  - 只输出替代专家减 fallback；
  - 输出 `float32` 和冻结 descriptor 名称；
  - 不修改现有 score-only 行为。
- **Exit proof**: 目标测试、multi-expert 回归和 Ruff 通过。

### Phase 3: v3 slice1 前向选择

- **Purpose**: 在不读取 slice2 的条件下判断 63 个新增 descriptor 是否有价值。
- **Rules**:
  - 复用 v2 的四组 validation score；
  - 重验所有输入 SHA-256；
  - 拼接 74 维 score-only、36 维 raw delta、27 维 proxy delta；
  - 配置网格固定为 depth `{1,2,3}`、min leaf `{250,500,1000}`、threshold `{0.0025,0.005,0.01}`；
  - selection report 不得包含 slice2/full 指标；
  - 未达到 `+0.001` 或 coverage 超过 25% 时停止。
- **Exit proof**: 合格配置被 SHA 锁定，或明确的 `no_eligible_candidate`。

### Phase 4: 独立 slice2 gate 与条件打包

- **Purpose**: 只验证锁定配置在下一时间片是否稳定。
- **Entry condition**: Phase 3 有合格配置且 selection SHA 匹配。
- **Rules**:
  - 用 `[0,13334)` 重训锁定配置；
  - 只报告 slice2 指标；
  - delta `< +0.001`、coverage `>25%` 或 fallback 不精确均拒绝；
  - rejected 不生成 checkpoint/CSV/zip；
  - accepted 才实现 test base/proxy 构建和生产路由。
- **Exit proof**: accepted 隔离候选包，或 rejected/no-candidate 报告。

## Dry Run

1. 用两行四候选人工数据制造 fallback 与 alternative 的 top1 不同及 top1 并列。
2. 人工计算候选特征集合均值之差，确认符号是 `alternative - fallback`。
3. 同步置换候选列，结果不变。
4. 用 validation tensor 的 feature schema 冻结 12+9 名称并构造 137 维 descriptor。
5. 只切出 slice0/slice1 进入 selector；若无合格配置，脚本在任何 slice2 指标计算前退出。

## Final Validation

- `uv run --no-sync pytest tests/test_hybrid_multi_expert_gate.py -q`
- `uv run --no-sync pytest tests/test_hybrid_multi_interest_gate.py tests/test_hybrid_temporal_robust_selection.py -q`
- `uv run --no-sync ruff check src/jgrec/rankers/hybrid/multi_expert_gate.py tests/test_hybrid_multi_expert_gate.py scripts/run_dataset2_multi_expert_top1_aligned_v3.py`
- 远端 frozen-config/input hashes/selection SHA 校验。
- 只有双门禁通过时才运行 checkpoint reload、CSV/zip validator 与 Dataset1 SHA 校验。

## First Execution Step

新增 `expert_top1_feature_deltas` 的失败测试，以 API 尚不存在为正确 RED 原因。

## Execution Result

- RED 因目标 API 不存在而正确失败；最小实现后目标/回归测试本地与 Linux 均为 `16 passed`，Ruff 通过。
- 最终 descriptor 为冻结的 137 维：74 score-only + 36 raw top1 delta + 27 multi-interest top1 delta。
- slice1 baseline MRR 为 `0.5510080326704802`。
- 全网格最高 delta 只有 `+0.00011593712508661813`，且 coverage 为 `56.93715314234288%`。
- coverage≤25% 的最佳非零路由 delta 为 `-0.00039888074711902366`，coverage 为 `11.864406779661017%`。
- 27 个配置均未达到 `+0.001`；selection status 为 `no_eligible_candidate`。
- selection report SHA-256 为 `ca22b016109d4a3499679b403474d8768316ad44717b0f4abdc1a2f35022dfb7`，sidecar 已核验。
- 按协议未运行 slice2 gate，没有 evaluation report、router 或候选包；当前线上冠军保持不变。
