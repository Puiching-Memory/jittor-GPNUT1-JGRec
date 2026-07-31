# Goal Document: Dataset2 时间窗口保守融合

## Go / No-Go

- **Judgment**: Online Go，分数 `1.3545839690981516`，成为新冠军。
- **Reason**: `alpha=0.30` 在 selection 两片和独立第三片均不退化，full MRR
  `+0.00097883`，已通过冻结的 `+0.0002` 门槛；production checkpoint reload、
  Dataset1 字节冻结和 zip 校验均通过；线上相对上一冠军提升
  `+0.0005506213794908`。

## Target Outcome

建立一个默认保持 Dataset2 当前冠军、只以小比例吸收时间窗口互补信号的可审计
融合协议。前两个可见时间片选择 residual 权重，锁定报告后才读取第三片；只有
full 达到最低增益且三个时间片均不下降，才授权生产化。

## Goal Definition

- **Type**: learning / quality / delivery
- **Boundary**:
  - Dataset2 only；
  - champion 固定为 `0.8 × recent200k Setwise + 0.2 × LightGBM`；
  - window candidate 固定为上一轮已经锁定的
    `0.8 × mean(recent100k, recent200k, recent200k_decay100k)
    + 0.2 × LightGBM`；
  - conservative score:
    `champion + alpha × (window_candidate - champion)`；
  - 非零候选固定为 `alpha ∈ {0.05, 0.10, 0.20, 0.30}`，不连续扫参；
  - selection 使用 `[0,13334)`，分为 slice0 `[0,6667)` 与
    slice1 `[6667,13334)`；
  - selection 资格：slice0、slice1 delta 均 `>=0`，prefix delta
    `>=+0.0001`；
  - 合格候选按 prefix MRR 选择；完全相同时优先较小 alpha；
  - selection report 与输入 SHA 锁定后，gate 才读取 `[13334,20000)`；
  - gate 要求三个 slice delta 均 `>=0` 且 full delta `>=+0.0002`。
- **Non-goals**:
  - 不重新训练 50k/100k/200k/decay 模型；
  - 不重新选择窗口专家子集；
  - 不修改 Dataset1 模型、CSV 或 checkpoint state；
  - 不在看到 slice2 后增补 alpha；
  - gate 失败时不打包。
- **Deferred work**:
  - 基于 query 置信度的动态 alpha；
  - Dataset2 rolling-origin 重新训练每个窗口 head；
  - 其他窗口子集或 alpha grid 搜索。
- **Verification rule**:
  - TDD 证明 residual 公式、alpha 边界、prefix-only selection 和独立 gate；
  - runner 在 selection 前核验上一轮 frozen/selection/evaluation report 和
    概率文件 SHA；
  - selection report 明确 `forward_metrics_read=false` 并生成 SHA sidecar；
  - gate 只接受 sidecar 锁定的单一 alpha；
  - Dataset1 冻结 CSV SHA 保持不变。
- **Evidence source**: RED/GREEN tests、输入 SHA、selection report/hash、
  independent gate report。
- **Pass criteria**:
  - 至少一个非零 alpha 通过 selection；
  - locked alpha 在 slice2 不下降；
  - 三片全部不下降；
  - full MRR 至少提升 `+0.0002`；
  - 输入 artifacts 的 SHA 全程不变。
- **Confidence note**: 这是缓存预测层的独立 forward gate，能验证固定 residual
  权重的时间稳健性；窗口模型的 early stopping 曾使用 selection prefix，因此不能替代
  更严格的模型级 rolling-origin。
- **Judgment owner**: selection helper 锁定 alpha；独立 gate helper决定是否授权
  production follow-up；线上 leaderboard 才能声明新冠军。

## Current State

- Dataset2 当前离线冠军 full MRR 为 `0.5469178184`。
- 激进窗口候选 full MRR 为 `0.5483139338`，delta `+0.0013961154`。
- 三片 delta 为
  `(+0.0020468235, -0.0002843004, +0.0024259775)`。
- 所需五份 validation probability 已存在并有 SHA-256；本轮只需读取
  recent100k、recent200k、recent200k_decay100k 和 LightGBM。
- 当前 Dataset1 线上包 CSV SHA-256 为
  `81285eddfba612b9c05e96075be29a15718d8a94839d143e3083bf5353644369`，
  本实验不得改写。

## Priority Rationale

1. 先锁 residual 算法和 prefix/gate 隔离，避免低成本扫权重变成事后挑点。
2. 复用已训练窗口预测，几秒内先证明固定小权重是否成立。
3. 只有缓存 gate 通过，才支付生产 test 推理和 4.7GB checkpoint 重写成本。

## Assumptions and Open Decisions

| Item | Status | Impact | Owner / Next step |
|---|---|---|---|
| 上轮 selected window blend 可作为固定 expert | confirmed | 不重复专家子集选择 | 锁定其 report/hash |
| 小 alpha 能减少 slice1 的排序翻转 | assumed | 决定是否有 eligible candidate | selection falsifier |
| `+0.0002` 是保守融合的最低可用 full 增益 | chosen | 避免把纯数值扰动当提升 | frozen gate |
| gate 通过后是否立即生产打包 | deferred | 需要 test 推理与 checkpoint 扩展 | gate 后决定 |

## Phases

### Phase 1: Conservative Contract RED

- **Purpose**: 在读取隐藏 slice 前锁定 shrinkage、selection 与 gate 行为。
- **Entry condition**: alpha grid、时间片和阈值已冻结。
- **Phase rules**:
  - 只写测试；
  - selection API 不接受 forward metric；
  - alpha 超界和输入 shape 不一致必须失败。
- **Todos**:
  - [x] 新增 residual 与 prefix-only selection RED
    - **Surface**: `tests/test_hybrid_conservative_window_blend.py`
    - **Proof**: 目标模块缺失导致正确 RED
    - **Depends on**: none
  - [x] 新增 independent gate RED
    - **Surface**: 同上
    - **Proof**: forward 回退或 full 增益不足均拒绝
    - **Depends on**: none
- **Exit proof**: focused pytest 因目标实现缺失失败。
- **Stop condition**: 现有 artifacts 无法重现冠军或 hash 不一致。

### Phase 2: Core GREEN and Locked Selection

- **Purpose**: 实现最小融合 contract，并只在前两个时间片选择 alpha。
- **Entry condition**: RED 原因正确。
- **Phase rules**:
  - 核心模块纯 NumPy；
  - frozen config 先于 selection report；
  - 不读取 slice2 指标；
  - 输入文件只读。
- **Todos**:
  - [x] 实现 conservative blend、selection 和 gate helpers
    - **Surface**: `src/jgrec/rankers/hybrid/conservative_window_blend.py`
    - **Proof**: focused GREEN
    - **Depends on**: Phase 1
  - [x] 实现缓存 selection/gate runner
    - **Surface**: `scripts/evaluate_dataset2_conservative_window_blend.py`
    - **Proof**: Ruff、py_compile、hash preflight
    - **Depends on**: core GREEN
  - [x] 运行 prefix selection并锁定报告
    - **Surface**: result report + SHA sidecar
    - **Proof**: `forward_metrics_read=false`
    - **Depends on**: runner verified
- **Exit proof**: non-zero locked alpha 或 evidence-backed stop。
- **Stop condition**: 无非零 eligible alpha、输入 hash 改变或冠军无法精确复现。

### Phase 3: Independent Forward Gate

- **Purpose**: 检验 locked alpha 向未来时间片迁移。
- **Entry condition**: selection report/hash 匹配且 alpha 非零。
- **Phase rules**:
  - 只评估 locked alpha；
  - 不修改 grid 或重新选择；
  - 失败即停止，不生成包。
- **Todos**:
  - [x] 执行 slice2/full gate
    - **Surface**: independent evaluation report
    - **Proof**: 三片 delta、full delta、source hashes unchanged
    - **Depends on**: Phase 2 pass
- **Exit proof**: production follow-up Go/No-Go。
- **Stop condition**: selection hash 不匹配或任一 gate 条件失败。

### Phase 4: Authorized Production Candidate

- **Purpose**: gate 通过后，把固定 `alpha=0.30` 与两个附加窗口 head 持久化到
  Dataset2 checkpoint，并生成可提交候选包。
- **Entry condition**: locked selection 和 independent gate 均通过。
- **Phase rules**:
  - Dataset1 state 与 CSV 必须字节冻结；
  - Dataset2 主 recent200k/LGBM 专家保持不变，只附加 recent100k 和
    recent200k_decay100k；
  - checkpoint reload 后必须复现保守融合；
  - 不提交线上。
- **Todos**:
  - [x] 增加 conservative window checkpoint round-trip RED/GREEN
    - **Surface**: hybrid ranker snapshot/hydrate/predict tests
    - **Proof**: Linux prediction round-trip
    - **Depends on**: Phase 3 pass
  - [x] 生成 Dataset2 test CSV、完整 checkpoint 与组合 zip
    - **Surface**: production builder / candidate artifacts
    - **Proof**: CSV row validation、Dataset1 SHA、checkpoint reload、`unzip -t`
    - **Depends on**: checkpoint GREEN
- **Exit proof**: candidate report 记录完整 SHA、行数和 gate provenance。
- **Stop condition**: Dataset1 字节变化、checkpoint reload 不一致或 CSV 校验失败。

## Dry-Run Findings

- 如果沿用上一轮 `+0.001` full 门槛，`alpha<=0.30` 在近似线性缩放下几乎
  不可能通过；因此按“保守融合”的收益尺度，在运行前冻结为 `+0.0002`。
- alpha 不能在 full 20k 上选择；否则 slice2 同时参与选权重和验权重。
- champion 必须从原始 recent200k + LightGBM 概率重建，不能只相信 JSON 指标。
- window subset 沿用上一轮已锁定结果，避免本轮同时搜索 subset 与 alpha。

## Final Validation

- `uv run --no-sync pytest tests/test_hybrid_conservative_window_blend.py -q`
- `uv run --no-sync ruff check` 覆盖 core、test、runner。
- Linux 重复 focused tests。
- frozen config、selection report 和源 probabilities SHA 全部复核。
- gate 通过前不生成 Dataset2 test 预测或提交包。

## First Execution Step

新增 residual 公式、prefix-only selection、alpha 安全边界和独立 gate 的 RED 测试。

## Execution Outcome

- locked alpha: `0.30`；
- selection report SHA-256:
  `a0bfcf22ffbaed9d09315aa504e54d7ecd0222d289628741cac69940153d76df`；
- selection slice deltas:
  `(+0.00038670, +0.00076711)`，prefix delta `+0.00057691`；
- gate slice deltas:
  `(+0.00038670, +0.00076711, +0.00178280)`；
- gate full delta: `+0.00097883`；
- Dataset1 CSV SHA 保持
  `81285eddfba612b9c05e96075be29a15718d8a94839d143e3083bf5353644369`；
- production checkpoint SHA-256:
  `d207cd0254c42061fbaea1be15f2abd76fb067be0da8204e5b7df85bd65b6c0a`；
- result.zip SHA-256:
  `7ff5957eaede18bbf4fc4aefc7ab32d7c516aedd26bb7bf932c9d39ada0efe8b`；
- leaderboard score: `1.3545839690981516`；
- leaderboard delta vs previous champion:
  `+0.0005506213794908`，确认新冠军。
