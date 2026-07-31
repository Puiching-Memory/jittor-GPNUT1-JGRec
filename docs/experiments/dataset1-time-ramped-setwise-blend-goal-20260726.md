# Goal Document: Dataset1 Time-Ramped Setwise Blend

## Go / No-Go

- **Judgment**: Go，已完成离线选择、独立门禁和生产拼包。
- **Reason**: recent-100k Setwise 已有 full `+0.0014274006`、slice2
  `+0.0038490563` 的正信号，唯一明显风险是最早 slice
  `-0.0000677395`。时间递增融合可以复用现成预测，以极低成本检验
  “越远离训练边界，Setwise 权重越高”。

## Target Outcome

在 Dataset1 冠军与 recent-100k Setwise expert 之间构造一个可部署、严格
时间单调的 query-level blend：

1. 最早 query exact champion，最晚 query exact Setwise；
2. 中间权重只由 query timestamp 决定；
3. slice0/slice1 选择固定曲线，slice2 只作独立门禁；
4. 只有 full 至少 `+0.001` 且三个 slice 均不下降才允许打包。

## Goal Definition

- **Type**: learning / quality / delivery
- **Boundary**:
  - Dataset1 only；Dataset2 字节保持当前线上冠军不变；
  - champion 固定为
    `d1_champion_d2_setwise_w080_seed60_20260725.pkl` 中 Dataset1
    `fixed MLP + LightGBM`；
  - expert 固定为现有 recent-100k Setwise validation prediction；
  - 不重训模型、不重建 200k×100 cache；
  - 时间进度固定为
    `progress=(time-min_time)/(max_time-min_time)`，clip 到 `[0,1]`；
  - 三个候选固定为 `w=progress**gamma`，
    `gamma ∈ {0.5, 1.0, 2.0}`；
  - 融合固定为
    `candidate=(1-w)*champion + w*setwise`；
  - slice0 `[0,6667)`、slice1 `[6667,13334)`、slice2
    `[13334,20000)`；
  - slice0/slice1 各自不下降且 prefix delta 至少 `+0.0002` 才有资格；
  - 资格候选按 prefix MRR 最高选择，同分时选择更大的 gamma
    （更保守、平均 Setwise 权重更低）。
- **Non-goals**:
  - 不搜索任意 breakpoint、最大权重、非单调曲线或 source segment；
  - 不加入 Dataset2 window expert；
  - 不用 slice2 选择 gamma；
  - 不降低最终 `+0.001` 或三片非退化门槛；
  - 未通过时不生成 test prediction、checkpoint、CSV 或 ZIP。
- **Deferred work**:
  - rolling-origin 历史回放；
  - source-recency ramp；
  - Dataset2 时间窗口 shrinkage。
- **Verification rule**:
  - 时间权重有限、位于 `[0,1]` 且随时间不下降；
  - 最早/最晚端点分别逐值等于 champion/Setwise；
  - 同时置换两专家 candidate 轴，只置换融合输出 candidate 轴；
  - constant-time 输入 exact champion；
  - selection 不读取 slice2 MRR/label；
  - selection report SHA 锁定后，独立 gate 才能读取 slice2。
- **Evidence source**: RED/GREEN 测试、冠军指标复现、selection report/hash、
  独立 evaluation report、CSV/ZIP validator（仅通过时）。
- **Pass criteria**:
  - selection：slice0、slice1 delta 均 `>=0`，prefix delta
    `>=+0.0002`；
  - gate：full delta `>=+0.001`、slice0/1/2 delta 全部 `>=0`；
  - Dataset2 CSV/checkpoint 字节与当前线上冠军一致；
  - 输出包通过行数、排序和 ZIP 校验。
- **Confidence note**: 当前 20k validation 已被历史实验观察，门禁能防止
  本轮直接使用 slice2 选参，但不能消除研究过程的元过拟合；线上提交仍是最终证据。
- **Judgment owner**: selection 脚本锁定 gamma；独立 gate 决定是否授权
  test/package；线上分数决定是否替换冠军。

## Current State

- Dataset1 champion validation full MRR `0.7894189977`。
- recent-100k Setwise：
  - full `+0.0014274006`
  - slice0 `-0.0000677395`
  - slice1 `+0.0005012482`
  - slice2 `+0.0038490563`
- validation Setwise prediction、模型、time/candidate/src/dst sidecars 和
  63 维 validation cache 均已存在且有 SHA-256。
- 现有代码只有全局静态 blend selector，没有时间单调 query-level blend。

## Priority Rationale

1. 先锁死时间权重和 exact endpoint，避免指标出来后改变曲线。
2. 只重算冠军 validation score；Setwise prediction 直接复用，避免训练方差。
3. slice0/slice1 分别非退化优先于 prefix 均值，防止一片掩盖另一片。
4. 未通过 selection 就停止，避免再次消费 slice2。

## Assumptions and Open Decisions

| Item | Status | Impact | Owner / Next step |
|---|---|---|---|
| validation time sidecar 与预测 candidate 轴完全对齐 | hash-confirmed | 决定 ramp 正确性 | preflight |
| query time min/max 可在 test 推理前全局获得 | confirmed | 决定生产可部署性 | test 全局范围 `129595642..134873285` 已固化进 checkpoint |
| score convex blend 的标度兼容 | confirmed by both being probabilities | 决定融合含义 | unit test |
| 三个 gamma 足以覆盖保守程度 | chosen | 避免连续扫参 | frozen config |
| Dataset2 当前冠军 CSV/checkpoint 可字节复制 | confirmed | 决定条件打包 | CSV SHA 与 checkpoint Dataset2 state pickle SHA 均一致 |

## Phases

### Phase 1: Ramp Contract RED

- **Purpose**: 锁定时间权重、融合不变性和安全 fallback。
- **Entry condition**: gamma/grid/slice/pass criteria 已冻结。
- **Phase rules**:
  - 纯 NumPy，不导入 Jittor；
  - 先 RED 后实现；
  - 测试只约束公共行为。
- **Todos**:
  - [x] 时间单调、端点、constant-time RED
    - **Surface**: `tests/test_hybrid_time_ramp.py`
    - **Proof**: 缺少目标模块/API 而 RED
    - **Depends on**: none
  - [x] candidate permutation 与 prefix selector RED
    - **Surface**: 同上
    - **Proof**: selection/fallback contract 失败
    - **Depends on**: none
- **Exit proof**: 最小测试因目标实现缺失正确失败。
- **Stop condition**: time sidecar 非有序或 prediction hash 不匹配。

### Phase 2: Minimal GREEN and Selection Runner

- **Purpose**: 实现通用 ramp 核心及不读取 slice2 的选择脚本。
- **Entry condition**: RED 原因正确。
- **Phase rules**:
  - 核心保持纯 NumPy；
  - selection report 不包含 slice2 metric；
  - 无合格候选时输出 exact champion fallback。
- **Todos**:
  - [x] 实现 ramp/selector/gate helpers
    - **Surface**: `src/jgrec/rankers/hybrid/time_ramp.py`
    - **Proof**: focused GREEN
    - **Depends on**: Phase 1
  - [x] 实现冠军复现和 selection artifact
    - **Surface**: experiment runner
    - **Proof**: baseline metric/hash 精确匹配
    - **Depends on**: core GREEN
- **Exit proof**: focused/regression tests 和 Ruff 通过。
- **Stop condition**: champion MRR 不能精确复现，或 Setwise prediction hash
  与原报告不匹配。

### Phase 3: Prefix Selection

- **Purpose**: 在 slice0/slice1 锁定一个 gamma 或停止。
- **Entry condition**: Linux tests GREEN，全部输入 hash 匹配。
- **Phase rules**:
  - 三个 gamma 一次性评估；
  - slice2 label/metric 不读取；
  - selection report 写 SHA sidecar。
- **Todos**:
  - [x] 运行三个 frozen ramps
    - **Surface**: existing 20k predictions/time sidecar
    - **Proof**: slice0/slice1/prefix delta
    - **Depends on**: Phase 2
  - [x] 锁定 selection
    - **Surface**: `selection-report.json/.sha256`
    - **Proof**: eligible config 或 no candidate
    - **Depends on**: trials
- **Exit proof**: selection artifact/hash。
- **Stop condition**: 无候选同时满足两片非退化和 prefix `+0.0002`。

### Phase 4: Independent Slice2 Gate and Conditional Package

- **Purpose**: 验证锁定 ramp 的下一时间片迁移并决定是否生产。
- **Entry condition**: selection pass 且 report/hash 匹配。
- **Phase rules**:
  - 只评估锁定 gamma；
  - full `+0.001` 与三片非退化缺一不可；
  - gate fail 不生成任何提交物。
- **Todos**:
  - [x] 独立 gate
    - **Surface**: `evaluation-report.json`
    - **Proof**: full/slice deltas、selection hash
    - **Depends on**: Phase 3 pass
  - [x] 条件 test/package
    - **Surface**: isolated CSV/checkpoint/ZIP
    - **Proof**: Dataset2 byte hash、submission validator、ZIP hash
    - **Depends on**: gate pass
- **Exit proof**: rejected evidence或验证通过的隔离包。
- **Stop condition**: gate 任一指标失败或 test 全局时间归一化不可部署。

## Dry-Run Findings

- 现有 `validation-setwise-recent_100k.npy` 只有约 8 MB；最大输入是现有
  500 MB validation cache，仅用于重算 champion score。
- ramp 计算只需 20k×100 的两次凸组合，分钟级完成。
- 端点设为 champion/Setwise 使曲线含义明确；三个 gamma 只改变中间权重，
  不引入额外模型。
- 当前 validation 已被看过，因此通过门禁只授权隔离候选，不等于保证线上提升。

## Final Validation

- `uv run --no-sync pytest tests/test_hybrid_time_ramp.py -q`
- `uv run --no-sync ruff check src/jgrec/rankers/hybrid/time_ramp.py tests/test_hybrid_time_ramp.py`
- Linux 重复 focused tests。
- champion/Setwise/time sidecar SHA 与原报告一致。
- selection SHA 与独立 gate 输入一致。
- 仅在 gate pass 时执行 package validator 和 Dataset2 byte hash。

## Execution Outcome

- prefix winner：`gamma=0.5`，prefix delta `+0.0014169720`；
  slice0 / slice1 分别 `+0.0009156032 / +0.0019183408`。
- independent gate：
  - full `0.7894189977 -> 0.7917380975`，delta `+0.0023190998`；
  - slice0 / slice1 / slice2 delta 分别
    `+0.0009156032 / +0.0019183408 / +0.0041236260`；
  - 三片非退化，门禁通过。
- production：
  - Dataset1 `61,051` rows，SHA-256
    `81285eddfba612b9c05e96075be29a15718d8a94839d143e3083bf5353644369`；
  - Dataset2 `153,420` rows，SHA-256 保持
    `d7e0d574789aa6b507e592fffb5da054839aba663f63ed58134ea2d951b1ae1e`；
  - source/output Dataset2 checkpoint state pickle SHA 均为
    `f9e6b7cefc7a5c49a854fa1cc52d5fabf94d5342053a5ccf69c51f7befdf3656`；
  - ZIP SHA-256
    `1ecb99bfd0983ad7de1bf3d71e03d355838356b781f91cd38e16e7ce021b20dd`，
    `unzip -t` 无错误。
- 最终判断：离线 Go，生产候选已生成；尚未提交线上，不能据此声明线上冠军。

## First Execution Step

新增时间权重单调性、端点、constant-time fallback、candidate permutation 和
prefix selection 的 RED 测试。
