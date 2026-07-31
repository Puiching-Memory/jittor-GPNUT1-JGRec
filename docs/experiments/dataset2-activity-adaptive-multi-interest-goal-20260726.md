# Goal Document: Dataset2 Activity-Adaptive Multi-Interest

## Go / No-Go

- **Initial judgment**: Go for implementation and the first two temporal slices.
- **Final judgment**: No-Go for slice2 and package.
- **Final reason**: slice1 adaptive expert 相对旧 multi-interest
  `-0.0025187067`，且 Q4 相对 v1 `-0.0060464163`。两个核心门槛同时失败，
  因此未读取 slice2 指标，未生成候选包。
- **Reason**: 现有 multi-interest 相对 v1 在 source activity Q1/Q2 分别提升
  `+0.0070620/+0.0070692`，但 Q4 下降 `-0.0027555`。本轮只改兴趣表示：
  用 event-recency 指数衰减、recent-16/recent-64/full 三层中心及带
  support/age/last-hit 的自适应 cluster，直接验证高活跃 source 的旧兴趣污染假设。

## Target Outcome

构建一个向后兼容的 19 维 multi-interest v2 proxy：

1. 原 9 维 `temporal2/cluster2/cluster4 × max/top2/coverage` 原样保留；
2. 增加 event-recency 指数衰减中心；
3. 增加 recent-16、recent-64、full 三层中心；
4. cluster4 使用指数衰减的 K-means 更新，并记录 decayed support、
   normalized age、last-hit recency；
5. 候选侧以 permutation-invariant 方式输出 weighted max/top2/coverage，
   以及最佳匹配 cluster 的 support/age/last-hit；
6. 高活跃 source 自动缩短 event half-life，降低旧 cluster 权重；
7. 用 200k × 100 训练、20k × 100 validation，slice0 选 epoch、slice1
   决定是否继续，slice2 只作不可见门禁。

## Goal Definition

- **Type**: learning / quality
- **Boundary**:
  - Dataset2 线上冠军固定为 `1.3521011401636023`；
  - 图塔、63 维 base cache、LightGBM、Setwise hidden size 和
    `0.8 Setwise + 0.2 LGBM` 融合权重不改；
  - graph history 固定为当前 `gnn_recent` 窗口；
  - old proxy 对每个 source 仍用 recent-64，保证 9 个旧 channel 的定义不变；
  - v2 full history 指当前 `gnn_recent` 窗口内该 source 的完整有序历史；
  - adaptive event half-life 固定为
    `clip(64 / sqrt(max(activity / 64, 1)), 8, 64)`；
  - event weight 固定为 `2 ** (-event_age / half_life)`；
  - cluster center 用 event weight 更新；support 为 cluster decayed mass
    占比；age 为 cluster weighted mean event age 除以 source 最大 event age；
    last-hit 为 `2 ** (-cluster_last_event_age / half_life)`；
  - cluster routing weight 固定为 `sqrt(support) * last_hit`；
  - adaptive 新增 10 维：decay1 cosine、recent16/recent64/full cosine、
    cluster weighted max/top2/positive coverage、best support/age/last-hit；
  - Setwise 只用 slice0 validation early stopping；slice1 不反向参与训练；
  - 不看 slice2，除非 slice1 通过冻结门槛。
- **Non-goals**:
  - 不搜索 K、window、half-life、cluster weight 公式；
  - 不增加随机种子或模型集成；
  - 不改 Dataset1 或现有冠军 checkpoint/package；
  - 不重训 LightGBM/GNN 配置，不实现端到端多兴趣塔；
  - 不在 slice1 未通过时生产 test proxy 或打包。
- **Verification rule**:
  - 时间整体平移不改变 event-recency proxy；
  - 候选置换只置换 candidate 轴输出；
  - cluster 顺序置换不改变聚合输出；
  - 同一旧 cluster 在高活跃 history 下权重严格更低；
  - cold source/destination 输出全零且旧 9 维契约不变；
  - frozen config 和 artifact SHA 在任何指标读取前落盘。
- **Evidence source**: RED/GREEN 测试、Ruff、proxy/model hashes、训练历史、
  slice0/slice1 MRR、source-activity quartile 诊断、独立 slice2 gate。
- **Pass criteria**:
  - slice1 adaptive expert 相对旧 multi-interest expert MRR 至少 `+0.001`；
  - slice1 Q4 相对 v1 的 delta 不低于 `0`；
  - slice1 Q1/Q2 各自相对旧 multi-interest 不低于 `-0.001`；
  - 通过后锁定模型和配置，slice2 相对旧 multi-interest 至少不下降，
    且 Q4 相对 v1 不低于 `0`；
  - 若继续到 production gate，最终必须相对当前线上冠军至少 `+0.001`，
    否则不打包。
- **Judgment owner**: slice1 selection 脚本决定是否解锁 slice2；独立 gate
  决定是否授权 production routing/package；线上分数决定是否替换冠军。

## Current State

- 旧 multi-interest expert validation full MRR 为 `0.5509812280`，
  相对 v1 `+0.0040634096`。
- 按 source activity 分段，旧 expert 相对 v1：
  Q1 `+0.0070620`、Q2 `+0.0070692`、Q3 `+0.0048672`、
  Q4 `-0.0027555`。
- 旧实现只保留 recent-64，并使用固定前后各半和静态 cosine K-means；
  没有 cluster support、age、last-hit，也没有 activity-adaptive 衰减。

## Plan Rewrite Notes

| User proposal | Decision | Reason |
|---|---|---|
| 指数时间衰减兴趣中心 | use event-recency exponential decay | 对数据集时间单位不敏感，整体时间平移不改变结果 |
| recent-16/recent-64/full 三层中心 | keep | 直接表达短、中、完整兴趣跨度 |
| cluster support、age、last-hit | keep as symmetric candidate features | 保留信息且不依赖 cluster 编号 |
| 高活跃 source 降低旧 cluster 权重 | freeze shrinking half-life | 使假设可测试，避免事后调参 |
| 替换现有 9 维 | reject; append instead | 保留旧能力，模型可学习 exact feature fallback |

## Assumptions and Open Decisions

| Item | Status | Impact | Owner / Next step |
|---|---|---|---|
| `_mapped_edges` 第三列是有序事件时间 | confirmed from code | 可保留时间/顺序语义 | implementation |
| event index 比原始时间间隔更稳健 | chosen | half-life 跨 source 可比 | RED contract |
| 19 维 proxy 的 1.52 GB train artifact 可接受 | assumed | 远端磁盘与训练时长 | preflight |
| v2 能修复 Q4 且不损伤 Q1/Q2 | unresolved | 决定是否继续 | slice1 |

## Phases

### Phase 1: Proxy Contract RED

- **Purpose**: 锁定衰减、层级中心、metadata 和不变性。
- **Todos**:
  - [x] RED：recent-16/recent-64/full 与指数中心
  - [x] RED：高活跃 source 的旧 cluster 权重更低
  - [x] RED：metadata 聚合的 cluster-permutation invariance
  - [x] RED：19 维 query proxy、cold ids、candidate permutation
- **Exit proof**: 测试因目标 API/schema 缺失而正确失败。

### Phase 2: Minimal GREEN

- **Purpose**: 在现有 proxy 模块内实现纯 NumPy v2。
- **Rules**:
  - 不改变旧函数输出；
  - 无 Jittor 依赖；
  - 所有 metadata 有限且位于 `[0,1]`。
- **Exit proof**: focused tests、旧 multi-interest 回归、Ruff 通过。

### Phase 3: Build and Slice0→Slice1 Selection

- **Purpose**: 生产 200k/20k proxy，训练一个冻结配置并检验 Q4。
- **Rules**:
  - 训练前写 frozen config；
  - slice0 只用于 early stopping；
  - slice1 用冻结 pass criteria 判断；
  - 未通过即停止，不读取 slice2 指标。
- **Exit proof**: selection report、model/proxy SHA、segment metrics。

### Phase 4: Independent Slice2 Gate and Conditional Production

- **Purpose**: 检验时间迁移并决定是否构建 test/package。
- **Rules**:
  - selection config/hash 不匹配即拒绝；
  - slice2 不达标即停止；
  - 只有最终相对当前冠军 `+0.001` 才允许打包。
- **Exit proof**: rejected evidence 或独立候选包及 validator/hash。

## Dry-Run Findings

- old 9 + new 10 的 train proxy 约 `1.52 GB`，validation 约 `152 MB`。
- 最大额外 CPU 成本是逐 source weighted K-means；每个 source 只在当前
  recent graph window 中计算一次，可在图塔训练后流式写 candidate proxy。
- candidate metadata 使用最佳相似 cluster 的属性，聚合对 cluster 编号置换
  不敏感，不需要把四个 cluster 展开成大量恒定 channel。
- 若 slice1 不达标，slice2/test 生产全部跳过。

## Final Validation

- `uv run --no-sync pytest tests/test_hybrid_multi_interest_proxy.py -q`
- `uv run --no-sync ruff check src/jgrec/rankers/hybrid/multi_interest_proxy.py tests/test_hybrid_multi_interest_proxy.py`
- Linux 重复 focused/regression tests。
- frozen config、proxy/model SHA、slice1 selection 状态。
- 只在 selection pass 后运行独立 slice2 gate。

## First Execution Step

新增 activity-adaptive center/cluster/query-schema 的 RED 测试。

## Execution Result

- RED 因缺少 `ACTIVITY_ADAPTIVE_FEATURE_NAMES`/目标 API 正确失败。
- GREEN：
  - 本地纯 NumPy/Setwise 回归 `14 passed`；
  - Linux 目标、Setwise、Jittor listwise 回归 `25 passed`；
  - 本地/远端 Ruff 均通过。
- 200k train adaptive proxy 为 `800,000,128` bytes；20k validation proxy
  为 `80,000,128` bytes。
- slice0 early stopping 在 epoch 4 达到最佳 MRR `0.5913958358`。
- 冻结 slice1：
  - adaptive MRR `0.5501556237`
  - 相对 v1 `+0.0019089324`
  - 相对旧 multi-interest `-0.0025187067`
  - Q4 相对 v1 `-0.0060464163`
  - Q4 相对旧 multi-interest `-0.0048429123`
- selection status 为 `no_eligible_candidate`；`slice2_metrics_read=false`。
- selection report SHA-256：
  `b576e15c86c9faabb0802bd83509113f06c5bf24e7c478d6bde0a79b416609ac`。
- 未运行 slice2 gate、未生产 test proxy、未打包，线上冠军保持
  `1.3521011401636023`。
