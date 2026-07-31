# Dataset2 多专家置信路由 v2：实验结果

## 结论

**No-Go：不打开 slice2，不生成候选包，继续保留线上 `1.3521011401636023`。**

多专家路由核心、74 个 score-only descriptors、exact fallback 和前向选择协议均已实现并验证；但新增窗口专家和纠错路由在 slice1 的增益不足。27 个冻结配置没有一个达到 `+0.001`，因此在 selection 阶段停止。

## 路由结构

- exact fallback：当前 multi-interest confidence gate；
- 可选专家：v1 champion、未门控 multi-interest、window ensemble；
- window ensemble：`0.80 × mean(recent100k, recent200k, decay100k) + 0.20 × LightGBM`；
- descriptor：每专家 margin/entropy/top-k mass，以及每对专家的 tie-neutral top-k Jaccard、交叉 top 偏好、交叉 rank 和分数差；
- reward：每个替代专家相对 current gate 的 query RR delta；
- 模型：每个替代专家一个浅层 `DecisionTreeRegressor`；
- 路由：最大 predicted lift 达到阈值才替换，否则逐值 fallback。

## 冻结验证协议

- slice0 `[0,6667)`：训练 reward trees；
- slice1 `[6667,13334)`：在 27 个 depth/leaf/threshold 配置中选择；
- eligibility：slice1 MRR delta `>= +0.001` 且 coverage `<=25%`；
- slice2 `[13334,20000)`：只允许在 selection 合格并写 SHA 锁后读取；
- 本次无合格 candidate，因此没有 slice2 指标。

## Slice1 结果

current gate baseline MRR 为 `0.5510080326704802`。

| 配置 | Slice1 delta | Coverage | 路由分布 | 结论 |
|---|---:|---:|---|---|
| depth2 / leaf250 / threshold0.0025 | +0.0001826203 | 11.7894% | current 5881 / MI 296 / window 490 / v1 0 | 覆盖合格，增益失败 |
| depth2 / leaf500 / threshold0.0025 | +0.0002532340 | 34.8433% | current 4344 / MI 1564 / window 679 / v1 80 | 增益、覆盖均失败 |
| depth3 / leaf250 / threshold0.0025 | +0.0003169309 | 57.0721% | current 2862 / MI 3315 / window 111 / v1 379 | 绝对最高，但增益、覆盖均失败 |

depth1 的所有配置均选择 100% current gate，delta 为 0。更深的树能找到少量互补，但 lift 远低于门槛，并且最佳数值依赖过高覆盖。

## 正确性证据

- RED：目标模块/API 缺失，测试按预期失败。
- GREEN：目标测试 `5 passed`。
- 本地新旧 gate 回归：`14 passed`。
- Linux/Jittor 新旧 gate 回归：`14 passed`。
- Ruff：`All checks passed!`。
- Python compile：通过。
- v1、multi-interest、window 与 current gate 已知 full/slice 指标和 current gate coverage 均精确复现。
- selection report SHA-256：`8aecf8cdd198b08aee12c8d852c9d3ba4a541bc39be73f72a2af7d9512d40b6a`，与 sidecar 一致。
- Dataset1 冻结 SHA-256：`6760fd966fde3a8de4693d06f27ec0f2458e49c44e2ea48fc79e1393434e6e2b`。

## 判断

score-only 的三专家路由能够识别一部分窗口/MI 互补查询，但从 slice0 学到的规则在 slice1 上只能提供噪声量级增益。继续降低 threshold、放宽 coverage 或打开 slice2 挑配置都会破坏预先冻结的验证纪律。

下一步若继续，不应扩大同一 score-only tree 网格；更有价值的是给 router 增加“专家所选 top1 对应的原始/多兴趣支持度差”，或者转向 champion top-k residual Setwise，并使用新的冻结实验验证。
