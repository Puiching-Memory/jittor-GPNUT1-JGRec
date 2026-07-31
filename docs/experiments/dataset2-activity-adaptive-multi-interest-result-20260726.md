# Dataset2 Activity-Adaptive Multi-Interest Result

## Verdict

**No-Go。** 这版 activity-adaptive center/cluster 不应进入当前冠军：
它在 slice0 有小幅增益，但到 slice1 相对旧 multi-interest
下降 `-0.0025187067`，且目标 Q4 退化扩大到相对 v1 `-0.0060464163`。

## Frozen Design

- 旧 9 维 multi-interest proxy 字节不变。
- 新增 10 维：
  - 指数衰减中心；
  - recent-16/recent-64/full；
  - adaptive cluster weighted max/top2/coverage；
  - best-cluster support/age/last-hit。
- activity-adaptive half-life：
  `clip(64 / sqrt(max(activity / 64, 1)), 8, 64)`。
- cluster weight：`sqrt(decayed_support) * last_hit`。
- 200k × 100 full-candidate train。
- slice0 early-stop，slice1 选型；slice2 仅在通过后解锁。

## Selection Evidence

| Segment | Adaptive MRR | vs v1 | vs old multi-interest |
|---|---:|---:|---:|
| slice0 | 0.5916129709 | +0.0053115387 | +0.0005571980 |
| slice1 | 0.5501556237 | +0.0019089324 | -0.0025187067 |
| slice1 Q1 | 0.6228408570 | +0.0062311149 | -0.0049095070 |
| slice1 Q2 | 0.5887482633 | +0.0041245630 | -0.0002111034 |
| slice1 Q3 | 0.5345036847 | +0.0033273188 | -0.0001098591 |
| slice1 Q4 | 0.4545203008 | -0.0060464163 | -0.0048429123 |

门槛要求 slice1 相对旧 multi-interest 至少 `+0.001`、Q4 相对 v1
非负、Q1/Q2 相对旧 expert 不低于 `-0.001`。实际同时失败三项：
slice1、Q1、Q4。

## What This Falsifies

- 不是“给 Setwise 更多时间尺度和 cluster metadata 就会自动修复 Q4”。
- 不是“高活跃用户统一缩短 half-life”这一单调规则；Q4 中可能同时存在
  短期漂移和稳定长期偏好，统一压低旧 cluster 会误杀后者。
- 新 channel 对 slice0 有效但跨时间片不稳，不能靠读取 slice2 后再调
  half-life 或 K 来追指标。

## Best Next Move

保留旧 multi-interest 排序专家，不再让自适应特征直接改候选分数。若继续，
应把 source activity、cluster staleness/support dispersion 只作为
**门控描述符**，学习“何时禁用旧 multi-interest”，并保持 current gate
逐 query exact fallback。这个方向直接针对 Q4 负迁移，且不需要再构建新的
4 GB base cache。

## Artifacts

- Selection report：
  `result/dataset2_activity_adaptive_multi_interest_seed60_20260726/selection-report.json`
- Report SHA-256：
  `b576e15c86c9faabb0802bd83509113f06c5bf24e7c478d6bde0a79b416609ac`
- Model SHA-256：
  `c15f996fee520707637a81d0e1d12b59d7c35411627f1793cf3559808c5cead9`
- Adaptive train proxy SHA-256：
  `ff22a3f2f8ae3860308f81643edd7e6a2bafad8c9a880326906307601f24ba4f`
- Adaptive validation proxy SHA-256：
  `e57a5336060521b5724c8d44172d786ea788a3a6bbe6749f62dd8109456a97c0`
- `slice2_metrics_read=false`
- 没有 evaluation report、test proxy、submission CSV 或 ZIP。

