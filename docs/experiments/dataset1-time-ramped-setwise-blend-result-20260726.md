# Dataset1 时间递增 Setwise 融合结果

## 结论

离线门禁通过，已生成可提交候选包；尚未提交线上。

固定融合为：

```text
progress = clip((query_time - test_min_time) / (test_max_time - test_min_time), 0, 1)
weight = progress ** 0.5
score = (1 - weight) * Dataset1_champion + weight * recent100k_Setwise
```

生产 test 全局时间范围为 `129595642..134873285`。该范围持久化在
checkpoint 中，保证 source 重排与 batch 切分不改变 query 权重。

## Prefix Selection

只用 slice0 和 slice1 选择：

| gamma | prefix delta | slice0 delta | slice1 delta | eligible |
|---:|---:|---:|---:|:---:|
| 0.5 | +0.0014169720 | +0.0009156032 | +0.0019183408 | yes |
| 1.0 | +0.0013828720 | +0.0010529212 | +0.0017128228 | yes |
| 2.0 | +0.0006915233 | +0.0004047038 | +0.0009783428 | yes |

按冻结规则选择 `gamma=0.5`。selection report SHA-256：

```text
e0af706fb402fb2679a90d436747755dfb0665e7d40e565bd7539e90d7081727
```

## Independent Gate

| 范围 | champion MRR | candidate MRR | delta |
|---|---:|---:|---:|
| full | 0.7894189977 | 0.7917380975 | +0.0023190998 |
| slice0 | 0.7936535922 | 0.7945691954 | +0.0009156032 |
| slice1 | 0.7957883902 | 0.7977067310 | +0.0019183408 |
| slice2 | 0.7788134199 | 0.7829370460 | +0.0041236260 |

full 超过 `+0.001`，三片均不退化，独立门禁通过。

## Production Artifacts

| Artifact | Rows / bytes | SHA-256 |
|---|---:|---|
| Dataset1 CSV | 61,051 rows | `81285eddfba612b9c05e96075be29a15718d8a94839d143e3083bf5353644369` |
| Dataset2 CSV | 153,420 rows | `d7e0d574789aa6b507e592fffb5da054839aba663f63ed58134ea2d951b1ae1e` |
| result.zip | 63,295,177 bytes | `1ecb99bfd0983ad7de1bf3d71e03d355838356b781f91cd38e16e7ce021b20dd` |
| checkpoint | 5,009,231,036 bytes | `7abcb6a53258ea65e792a2c21e97818e1c6e419c9ff426e5b65a943afc07465b` |

Dataset2 采用冠军 CSV 字节复制；source/output checkpoint 中 Dataset2
state 的独立 pickle SHA-256 均为：

```text
f9e6b7cefc7a5c49a854fa1cc52d5fabf94d5342053a5ccf69c51f7befdf3656
```

CSV 行数校验、checkpoint reload 和 `unzip -t` 全部通过。

## Locations

- 本地提交包：
  `result/d1_time_ramp_g050_d2_setwise_w080_seed60_20260726/result.zip`
- 本地 package report：
  `result/d1_time_ramp_g050_d2_setwise_w080_seed60_20260726/candidate-report.json`
- 本地 selection/gate evidence：
  `result/dataset1_time_ramped_setwise_blend_20260726/artifacts/`
- 服务器 checkpoint：
  `/home/edu/workspace/jittor-GPNUT1-JGRec/checkpoints/d1_time_ramp_g050_d2_setwise_w080_seed60_20260726.pkl`

## Decision

该候选值得进行一次线上提交；在 leaderboard 返回前，不替换当前冠军，也不把
离线增益解释为确定的线上增益。
