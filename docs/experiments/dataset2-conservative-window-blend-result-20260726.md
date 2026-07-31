# Dataset2 时间窗口保守融合结果

## 结论

**线上分数 `1.3545839690981516`，成为新冠军。**

保守融合只吸收上一轮固定窗口候选 30% 的 residual：

```text
window =
  0.8 * mean(recent100k, recent200k, recent200k_decay100k)
  + 0.2 * LightGBM

champion = 0.8 * recent200k + 0.2 * LightGBM

candidate = champion + 0.30 * (window - champion)
```

相对激进窗口融合，它保留了约 70% 的 full 离线增益，同时把原来的 slice1
`-0.00028430` 改为 `+0.00076711`，三个时间片全部为正。

## Leaderboard Result

| Package | Score | Delta vs previous champion |
|---|---:|---:|
| Previous champion | 1.3540333477186608 | — |
| Conservative window `alpha=0.30` | **1.3545839690981516** | **+0.0005506213794908** |

Dataset1 CSV 保持字节不变，因此本次线上增益来自 Dataset2 保守时间窗口融合。
离线 `+0.0009788321` 与线上提升方向一致。

## Prefix Selection

只读取 `[0,13334)`，其中 slice0 为 `[0,6667)`、slice1 为
`[6667,13334)`。第三片在 selection report SHA 锁定前未读取。

| Alpha | Slice0 ΔMRR | Slice1 ΔMRR | Prefix ΔMRR | Eligible |
|---:|---:|---:|---:|:---:|
| 0.05 | -0.00011068 | +0.00008900 | -0.00001084 | no |
| 0.10 | -0.00036061 | +0.00025099 | -0.00005481 | no |
| 0.20 | -0.00047040 | +0.00078780 | +0.00015870 | no |
| 0.30 | +0.00038670 | +0.00076711 | +0.00057691 | **yes** |

锁定 `alpha=0.30`。selection report SHA-256：

```text
a0bfcf22ffbaed9d09315aa504e54d7ecd0222d289628741cac69940153d76df
```

## Independent Gate

| 范围 | Champion MRR | Candidate MRR | Delta |
|---|---:|---:|---:|
| full | 0.5469178184 | 0.5478966506 | +0.0009788321 |
| slice0 | 0.5863014322 | 0.5866881339 | +0.0003867017 |
| slice1 | 0.5482466914 | 0.5490138054 | +0.0007671140 |
| slice2 | 0.5061992242 | 0.5079820256 | +0.0017828013 |

- full gate `>=+0.0002`：通过；
- 三片不下降：通过；
- source hashes unchanged：通过；
- production follow-up：授权。

## Production Artifacts

| Artifact | Rows / bytes | SHA-256 |
|---|---:|---|
| Dataset1 CSV | 61,051 rows | `81285eddfba612b9c05e96075be29a15718d8a94839d143e3083bf5353644369` |
| Dataset2 CSV | 153,420 rows | `a34076a982b3b8cddf7a8d0b79ac5e3a0f33368813061caf76aa588dfd78336d` |
| result.zip | 63,309,837 bytes | `7ff5957eaede18bbf4fc4aefc7ab32d7c516aedd26bb7bf932c9d39ada0efe8b` |
| checkpoint | 5,009,289,511 bytes | `d207cd0254c42061fbaea1be15f2abd76fb067be0da8204e5b7df85bd65b6c0a` |

Dataset1 state pickle SHA 在 source/output checkpoint 中均为：

```text
ddf8a992bc0607c33942a3d865a25befa3fa02674c444614fc14e3007b7431bd
```

checkpoint reload 后在六个跨时间边界行上，用锁定 causal validation cache
重放模型，最大绝对误差 `1.9908e-7`，完整排序一致。`unzip -t` 两个 CSV
成员均通过。本地相关回归 `17 passed, 4 skipped`，Linux 合并回归
`12 passed`，Ruff 全部通过。

## Locations

- 本地提交包：
  `result/d1_time_ramp_g050_d2_window_conservative_a030_seed60_20260726/result.zip`
- 本地 candidate report：
  `result/d1_time_ramp_g050_d2_window_conservative_a030_seed60_20260726/candidate-report.json`
- 本地 selection/gate evidence：
  `result/dataset2_conservative_window_blend_20260726/`
- 服务器 checkpoint：
  `/home/edu/workspace/jittor-GPNUT1-JGRec/checkpoints/d1_time_ramp_g050_d2_window_conservative_a030_seed60_20260726.pkl`

## Decision

线上结果已经确认保守时间窗口融合为新冠军，替换上一版
`1.3540333477186608`。冻结 `alpha=0.30`、提交包和全部 provenance；不根据这一次
leaderboard 结果继续细扫 alpha。
