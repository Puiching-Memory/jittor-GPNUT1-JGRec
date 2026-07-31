# Result: Dataset2 Champion top-k residual Setwise

## Verdict

**No-Go for slice2/package。** Champion top-k residual Setwise 已实现并完成 slice0→slice1 前向选型。top20 出现稳定的小正增益，但最高仅 `+0.0003611965`，未达到冻结的 `+0.001`，因此不读取 slice2、不接入当前冠军。

## Frozen Protocol

- Training rows: slice0 `[0,6667)`
- Selection rows: slice1 `[6667,13334)`
- Hard-negative widths: top10 / top20
- Group: positive + champion top-k negatives
- Features: 63 base + 9 multi-interest proxy，Setwise v1 后 216 维
- Loss: static champion-rank `ΔMRR` weighted pairwise softplus
- Residual MLP: hidden 32，4 epochs，batch 256
- Switch thresholds: `0.05 / 0.10 / 0.20 / 0.40`
- Selection gate: delta≥`+0.001` and coverage≤`25%`

## Training

| Model | Epoch 1 loss | Epoch 4 loss | Trend |
|---|---:|---:|---|
| top10 | `0.3646783` | `0.3442800` | decreasing |
| top20 | `0.1889222` | `0.1823755` | decreasing |

不同 top-k 的 pair 数和权重总量不同，loss 绝对值不可横向比较；两条曲线都证明优化器实际学习了 pairwise objective。

## Slice1 Selection

Baseline MRR: `0.5510080326704802`

| Top-k | Switch gain | Coverage | Delta |
|---:|---:|---:|---:|
| 10 | 0.05 | 16.6192% | `-0.0043655555` |
| 10 | 0.10 | 14.4893% | `-0.0035971416` |
| 10 | 0.20 | 10.8895% | `-0.0015974201` |
| 10 | 0.40 | 6.1647% | `-0.0009451313` |
| 20 | 0.05 | 14.2943% | `+0.0000813853` |
| 20 | 0.10 | 12.6294% | `+0.0003392998` |
| 20 | 0.20 | 9.3745% | **`+0.0003611965`** |
| 20 | 0.40 | 4.7998% | `+0.0002596207` |

最佳配置为 top20 / switch gain 0.20，路由 625/6,667 行，但只达到准入门槛的约 36%。

## Safety Contract

所有 8 个配置均满足：

- 未路由 query 逐值等于冠军；
- routed query 只重新分配冠军 top-k 的原分值；
- 每行 score multiset 保持；
- top-k 外候选逐值不变。

因此失败原因是泛化增益不足，不是 residual 改坏了 score 契约。

## Interpretation

- top10 全部下降，说明只看最靠前的难负例过窄，模型学到的 top1 switch 规则明显偏置。
- top20 四个阈值全部非负，且在 0.10–0.20 达到峰值，说明更宽的 hard-negative context 确实提供了弱但真实的纠错信号。
- 提高阈值能减少错误切换，但 precision 上限不足；单靠 residual switch gain 无法把 `+0.00036` 放大到生产门槛。
- 这是 slice0→slice1 的前向结果；由于门禁未通过，没有使用 slice2 验证上述解释。

## Artifact Status

- Output: `result/dataset2_champion_topk_residual_setwise_20260726`
- Selection status: `no_eligible_candidate`
- Selection report SHA-256: `3432b1b8545c939fcc59421af9a0c2ee2146dccfd003f7073ed5c1f8d963fa4b`
- SHA sidecar verified: true
- top10/top20 model hashes verified locally: true
- evaluation report: not generated
- prefix/full residual model: not generated
- candidate package: not generated
- Dataset1 frozen SHA-256: `6760fd966fde3a8de4693d06f27ec0f2458e49c44e2ea48fc79e1393434e6e2b`

## Decision

保留 residual 核心和 top20 模型作为研究组件，但不接入线上 `1.3521011401636023` 冠军。若继续这条线，新的验证点应是“校准 residual switch 的正确率”，而不是继续扩大 top-k 或放宽阈值。
