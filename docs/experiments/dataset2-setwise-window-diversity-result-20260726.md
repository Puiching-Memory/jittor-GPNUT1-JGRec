# Dataset2 Setwise 时间窗口多样性实验结果（2026-07-26）

## Verdict

**方向有效，但本轮不换冠军。**

`100k + 200k + 200k-decay100k` 的锁定融合相对当前 Dataset2 冠军取得 full MRR `+0.00139612`，第三片也取得 `+0.00242598`；但 slice1 下降 `-0.00028430`，未通过预先冻结的“三片均不退化”门禁，因此状态为 `rejected`，没有打包。

## Frozen Selection

选择只使用验证行 `[0, 13334)`；选择报告在打开第三片前已写入 SHA-256 锁：

- selection report SHA-256：`58af5c284499f99c17e2f8a9666018132bd894dae6573c181cca170f41a91598`
- 当前 200k 冠军前缀 MRR：`0.5672740618048593`
- 锁定子集前缀 MRR：`0.5681553233702769`
- 前缀增量：`+0.0008812615654176`
- 锁定子集：`recent100k, recent200k, recent200k_decay100k`

前缀最强候选：

| 候选 | 前缀 MRR |
|---|---:|
| 100k + 200k + decay100k | 0.5681553234 |
| 100k + 200k | 0.5681477009 |
| 50k + 200k + decay100k | 0.5678284370 |
| 200k + decay100k | 0.5677858711 |
| 200k champion | 0.5672740618 |

固定衰减加入 `100k + 200k` 后只贡献 `+0.00000762` 前缀 MRR；50k 没有进入最终子集。

## Forward Gate

| 指标 | 当前冠军 | 锁定候选 | Delta | Gate |
|---|---:|---:|---:|---|
| full | 0.5469178184 | 0.5483139338 | +0.0013961154 | pass |
| slice0 | 0.5863014322 | 0.5883482557 | +0.0020468235 | pass |
| slice1 | 0.5482466914 | 0.5479623910 | -0.0002843004 | **fail** |
| slice2 | 0.5061992242 | 0.5086252017 | +0.0024259775 | pass |

最终状态：

- `gate_passed: false`
- `package_authorized: false`
- `package_generated: false`
- Dataset1 冻结 SHA-256：`6760fd966fde3a8de4693d06f27ec0f2458e49c44e2ea48fc79e1393434e6e2b`

## Interpretation

1. 时间窗口确实比随机种子提供了更有价值的多样性：随机种子融合 full 下降 `-0.00165669`，本轮窗口融合 full 上升 `+0.00139612`。
2. 核心互补来自 `100k + 200k`；固定衰减在当前均匀融合中几乎没有额外前缀收益。
3. 50k 过短，单模型前缀 MRR 只有 `0.56128339`，且未被选择。
4. 当前选择器最大化“两片合并 MRR”，允许 slice0 的较大收益掩盖 slice1 的轻微下降。下一轮若继续，应在训练前冻结为：先要求 slice0、slice1 都不低于冠军，再最大化两片合并 MRR；第三片仍只做一次门禁。

## Evidence

- `result/dataset2_setwise_window_diversity_20260726/frozen-config.json`
- `result/dataset2_setwise_window_diversity_20260726/selection-report.json`
- `result/dataset2_setwise_window_diversity_20260726/evaluation-report.json`
- `result/dataset2_setwise_window_diversity_20260726/training.log`

