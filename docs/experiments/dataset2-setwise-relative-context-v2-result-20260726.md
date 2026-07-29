# Dataset2 Setwise 行内相对特征 v2：实验结果

## 结论

**No-Go，不替换冠军，不生成提交包。**

percentile midrank 与 robust z-score 已以动态 batch transform 实现，无需重建 4.7GB 特征缓存；但单模型 v2 在前两个时间片均退化，v1/v2 均匀融合也未通过“两片均不下降”的选择约束。选择报告锁定回原 v1 冠军后，第三片门禁的 full 增益为 0，低于 `+0.001`。

## 冻结协议

- 训练：Dataset2 `200k × 100`、seed60、10 epochs、patience2、batch256、hidden32、lr0.001。
- v2 通道：`raw`、`raw-row_mean`、`raw-row_max`、tie-neutral ascending percentile midrank、`(x-median)/(1.4826*MAD)`。
- early stopping / 选择：只读取 `[0,13334)`，分为 slice0 `[0,6667)` 与 slice1 `[6667,13334)`。
- 候选：v1、v2、v1/v2 uniform；外层均固定 `0.80 Setwise + 0.20 LightGBM`。
- 资格：slice0 和 slice1 均不得低于 v1；合格候选再按联合前缀 MRR 选择。
- gate：锁定后才读取 slice2；full MRR 至少 `+0.001` 且三片均不下降才可打包。

## 前两片选择结果

| Candidate | Slice0 MRR | Δ Slice0 | Slice1 MRR | Δ Slice1 | Prefix MRR | Δ Prefix | Eligible |
|---|---:|---:|---:|---:|---:|---:|---|
| v1 champion | 0.5863014322 | 0 | 0.5482466914 | 0 | 0.5672740618 | 0 | yes |
| v2 relative | 0.5833335086 | -0.0029679236 | 0.5441074632 | -0.0041392282 | 0.5637204859 | -0.0035535759 | no |
| v1/v2 uniform | 0.5859838878 | -0.0003175444 | 0.5483494709 | +0.0001027795 | 0.5671666793 | -0.0001073825 | no |

v2 的最佳纯 Setwise 前缀 MRR 为 `0.5616044917365675`（epoch 3）；epoch 5 达到 patience2 并早停。训练与选择耗时 `850.827s`。

## 第三片门禁

SHA 锁定候选为 `v1_champion`，因此只评估该候选：

| Metric | Candidate | Champion | Delta |
|---|---:|---:|---:|
| Full MRR | 0.5469178184 | 0.5469178184 | 0 |
| Slice0 MRR | 0.5863014322 | 0.5863014322 | 0 |
| Slice1 MRR | 0.5482466914 | 0.5482466914 | 0 |
| Slice2 MRR | 0.5061992242 | 0.5061992242 | 0 |

- 三片不下降：通过。
- full `+0.001`：失败。
- 最终：`gate_passed=false`、`package_authorized=false`、`package_generated=false`。

## 正确性与产物

- 本地目标回归：`12 passed`。
- Linux/Jittor 与 checkpoint 回归：`28 passed`。
- Ruff：`All checks passed!`。
- 256×100×63 动态 v2 benchmark：约 `0.133s`，输出 315 维、约 32.3MB；没有持久化新缓存。
- frozen config SHA-256：`21dfe3416a39b18be5bc37b2299681dffc78d6cf37bf9655a2ba05aa6e93dfc8`。
- selection report SHA-256：`44d9a244b77c209062ad7e5d6093d3839bb1947497db13f09cb4a4b57f03d87a`。
- v2 model SHA-256：`b05c069071e67403baa01a3197e0b1c298bd3497c29c661e6cc6a528fc48e9aa`。
- Dataset1 CSV SHA-256：`6760fd966fde3a8de4693d06f27ec0f2458e49c44e2ea48fc79e1393434e6e2b`，保持冻结。

## 判断

这个改造的工程成本低且实现可复用，但当前信号价值不足。percentile 与 robust z 对已有 raw/mean/max 的增量信息没有转化为时间稳健增益；均匀融合只在 slice1 获得极小提升，同时损失 slice0。保留 transform 能力与测试，停止为本次模型追加权重搜索或生产打包。
