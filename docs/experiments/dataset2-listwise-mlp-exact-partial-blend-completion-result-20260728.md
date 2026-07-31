# Dataset2 listwise-MLP 精确部分混合完成结果（2026-07-28）

## 结论

**A2 已完成，结论为 Rolling No-Go。**

三个 rolling-origin 折均已按冻结协议重新训练 fold-exact Setwise 基线头和
63 维 listwise-MLP 辅助头。六个事先冻结的权重
`0.05/0.10/0.20/0.30/0.40/0.50` 全部未通过跨折稳定性硬门禁，
因此：

- `selected_weight = null`；
- 未生成 selection lock；
- 未读取 official 20k external holdout；
- 未训练 full auxiliary；
- 未生成提交包；
- 不允许根据本次结果追加更小权重或邻近点。

这不是“实验没做完”，而是预先定义的停止条件被触发后的正式负结论。

## 冻结候选

- Integration ID：`listwise_mlp_exact_current_champion_v1`
- Fold champion：
  `0.80 * fold Setwise(short_none 50/40k) + 0.20 * frozen LGBM`
- Auxiliary：63 维基础特征上的 fixed-5-epoch listwise-MLP
- 精确混合：
  `candidate_w = (1 - w) * fold_champion + w * fold_listwise_mlp`
- Setwise：4 epochs
- Auxiliary：5 epochs
- Batch size：256
- Learning rate：0.001
- Seed：60（各折只使用冻结的 seed salt）
- Frozen weight source SHA-256：
  `1fb8170804e5781fc6d260612c1838890d3f2bc3aba11e19814c5bddc6e5b979`
- 全过程：`external_scores_read = false`

## 三折训练结果

| Fold | Train rows | Score rows | Baseline MRR | Auxiliary MRR |
|---|---:|---:|---:|---:|
| fold-0 | `[0, 79909)` | `[79909, 118816)` | 0.429038891 | 0.425124760 |
| fold-1 | `[0, 118816)` | `[118816, 159804)` | 0.426791956 | 0.424005508 |
| fold-2 | `[0, 159804)` | `[159804, 200000)` | 0.425231824 | 0.419516942 |

## 六个最终集成候选

以下均为三个 score fold 合并后的指标差值，正数表示候选更高；平均排名差值为
负数才是改善。

| Weight | Eligible | ΔMRR | ΔNDCG@10 | ΔHit@1 | ΔHit@3 | ΔHit@10 | ΔMean rank | Improved / worsened | Worst-fold ΔMRR |
|---:|:---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.05 | no | -0.000004540 | -0.000002559 | -0.000058289 | +0.000024981 | 0 | -0.001956849 | 3107 / 3020 | -0.000133340 |
| 0.10 | no | -0.000024860 | -0.000068818 | -0.000058289 | -0.000091597 | -0.000174867 | -0.002681300 | 5652 / 5613 | -0.000290959 |
| 0.20 | no | -0.000162233 | -0.000179335 | -0.000233157 | -0.000374716 | -0.000183194 | -0.002464798 | 9619 / 9986 | -0.000477872 |
| 0.30 | no | -0.000186698 | -0.000149948 | -0.000091597 | -0.000166540 | -0.000008327 | -0.000849356 | 12777 / 13548 | -0.000716012 |
| 0.40 | no | -0.000537937 | -0.000452252 | -0.000474640 | -0.000965934 | -0.000133232 | +0.004529898 | 15242 / 16611 | -0.000894194 |
| 0.50 | no | -0.000780343 | -0.000628800 | -0.000691143 | -0.001598788 | -0.000083270 | +0.009551090 | 17434 / 19099 | -0.001218339 |

最接近通过的是 `w=0.05`，但它仍同时违反：

- fold-0 与 fold-1 MRR 不下降门禁；
- fold-0 NDCG@10 不下降门禁；
- pooled Hit@1 不下降门禁；
- pooled MRR 本身也为负增益。

`w>=0.10` 的退化随权重总体扩大，说明该 auxiliary 与当前冠军并未形成可靠的
外层互补信号。

## 判断

历史上的“listwise-MLP 独立 +0.0067”不能直接迁移到当前
`short_none + Setwise + LGBM` 冠军结构。此次实验已经对“最终集成后的精确候选”
重新训练、重新打分并跨三折验证；结论是即使降到 `w=0.05`，信号也不稳定。

因此不应：

- 用单折或仅 MRR 选择 `w=0.05`；
- 打开 external 后再回扫 `0.01/0.02/0.03`；
- 为了“必须有提交包”绕过门禁。

## 证据与可复现资产

- 冻结配置：
  `result/dataset2_listwise_mlp_exact_rolling_20260728/artifacts/frozen-config.json`
  (`91b08ccf978733588a7b5b853c3d5b26abe4d6e4716655fe2f157cc5789dbc98`)
- 三折训练报告：
  `result/dataset2_listwise_mlp_exact_rolling_20260728/artifacts/rolling-training-report.json`
  (`8e5510ef6ccc04a285a1d51a0ab6d992889a82c13c6612bf9d715221421638c8`)
- Rolling manifest：
  `result/dataset2_listwise_mlp_exact_rolling_20260728/artifacts/rolling-manifest.json`
  (`690aa345af3b303c98b27e329901105b9285899f4ff3b847cbdae14afb98796f`)
- 完整选择报告：
  `result/dataset2_listwise_mlp_exact_rolling_20260728/selection/selection-report.json`
  (`72aec2e751901dc1996acabcfa8c3db8937b11beae3961bfa1b0abbdf06c6934`)
- 远端 fold 模型和分数仍保留在：
  `/home/edu/workspace/jittor-GPNUT1-JGRec/result/dataset2_listwise_mlp_exact_rolling_20260728`

## 限制

除 `gnn_short` 使用 winning `short_none` 覆盖外，200k cache 的其余 encoder
特征是冻结资产，并未逐折重编码。因此本结论严格适用于“当前缓存表示上的融合头与
外层权重稳定性”，不等价于所有 encoder 完全 fold-pure 的重新训练。

