# A3 结果：post-refit 服务口径归一化校准

## 结论

**A3 已按预定协议完成，结论为 rolling No-Go。**

训练—服务漂移客观存在：三个折的最大标准化均值漂移为
`0.1370 / 0.0704 / 0.1080`，服务/训练标准差比最大达到 `1.2648`。
但“直接用服务候选重算完整 `mean/std`”没有跨折稳定提升最终
`0.80 Setwise + 0.20 LightGBM` 集成，因此不打开 official 20k、不生成 checkpoint
或提交 ZIP，也不扫描部分校准系数。

## 固定候选

- Integration ID：`post_refit_service_normalizer_calibration_v1`
- 公式：`0.80 * same Setwise state with service mean/std + 0.20 * unchanged LGBM`
- 每折 normalizer population：该折全部无标签 score query/candidate 特征
- 模型 state、feature indices、GNN、LightGBM 和 `0.80/0.20` 权重不变
- 候选数量：1；没有校准强度或融合权重扫描
- external scores read：`false`

## 漂移与状态审计

| 折 | 候选行数 | 最大均值漂移 / train std | service/train std 范围 | 最大概率变化 | state SHA 前后 |
|---|---:|---:|---:|---:|---|
| fold-0 | 3,890,700 | 0.137021 | 0.931651–1.264811 | 0.072475 | 相同 |
| fold-1 | 4,098,800 | 0.070421 | 0.946657–1.150434 | 0.044230 | 相同 |
| fold-2 | 4,019,600 | 0.108046 | 0.899776–1.203417 | 0.058198 | 相同 |

三个折的 `feature_indices_unchanged=true`，模型 state SHA-256 在校准前后逐折完全
一致。变化仅来自 `mean/std`。

## Rolling 多指标结果

下表均为 candidate − baseline；平均排名为负代表改善。

| 折 | MRR | Hit@1 | Hit@3 | Hit@10 | NDCG@10 | 平均排名 | 改善 / 恶化 query |
|---|---:|---:|---:|---:|---:|---:|---:|
| fold-0 | -0.00029344 | -0.00056545 | -0.00007711 | +0.00097669 | +0.00004744 | -0.00395816 | 2,026 / 1,783 |
| fold-1 | -0.00014498 | -0.00009759 | -0.00048795 | -0.00009759 | -0.00013913 | -0.00082951 | 1,411 / 1,299 |
| fold-2 | +0.00013601 | +0.00004976 | -0.00014927 | -0.00049756 | -0.00002523 | -0.00373171 | 2,742 / 2,362 |
| pooled | -0.00009902 | -0.00019985 | -0.00024148 | +0.00011658 | -0.00004056 | -0.00281453 | 6,179 / 5,444 |

这不是“只看 MRR”的拒绝。候选在 pooled 平均排名、Hit@10、改善/恶化 query
三个维度上更好，但同时：

- 两个折 MRR 下降；
- 两个折 NDCG@10 下降；
- pooled Hit@1 下降；
- pooled Hit@3 下降。

因此四个硬门失败：

```text
all_folds_mrr_non_decreasing
all_folds_ndcg_at_10_non_decreasing
pooled_hit_at_1_non_decreasing
pooled_hit_at_3_non_decreasing
```

## 工程落点

已实现可复用、可审计的轻量修复路径：

- 流式、batch-order invariant 的服务特征 normalizer；
- 只替换 neural result `mean/std` 的不可变 API；
- hybrid 在 final encoder refit 后的显式校准接线；
- 支持 base Fusion、Setwise、time-ramp 和 conservative-window 神经头；
- 漂移摘要进入 `TrainingReport.metrics`。

由于固定策略未通过 rolling，配置默认值最终锁为：

```text
service_normalizer_calibration_enabled = False
```

这保留了诊断/后续独立实验能力，同时保证当前冠军和默认训练行为不被一个已证伪候选
静默改变。

## Stop Rule 执行

- 没有生成 `selection-lock.json`；
- 没有打开 official 20k external；
- 没有生成真实 test 预测、checkpoint 或提交 ZIP；
- 没有根据结果追加 `0.10/0.25/0.50` 部分校准扫描；
- 当前线上冠军 `1.3557002251184347` 保持不变。

## 审计产物

- `result/dataset2_service_normalizer_rolling_20260728/reports/selection-report.json`
  - SHA-256:
    `cabe7c43326145e13ac26924fc2295a45634b718359fec95b6e316b99bc7e234`
- `result/dataset2_service_normalizer_rolling_20260728/reports/rolling-report.json`
  - SHA-256:
    `d7872b413234f33739144857df7c04d51ae18bc1d00ed531423283efe1c31d30`
- `result/dataset2_service_normalizer_rolling_20260728/reports/frozen-config.json`
  - SHA-256:
    `c019d4ac9bae89dee5bc19859c5bf8a5667a420deefd53cd1d703f665019dc05`

## 后续方向

若继续研究 A3，应另立目标解决“如何在不见目标边的前提下，用 full-encoder
口径训练融合头”。这需要真正的逐折 encoder refit 或严格 causal 的二阶段
head fine-tune，不能把本次 rolling 结果当作系数调参集继续扫描。
