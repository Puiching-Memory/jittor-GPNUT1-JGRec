# Dataset2 纯 Jittor OOF Stacking 实验结果

## 结论

实验完整跑通，但候选被拒绝，不能替换当前线上冠军 `1.3545839690981516`。

首版报告中的 external validation `+0.003176` 是并列分数与“正样本固定在候选 0”共同造成的指标假增益。改用平均秩的 tie-neutral MRR，并从 meta early-stop 到最终门禁统一口径后，stacking 明显低于冠军。

## 已完成协议

- 训练缓存：Dataset2 `200k × 100`。
- OOF 覆盖：160,165 行。
- rolling-origin：
  - fold 0：train `[0, 39835)`，score `[39835, 79909)`；
  - fold 1：train `[0, 79909)`，score `[79909, 118816)`；
  - fold 2：train `[0, 118816)`，score `[118816, 159804)`；
  - fold 3：train `[0, 159804)`，score `[159804, 200000)`。
- 每折均满足 `train_time_max < score_time_min`。
- 专家：
  - CST main；
  - CST pointwise-residual；
  - Setwise MLP。
- meta learner：纯 Jittor 小 MLP。
- meta 只在前三个 OOF fold 训练，第四个 fold 选择固定融合权重。
- 三个专家均在全量 200k 行重训。
- external 20k validation 未进入专家或 meta 训练。

## 最终诚实门禁

当前冠军 tie-neutral validation MRR：

```text
full     0.5478966506
slice 0  0.5867386422
slice 1  0.5490138054
slice 2  0.5079433302
```

三个全量 Jittor 专家：

| 专家 | Full MRR | 相对冠军 |
|---|---:|---:|
| CST main | 0.5443915763 | -0.0035050743 |
| CST residual | 0.5453518593 | -0.0025447913 |
| Setwise MLP | 0.5439172359 | -0.0039794146 |

稳定版 OOF meta 在第四折选择 `meta_weight = 0.70`。external validation：

```text
full     0.3044341031  delta -0.2434625475
slice 0  0.3299788259  delta -0.2567598163
slice 1  0.3117311703  delta -0.2372826351
slice 2  0.2715961447  delta -0.2363471855
```

门禁：

```text
score_gate_passed: false
all_three_slices_non_decreasing: false
replay_passed: true
different_ranking_rows: 0
```

## 最重要的新发现

旧候选的 optimistic MRR 为 `0.7843766300`，tie-neutral MRR 只有 `0.3044341031`；正样本与负样本发生了 262,054 次精确并列。

这说明问题不在 meta MLP 容量，而在信息量：

1. 三个专家对大量候选给出语义上近似相同的 logits；
2. 行内 percentile/平均 rank 把这些近似同分进一步离散化；
3. validation 把正样本固定在第 0 列；
4. 使用稳定排序或“只数严格更高候选”的 MRR，会把并列错误地算成正样本第一；
5. CUDA 的几微小数值漂移还会随机打散近似同分，导致 checkpoint 重放排序变化。

因此，继续增加相同输入特征上的 OOF 专家或 seed 不会解决问题。后续若再做 stacking，必须先增加能区分这些候选的稳定信息，例如真实 candidate identity/embedding、连续且可重放的语义 residual，或基于冠军难负例的无位置泄漏训练；不能再依赖候选数组位置打破并列。

## 纯 Jittor 边界

- 专家和 meta learner 的可训练模块全部为 `jt.nn.Module`。
- OOF、full-data 训练和 meta 训练未使用 LightGBM/sklearn。
- NumPy 只用于 memmap、稳定行内变换、固定融合和离线指标。
- checkpoint hydrate/replay 阻断 LightGBM、sklearn 和旧 fusion 导入。
- 稳定版 meta checkpoint 重放：最大 logit/score 误差均为 `0.0`。

## Artifact

远端基础实验：

```text
result/dataset2_pure_jittor_oof_stacking_20260726
```

最终 tie-neutral 稳定实验：

```text
result/dataset2_pure_jittor_oof_stacking_stable_v2_tieneutral_20260727
```

关键文件：

```text
oof-expert-logits.npy
full-validation-expert-logits.npy
full-experts/cst_main.npz
full-experts/cst_residual.npz
full-experts/setwise_mlp.npz
meta-stacking-mlp.npz
evaluation-report.json
```

测试专家 logits 已生成到：

```text
result/dataset2_pure_jittor_oof_stacking_test_logits_20260727
```

产物：

```text
dataset2-test-expert-logits.npy
shape: [3, 153420, 100]
sha256: 7be9112b9a75f94805126497ceacea41b5136a6b4c18ff027b81f93da483c9d0

dataset2-test-stacking-scores.npy
shape: [153420, 100]
sha256: 7d795c1289a3bf6cc32cdb9683b8e7d0ff3d20b6b0b6c88936b79776ae0b7c72
```

生成耗时 `1058.51s`。报告明确写入 `submission_generated: false`，并在 hydrate/推理时阻断 LightGBM、sklearn 和旧 fusion。由于 external gate 失败，本实验不生成提交 ZIP，也不晋升 checkpoint。
