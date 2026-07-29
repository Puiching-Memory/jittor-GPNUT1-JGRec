# Dataset2 Confidence-Routed Top-k ID Correction Result

## 结论

**置信路由 top-k ID 纠错已完成三折，但 Fold0/1 selection 未通过，因此没有读取
external 20k，也没有生成提交。**

最接近晋级的是 `top10 × 5% rows`：

- Fold0：`+0.00018236`
- Fold1：`-0.00000088`
- 两折均值：`+0.00009074`

它同时违反“每折不退化”和“两折均值至少 `+0.0001`”两条冻结规则。
Fold2 lock 后诊断进一步显示所有配置的 MRR delta 都为 `0`。

这次最有价值的发现是：**router 不是主要瓶颈，真正可纠正的 ID rank-change
样本随时间几乎消失。** top10 时间留出段的改善样本比例从 `1.65%` 降到
`0.72%`，再降到 `0.035%`；到 Fold2 score 段，proposal 几乎不再改变任何排名。

## 固定结构

所有候选使用：

- frozen pure-Jittor A/CST logits；
- absolute cap `0.10`；
- 32 维 item embedding；
- correction 固定 3 epoch；
- router hidden dim 16、固定 8 epoch；
- router supervision 来自每折训练末尾约 20k 的时间留出段；
- router features 只含：
  - base margin/entropy；
  - proposal margin、top1 change、correction magnitude/rank-change；
  - base/proposal top1 与 top-k 的历史 item support；
- probability 至少 `0.5`；
- hard route quota；
- tie-neutral MRR、seed 60。

候选：

| 名称 | Candidate 范围 | 最大路由行 |
|---|---:|---:|
| top5-route05 | base top5 | 5% |
| top10-route05 | base top10 | 5% |
| top10-route10 | base top10 | 10% |

## Fold0/1 Selection

| 候选 | Fold0 delta | Fold1 delta | 两折均值 | 晋级 |
|---|---:|---:|---:|---|
| top5-route05 | +0.00001285 | -0.00000325 | +0.00000480 | 否 |
| **top10-route05** | **+0.00018236** | **-0.00000088** | **+0.00009074** | 否 |
| top10-route10 | +0.00018236 | -0.00000088 | +0.00009074 | 否 |

top10 的 5% 与 10% 结果完全相同，因为 `p>=0.5` 的行本来就只有：

- Fold0：4.593%
- Fold1：4.160%

因此 10% 配额没有新增路由行。这说明继续扩大 row budget 没有价值。

## Fold2 诊断

selection lock 已先写入 `selected_candidate=null`，随后才运行 Fold2 诊断：

| 候选 | Routed rows | Route rate | Full delta |
|---|---:|---:|---:|
| top5-route05 | 0 | 0.000% | 0.000000 |
| top10-route05 | 189 | 0.470% | 0.000000 |
| top10-route10 | 189 | 0.470% | 0.000000 |

top10 的 189 个“高置信”行全部是 rank-neutral：proposal 有微小 logit 变化，
但没有改变正例 reciprocal rank。top5 的 router holdout 没有任何改善样本，
按契约安全退回纯 Jittor no-route constant router。

## Correction Opportunity 的时间衰减

router 训练段约 20k 行上的 proposal outcome：

| Top-k | Fold | 改善行 | 伤害行 | 中性行 | 改善比例 | Proposal MRR delta |
|---:|---:|---:|---:|---:|---:|---:|
| 5 | 0 | 74 | 58 | 20,441 | 0.360% | +0.00011550 |
| 5 | 1 | 9 | 5 | 20,170 | 0.0446% | -0.00000330 |
| 5 | 2 | 0 | 0 | 20,263 | 0.000% | 0.00000000 |
| 10 | 0 | 340 | 228 | 20,005 | 1.653% | +0.00039382 |
| 10 | 1 | 146 | 80 | 19,958 | 0.723% | +0.00018789 |
| 10 | 2 | 7 | 3 | 20,253 | 0.0345% | -0.00000646 |

top10 比 top5 保留了更多潜在纠错机会，但信号仍随时间单调崩塌。Fold2
score proposal 的 MRR delta 仅约 `+0.00000003`，router 已没有可提纯的
有效排序变化。

## 稀疏性与安全审计

全部 9 个正式 candidate/fold report：

- `topk_outside_exact = true`
- `unrouted_rows_exact = true`
- `routed_rows_match_proposal = true`
- route rate 均未超过冻结预算
- `max_absolute_residual <= 0.10`
- `trainable_frameworks = ["jittor"]`
- `non_jittor_trainable_models = []`

Fold2：

- top5 max residual：`0.00006325`
- top10 max residual：`0.00152135`

因此零增益不是越界、主干漂移或路由污染造成，而是晚期 ID correction
本身已经无法改变排名。

## 后续判断

1. **不要继续扫 top-k、route budget 或 probability threshold。**
   5% 和 10% 已经等价；晚期 proposal 没有 rank-change，放宽阈值只会增加
   中性或有害路由。

2. **candidate ID prior 这条线可以停止。**
   从无约束 ID、absolute bounded ID 到 confidence-routed top-k，风险逐步被
   控制，但可迁移增益也收敛到零，证据已经足够。

3. **下一轮应改 correction signal，而不是改 router。**
   更值得进入同一稀疏框架的是：
   - champion 与 CST/Setwise 的 OOF disagreement；
   - recent repeat/recent-neighbor 的严格时间支持差；
   - 只针对 base rank2–rank5 且存在多模型一致支持的 pairwise correction。

这类信号能随 query/context 变化；静态 item ID 无法表达这种变化。

## 产物

远端：

- `result/dataset2_confidence_routed_topk_id_20260727`

本地证据：

- `artifacts/dataset2_confidence_routed_topk_id_20260727`

实现：

- `src/jgrec/rankers/hybrid/confidence_routed_topk_id.py`
- `scripts/train_dataset2_confidence_routed_topk_id.py`
- `tests/test_hybrid_confidence_routed_topk_id.py`

提交状态：

- Fold0/1 selection：rejected
- Fold2：仅诊断
- external 20k：未读取
- submission：未生成
