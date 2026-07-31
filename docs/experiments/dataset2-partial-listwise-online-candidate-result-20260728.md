# Dataset2 Partial-Listwise Online Candidate Result

## Outcome

已生成一个只等待线上裁决的提交包：

```text
result/d1_time_ramp_g050_d2_short_none50k_setwise_w080_twotower_full_w020_20260728/result.zip
```

- ZIP bytes: `63,251,836`
- ZIP SHA-256:
  `10fe35d73d7981e29a33a3bab45e8e7737fdc9686f5c48c5a76679e0e263a1c6`
- Status: `online_candidate`
- Promotion authorized: `false`
- Current champion threshold: `1.3557002251184347`

## Frozen Formula

```text
candidate =
    0.80 * current short_none50/40k + Setwise0.80 champion
  + 0.20 * completed listwise Two-Tower full-reranker candidate
```

本轮没有根据新增指标修改专家或权重。`0.20` 来自上一轮预先冻结的
partial-listwise 权重；完整直接候选和当前冠军的精确 ZIP/member 哈希在计算
新混合结果前写入：

```text
result/dataset2_partial_listwise_expert_blend_20260728/
online-candidate-delivery-lock.json
```

delivery-lock SHA-256:
`37741f3dd39a1f5314dc52cdd19b60f78479af659bd1b367579a42c5d147ecc8`。

## Source Provenance

### Champion trunk

- ZIP:
  `result/d1_time_ramp_g050_d2_gnn_short_none_e50_edges40000_setwise_w080_seed60_20260727/result.zip`
- ZIP SHA-256:
  `104f68dc82aed862600be3328f779d80e04746283c0ec75193a3582266438193`
- Dataset1 member SHA-256:
  `81285eddfba612b9c05e96075be29a15718d8a94839d143e3083bf5353644369`
- Dataset2 member SHA-256:
  `b5544d15fac4bd6d5737c5e7e30d5d413553d1e11109d5ee23acb7b18513cc3a`

### Auxiliary full reranker

- ZIP:
  `result/d1_champion_d2_twotower200k_exploratory_seed60_20260724/result.zip`
- ZIP SHA-256:
  `f0b637fab7ff65dfc64b6b1d8175a475cf3e329864776ed547b7687e8fedede7`
- Dataset2 member SHA-256:
  `d88fe864c472c75083d386c9c823f44207ce791ed5afe4971752312cb4f7dcb9`
- Historical checkpoint SHA-256:
  `b46a5514ebfd9e0e5b4ec11b4c8e2d1e8e1ab15e65a8b6b940e5bd2ad7732caa`
- Materialized score SHA-256:
  `1730423d56492c0a35205c56b63f7b7a0e03846464a7d2afecef47ae6a303c5e`

资产审计纠正了“Phase 2/3 未执行”的旧判断：本地已有完整 reranker 的
checkpoint 验证报告和双数据集探索包。因此本轮直接对这个最终替换方案做
部分混合，比重新把 standalone raw tower 当最终专家更符合目标。

## Structural Verification

- ZIP 根目录成员严格为
  `dataset1.csv`、`dataset2.csv`。
- Dataset1: `61051 × 100`，与当前线上冠军逐字节相同。
- Dataset2: `153420 × 100`，所有值有限并位于 `[0, 1]`。
- Dataset2 CSV SHA-256:
  `6712c2bc7af810d8adbce6ee7b22082df0ff7b7e45bd2eab8b4d9b7c791c1caa`。
- Dataset2 最小/最大概率:
  `0.00000459 / 0.95622302`。
- 行和范围:
  `[0.99999962, 1.00000035]`。
- 保存后对固定混合公式的全量最大误差:
  `4.0000002199391815e-09`。
- 相对当前冠军，top-1 改变 `7370` 行，占 `4.8038065%`。
- 相对当前冠军，平均绝对分数变化:
  `0.00054932355`。

## Offline Evidence Is Not the Online Decision

为回应“不能只看 MRR”，此前对 standalone Two-Tower `0.20` 的对齐验证额外
报告了 Hit@1/3/5/10、平均/中位排名和 rank movement：

```text
result/dataset2_partial_listwise_expert_blend_20260728/
online-candidate-multi-metric-report.json
```

其中 full Hit@1 为 `+0.00025`，但 Hit@3/5/10 和平均排名变差。该报告描述的是
standalone tower 混合风险，不冒充本次“完整 reranker 混合”的离线成绩，也不
阻止打包。最终判断只使用用户提交后返回的线上分数。

## Verification

```text
uv run --no-sync pytest \
  tests/test_partial_listwise_submission.py \
  tests/test_hybrid_partial_listwise_blend.py \
  tests/test_submission.py -q
```

Result: `18 passed`。

```text
uv run --no-sync ruff check \
  src/jgrec/partial_listwise_submission.py \
  scripts/report_dataset2_partial_listwise_online_candidate.py \
  scripts/materialize_dataset2_submission_expert.py \
  scripts/package_dataset2_partial_listwise_online_candidate.py \
  tests/test_partial_listwise_submission.py
```

Result: passed。

## Promotion Rule

当前冠军、checkpoint 和原提交包都未修改。用户提交本包后：

- 线上分数 `> 1.3557002251184347`: 再做正式 checkpoint 接线与双重回放；
- 线上分数 `<= 1.3557002251184347`: 淘汰该 `0.20` 完整候选混合，不反向用
  leaderboard 调权重。

## Online Outcome

- 用户回报线上分数：`1.3545061936665996`。
- 相对当前冠军：`-0.0011940314518351`。
- **Judgment**: rejected。
- `two_tower_full_reranker_partial_v1 / w=0.20` 候选族关闭；不根据该线上结果
  回扫邻近权重，当前冠军与 checkpoint 保持不变。
