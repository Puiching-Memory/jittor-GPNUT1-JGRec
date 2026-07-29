# Dataset2 精确集成候选稳健选权重：实施结果

## Verdict

协议与工具已完成，真实 listwise 选权重未启动，也没有生成新提交包。

这是有意的 stop condition：当前本地资产只有重复使用的单切分 standalone
分数，以及已经线上回归的完整 reranker `w=0.20` 测试包；它们不能冒充三折
“最终集成后的精确候选”。继续从这些资产扫描邻近权重会直接使用
leaderboard 反馈过拟合。

## Implemented Contract

- rolling-origin 至少三折，训练终点严格早于评分起点，评分折不重叠且 origin
  单调递增；
- 每个权重在每折都必须是物化后的最终集成分数，并绑定同一个
  `integration_id` 与折内 candidate fingerprint；
- 指标面板：MRR、Hit@1/3/10、NDCG@10、平均排名、
  improved/unchanged/worsened query 数；
- 硬门禁：
  - 每折 MRR、NDCG@10 不下降；
  - pooled Hit@1/3/10 不下降；
  - pooled 平均排名不变差；
  - pooled improved query 多于 worsened query；
- 通过者按最差折 MRR 增益优先，而不是单折或平均峰值优先；
- selection 命令没有 external 参数，也不会读取 external；
- external 命令要求 selection lock SHA、相同 integration/weight/fingerprint；
- external 在读取分数前独占创建 receipt，第二次调用确定性失败；
- external 报告明确写入
  `weight_rescan_authorized=false` 和
  `leaderboard_tuning_authorized=false`。

## Delivered Files

- 核心实现：`src/jgrec/robust_weight_selection.py`
- rolling 选择 CLI：`scripts/select_robust_integrated_weight.py`
- external 一次性评估 CLI：`scripts/evaluate_locked_weight_external.py`
- 测试：`tests/test_robust_weight_selection.py`
- 目标文档：
  `docs/experiments/dataset2-exact-integrated-robust-weight-selection-goal-20260728.md`
- TDD 证据：
  `docs/experiments/dataset2-exact-integrated-robust-weight-selection-tdd-20260728.md`
- preflight 与 manifest 契约：
  `result/dataset2_exact_integrated_robust_weight_selection_20260728/`

## Real-Asset Preflight

机器可读报告：

```text
result/dataset2_exact_integrated_robust_weight_selection_20260728/preflight-report.json
```

状态：

- `framework_ready_real_selection_not_started`
- `external_holdout_read=false`
- `selection_lock_created=false`
- `submission_package_created=false`

已有的 `20,000 × 100` champion、listwise MLP 和 Two-Tower 分数都来自同一个
反复使用的 validation，后两者还是 standalone 语义；不能进入新 selector。
线上完整 reranker 包只有已拒绝的 `w=0.20` 无标签 test 预测，同样不能用于
离线选权重。

## Closed Candidate Family

- champion：`1.3557002251184347`
- rejected candidate：`1.3545061936665996`
- delta：`-0.0011940314518351`
- result.zip SHA-256：
  `10fe35d73d7981e29a33a3bab45e8e7737fdc9686f5c48c5a76679e0e263a1c6`

`two_tower_full_reranker_partial_v1` 已关闭，不回扫 `0.10/0.15/0.25` 或其他
邻近权重。下一次真实运行必须有一个实质变化、在读任何新折指标前登记的新
`integration_id` 和预声明权重集合。

## Verification

```text
uv run --no-sync pytest \
  tests/test_robust_weight_selection.py \
  tests/test_partial_listwise_submission.py \
  tests/test_hybrid_partial_listwise_blend.py \
  tests/test_hybrid_rolling_origin.py \
  tests/test_submission.py -q
```

Result: `30 passed`。

```text
uv run --no-sync ruff check \
  src/jgrec/robust_weight_selection.py \
  scripts/select_robust_integrated_weight.py \
  scripts/evaluate_locked_weight_external.py \
  tests/test_robust_weight_selection.py
```

Result: `All checks passed!`。

## Next Valid Move

不是再做一包。先定义一个新的 listwise 集成假设，并在任何指标读取前冻结：

1. 新 `integration_id`；
2. 权重集合；
3. 三个以上 rolling-origin 边界；
4. 每折完整 reranker 的基线与每个权重精确分数；
5. long-span external 的最小时间间隔。

只有这些资产齐全并通过 rolling 门禁后，才会生成 selection lock；随后 external
只开启一次。
