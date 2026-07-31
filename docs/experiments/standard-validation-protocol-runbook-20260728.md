# 标准验证协议运行手册

## 结论

所有新的特征组选择和 ensemble 权重选择都应走同一条晋升链：

```text
预登记完整精确候选空间
  -> 冻结 plan lock
  -> 至少三折 rolling-origin 精确最终集成分数
  -> 等权多折平均 + 跨折稳定性硬门禁
  -> 唯一 selection lock
  -> 一次性 468 天 external gate
  -> 仅通过时允许构建 checkpoint / replay / 提交包
```

旧的单切分内部搜索仍可用于探索或每折 early-stop，但不能再直接决定正式
feature group、ensemble 权重或候选晋升。

## 1. 预登记候选

复制
`docs/experiments/standard-validation-plan.example.json`，并在读取任何新
rolling 指标前替换：

- `experiment_id`；
- 完整候选空间；一个候选必须代表“最终集成后的精确组合”，例如
  `feature mask + MLP/LGBM weight + calibration + 其他共享专家配置`；
- 每个候选完整配置文件的 SHA-256；
- 明确的 tie-break 顺序；
- external lineage、468 天 horizon 和全部 gate 阈值。

不得先分别在 standalone 专家上选权重，再把该权重迁移到完整融合候选。若要比较
两个 feature mask 和三个 ensemble 权重，plan 中必须预登记全部允许比较的精确组合。

冻结：

```bash
uv run --no-sync python scripts/freeze_standard_validation_plan.py \
  --plan result/<experiment>/validation-plan.json \
  --output-dir result/<experiment>/plan
```

产物：

- `validation-plan-lock.json`；
- `preflight-report.json`，必须为
  `ready_for_rolling_selection`；
- `selection_metrics_read=false`；
- `external_holdout_read=false`；
- `package_authorized=false`。

## 2. 生成 fold-exact 候选

每个候选、每个 selection fold 都要重新训练/重放，并输出当前 fold 上完整最终集成
的 `[query, candidate]` 分数矩阵。Hybrid 的两个新冻结入口用于关闭折内单切分扫描：

```bash
uv run --no-sync jgrec-build \
  --frozen-fusion-feature-candidate <exact-mask-name> \
  --frozen-ensemble-mlp-weight <exact-weight> \
  <other-frozen-fold-arguments>
```

训练 runner 应为每折写入：

- causal `train_time_max < score_time_min <= score_time_max`；
- baseline 分数路径和 SHA-256；
- plan 中每个候选的分数路径、SHA-256、`candidate_id`、`config_sha256`；
- 同一折统一的 candidate-order fingerprint；
- 至少三个 `role=selection` 折；
- 预留折只写时间边界，不把 score artifact 放进 selection manifest。

内部训练 early-stop 可以使用当前 fold 训练前缀内的 tune 段；feature mask 和
ensemble 权重必须来自冻结 plan，不得由该 tune 段重新选择。

### 2.1 时间局部候选的 gapped 远视界折

候选只要使用短窗、近期计数、近期邻居或其他会随训练终点后的时间间隔自然塌缩的
特征，就必须改用
`standard-validation-time-local-plan.example.json`，并在读取任何候选指标前
冻结：

- `temporal_scope.kind=time_local` 与 `short_window_seconds`；
- 至少两档、正式建议三档 `gapped_fold_specs`；
- 每档部署视界分位点和 `minimum_gap_seconds`，且 gap 必须 `>=` 短窗；
- 部署中 short 完全塌缩行的预登记占比；
- 可选 `zero_short` 反事实臂；该臂固定
  `participates_in_selection=false`；
- external 的 `safety_gate_only` 角色与 `19.5x` 校准折扣。

Dataset2 cooccur-lift 首版模板按已审计的测试视界冻结：

| 档位 | 部署 gap | 秒数 |
|---|---:|---:|
| P75 | 251 天 | `21,686,400` |
| P90 | 308 天 | `26,611,200` |
| P100 | 349 天 | `30,153,600` |

短窗为 `17,038,080` 秒（197.2 天），三档都满足 `gap >= w`。部署 short
全零占比为 `0.39971972363446745`。

每个 `gapped_folds[]` 条目使用 `role=gapped`，除普通 fold 字段外还必须写入
与 plan 完全一致的 `deployment_horizon_quantile`。候选空间、config hash 和
candidate fingerprint 与近视界三折相同。若启用 `zero_short`，每个近视界 fold
还必须在独立的 `counterfactual_arms.zero_short` 命名空间写 baseline 与所有
候选分数；不得覆盖主分数。

## 3. 多折选择

```bash
uv run --no-sync python scripts/select_standard_rolling_candidate.py \
  --manifest result/<experiment>/rolling-manifest.json \
  --plan-lock result/<experiment>/plan/validation-plan-lock.json \
  --output-dir result/<experiment>/selection
```

标准 selector：

- 按折等权计算 MRR、Hit@1/3/10、NDCG@10、平均排名；
- 汇总 improved / unchanged / worsened query 数；
- 每折 MRR、NDCG@10 必须达到预登记下限；
- 多折平均的 Hit、NDCG、MRR 与平均排名必须通过预登记阈值；
- improved 减 worsened 必须达到预登记下限；
- 只在硬门禁通过的候选中，按“多折平均 MRR、最差折 MRR、多折平均
  NDCG@10、预登记 tie-break”排序；
- 不读取预留折或 external。

对 `time_local` plan，上述普通规则保留为近视界报告，但不再让近折逐折非退化
提前淘汰候选。正式裁决改为：

1. 近视界三折内部等权；
2. gapped 各档内部等权，并逐折通过 MRR/NDCG 安全下限；
3. 用预登记的 collapse fraction 计算
   `near × (1-collapse) + gapped × collapse`；
4. 用该 deployment mixture 的完整指标和 movement rate 过门；
5. 按 deployment-mixture MRR、最差 gapped MRR、deployment-mixture
   NDCG@10、预登记 tie-break 排序。

因此“近视界略退、short 塌缩视界明显改善、部署混合为正”的候选可以进入
selection lock；`zero_short` 只提供机制交叉验证，不进入 gate 或排序。

没有 `selection-lock.json` 时停止，不得打开 external。

## 4. 一次性 external

先由锁定候选生成 external baseline/candidate 精确最终集成分数和 manifest。
manifest 必须绑定：

- selection-lock SHA-256；
- `experiment_id`；
- `holdout_id` 和 lineage SHA-256；
- 锁定的 candidate ID 和 config SHA-256；
- candidate-order fingerprint；
- `training_time_max`、`score_time_min`、`score_time_max`。

这里的长跨度定义是：

```text
score_time_max - training_time_max >= 40,435,200 秒 = 468 天
```

不是仅检查 external 起点与训练终点之间的 gap。

运行：

```bash
uv run --no-sync python scripts/evaluate_standard_external_gate.py \
  --manifest result/<experiment>/external-manifest.json \
  --selection-lock result/<experiment>/selection/selection-lock.json \
  --state-dir result/<experiment>/external-state
```

命令先校验 lock、lineage、候选身份和时间跨度，再独占写
`external-open-receipt.json`，然后才读取分数。即使读取或评估失败，该 state
目录也已消费，不能覆盖或重开。external 报告固定写入：

- `weight_rescan_authorized=false`；
- `feature_rescan_authorized=false`；
- `leaderboard_tuning_authorized=false`；
- 只有所有 gate 通过时 `package_authorized=true`。

对 `time_local` 候选，external 仍用 raw delta 执行上述安全门，但报告必须同时
写入：

- `decision_role=safety_gate_only`；
- `effect_size_estimation_authorized=false`；
- `calibration_discount_factor=19.5`；
- `calibrated_effect_size_proxy = raw_delta / 19.5`。

该 proxy 是保守的 transport 校准标记，不是可识别的线上因果效应；禁止用 raw
external delta 宣称线上收益幅度。

## 5. 当前本地状态

本地元数据预检：

```bash
uv run --no-sync python scripts/preflight_standard_validation_local.py \
  --output-dir result/standard_validation_protocol_local_20260728
```

当前证据：

- rolling selection 可用 3 折，窗口分别约 31、34、36 天；
- 这三折只满足非时间局部候选的预注册前置条件；时间局部候选仍须先物化 plan
  中冻结的 gapped 折；
- Dataset2 external 从最终 rolling origin 开始，终点 horizon 精确为 468 天；
- 本次 preflight 未读取 external 数组或指标；
- official external 已被历史实验重复使用，统计独立性有限，只能作为新候选族的一次性
  工程 gate；
- 尚未预登记新的真实候选，未创建 selection lock、external receipt 或提交包。
