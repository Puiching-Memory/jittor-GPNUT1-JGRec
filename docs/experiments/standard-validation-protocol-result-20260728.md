# 标准验证协议升级：本地实施结果

## Verdict

**本地工程部分已完成；真实候选验证尚未开始。**

特征组与 ensemble 权重现在可以作为“最终集成后的精确候选”先预登记，再由至少三折
rolling-origin 的等权平均与跨折硬门禁选择。唯一候选锁定后，external gate 会校验
从训练 origin 到评分终点的完整 468 天 horizon，并且只能打开一次。

当前没有真实新候选 plan、selection lock、external receipt 或获准提交包。

## Delivered

- `src/jgrec/standard_validation_protocol.py`
  - plan freeze；
  - 候选空间、rolling policy、external policy 独立 SHA-256；
  - 候选无关的等权多折选择；
  - MRR、Hit@1/3/10、NDCG@10、平均排名、query movement；
  - 跨折稳定性硬门禁；
  - selection lock；
  - 468 天 external horizon；
  - one-shot receipt 和禁止反向扫描标志。
- Hybrid 正式候选冻结入口：
  - `frozen_fusion_feature_candidate`；
  - `frozen_ensemble_mlp_weight`；
  - CLI 已接线。
- 标准命令：
  - `scripts/freeze_standard_validation_plan.py`；
  - `scripts/select_standard_rolling_candidate.py`；
  - `scripts/evaluate_standard_external_gate.py`；
  - `scripts/preflight_standard_validation_local.py`。
- 模板与手册：
  - `standard-validation-plan.example.json`；
  - `standard-validation-protocol-runbook-20260728.md`。

## Selection Rule

候选首先必须满足：

- 每折 MRR 与 NDCG@10 达到预登记下限；
- 多折等权平均的 MRR、Hit@1/3/10、NDCG@10 达到下限；
- 多折等权平均排名变化不超过上限；
- improved query 减 worsened query 达到下限。

只有全部硬门禁通过的候选才进入排序，顺序为：

1. 最大多折平均 MRR 增益；
2. 最大最差折 MRR 增益；
3. 最大多折平均 NDCG@10 增益；
4. 预登记 tie-break。

因此单折峰值不再能直接决定 feature group 或 ensemble 权重。

## External Rule

external 的长跨度校验使用：

```text
score_time_max - training_time_max
```

而不是仅看 external 起点 gap。Dataset2 当前元数据：

- reference origin：`1255824000`；
- external start：`1255824000`；
- external end：`1296259200`；
- horizon：`40435200` 秒，精确 `468` 天。

本次本地 preflight 只读取已有 JSON 元数据，没有打开 external 特征、标签或分数数组。
official external 已被历史实验重复使用，因此报告明确标记
`historically_reused_holdout=true`、`statistical_independence=limited`。

## Local Preflight

机器可读报告：

```text
result/standard_validation_protocol_local_20260728/preflight-report.json
```

状态：

- 非时间局部候选仍为 `ready_for_candidate_preregistration`；
- 时间局部候选为 `far_horizon_folds_required_before_preregistration`；
- rolling 3 折，窗口约 31、34、36 天；
- 当前 gapped fold 数为 `0`，不得先预注册 `full-only v2`；
- external horizon 468 天；
- `candidate_preregistered=false`；
- `selection_metrics_read=false`；
- `external_holdout_read_by_this_preflight=false`；
- `selection_lock_created=false`；
- `external_open_receipt_created=false`；
- `package_authorized=false`。

## Protocol Amendment: 2026-07-29 Far-Horizon Folds

cooccur-lift transport audit 证明旧协议存在一个系统性盲点：strict external 的
short 通道一层能量占 `36.18%`，但线上 `39.9720%` 行的 short 已完全塌缩。
因此只用 31–36 天近视界 rolling，会在看到塌缩状态前偏向保留 short，并可能
错误淘汰线上中性或正向的 full-only 候选。

标准协议现已增加显式 `time_local` 路径：

- plan lock 绑定 `short_window_seconds`、部署 collapse fraction、gapped 视界
  分位点、gap 下限和 far-horizon selection order；
- rolling manifest 必须提供至少两档 gapped folds；Dataset2 模板冻结 P75/P90/
  P100 三档 251/308/349 天，全部不小于 197.2 天短窗；
- 近折与 gapped 折分别等权，再按部署 `60.0280% near / 39.9720% collapsed`
  形成 selection 指标；
- gapped 折逐折保留 MRR/NDCG 安全门，近折不再对 time-local 候选一票否决；
- 可选 `zero_short` 近折反事实臂独立报告且永不参与选择；
- time-local external 只作 `safety_gate_only`，不授权效应量估计，并输出
  `raw_delta / 19.5` 的保守校准 proxy。

本次升级只完成通用协议和本地行为验证，没有生成真实 gapped 分数、v2 plan
lock、selection lock、external receipt 或 package。下一步必须先物化冻结的
gapped folds，再允许 `full-only v2` 进入预注册。

## Verification

相关回归：

```text
69 passed
```

覆盖：

- standard protocol；
- legacy robust weight selection；
- rolling-origin；
- Fusion/LGBM；
- CLI；
- hybrid checkpoint；
- base-context gate/head。

Ruff：

```text
All checks passed!
```

四个标准脚本和主 CLI 的 `--help` smoke 均通过。

## Remaining Remote Work

服务器恢复后才可继续：

1. 为一个新的、实质不同的候选族预登记完整精确候选空间；
2. 冻结 plan lock；
3. 在每个 rolling fold 上用冻结 feature mask/weight 生成完整最终集成分数；
4. 执行标准多折 selector；
5. 只有 selection lock 存在时，生成锁定候选的 external 分数并打开一次 gate；
6. 只有 external `package_authorized=true` 时才构建 checkpoint、双重 replay 和候选包。

external 结果不能用于回扫相邻权重或删改候选空间；新的假设必须使用新的
`experiment_id`、新 plan 和新的证据链。
