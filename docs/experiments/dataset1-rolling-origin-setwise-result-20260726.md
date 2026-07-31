# Dataset1 Rolling-Origin 多时间折结果

## 结论

rolling-origin 基础设施已经建立并通过本地、Linux 验证，但本轮
`Setwise + 时间递增融合` 没有通过前三折 selection。三个 gamma 都在后两个
origin 回退，因此没有锁定候选，独立 fold3 gate 按预先冻结的协议未读取、
未运行。Dataset1 线上冠军 `1.3540333477186608` 保持不变。

这不是一次“没跑完”的实验，而是有效的稳定性否证：此前单 validation 和线上
提升不能证明该增益能跨时间 origin 复现。

## 冻结协议

| Fold | Role | Train rows | Score rows | Train time max | Score time min |
|---:|---|---:|---:|---:|---:|
| 0 | selection | `[0,100000)` | `[100000,125000)` | 112468656 | 112468770 |
| 1 | selection | `[25000,125000)` | `[125000,150000)` | 114914396 | 114914410 |
| 2 | selection | `[50000,150000)` | `[150000,175000)` | 117156968 | 117156994 |
| 3 | gate | `[75000,175000)` | `[175000,200000)` | 119303644 | 119303654 |

- 每折 raw 63-feature head 和 Setwise v1 189-feature head 都从头训练；
- train window 100k，score horizon/step 25k；
- seed 60、hidden 32、learning rate `0.001`、batch 256、固定 4 epochs；
- 不使用 score fold early stopping；
- fold0/1/2 选择，fold3 只在 selection report hash 锁定且有合格候选后解锁；
- selection 要求每折 delta `>=0` 且三折 mean delta `>=+0.0002`。

## Selection 结果

| Candidate | Fold0 ΔMRR | Fold1 ΔMRR | Fold2 ΔMRR | Mean ΔMRR | Worst | Eligible |
|---|---:|---:|---:|---:|---:|---|
| gamma 0.5 | +0.00087314 | -0.00027292 | -0.00016395 | +0.00014543 | -0.00027292 | No |
| gamma 1.0 | +0.00105828 | -0.00024774 | -0.00022269 | +0.00019595 | -0.00024774 | No |
| gamma 2.0 | +0.00054065 | -0.00009108 | -0.00020514 | +0.00008148 | -0.00020514 | No |

`gamma=1.0` 的三折平均值最接近阈值，但仍低于 `+0.0002`，并且两折为负；
不能因为均值接近就事后放宽“每折不退化”的门槛。

## 审计证据

- manifest SHA-256:
  `3a5cc87efbee49c52367b19f16d41e548a4e47fa932d2cc92449b9cc638572af`
- selection report SHA-256:
  `1a6ea48d62ea25b3c179b513a9c4e7235b82eb557b78c310340426c690282363`
- feature cache SHA-256:
  `a8f4b5d71dedd1b5aa89a9f0c40e1501afc882929d036ddaaa73076e5be6a6ef`
- `gate_fold_metrics_read=false`
- `gate_fold_unlocked=false`
- `source_hashes_unchanged=true`
- 本地测试：`12 passed`
- Linux 测试：`23 passed`
- 本地与 Linux Ruff：通过

## 产物

- `src/jgrec/rankers/hybrid/rolling_origin.py`：通用 fold、time、
  selection、gate contract；
- `tests/test_hybrid_rolling_origin.py`：5 个 contract tests；
- `scripts/build_dataset1_rolling_origin_manifest.py`：输入、schema、行序、
  时间和 SHA preflight；
- `scripts/train_select_dataset1_rolling_origin_setwise.py`：前三折固定协议训练
  与 selection report 锁定；
- `scripts/evaluate_dataset1_rolling_origin_setwise_gate.py`：独立 gate runner，
  只接受通过 hash 校验的 locked selection；
- `result/dataset1_rolling_origin_setwise_20260726/manifest.json`；
- `result/dataset1_rolling_origin_setwise_20260726/selection/selection-report.json`。

## 下一步

本轮不应继续做 full-pipeline Level-2，也不应提交新包。rolling-origin 框架可直接
用于下一项候选，但下一候选应改变信号来源，而不是继续微调 gamma。优先验证：

1. recent-only / repeat-only 专家的 OOF residual，而不是全量 Setwise 分数融合；
2. 将目标改成“只纠正 raw top-k 高置信错误”，并保持同一四折门禁；
3. 若仍只在 fold0 增益，则把它视为 regime-specific 信号，做显式时间状态路由，
   不再追求一个全时段固定权重。
