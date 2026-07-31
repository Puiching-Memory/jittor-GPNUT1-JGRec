# Runbook: Base Fusion Context Rolling → External → Package

## Frozen inputs

- Source checkpoint:
  `checkpoints/d1_time_ramp_g050_d2_gnn_short_none_e50_edges40000_setwise_w080_seed60_20260727.pkl`
- Rolling manifest:
  `result/dataset1_rolling_origin_setwise_20260726/manifest.json`
- Rolling feature cache:
  `cache/supervised_features/dataset1_joint_recent200k_full100_seed60_20260726.train.npy`
- Rolling time cache:
  `cache/supervised_features/dataset1_joint_recent200k_full100_seed60_20260726.train-time.npy`
- External feature cache:
  `cache/supervised_features/dataset1_joint_recent200k_full100_val_seed60_20260726.val.npy`
- External time cache:
  `cache/supervised_features/dataset1_joint_recent200k_full100_val_seed60_20260726.val-time.npy`
- Local Dataset2 source:
  `artifacts/b_prediction_contract_tiesafe_20260728/result.zip`
  (`085da277…fb50730`).

## State machine

```text
local preflight
  -> rolling selected + selection-lock
    -> one-shot external accepted
      -> candidate checkpoint
        -> two byte-identical Dataset1 replays with zero ties
          -> local result.zip
```

任何箭头左侧失败，立即停止；不得扫描 v1 权重、γ、seed、epoch 或 window。

## 1. Rolling-origin

```bash
source .workspace-env.sh
uv run --no-sync python scripts/train_select_dataset1_base_context_rolling.py \
  --rolling-manifest result/dataset1_rolling_origin_setwise_20260726/manifest.json \
  --source-checkpoint checkpoints/d1_time_ramp_g050_d2_gnn_short_none_e50_edges40000_setwise_w080_seed60_20260727.pkl \
  --output-dir result/dataset1_base_context_rolling_20260728
```

只有以下文件存在且 report 为 `selected` 才继续：

```text
result/dataset1_base_context_rolling_20260728/selection/selection-lock.json
result/dataset1_base_context_rolling_20260728/selection/selection-report.json
```

## 2. External one-shot

```bash
source .workspace-env.sh
uv run --no-sync python scripts/evaluate_dataset1_base_context_external.py \
  --selection-lock result/dataset1_base_context_rolling_20260728/selection/selection-lock.json \
  --rolling-frozen-config result/dataset1_base_context_rolling_20260728/frozen-config.json \
  --source-checkpoint checkpoints/d1_time_ramp_g050_d2_gnn_short_none_e50_edges40000_setwise_w080_seed60_20260727.pkl \
  --train-features cache/supervised_features/dataset1_joint_recent200k_full100_seed60_20260726.train.npy \
  --train-times cache/supervised_features/dataset1_joint_recent200k_full100_seed60_20260726.train-time.npy \
  --external-features cache/supervised_features/dataset1_joint_recent200k_full100_val_seed60_20260726.val.npy \
  --external-times cache/supervised_features/dataset1_joint_recent200k_full100_val_seed60_20260726.val-time.npy \
  --output-dir result/dataset1_base_context_external_20260728
```

只运行一次。只有
`external-state/external-evaluation-report.json` 的 `status=accepted` 才继续。

## 3. Build candidate checkpoint

```bash
source .workspace-env.sh
uv run --no-sync python scripts/build_dataset1_base_context_checkpoint.py \
  --source-checkpoint checkpoints/d1_time_ramp_g050_d2_gnn_short_none_e50_edges40000_setwise_w080_seed60_20260727.pkl \
  --external-result result/dataset1_base_context_external_20260728/external-result.json \
  --external-evaluation result/dataset1_base_context_external_20260728/external-state/external-evaluation-report.json \
  --candidate-head result/dataset1_base_context_external_20260728/base-context-v1-head.npz \
  --output-checkpoint checkpoints/d1_base_context_v1_g050_d2_gnn_short_setwise_seed60_20260728.pkl \
  --output-report result/dataset1_base_context_checkpoint_20260728/checkpoint-report.json
```

允许变化的 Dataset1 顶层键只有 `fusion_state`、`fusion_result`、
`fusion_hidden_dim`；Dataset2 state 不训练。

## 4. Two standard replays

```bash
source .workspace-env.sh
uv run --no-sync python scripts/replay_dataset1_base_context_checkpoint.py \
  --checkpoint checkpoints/d1_base_context_v1_g050_d2_gnn_short_setwise_seed60_20260728.pkl \
  --checkpoint-report result/dataset1_base_context_checkpoint_20260728/checkpoint-report.json \
  --output-dir result/dataset1_base_context_replay_20260728 \
  --data-dir data
```

要求两次 Dataset1 CSV 字节一致，并且精确并列为 0。

## 5. Local package

下载以下三个远端 artifact：

```text
result/dataset1_base_context_checkpoint_20260728/checkpoint-report.json
result/dataset1_base_context_replay_20260728/replay-report.json
result/dataset1_base_context_replay_20260728/replay-a/dataset1.csv
```

本地执行：

```powershell
uv run --no-sync python scripts\package_dataset1_base_context_candidate.py `
  --dataset1-csv result\dataset1_base_context_replay_20260728\replay-a\dataset1.csv `
  --replay-report result\dataset1_base_context_replay_20260728\replay-report.json `
  --checkpoint-report result\dataset1_base_context_checkpoint_20260728\checkpoint-report.json `
  --champion-result-zip artifacts\b_prediction_contract_tiesafe_20260728\result.zip `
  --output-dir result\d1_base_context_v1_g050_d2_tiesafe_champion_20260728 `
  --data-dir data
```

Windows `uv` 若仍因 `.venv/lib64` 权限失败，使用已由 uv 准备的 WSL 环境运行
同一脚本；不得重新解析依赖或修改 lockfile。

最终包只有在 `candidate-report.json` 中
`package_gate_passed=true` 后才可交给用户决定是否提交。
