# Dataset1 Full-100 Setwise Runbook

Run only on the Linux CUDA host after SSH and resource preflight pass.

## Frozen Inputs

- Checkpoint:
  `checkpoints/d1_champion_d2_setwise_w080_seed60_20260725.pkl`
- Dataset1 train:
  `data/dataset1/train.csv`
- Frozen Dataset2 CSV:
  `result/d1_champion_d2_setwise_w080_seed60_20260725/csv/dataset2.csv`
- Frozen Dataset2 SHA-256:
  `d7e0d574789aa6b507e592fffb5da054839aba663f63ed58134ea2d951b1ae1e`

## Preflight

```bash
.deps/uv/bin/uv run --no-sync pytest \
  tests/test_hybrid_full100_training.py \
  tests/test_hybrid_fusion_analysis.py \
  tests/test_hybrid_checkpoint.py -q

.deps/uv/bin/uv run --no-sync ruff check \
  src/jgrec/rankers/hybrid/full100_training.py \
  src/jgrec/rankers/hybrid/fusion_analysis.py \
  scripts/build_dataset2_full100_train_cache.py \
  scripts/build_dataset1_full100_train_cache.py \
  scripts/train_evaluate_dataset1_full100_setwise.py \
  scripts/build_dataset1_full100_setwise_candidate.py \
  tests/test_hybrid_full100_training.py \
  tests/test_hybrid_fusion_analysis.py

nvidia-smi
df -h .
test -f checkpoints/d1_champion_d2_setwise_w080_seed60_20260725.pkl
test -f data/dataset1/train.csv
```

## Joint Cache Build

```bash
mkdir -p result/dataset1_joint_recent200k_full100_seed60_20260726

.deps/uv/bin/uv run --no-sync python scripts/build_dataset1_full100_train_cache.py \
  --checkpoint checkpoints/d1_champion_d2_setwise_w080_seed60_20260725.pkl \
  --train-csv data/dataset1/train.csv \
  --output-prefix cache/supervised_features/dataset1_joint_recent200k_full100_seed60_20260726 \
  --report result/dataset1_joint_recent200k_full100_seed60_20260726/train-cache-report.json \
  --validation-output-prefix cache/supervised_features/dataset1_joint_recent200k_full100_val_seed60_20260726 \
  --validation-report result/dataset1_joint_recent200k_full100_seed60_20260726/validation-cache-report.json \
  --candidate-count 100 \
  --train-rows 200000 \
  --validation-rows 20000 \
  --train-selection recent \
  --batch-rows 4096
```

For Dataset1, the builder must report `configured_context_end=440415`,
`context_end=387221`, and `context_backoff_rows=53194`. This is the minimum
context adjustment needed to leave exactly 200,000 chronological train rows;
all selected rows remain before `train_end=587221`.

## Dual-Scale Setwise and Unseen Gate

```bash
.deps/uv/bin/uv run --no-sync python scripts/train_evaluate_dataset1_full100_setwise.py \
  --checkpoint checkpoints/d1_champion_d2_setwise_w080_seed60_20260725.pkl \
  --train-cache-prefix cache/supervised_features/dataset1_joint_recent200k_full100_seed60_20260726 \
  --train-cache-report result/dataset1_joint_recent200k_full100_seed60_20260726/train-cache-report.json \
  --validation-cache-prefix cache/supervised_features/dataset1_joint_recent200k_full100_val_seed60_20260726 \
  --validation-cache-report result/dataset1_joint_recent200k_full100_seed60_20260726/validation-cache-report.json \
  --output-dir result/dataset1_full100_setwise_seed60_20260726
```

Exit code `2` means the frozen gate rejected the candidate. Do not package or
retry with changed settings after seeing the forward slice.

## Conditional Package

Read `selection-report.json` only after `evaluation-report.json` reports
`status=passed`, `gate_passed=true`, and `package_authorized=true`. Pass the
selected model path as `--setwise-model`:

```bash
.deps/uv/bin/uv run --no-sync python scripts/build_dataset1_full100_setwise_candidate.py \
  --source-checkpoint checkpoints/d1_champion_d2_setwise_w080_seed60_20260725.pkl \
  --evaluation-report result/dataset1_full100_setwise_seed60_20260726/evaluation-report.json \
  --setwise-model result/dataset1_full100_setwise_seed60_20260726/dataset1-setwise-<selected-scale>.npz \
  --champion-dataset2 result/d1_champion_d2_setwise_w080_seed60_20260725/csv/dataset2.csv \
  --output-checkpoint checkpoints/d1_full100_setwise_d2_champion_seed60_20260726.pkl \
  --output-dir result/d1_full100_setwise_d2_champion_seed60_20260726 \
  --data-dir data
```

## Production Result: 2026-07-26

The frozen run selected recent-100k with Setwise weight `1.00`, then failed the
gate: full MRR delta was `+0.0014274006` and slice 0 delta was
`-0.0000677395`. `evaluation-report.json` records `status=rejected` and
`package_authorized=false`. The conditional package command was not run.
