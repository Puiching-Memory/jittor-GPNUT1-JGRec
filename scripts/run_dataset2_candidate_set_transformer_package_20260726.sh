#!/usr/bin/env bash
set -euo pipefail

cd /home/edu/workspace/jittor-GPNUT1-JGRec

.venv/bin/python scripts/package_dataset2_candidate_set_transformer.py \
  --source-checkpoint checkpoints/d1_time_ramp_g050_d2_window_conservative_a030_seed60_20260726.pkl \
  --source-candidate-report result/d1_time_ramp_g050_d2_window_conservative_a030_seed60_20260726/candidate-report.json \
  --ensemble-checkpoint result/dataset2_candidate_set_transformer_ensemble_20260726/candidate-set-transformer-ensemble.npz \
  --ensemble-evaluation-report result/dataset2_candidate_set_transformer_ensemble_20260726/evaluation-report.json \
  --validation-cache-prefix cache/supervised_features/dataset2_joint_recent200k_full100_val_seed60_20260725 \
  --expected-validation-probabilities result/dataset2_candidate_set_transformer_ensemble_20260726/validation-probabilities.npy \
  --frozen-dataset1-csv result/d1_time_ramp_g050_d2_window_conservative_a030_seed60_20260726/csv/dataset1.csv \
  --output-checkpoint checkpoints/d1_time_ramp_g050_d2_pure_jittor_cst_seed60_20260726.pkl \
  --output-dir result/d1_champion_d2_pure_jittor_cst_seed60_20260726 \
  --data-dir data \
  --batch-size 512
