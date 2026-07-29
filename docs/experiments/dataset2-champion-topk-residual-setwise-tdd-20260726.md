# Hai TDD: Champion top-k residual Setwise

## Target Behavior

训练只面对 candidate 0 positive 与冠军 top-k hard negatives；loss 使用冠军静态 rank 的 `ΔMRR` 权重。推理时 residual 只能重排冠军 top-k 的原分值，未达到 top1 switch gain 阈值的 query 必须逐值返回冠军。

## RED 1: Hard Negatives and Safe Routing

- **Test added**: `tests/test_hybrid_champion_residual.py`
- **Behavior asserted**:
  - hard negatives 排除 positive column 0；
  - 分数并列时按原 candidate 顺序稳定选择；
  - LambdaMRR weight 等于 positive/negative champion rank 的 reciprocal-rank 差；
  - threshold 下 exact fallback；
  - routed query 的 score multiset 保持；
  - top-k 外分值保持；
  - candidate 同步置换后 route 同步置换。
- **Command**:

  ```bash
  uv run --no-sync pytest tests/test_hybrid_champion_residual.py -q
  ```

- **Observed failure**:

  ```text
  ModuleNotFoundError:
  No module named 'jgrec.rankers.hybrid.champion_residual'
  ```

- **Failure is correct because**: 目标模块和 API 尚未实现，失败不是 fixture、语法或环境问题。

## GREEN 1

- **Minimal implementation**:
  - `champion_hard_negative_indices`
  - `lambda_mrr_pair_weights`
  - `route_champion_topk_residual`
  - `ChampionResidualRoutingResult`
- **Command**:

  ```bash
  uv run --no-sync pytest tests/test_hybrid_champion_residual.py -q
  ```

- **Observed pass**: `3 passed`

## RED 2: Pairwise Loss Direction

- **Test added**: `test_lambda_mrr_pairwise_loss_rewards_larger_positive_margin`
- **Behavior asserted**: 当 positive logit 提高且 hard-negative logit 降低时，固定 `ΔMRR` 加权 pairwise logistic loss 必须下降。
- **Command**:

  ```bash
  uv run --no-sync pytest tests/test_hybrid_champion_residual.py -q
  ```

- **Observed failure**:

  ```text
  ImportError:
  cannot import name 'lambda_mrr_pairwise_loss'
  ```

- **Failure is correct because**: loss 数值契约 API 尚未实现。

## GREEN 2

- **Minimal implementation**: 用稳定的 `np.logaddexp(0, -margin)` 实现 `lambda_mrr_pairwise_loss`，并验证 logits/weights shape、finite、非负权重和正权重总量。
- **Command**:

  ```bash
  uv run --no-sync pytest \
    tests/test_hybrid_champion_residual.py \
    tests/test_hybrid_setwise.py -q
  ```

- **Observed pass**: `9 passed`

## REFACTOR

- **Refactor done**: yes
- **Change**:
  - 纯 NumPy 的 hard-negative/loss/route 放入独立生产模块；
  - Jittor residual MLP、streaming normalization、训练/选型/gate 保留在隔离实验脚本；
  - 不修改现有 `fusion.py`、`setwise.py` 或冠军 checkpoint；
  - routed score 采用冠军 top-k score multiset 重新赋值，避免 residual 产生新的 score scale。
- **Command after refactor**:

  ```bash
  uv run --no-sync pytest \
    tests/test_hybrid_champion_residual.py \
    tests/test_hybrid_setwise.py \
    tests/test_hybrid_multi_interest_gate.py -q

  uv run --no-sync ruff check \
    src/jgrec/rankers/hybrid/champion_residual.py \
    tests/test_hybrid_champion_residual.py \
    scripts/run_dataset2_champion_topk_residual_setwise.py
  ```

- **Observed result**:
  - Local: `16 passed`; Ruff `All checks passed!`
  - Linux: `16 passed in 1.80s`; Ruff `All checks passed!`

## Integration Evidence

- top10 training loss: `0.364678 → 0.344280`
- top20 training loss: `0.188922 → 0.182376`
- 两个模型均完成保存/重载后再生成 validation residual。
- 所有 8 个 slice1 trial：
  - fallback exact = true
  - score multiset preserved = true
  - outside top-k exact = true
- selection status: `no_eligible_candidate`
- selection SHA-256: `3432b1b8545c939fcc59421af9a0c2ee2146dccfd003f7073ed5c1f8d963fa4b`
- `evaluation-report.json`: absent

## Next Behavior

本轮 done。实现契约成立，但 slice1 增益不足以解锁 slice2 或生产接入。
