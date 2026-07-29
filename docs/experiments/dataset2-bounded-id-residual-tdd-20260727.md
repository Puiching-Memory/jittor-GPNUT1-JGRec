# Dataset2 Frozen CST + Bounded ID Residual TDD Evidence

## Target behavior

冻结现有纯 Jittor CST 主干，只训练 candidate-ID embedding 与线性 residual
head，并保证：

- `cap=0` 或零初始化 head 时逐元素精确复现 frozen base logits；
- 每个 candidate 的绝对 logit residual 不超过固定 cap；
- 实验 API 不允许 cap 大于 `0.10`；
- 候选同步置换只同步置换输出；
- 固定 epoch 训练、保存和重载只依赖 Jittor/NumPy；
- rolling-origin score 段只使用已冻结的 CST logits。

正式公式为：

```text
score = frozen_base
      + cap * tanh(raw_id_logit - row_mean(raw_id_logit))
```

## RED

实现文件创建前首次运行：

```bash
.venv/bin/python -m pytest -q \
  tests/test_hybrid_bounded_id_residual.py
```

预期失败并实际失败：

```text
ModuleNotFoundError:
  No module named 'jgrec.rankers.hybrid.bounded_id_residual'
```

第一次实现把 cap 错误解释成 `cap × row_std(base)`。补入绝对上限回归测试后：

```text
FAILED test_bounded_scores_never_exceed_absolute_cap
abs(scores - base) = 3.95285
requested cap = 0.05
```

这个第二次 RED 证明 normalized cap 虽然有界，但不满足“最大幅度
`0.02–0.10`”的字面契约。因此 normalized v1 仅保留为失败诊断，
正式结果全部来自 absolute-cap v2。

## GREEN

最小实现：

- `BoundedIDResidual`
  - Jittor item embedding、dropout 和线性 head；
  - output head 零初始化，初始状态精确等于 frozen base；
  - row-centered raw ID logit 经 `tanh` 后乘绝对 cap；
  - cap 配置硬拒绝大于 `0.10` 的值。
- fixed trainer
  - frozen base logits 仅作为常量 batch 输入；
  - 3 epoch listwise cross-entropy；
  - Adam、weight decay `1e-3`、ID dropout `0.10`；
  - 无 cap-specific early stopping。
- checkpoint
  - NumPy 容器保存 Jittor state；
  - checkpoint version 2 与 normalized v1 隔离；
  - 保存/重载 logits 一致。

远端 Linux/Jittor：

```bash
.venv/bin/python -m pytest -q \
  tests/test_hybrid_bounded_id_residual.py
```

结果：

```text
5 passed in 2.59s
```

## Refactor

- residual 的边界写入前向公式，而不是依赖 weight decay 或训练是否收敛；
- base logits 预计算后复用，residual 训练图中不存在 CST 参数；
- Fold0/1 selection、Fold2 gate 和 external evaluation 使用独立原子产物；
- normalized v1 和 absolute v2 使用不同结果目录与 checkpoint version，
  避免断点复用污染正式结论；
- 外部 20k 只在 selection lock 与 Fold2 gate 通过后读取。

## CUDA smoke

命令：

```bash
.venv/bin/python scripts/train_dataset2_bounded_id_residual.py \
  --phase smoke \
  --train-cache-prefix \
    cache/supervised_features/dataset2_joint_recent200k_full100_seed60_20260725 \
  --sequence-cache-dir \
    cache/source_conditioned/dataset2_abcd_recent200k_full100_20260727 \
  --base-result-dir \
    result/dataset2_source_conditioned_cst_abcd_20260727 \
  --base-cache-dir \
    cache/bounded_id_residual/dataset2_frozen_a_20260727 \
  --output-dir \
    result/dataset2_bounded_id_residual_v2_20260727 \
  --device cuda
```

结果：

```text
cap=0.020 epoch=1 train_loss=4.605040
cap=0.050 epoch=1 train_loss=4.604839
cap=0.100 epoch=1 train_loss=4.604960
status=complete
```

三个 cap 均完成 CUDA 前向、反向和绝对边界审计。Smoke 指标不参与选择。

## Verification commands

```bash
.venv/bin/ruff check \
  src/jgrec/rankers/hybrid/bounded_id_residual.py \
  tests/test_hybrid_bounded_id_residual.py \
  scripts/train_dataset2_bounded_id_residual.py

.venv/bin/python -m pytest -q \
  tests/test_hybrid_bounded_id_residual.py \
  tests/test_hybrid_source_conditioned_training.py \
  tests/test_hybrid_source_conditioned_cst.py
```

Ruff 结果：`All checks passed!`

完整相关回归结果：`19 passed in 2.78s`。
