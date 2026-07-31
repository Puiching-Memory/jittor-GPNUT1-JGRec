# Dataset2 Source-Conditioned CST A/B/C/D TDD Evidence

## Target behavior

在不改变现有 63 维候选特征语义的前提下，增加可消融的 candidate ID、严格因果 source history 和 candidate self-attention，并保证：

- A 与现有 raw-feature CST 架构等价；
- B/C/D 的 candidate 与 source item 共用一张 Jittor embedding；
- source history 不含同时间或未来事件；
- rolling-origin score history 冻结在 fold origin；
- 候选同步置换只同步置换 logits；
- padding 不影响有效历史；
- 训练、保存和重载均只依赖 Jittor/NumPy。

## RED

首次运行：

```bash
.venv/bin/python -m pytest -q \
  tests/test_hybrid_source_sequence_cache.py \
  tests/test_hybrid_source_conditioned_cst.py
```

预期失败并实际失败：

```text
ModuleNotFoundError:
  No module named 'jgrec.rankers.hybrid.source_sequence_cache'
ModuleNotFoundError:
  No module named 'jgrec.rankers.hybrid.source_conditioned_cst'
```

训练/checkpoint 契约的独立 RED：

```bash
.venv/bin/python -m pytest -q \
  tests/test_hybrid_source_conditioned_training.py
```

预期失败并实际失败：

```text
ModuleNotFoundError:
  No module named 'jgrec.rankers.hybrid.source_conditioned_training'
```

这些失败发生在实现文件创建前，证明测试确实约束了新增行为，而不是事后补写的恒真测试。

## GREEN

最小实现：

- `source_sequence_cache.py`
  - timestamp-aligned 40/20、60/20、80/20 folds；
  - 严格 `event_time < query_time` 的 recent-64 history；
  - fold score history 的固定 origin 上界。
- `source_conditioned_cst.py`
  - A/B/C/D 单一实现、三个布尔结构开关；
  - 共享 candidate/source item embedding；
  - source positional/time embedding、source encoder、candidate-to-source cross-attention；
  - 可开关 candidate self-attention；
  - candidate 维不使用位置编码。
- `source_conditioned_training.py`
  - Jittor listwise cross-entropy；
  - tie-neutral MRR early stopping；
  - fixed-epoch 全量重训；
  - NumPy-only checkpoint 容器中的 Jittor state；
  - checkpoint 保存/重载和批量推理。

远端 Linux/Jittor GREEN：

```bash
.venv/bin/python -m pytest -q \
  tests/test_hybrid_source_sequence_cache.py \
  tests/test_hybrid_source_conditioned_cst.py \
  tests/test_hybrid_source_conditioned_training.py
```

结果：

```text
18 passed in 2.85s
```

覆盖的关键行为：

- 相同 timestamp 不跨 fold；
- causal history 排除 equal/future；
- recent truncation 与 origin freeze；
- A 加载现有 CST 权重后逐 logits 等价；
- A/B/C/D 候选置换等价；
- A/B 忽略 source tensor；
- C/D 对有效 source history 敏感；
- padding token 不影响 logits；
- candidate/source 路径共享同一 embedding；
- D 能完成训练、推理、保存和重载。

## Refactor

- 四个变体由同一 config 和同一模型实现，避免四套 trainer 漂移；
- 复用现有 CST relative features、candidate block、listwise loss、normalizer 和 state snapshot；
- item ID 直接使用 Dataset2 原始连续整数 ID，不增加数据拟合型词表；
- 缓存、fold artifact、selection lock、gate report 均采用原子写入，并支持完整 artifact 的断点复用。

## CUDA smoke

命令：

```bash
.venv/bin/python scripts/train_dataset2_source_conditioned_abcd.py \
  --phase smoke \
  --train-cache-prefix \
    cache/supervised_features/dataset2_joint_recent200k_full100_seed60_20260725 \
  --train-cache-report \
    result/dataset2_joint_recent200k_full100_seed60_20260725/train-cache-report.json \
  --sequence-cache-dir \
    cache/source_conditioned/dataset2_abcd_recent200k_full100_20260727 \
  --output-dir result/dataset2_source_conditioned_cst_abcd_20260727 \
  --device cuda
```

结果：

```text
A epoch=1 val_mrr=0.434363
B epoch=1 val_mrr=0.425179
C epoch=1 val_mrr=0.421804
D epoch=1 val_mrr=0.429869
status=complete
```

四个变体都完成 CUDA 前向、反向、checkpoint 保存与重载。Smoke 指标只用于工程验收，不参与模型选择。

## Verification commands

```bash
.venv/bin/ruff check \
  src/jgrec/rankers/hybrid/source_sequence_cache.py \
  src/jgrec/rankers/hybrid/source_conditioned_cst.py \
  src/jgrec/rankers/hybrid/source_conditioned_training.py \
  scripts/build_dataset2_source_sequence_cache.py \
  scripts/train_dataset2_source_conditioned_abcd.py \
  tests/test_hybrid_source_sequence_cache.py \
  tests/test_hybrid_source_conditioned_cst.py \
  tests/test_hybrid_source_conditioned_training.py
```

结果：`All checks passed!`
