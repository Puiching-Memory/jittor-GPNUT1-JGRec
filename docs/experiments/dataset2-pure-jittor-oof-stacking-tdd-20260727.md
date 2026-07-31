# Dataset2 纯 Jittor OOF Stacking：TDD 证据

## 目标行为

- rolling-origin 每折训练数据严格早于下一时间段的 OOF 行。
- 三类专家 logits 只通过行内 rank、robust z、margin、entropy、top1 support 等稳定特征进入 meta learner。
- 专家和 meta learner 都是 `jt.nn.Module`，checkpoint 推理不依赖 LightGBM/sklearn。
- validation 中候选 0 是正样本时，并列分数不得被当成“正样本排第一”。
- full-data 专家、meta checkpoint 和生产 hydrate 能完整 round-trip。

## RED

新增 `test_tie_neutral_mrr_does_not_reward_positive_candidate_position` 后，远端执行：

```bash
.venv/bin/python -m pytest tests/test_hybrid_oof_stacking.py -q
```

按预期在收集阶段失败：

```text
ImportError: cannot import name 'tie_neutral_mrr'
```

这条 RED 锁定了实际暴露的指标漏洞：旧 MRR 只统计严格高于正样本的候选，正样本固定放在第 0 列时，所有并列都会被错误地记成正样本领先。

## GREEN

最小实现：

- 新增 `tie_neutral_mrr`，精确并列使用平均秩；
- OOF runner 的 meta 扫权重和 external gate 统一使用 tie-neutral MRR；
- `CandidateSetMLP` validation early-stop 同样切换为 tie-neutral MRR；
- 近似 CUDA 数值噪声的 rank/top1 特征使用容差并列和稳定量化。

验证：

```text
6 passed in 1.08s
```

对应命令：

```bash
.venv/bin/python -m pytest tests/test_hybrid_oof_stacking.py -q
```

## 已有回归

最终 OOF/checkpoint 实现完成后，以下定向回归为：

```text
20 passed, 4 warnings in 3.51s
```

覆盖：

- `tests/test_hybrid_oof_stacking.py`
- `tests/test_hybrid_oof_models.py`
- `tests/test_hybrid_candidate_set_transformer.py`
- `tests/test_hybrid_checkpoint.py::test_pure_candidate_set_snapshot_bypasses_legacy_fusion`

最终版本已再次执行同一组测试，确认 tie-neutral early-stop、稳定特征修订和 stable-feature 版本拒绝没有破坏 checkpoint round-trip。

Windows 本地同一命令在 Jittor/MSVC 编译阶段失败，错误发生在测试收集前；最终证据以比赛目标环境 Linux/CUDA 为准。

## Refactor 判断

保留 `stable_expert_logit_features` 作为唯一 stable-feature 入口，并把 tie-neutral MRR 放在同一 OOF 核心模块。训练脚本不再各自复制“严格大于”的乐观指标。没有把指标修复扩散到与本实验无关的历史模型比较代码，避免改变旧实验口径。
