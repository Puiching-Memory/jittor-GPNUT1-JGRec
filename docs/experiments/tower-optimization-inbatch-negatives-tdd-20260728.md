# Hai TDD：各塔优化器与双塔 In-Batch Negatives

## Target Behavior

- GNN、GRU sequence、Two-Tower、SourceProfile item2vec 可分别配置学习率、
  `constant/cosine` 调度、最低学习率比例和 weight decay。
- 所有新配置默认保持关闭：旧 checkpoint 缺字段时仍解析为固定学习率、
  weight decay 0、无 in-batch auxiliary loss。
- Two-Tower 可选用 multi-positive in-batch softmax；同一 batch 内重复的
  destination 被视为多个正例，不能互相充当负例。
- 冻结的 2×2 实验只比较 control、optimizer-only、inbatch-only、combined，
  并报告 MRR、Hit@1/3/10、NDCG@10、平均排名和 query 改善/恶化数。

## RED

### 学习率与旧配置兼容

- **Tests added**：`tests/test_tower_optimization.py`
- **Behavior asserted**：
  - cosine 首尾学习率和单 epoch 边界；
  - 非法 schedule、epoch、最低比例被拒绝；
  - 四塔独立配置映射；
  - 旧 `TrainingConfig` 缺新字段时保持历史默认。
- **Command**：
  `C:\Users\75556\anaconda3\python.exe -m pytest tests/test_tower_optimization.py -q`
- **Observed failure**：
  `ModuleNotFoundError: jgrec.rankers.common.optimization`，随后配置测试因
  `gnn_lr_schedule` 等字段不存在而失败。
- **Why this was the right failure**：共享调度契约和塔级配置在实现前确实不存在。

### Two-Tower multi-positive loss

- **Tests added**：
  - 重复 destination 的正例 mask；
  - Jittor loss 与 NumPy multi-positive softmax 参考实现一致；
  - CLI/config 能接通开关、权重和温度。
- **Local RED**：缺少 `in_batch_negatives` 模块和配置字段。
- **Remote CUDA RED**：
  `ImportError: cannot import name '_multi_positive_in_batch_loss'`。
- **Why this was the right failure**：旧 Two-Tower 只有显式候选组 BCE/listwise
  loss，没有 batch 内检索约束。

### 冻结筛选协议

- **Tests added**：`tests/test_tower_optimization_experiment.py`
- **Behavior asserted**：
  - 四臂只改变冻结的两个因子；
  - 完整 ranking 指标计算；
  - paired query movement；
  - 任一完整指标、时间片 MRR/NDCG 或 movement 门禁回归即拒绝。
- **Observed failure**：
  `ModuleNotFoundError:
  jgrec.rankers.hybrid.tower_optimization_experiment`，随后 gate 测试因缺少
  `two_tower_screen_gate` 导入失败。
- **Why this was the right failure**：实验选择协议尚未实现，不能依赖 runner 内
  临时计算或只看 MRR。

## GREEN

- **Minimal implementation**：
  - 新增共享 epoch LR 计算与 optimizer 更新 helper；
  - 四塔训练循环逐 epoch 应用并记录 LR；
  - `TrainingConfig` 和 CLI 增加四塔独立 optimizer 字段；
  - Two-Tower 训练阶段叠加可选 multi-positive in-batch loss，验证与 early
    stopping 仍使用原始完整候选组；
  - destination 侧 auxiliary representation 使用 destination ID 加统一中性
    context bucket，避免借用另一事件的时间上下文；
  - 新增冻结四臂、多指标和 query movement gate。
- **Local command**：
  `C:\Users\75556\anaconda3\python.exe -m pytest
  tests/test_tower_optimization.py
  tests/test_tower_optimization_experiment.py
  tests/test_hybrid_two_tower.py
  tests/test_hybrid_sequence_gru.py
  tests/test_hybrid_source_profile.py
  tests/test_hybrid_checkpoint.py tests/test_cli.py -q`
- **Local result**：`70 passed, 16 skipped`；跳过项均为本机未安装 Jittor 的
  CUDA 测试。
- **Remote CUDA command**：
  `uv run --no-sync pytest tests/test_tower_optimization.py
  <four tower focused tests> tests/test_cli.py::<focused test> -q`
- **Remote CUDA result**：`22 passed`；额外开启 cosine、weight decay 和
  in-batch negatives 的真实 Two-Tower 小数据训练 smoke 为 `1 passed`。
- **Preflight result**：
  1k events / 1 epoch 全链路预检完成；历史 control 的 full MRR
  `0.46412480606397805` 与三个时间片 MRR 全部精确复现。

## REFACTOR

- **Refactor done**：yes。
- **Change**：
  - 把 scheduler 数学与 Jittor optimizer 更新隔离到 common helper；
  - 把无 Jittor 依赖的 multi-positive mask、ranking metrics 和 gate 放入纯
    NumPy 模块，便于本地快速回归；
  - 保留所有新功能的关闭默认值，不改变冠军 checkpoint 回放路径；
  - 修正 paired-movement fixture 中一处测试数据算术错误，该错误发生在生产
    实现之后，不属于 GREEN 行为缺陷。
- **Ruff command**：覆盖本任务所有生产、测试和实验 runner 文件。
- **Observed result**：`All checks passed`。

## Next Behavior

2×2 筛选已经完成。历史 control 回放后又补做了 same-code/same-seed matched
control；权威重算没有候选通过全部硬门禁，因此停止在 Stage 1，不运行最终集成
rolling-origin、不打开 external、不生成提交包。完整结论见
`tower-optimization-inbatch-negatives-result-20260728.md`。

## Environment Note

本地按项目要求首先尝试 `uv run`，但 Windows 工作区的 `.venv/lib64` reparse
point 导致 uv 在测试启动前失败。该失败与代码无关；本地纯 Python 测试改用同为
Python 3.12 的 Anaconda 解释器，真实 Jittor/CUDA 验证全部在服务器通过
`uv run --no-sync` 执行。
