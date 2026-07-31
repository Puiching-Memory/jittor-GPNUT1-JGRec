# TDD Evidence: A3 全量 refit 后服务口径归一化

## Target Behavior

在 hybrid 完成 final encoder 全量 refit 后，允许用无标签服务
query/candidate 特征流式重算神经融合头的 `mean/std`。校准必须：

- 与直接 NumPy population moments 一致，且不依赖 batch 划分或顺序；
- 只替换 `mean/std`，不改变模型 state、特征索引或其他融合元数据；
- 同时支持基础 Fusion、主 Setwise、time-ramp Setwise 和 conservative-window
  神经头，LightGBM 保持不变；
- 在 `fit()` 中位于 final encoder refit/compact 之后；
- rolling 硬门拒绝后默认关闭，只能显式 opt-in。

## RED

### RED 1：流式统计模块缺失

命令：

```text
uv run --no-sync pytest tests/test_service_normalizer.py -q
```

预期失败：

```text
ModuleNotFoundError: No module named 'jgrec.service_normalizer'
```

该失败证明测试命中了缺失能力，而不是环境或断言错误。

### RED 2：不可变结果替换入口缺失

加入模型契约测试后，同一命令按预期失败：

```text
ImportError: cannot import name 'replace_result_normalizer'
```

测试要求新结果保留原 `state` 对象、`feature_indices` 和额外元数据，只复制新的
`mean/std`。

### RED 3：ranker 公开校准入口缺失

远端 Linux/Jittor 命令：

```text
source .workspace-env.sh
uv run --no-sync pytest tests/test_hybrid_service_normalizer.py -q
```

预期失败：

```text
AttributeError: 'TemporalHybridRanker' object has no attribute
'recalibrate_service_normalizers'
```

### RED 4：`fit()` 未接入 post-refit 校准

为 final encoder 顺序写集成测试后，首先按预期失败：

```text
TypeError: TrainingConfig.__init__() got an unexpected keyword argument
'service_normalizer_calibration_enabled'
```

测试使用 fake final encoder，要求校准后的均值来自最终 encoder 的服务特征，并要求
`TrainingReport.metrics` 记录头数量、候选行数和最大标准化均值漂移。

### RED 5：rolling 拒绝后仍默认启用

三折门禁拒绝后增加 stop-rule 测试，按预期失败：

```text
AssertionError: assert True is False
```

目标是防止已被跨折证伪的策略静默改变后续默认训练。

## GREEN

最小实现包括：

- `StreamingFeatureNormalizer`：float64 batch moment 合并，输出 float32
  population `mean/std`，退化列 `std=1`；
- `normalizer_drift_report()`：报告均值绝对漂移、按训练标准差归一化的漂移和
  service/train 标准差比；
- `replace_result_normalizer()`：以 dataclass replace 只更新统计量；
- `TemporalHybridRanker.recalibrate_service_normalizers()`：分批编码服务候选，
  分别累计 raw/Setwise 头统计量；
- `TemporalHybridRanker.fit()`：显式启用时在 final encoder
  refit/compact 之后校准，并把漂移摘要写入训练报告；
- `TrainingConfig.service_normalizer_calibration_enabled=False`：rolling No-Go 后
  保持 opt-in。

GREEN 证据：

```text
# 本地纯 NumPy/配置行为
6 passed, 3 skipped

# 远端 Linux/Jittor 集成行为
9 passed
```

## REFACTOR

- 数学核心放在 `src/jgrec/service_normalizer.py`，不导入 Jittor。
- ranker 只负责选择各头特征、流式调用和回写结果，统计合并逻辑不重复。
- 所有 Setwise 变体共用同一校准路径；candidate-set transformer、OOF stacking 和
  LightGBM 不被静默改写。
- 校准使用完整候选矩阵且不读取正样本列、标签或指标；对真实 test 必须先完成统计遍，
  再开始固定 normalizer 的预测遍，避免顺序依赖。

## Verification

已覆盖：

- 分批统计与直接 flatten 统计一致；
- batch 逆序不改变结果；
- 空输入、非有限值、维度变化和不匹配结果被拒绝；
- constant feature 使用安全标准差；
- raw Fusion 与 Setwise 同时校准；
- state 对象、state SHA-256、feature indices 不变；
- `fit()` 校准顺序位于 final encoder 之后；
- rolling No-Go 后默认关闭。

验证命令：

```text
uv run --no-sync ruff check \
  src/jgrec/service_normalizer.py \
  src/jgrec/rankers/hybrid/config.py \
  src/jgrec/rankers/hybrid/ranker.py \
  scripts/evaluate_dataset2_service_normalizer_rolling.py \
  tests/test_service_normalizer.py \
  tests/test_hybrid_service_normalizer.py

uv run --no-sync pytest \
  tests/test_service_normalizer.py \
  tests/test_hybrid_service_normalizer.py -q
```

## Residual Risk

三折 proxy 复用了冻结 200k cache，不能逐折重演 encoder full refit；它能验证
时间/候选分布适配是否稳定，但不能证明每个 tower 的逐折 refit 漂移。由于该固定方法
已经在 proxy 硬门失败，按预先锁定的 stop rule 没有打开 official 20k。

