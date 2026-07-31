# Hai TDD: Dataset2 Setwise 行内相对特征 v2

## Target Behavior

Setwise 默认继续生成 v1 的 `raw / raw-row_mean / raw-row_max`；显式选择 v2 时，再动态生成不依赖候选位置的行内 percentile midrank 和 median/MAD robust z-score。前缀候选必须同时不降低 slice0、slice1 才有资格被选择。

## RED

- **Test added**:
  - `tests/test_hybrid_setwise.py`
  - `tests/test_hybrid_temporal_robust_selection.py`
- **Behavior asserted**:
  - v1 默认值与显式 version1 逐值一致；
  - v2 从 3 倍扩展为 5 倍通道；
  - 并列值得到相同 percentile，候选重排只重排输出，不改变其值；
  - robust z 使用 `(x-median)/(1.4826*MAD)`，MAD=0 输出 0；
  - 选择器拒绝“合并前缀 MRR 更高但 slice1 下降”的候选；
  - 平分时依次选择更少模型和冻结顺序。
- **Command**:
  - `uv run --no-sync pytest tests/test_hybrid_setwise.py -q`
  - `uv run --no-sync pytest tests/test_hybrid_temporal_robust_selection.py -q`
- **Observed failure**:
  - 5 个 transform 测试均因 `transform_version` 参数不存在而失败；
  - 选择器测试收集时因 `select_temporally_robust_candidate_on_prefix` 不存在而失败。
- **Failure is correct because**: v2 transform API、两片约束选择器和对应行为均尚未实现；失败不是语法、fixture 或环境错误。

## GREEN

- **Minimal implementation**:
  - `setwise_context_features` 和 `SetwiseFeatureView` 增加默认 v1 的显式版本；
  - 在 sorted candidate 轴识别 tie group 的起止位置，回填 ascending midrank percentile；
  - 动态计算 median/MAD robust z，零 MAD 通道写 0；
  - 新增只读取 `[0, selection_stop)` 的两片稳健选择器。
- **Command**:
  - `uv run --no-sync pytest tests/test_hybrid_temporal_robust_selection.py tests/test_hybrid_setwise.py tests/test_hybrid_window_diversity.py -q`
  - Linux: `.deps/uv/bin/uv run pytest tests/test_hybrid_setwise.py tests/test_hybrid_temporal_robust_selection.py tests/test_hybrid_fusion_listwise.py tests/test_hybrid_checkpoint.py -q`
- **Observed pass**:
  - 本地纯逻辑：`12 passed`
  - Linux/Jittor + checkpoint 回归：`28 passed`

## REFACTOR

- **Refactor done**: yes
- **Change**: 把 transform 倍数、percentile 和 robust z 分离为私有纯函数；把两片稳健选择抽成可复用、可用 NaN forward 行验证隔离的纯选择器。
- **Command after refactor**:
  `uv run --no-sync ruff check scripts/run_dataset2_setwise_relative_context_v2.py src/jgrec/rankers/hybrid/setwise.py src/jgrec/rankers/hybrid/window_diversity.py tests/test_hybrid_setwise.py tests/test_hybrid_temporal_robust_selection.py`
- **Observed result**: `All checks passed!`

## Next Behavior

已完成 200k full-100 context v2 seed60 训练、前两片锁定和独立第三片门禁。v2 两片均退化，v1/v2 uniform 因 slice0 退化而不合格，锁定候选回退为 `v1_champion`；full 增益为 0，门禁拒绝，未生成包。后续若继续探索，应建立新的冻结实验，不在本轮结果上调 rank、robust scale 或融合权重。
