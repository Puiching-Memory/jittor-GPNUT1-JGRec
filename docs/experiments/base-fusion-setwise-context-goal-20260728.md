# Goal Document: Base Fusion Setwise Context

## Go / No-Go

- **Judgment**: Go
- **Reason**: `setwise_context_features()` 已经在独立 Setwise 专家中验证，并且三通道变换可按 query batch 计算；主要风险可由输入维度契约和旧 checkpoint 回放测试约束。

## Target Outcome

新训练的 hybrid 基础 `FusionMLP` 对每个被选中的原始特征同时接收：

1. `value`
2. `value - row_mean`
3. `value - row_max`

LGBM 继续消费原始特征；现有 Setwise 专家不重复变换；旧 checkpoint 在不含新配置字段时保持原输入维度和原预测语义。

## Goal Definition

- **Type**: technical / quality
- **Boundary**:
  - 基础 FusionMLP 的训练、验证、推理、服务归一化与 checkpoint hydrate。
  - hybrid CLI/TrainingConfig 的显式变换版本配置。
  - MLP/LGBM ensemble 中仅 MLP 使用行内相对化，LGBM 保持 raw。
- **Non-goals**:
  - 不改变 GNN、two-tower、structure 等编码塔。
  - 不改变独立 Setwise、time-ramp Setwise、conservative-window 专家的既有输入。
  - 不扫描线上权重，不以本次实现直接宣称 MRR 提升。
- **Deferred work**:
  - rolling-origin 和 external holdout 的正式候选选择与提交包生成。
  - transform v2 的 percentile / robust-z 通道是否进入基础融合。
- **Verification rule**: 新训练基础 MLP 的首层输入维度为原始已选特征数的 3 倍；推理自动应用同一变换；旧结果的均值维度与原始特征相同时不变换。
- **Evidence source**: 单元测试、checkpoint hydrate 测试、Linux/Jittor 全回归。
- **Pass criteria**:
  - raw 两维输入经基础融合准备后得到六维且数值等于 `setwise_context_features()`。
  - 新结果能完成 checkpoint round-trip，并以六维模型预测。
  - 旧结果仍以两维模型预测，数值路径不变。
  - LGBM 收到的仍是原始特征维度。
  - 全回归通过。
- **Confidence note**: 自动测试能证明训练/服务契约一致与兼容性；实际泛化收益仍必须由多折和 external gate 判断。
- **Judgment owner**: 自动化测试负责工程完成判断；离线多折与 external 指标负责候选晋级判断。

## Current State

- `setwise_context_features()` 已实现 v1：raw、row-mean、row-max。
- 基础 FusionMLP 当前只对 raw `feature_indices` 做归一化和训练。
- 独立 Setwise 专家已先变换整套特征再送入 FusionMLP。
- `FusionResult.feature_indices` 同时承担原始特征选择、最终 encoder 裁剪和报告命名，不能改成三倍展开后的索引。
- checkpoint hydrate 当前用 `len(feature_indices)` 重建首层；新实现必须改为按 normalizer/input contract 重建。

## Priority Rationale

- 先固定“feature_indices 仍指原始特征、mean/std 长度代表实际 MLP 输入维度”，可避免破坏 encoder 裁剪和 LGBM。
- 再让训练与推理共用同一个输入准备函数，降低分布不一致风险。
- 最后接服务归一化与 checkpoint hydrate，覆盖训练后部署链路。

## Assumptions and Open Decisions

| Item | Status | Impact | Owner / Next step |
|---|---|---|---|
| 基础融合采用 transform v1 | confirmed | 输入维度 3x，不引入 v2 的额外噪声与成本 | 本目标固定 |
| LGBM 保持 raw | confirmed | 避免改变树模型尺度和已有融合语义 | 测试锁定 |
| 新训练默认启用 v1 | assumed | 无参数训练直接获得相对化特征 | CLI/config 测试锁定 |
| 旧 checkpoint 缺字段时不变换 | confirmed | 当前冠军可继续回放 | hydrate/predict 测试锁定 |

## Phases

### Phase 1: 输入契约 RED

- **Purpose**: 用测试冻结三通道数值、默认开关与旧结果兼容行为。
- **Entry condition**: 当前基础 FusionMLP 只消费 raw。
- **Phase rules**:
  - 只写测试，不改生产实现。
  - RED 必须因缺少基础上下文化行为失败。
- **Todos**:
  - [x] 增加基础融合输入准备测试。
    - **Surface**: fusion tests
    - **Proof**: 期望六维、实际两维。
    - **Depends on**: none
  - [x] 增加新旧 checkpoint 输入维度测试。
    - **Surface**: checkpoint tests
    - **Proof**: 新结果无法按 mean/std 六维重建或预测。
    - **Depends on**: 输入契约测试
- **Exit proof**: 测试以目标行为缺失为唯一失败原因。
- **Stop condition**: 若必须改变 `feature_indices` 的原始语义，停止并重新设计。

### Phase 2: 基础 MLP 上下文化 GREEN

- **Purpose**: 训练、验证和推理使用同一三通道准备逻辑。
- **Entry condition**: Phase 1 RED 已确认。
- **Phase rules**:
  - `FusionConfig` 默认保持 raw，只有 hybrid 基础训练显式传入 v1，避免独立 Setwise 双重变换。
  - 上下文化按 query batch 计算，不物化整份 3x 特征缓存。
  - LGBM 路径不得调用上下文化函数。
- **Todos**:
  - [x] 扩展 FusionMLP 训练与评估输入准备。
    - **Surface**: `fusion.py`
    - **Proof**: 数值、维度与训练 smoke 通过。
    - **Depends on**: Phase 1
  - [x] 将 hybrid 基础配置默认接到 transform v1。
    - **Surface**: `config.py`, `cli.py`, `ranker.py`
    - **Proof**: config wiring 测试通过。
    - **Depends on**: fusion input helper
- **Exit proof**: 新基础 FusionResult 的 mean/std 为 `3 * len(feature_indices)`，训练和预测均通过。
- **Stop condition**: 若实现需要整份 3x 内存物化，停止并改为分批。

### Phase 3: 服务与兼容闭环

- **Purpose**: 消除 checkpoint hydrate、服务归一化和直接脚本调用的训练-服务差异。
- **Entry condition**: Phase 2 GREEN。
- **Phase rules**:
  - 推理根据实际 normalizer 维度自动对齐，旧结果保持 raw。
  - 已经是三通道的独立 Setwise 输入不得再次变换。
- **Todos**:
  - [x] 按 `len(result.mean)` 重建 FusionMLP。
    - **Surface**: checkpoint hydrate
    - **Proof**: 新旧 checkpoint 测试通过。
    - **Depends on**: Phase 2
  - [x] 服务归一化使用与推理相同的输入对齐。
    - **Surface**: service normalizer
    - **Proof**: normalizer feature_dim 与 result.mean 一致。
    - **Depends on**: Phase 2
  - [x] 更新操作文档并运行全回归。
    - **Surface**: docs/tests
    - **Proof**: Linux/Jittor 全回归通过。
    - **Depends on**: 前两项
- **Exit proof**: 训练、checkpoint、服务、旧冠军四条路径均通过。
- **Stop condition**: 旧 checkpoint 预测出现维度或排序变化。

## Dry-Run Findings

- 不能直接把 `feature_indices` 展开成三倍索引，否则最终 encoder 特征裁剪、报告命名和 LGBM 都会被破坏。
- 可保持 `feature_indices` 为 raw 选择，并让 `mean/std` 长度成为实际 MLP 输入维度；推理通过维度比值自动识别是否需要 v1。
- 独立 Setwise 专家已经传入三通道特征，其输入维度等于 normalizer，因此自动对齐不会重复变换。
- normalizer 必须分批计算，否则 dataset2 全量监督特征的 3x 临时数组会造成不可接受的峰值内存。

## Final Validation

- `uv run --no-sync pytest` 的本地纯逻辑测试。
- 远端 Linux/CUDA：`uv run --no-sync python -m pytest -q --no-cov`。
- Ruff、compileall、`git diff --check`。
- 旧冠军 checkpoint dataset1 小批回放 smoke。

## First Execution Step

先增加基础融合输入准备、默认配置和 checkpoint 输入维度的失败测试，确认 RED 后再修改生产代码。
