# TDD Evidence: Base Fusion Setwise Context

## Target behavior

- 新训练的 hybrid 基础 `FusionMLP` 对每个 raw 特征使用
  `value`、`value-row_mean`、`value-row_max` 三组通道。
- mask 和 `FusionResult.feature_indices` 仍描述 raw 特征；LGBM 仍使用 raw。
- 训练、验证、推理和服务归一化使用同一输入契约。
- 已经上下文化的 Setwise 专家不重复变换。
- 旧 checkpoint 缺少新配置字段时保持 raw v0；新 checkpoint 按首层状态宽度恢复。

## RED

1. `test_safe_hybrid_defaults_select_query_metric_and_ensemble`
   首先因 `CLIConfig` 没有 `fusion_context_transform_version` 而失败，证明默认接线尚不存在。
2. 新增基础融合三通道、自动推理对齐和服务归一化测试；旧实现只能产生
   `k` 维 normalizer，无法满足 `3*k` 输入契约。
3. `test_build_fusion_from_state_uses_stored_context_input_width`
   在实现外围兼容前明确失败：
   `linear1.weight` 期望 `[8,2]`、checkpoint 实际为 `[8,6]`。

这些失败均由目标行为缺失引起，不是夹具、随机性或环境错误。

## GREEN

- `FusionConfig.context_transform_version` 保持默认 v0，避免独立 Setwise 脚本二次变换。
- `TrainingConfig` / CLI 对新训练显式默认 v1。
- 四条 FusionMLP 训练路径、normalizer、验证指标和推理共用批级输入准备。
- `predict_logits()` 根据 stored normalizer 宽度将 raw 输入自动对齐到 v1/v2。
- ranker hydrate 使用 `len(result.mean)`；通用 `build_fusion_from_state()` 使用
  checkpoint 的 `linear1.weight` 宽度，覆盖离线评估脚本。
- 旧 pickle 兼容判断读取实例字典而不是 dataclass 类默认值，缺字段严格解析为 v0。

## Refactor decision

没有展开 `feature_indices`，也没有物化全量三倍缓存。这样保留 encoder 裁剪、
特征 mask、报告命名和 LGBM 语义，并把额外内存限制在当前 query batch。

## Verification evidence

- Windows 纯逻辑：`34 passed, 6 skipped`。
- WSL/Linux Jittor 融合组最终状态：`52 passed`。
- WSL/Linux 服务归一化与 checkpoint：`18 passed`。
- WSL/Linux 其余广泛回归：`487 passed`；唯一失败是工作区缺少
  `数学建模作业/实验代码_图论模型动态推荐预测.py`。
- CPU-only WSL 的全量收集另有两个既有阻断：
  `JittorGeometric/spmmcsr.py` 导入时强制 `jt.flags.use_cuda=1`。
- Ruff、compileall、`git diff --check` 均通过。

CUDA 远端在验证期间持续于 SSH protocol banner 阶段断开，因此未把远端结果伪装成已通过。

