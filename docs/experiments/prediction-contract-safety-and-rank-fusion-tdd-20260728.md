# Prediction Contract Safety and Rank Fusion — TDD Evidence

## Target Behavior

- 提交 CSV 每个 query 的 100 个分数在 float64 round-trip 后严格唯一。
- 原本非并列候选的相对顺序不变；并列内部按 candidate prior、candidate id、原列号稳定排序。
- CLI / hybrid / fusion 默认使用 MRR，hybrid 默认 ensemble + RRF。
- `hybrid --no-refit-full` 不调用最终全量 encoder refit。
- probability、temperature、RRF 三种专家融合可选；旧 checkpoint 默认解释为 probability。

## RED

| Behavior | Failing evidence |
|---|---|
| CSV exact-tie elimination | runner fixture 写回后 100 列只有 2 个唯一值 |
| Safe defaults | `CLIConfig.selection_metric == "ap"`、`fusion_mode == "mlp"` |
| Hybrid no-refit wiring | `TrainingConfig` 不存在 `refit_full` |
| RRF / temperature | 六个行为测试均以 `NotImplementedError` 失败 |
| Linux 0-boundary | flush-to-zero 环境中 `nextafter(0)` 塌回 `-0.0`，唯一值断言失败 |
| Float-neighbor exhaustion | 真实 checkpoint 重放触发 `FloatingPointError`：并列组上下没有可插入 float64 |

## GREEN

| Behavior | Minimal implementation |
|---|---|
| Submission contract | runner clip 后调用 `break_prediction_ties()`，以 `%.17g` 保存 |
| Candidate prior | hybrid 暴露 test-candidate frequency 作为 tie-break priority |
| Boundary safety | 0 边界使用最小正规 float64 间隔，避免 FTZ |
| Exhausted interval | 仅对无法插值的行使用严格保序的等距秩分数 |
| Defaults | CLI、TrainingConfig、FusionConfig 改为 MRR；hybrid 改为 ensemble + RRF |
| No-refit | 非缓存路径保留 val encoder；缓存路径恢复 train-end encoder；跳过 final full refit |
| Expert fusion | probability / temperature / RRF 统一接口；温度按 validation positive-column NLL 标定 |
| Compatibility | `LGBMFusionResult` 新增持久化字段，字段缺失时使用 legacy probability |

## Refactor Decision

- 将异构专家融合独立到纯 NumPy `expert_fusion.py`，避免把尺度处理继续堆进 `ranker.py`。
- 将输出并列处理放在 runner 边界，覆盖 hybrid、temporal-graph、checkpoint replay 和 scheduled prediction。
- 保留 `segment_fusion.blend_expert_probabilities()`，避免破坏历史 segment-gate checkpoint。
- 没有把 external/线上结果纳入温度或权重搜索。

## Verification

- Windows 纯 NumPy/CLI 定向测试：51 passed。
- Linux/Jittor 定向集成：61 passed；后续 no-refit / LGBM 定向：9 passed。
- Linux/Jittor 全回归：491 passed。
- Ruff：所有本次涉及的 Python 文件通过。
- `compileall`：`src` 与 `tests` 通过。
- 真实旧提交诊断：
  - dataset1：61,051 行，41,645 行含精确并列；
  - dataset2：153,420 行，153,420 行含精确并列。
- 冠军 checkpoint tie-safe 重放：
  - dataset1：61,051 行，0 行含精确并列；
  - dataset2：153,420 行，0 行含精确并列；
  - ZIP CRC、成员清单和本地/远端 SHA-256 一致性均通过。
