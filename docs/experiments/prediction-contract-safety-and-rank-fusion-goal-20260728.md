# Prediction Contract Safety and Rank Fusion Goal — 2026-07-28

## 1. Goal

完成 B 组工程防呆，使新训练与新提交默认满足以下可验证契约：

1. 提交文件中的同一 query 不再包含精确相同的预测分数；只打破原本相同的分数，优先按 `candidate_prior`、再按 candidate id 确定顺序。
2. 默认模型选择指标改为 query-level MRR，默认融合模型改为 ensemble，避免无参数运行时退回全局 AP / 单 MLP。
3. `hybrid --no-refit-full` 真正跳过全量编码器 refit，并复用验证口径编码器。
4. 异构 MLP 与 LGBM 专家支持 probability、temperature、RRF 三种融合；新训练默认 RRF，旧 checkpoint 保持 probability 语义。

## 2. Boundary

### In scope

- `src/jgrec/core/runner.py` 的提交输出契约和 round-trip 精度。
- hybrid candidate-prior tie-break signal。
- CLI、`TrainingConfig`、`FusionConfig` 的安全默认值。
- hybrid 的 `refit_full` 接线、训练行为和报告。
- 异构专家 RRF、温度标定、权重搜索和 checkpoint 持久化。
- 纯 NumPy 单元测试、CLI/config 回归测试，以及可运行环境中的 hybrid 集成测试。

### Out of scope

- 不用本次改动反扫 leaderboard 权重。
- 不修改已有 checkpoint 的模型参数或把新融合默认强加给旧 checkpoint。
- 不重训冠军模型，不改变非并列候选的排序。
- 不把 AP 指标删除；只把默认选择改为 MRR，显式配置仍可用 AP。

## 3. Current State

- CSV 固定以 8 位小数输出，模型内微小差异可能再次被序列化为精确并列。
- `selection_metric="ap"`、`fusion_mode="mlp"` 是 CLI 和 hybrid 配置默认值。
- CLI 仅将 `refit_full` 传给 temporal-graph；hybrid 配置无该字段，`fit()` 无条件全量 refit。
- MLP softmax 概率与 LGBM LambdaRank 分数最终只走概率线性混合，没有尺度无关的秩融合，也没有温度标定。

## 4. Target State

- 输出边界执行确定性 total-order tie-break，并使用可 round-trip 保存 float64 微扰的格式。
- CLI、训练配置和底层融合配置默认分别为 `mrr`、`ensemble`。
- hybrid 在 `refit_full=False` 时保留验证口径 encoder，日志/报告明确记录未全量 refit。
- `expert_blend_mode` 支持：
  - `probability`：兼容旧行为；
  - `temperature`：分别以验证 NLL 标定两专家温度后混合；
  - `rrf`：按候选秩做 reciprocal-rank fusion，不受专家分数量纲影响。
- 新 checkpoint 保存 blend mode、温度、RRF k 和 MLP 权重；旧 checkpoint 缺字段时解析为 probability。

## 5. Phased Route

### Phase 1 — Submission contract

- 先写 exact-tie、边界 0/1、序列化 round-trip 的失败测试。
- 实现确定性 tie-break，接入 runner，并暴露 hybrid candidate-prior。

### Phase 2 — Safe defaults and no-refit

- 先写 CLI/config 默认值和 hybrid refit 调用次数测试。
- 改默认值，补齐 hybrid 接线；`no-refit` 复用验证 encoder，缓存命中时最多恢复验证口径 encoder，不做全量训练。

### Phase 3 — Heterogeneous expert fusion

- 先写 RRF 尺度不变、温度 NLL 改善、旧 checkpoint 兼容测试。
- 实现三种融合并接入训练期权重选择与服务期预测。

### Phase 4 — Verification

- 跑定向测试、CLI smoke、完整可运行测试集。
- 对真实提交分数做并列诊断；若素材完整，生成一个只做防并列处理的验证包，不用其线上结果反调参数。

## 6. Phase Rules

- RED 测试必须因缺失目标行为失败，不能因导入或 fixture 错误失败。
- tie-break 只允许改变精确并列组内部顺序；非并列 pair 的相对次序必须保持。
- 温度只从训练内验证数据标定；external/线上结果不进入搜索。
- 旧 checkpoint 缺少新字段时严格走 legacy probability。
- `--no-refit-full` 不得静默退化为 full refit。

## 7. Todos with Proof

- [x] Tie-break 单元测试从 RED 到 GREEN；CSV 读回后每行唯一。
- [x] 默认值测试证明 CLI、TrainingConfig、FusionConfig 一致。
- [x] hybrid no-refit 测试证明全量 encoder fit 次数为 0，并能预测。
- [x] RRF 测试证明单调重标度前后输出一致。
- [x] temperature 测试证明合成过置信 logits 的 NLL 不升。
- [x] checkpoint 测试证明新字段 round-trip、旧字段 fallback。
- [x] 定向与完整回归通过；任何环境限制单独记录。

## 8. Dry Run

1. 用户无参数启动 hybrid：选择 MRR、训练 MLP+LGBM，专家以 RRF 融合。
2. 权重只在该精确集成候选的训练内验证集搜索并锁定。
3. `--no-refit-full` 时，fusion 训练结束后留下验证口径 encoder；`fit()` 不再用全量交互重训。
4. 推理得到概率后，runner 找出精确并列组，用 candidate prior 和 id 建立稳定次序，以 float64 round-trip 文本写出。
5. 加载历史 checkpoint 时无新融合元数据，解析为 legacy probability，因此历史预测不变（除提交边界的并列消除）。

## 9. Go / No-Go

**Go**。三个改动均有清晰边界和可自动验证的不变量；主要风险是旧 checkpoint 兼容与输出排序漂移，已分别用 legacy fallback 和 pairwise-order 测试约束。

## 10. Plan Rewrite Notes

- 原 B 清单把“秩融合/温度标定”写成一个方向；执行计划拆成两个可独立选择的 mode，并保留 probability 兼容模式。
- “epsilon”不直接固定为任意十进制常数：改用 float64 相邻值微扰并提高输出精度，避免越过临近非并列分数或被 8 位小数重新压平。
- `--no-refit-full` 不仅补 CLI 字段，还要求训练路径复用验证 encoder，否则只把“全量 refit”替换成“前缀 refit”并不能兑现迭代加速。

## 11. Drift Diagnosis

- 当前实现相对 B 目标的最大漂移不是缺少一个开关，而是三处默认/边界分散在 CLI、训练配置、runner 和 checkpoint 运行时。
- 因此验收以端到端行为为准，不以“字段已添加”或“函数已实现”为完成。
