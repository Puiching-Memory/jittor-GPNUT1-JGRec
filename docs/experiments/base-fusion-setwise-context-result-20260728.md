# Result: Base Fusion Setwise Context

## Verdict

工程接入完成，可以进入 rolling-origin / external holdout 候选验证；本次不根据单折
MRR 宣称涨分，也没有扫描线上权重。

## Delivered

- 基础 FusionMLP 默认接收 raw、减 query 行均值、减 query 行最大值三组通道。
- 变换在 batch 内即时计算，监督特征缓存仍保存 raw，不扩大三倍。
- LGBM 输入不变；已有 Setwise、time-ramp 和 conservative-window 头不重复变换。
- 训练、验证、ensemble 权重选择、服务推理、refit 后 normalizer 重算保持同一维度契约。
- 新 checkpoint 以 stored normalizer / 首层权重宽度恢复；旧 checkpoint 和旧
  `TrainingConfig` 缺字段时保持 raw v0。
- CLI 新增 `--fusion-context-transform-version`，默认 `1`，可显式设 `0` 回放旧训练。

## Verification

| Gate | Result |
|---|---:|
| 融合、Setwise、LGBM、CLI（WSL/Linux Jittor） | 52 passed |
| 服务 normalizer + checkpoint（WSL/Linux Jittor） | 18 passed |
| 其余非 temporal-graph 广泛回归 | 487 passed, 1 unrelated missing-file failure |
| Windows 纯逻辑 | 34 passed, 6 environment skips |
| Ruff / compileall / diff-check | passed |

未完成的是 CUDA 远端复核：主机持续无法返回 SSH banner。CPU-only WSL 也不能收集
两个强制 CUDA 的 temporal-graph 测试。这两个环境限制不影响本次基础融合目标路径的
Linux/Jittor GREEN 证据。

## Next gate

用最终集成候选做 rolling-origin 多折与长跨度 external holdout，同时报告 MRR、
Hit@1/3/10、NDCG@10、平均排名及改善/恶化 query 数。锁定候选后再生成提交包，
不使用线上分数反向扫描该开关或融合权重。

