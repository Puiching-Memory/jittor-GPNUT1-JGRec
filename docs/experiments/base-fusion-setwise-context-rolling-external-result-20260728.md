# Result: Base Fusion Context Rolling / External Gate

## Verdict

`LOCAL_COMPLETE_REMOTE_PENDING`。

候选尚未被接受或拒绝。三折真实 rolling 尚未产出，因此 selection lock 不存在；
external 未打开，提交包未生成。

## Completed

- 冻结 Dataset1 最终集成对照：
  `base MLP v0/v1 -> shared LGBM -> shared Setwise γ=0.5`。
- rolling 使用三个 selection folds；每折 100k 窗口内保留最后 20k 做因果
  tune，基础 MLP/LGBM fit 区遵守 source checkpoint 的
  `max_train_events`，共享 Setwise 使用前 80k；随后的 25k 才用于评分。
- v0/v1 配置除 context version 外任一差异都会失败。
- 多指标稳定门禁复用 robust selection 状态机。
- external runner 在读取 holdout 字节前写独占 receipt，且只接受 selection lock。
- 候选 head 使用严格 NPZ schema，绑定 context version、输入宽度、normalizer、
  state、fit/tune 边界和 validation 指标。
- checkpoint builder 只有在 external 接受且 report/head/source/lock 哈希一致时
  才运行；只允许替换 Dataset1 三个 fusion 键。
- 两次标准 hydrate/replay 必须字节一致且无精确并列，才能与线上 tie-safe
  Dataset2 字节组合。
- 本地真实 metadata preflight 已通过：
  `result/dataset1_base_context_local_preflight_20260728/preflight-report.json`。
- 15 个核心 gate/schema 测试通过；相关 fusion/checkpoint/submission 回归
  共 `84 passed`；Ruff 和六个 runner 的 `--help` smoke 通过。

## Blocker

真实 feature cache 和当前约 5GB checkpoint 仅存在 CUDA 远端。连续 SSH 重试均在
认证前失败：

```text
paramiko.ssh_exception.SSHException: Error reading SSH protocol banner
```

本地没有
`dataset1_joint_recent200k_full100_seed60_20260726.train.npy`，不能执行等价三折；
用旧 checkpoint、缩小样本或 standalone MLP 分数代替会违反已冻结协议。

## Package decision

`package_authorized=false`。

这不是候选被 gate 拒绝，而是 gate 尚未执行。远端恢复后应从 rolling runner
继续；只有 selection lock 存在才执行 external，只有 external
`status=accepted` 才生成提交包。
