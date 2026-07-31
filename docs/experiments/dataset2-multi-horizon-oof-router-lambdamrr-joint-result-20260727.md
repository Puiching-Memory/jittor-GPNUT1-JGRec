# Result: Dataset2 多时间尺度 OOF 路由 + top-k LambdaMRR 联训

## Verdict

**实现完成，候选 No-Go。** 纯 Jittor 联合模型、严格时间 selection/gate、
checkpoint replay 和 bounded top-k 审计均已完成。selection 正增益，但最终不可见
gate 为 `-0.0000061120`，因此不全量重训、不生成提交、不替换当前冠军。

## Joint Model

- 共享行级 MLP 主干。
- route head 输出 medium/long 两路 OOF advantage。
- candidate head 为每个候选输出 medium/long 两路 LambdaMRR residual。
- 训练 loss：
  - 稀疏 reward 加权 route MSE；
  - positive 对默认 short top-k 难负例的静态 `ΔMRR` 加权 pairwise softplus；
  - 两个 loss 同时更新共享主干。
- 推理默认保持 short；只在高置信路由行启用 route-specific residual。
- 时间尺度差与 LambdaMRR residual 作为整体重新投影，保证 top-k 外逐值不变、
  行内改分和为零且最大绝对改分不超过 cap。
- 所有可训练模块仅使用 Jittor；无 LightGBM/sklearn 训练依赖。

## Frozen Protocol

- Common OOF rows: `[159804, 200000)`，共 `40,196` 行。
- Train: `23,334` 行。
- Selection: `8,719` 行。
- Final gate: `8,143` 行。
- Search:
  - top-k: `10 / 20`
  - cap: `0.01 / 0.02`
  - LambdaMRR loss weight: `0.03 / 0.10 / 0.30`
  - route fraction: `0.25% / 0.5% / 1% / 2% / 3% / 5%`
- Model: hidden `128`，dropout `0.05`，12 epochs。
- Selection 必须整体正增益且最差时间子片不低于 `-0.00005`。
- selection lock 写盘后才允许计算 gate 标签和指标。

## Selection

锁定配置：

- `topk-20-cap-0.01-lambda-0.10`
- route fraction: `4.989104%`，435/8,719 行
- default short MRR: `0.4188148523`
- candidate MRR: `0.4189538166`
- delta: **`+0.0001389643`**
- time-slice deltas:
  - `+0.0000018048`
  - `+0.0003998420`
  - `+0.0000152888`
- gain/loss/unchanged rows: `10 / 4 / 8,705`

12 个联合变体中，top20/cap0.01 的三个 Lambda 权重均为正且三个 selection
子片非负；top10/cap0.02 更不稳定。

## Final Gate

- default short MRR: `0.4171508932`
- candidate MRR: `0.4171447812`
- delta: **`-0.0000061120`**
- route fraction: `4.998158%`，407/8,143 行
- medium/long rows: `349 / 58`
- gain/loss/unchanged rows: `9 / 7 / 8,127`
- time-slice deltas:
  - `+0.0000495409`
  - `-0.0000391987`
  - `-0.0000286699`
- worst time-slice delta: `-0.0000391987`
- oracle delta at最多 5% coverage: `+0.0017491889`

gate 的最差子片仍在稳定性容差内，但整体 delta 小于零，所以按冻结规则拒绝。

## Safety and Reproduction

- top-k 外逐值保持：passed
- 未路由行逐值保持：passed
- routed 行与候选 alternative 对齐：passed
- 行内 residual 居中：passed
- cap `0.01`：passed；实际最大绝对改分 `0.01000005`
- selected checkpoint replay error: `0.0`
- selected checkpoint SHA-256:
  `2a233f2e4d0451d9181ac529006f1d2f051b79478faf071d205bc6b8524f31be`
- 相关测试：16 passed
- ruff / py_compile：passed

## Interpretation

- LambdaMRR 把 selection 信号从原高置信路由器的微弱正值放大到约
  `+0.000139`，但增益主要集中在 selection 中间子片，没有稳定迁移到 gate。
- gate 中 407 条路由只有 16 条真正改变 MRR，说明 5% coverage 仍过宽；大量
  “高置信”样本实际上不改变正样本名次。
- gate oracle 仍有 `+0.001749`，空间没有消失；主要瓶颈仍是纠错行识别和置信度
  校准，而不是 LambdaMRR residual 的安全幅度。
- 不能用已读取的 gate 回头选择更小 coverage，否则会污染时间门禁。后续若继续，
  应在更早 rolling-origin 折内校准“是否改变正样本 rank”的概率，再冻结策略。

## Artifacts

- Result directory:
  `result/dataset2_joint_oof_lambdamrr_20260727`
- Evaluation:
  `result/dataset2_joint_oof_lambdamrr_20260727/evaluation-report.json`
- Selection lock:
  `result/dataset2_joint_oof_lambdamrr_20260727/selection-lock.json`
- Selected checkpoint:
  `result/dataset2_joint_oof_lambdamrr_20260727/variants/topk-20-cap-0.01-lambda-0.10/model.npz`
- Remote artifact archive:
  `result/dataset2_joint_oof_lambdamrr_20260727.remote.tar.gz`
- Training log:
  `result/dataset2_joint_oof_lambdamrr_20260727.training.log`

## Decision

保留联训模块与 OOF 制品作为研究组件；本轮不执行条件 Phase 4，不生成测试候选或
提交包，当前线上冠军保持不变。
