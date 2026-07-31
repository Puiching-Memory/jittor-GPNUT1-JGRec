# Result: Dataset2 多专家 top1 对齐路由 v3

## Verdict

**No-Go。** “各专家 top1 对应的原始特征/多兴趣支持度差”已经实现并完成前向选型，但没有带来可泛化的 slice1 增益。按冻结协议不读取 slice2、不生成路由模型、不打包，线上冠军 `1.3521011401636023` 保持不变。

## What Changed

- 保留 v2 的 74 维 score-only descriptor。
- 对 `v1_champion`、`multi_interest`、`window_ensemble` 分别增加其 top1 相对 `current_gate` top1 的：
  - 12 维候选原始特征均值差；
  - 9 维 multi-interest proxy 支持度均值差。
- top1 并列按集合处理，集合内求均值；candidate 同步换列不改变结果。
- descriptor 从 74 维增加到 137 维。

## Slice1 Result

当前门控 slice1 baseline MRR：`0.5510080326704802`

| Scope | Config | Delta | Coverage | Judgment |
|---|---|---:|---:|---|
| 全网格最高 delta | depth=3, leaf=250, threshold=0.0025 | `+0.0001159371` | `56.9372%` | 覆盖率超限，且增益远低于门槛 |
| coverage≤25% 的最佳非零路由 | depth=3, leaf=250, threshold=0.005 | `-0.0003988807` | `11.8644%` | 下降 |
| 冻结准入门槛 | 任意 | `≥+0.001` | `≤25%` | 无配置通过 |

coverage≤25% 的最佳非零路由专家计数：

- `current_gate`: 5,876
- `v1_champion`: 229
- `multi_interest`: 246
- `window_ensemble`: 316

## Comparison with Score-Only v2

| Router | Best delta under coverage≤25% | Coverage |
|---|---:|---:|
| score-only v2 | `+0.0001826203` | `11.7894%` |
| top1-aligned v3（非零路由） | `-0.0003988807` | `11.8644%` |

新增 top1 支持度差没有补上 v2 缺失的信息，反而让浅树在相近覆盖率下选择了更差的时间片规则。

## Interpretation

- 单个 top1 候选的原始属性差是高方差信号；它能描述“专家选了谁”，但不直接说明“谁会取得更高 reciprocal rank”。
- 137 维输入对 6,667 行 slice0 训练数据仍偏宽，浅树容易选择只在 slice0 成立的阈值。
- 九维多兴趣支持度已经进入 multi-interest 专家本身；再次以 top1 差进入路由器，边际信息可能很小。
- 这是基于 slice0→slice1 结果的推断，不是单独的因果消融结论。

## Gate and Artifact Status

- Output: `result/dataset2_multi_expert_top1_aligned_v3_r2_20260726`
- Selection status: `no_eligible_candidate`
- Selection report SHA-256: `ca22b016109d4a3499679b403474d8768316ad44717b0f4abdc1a2f35022dfb7`
- SHA sidecar verified: true
- slice2 evaluation report: not generated
- selection/final router: not generated
- candidate package: not generated
- Dataset1 frozen SHA-256: `6760fd966fde3a8de4693d06f27ec0f2458e49c44e2ea48fc79e1393434e6e2b`

## Decision

保留通用的 tie-neutral top1 feature-delta API 和实验脚本，作为可复用研究组件；不把本轮 descriptor 接入当前 Dataset2 生产 checkpoint。
