# Dataset2 部分混合抢救 Listwise 专家结果

## Decision

**No-Go。** 两条历史 listwise 正信号都没有通过相对当前
`short_none 50/40k + 0.80 Setwise + 0.20 LightGBM` 冠军的冻结前向门禁。

- listwise MLP：六个辅助权重在 slice0 全部下降，未产生 selection lock。
- listwise Two-Tower：slice0 锁定 `0.20`，但 slice1 回归，且前两片合并增益
  只有 `+0.0000220828`，低于冻结的 `+0.0003`。
- 按协议没有读取任何候选的 slice2 指标。
- 没有修改冠军 checkpoint，没有生成候选 checkpoint 或提交 ZIP。

这说明“部分混合”方法本身没有问题，但旧 listwise 专家的正信号没有迁移到
已经更强的 A1 新冠军上；不能因为 γ/α 曾在线上成功，就跳过当前主干上的重新
校准与时间稳定性门禁。

## Aligned Score Contract

三个分数源使用同一份 `20000 × 100` candidate sidecar：

```text
dec159209d9c6913825591b585afa0689b7b7323912543204ca6190dad4e4a95
```

当前冠军四项指标精确复现：

| Metric | MRR |
|---|---:|
| Full | 0.5485470648527594 |
| Slice 0 | 0.5882028774417708 |
| Slice 1 | 0.5493313411199712 |
| Slice 2 | 0.5081009093765456 |

同一验证段上的独立专家 full MRR：

| Expert | Full MRR |
|---|---:|
| cached-feature listwise MLP | 0.5387816622304992 |
| listwise Two-Tower raw | 0.5087423429616291 |
| listwise Two-Tower midrank transform | 0.5087423429616291 |

历史报告中的 MLP `+0.0066839` 和 Two-Tower `0.4641` 是旧主干/旧时间段证据，
不是相对当前冠军的可直接继承增益。

## Slice0 Frozen Scan

### Listwise MLP

| Auxiliary weight | Slice0 delta | Eligible |
|---:|---:|---|
| 0.05 | -0.0000361286 | no |
| 0.10 | -0.0004974899 | no |
| 0.20 | -0.0012346287 | no |
| 0.30 | -0.0009574432 | no |
| 0.40 | -0.0005828532 | no |
| 0.50 | -0.0012134347 | no |

没有非退化权重，因此该支路在读取 slice1 前结束。

### Listwise Two-Tower

| Auxiliary weight | Slice0 delta | Eligible |
|---:|---:|---|
| 0.05 | +0.0000073034 | yes |
| 0.10 | +0.0000599637 | yes |
| 0.20 | +0.0000790706 | yes, selected |
| 0.30 | -0.0000413820 | no |
| 0.40 | -0.0003715240 | no |
| 0.50 | -0.0001469709 | no |

锁定 selection：

```text
expert = listwise_two_tower
weight = 0.20
selection_lock_sha256 =
8ec5afadbfc5a904a074a460d7379845dc8adfc77311223c16b0adf0a32f6eb6
```

## Slice1 Forward Gate

| Metric | Baseline | Candidate | Delta |
|---|---:|---:|---:|
| Slice 1 | 0.5493313411 | 0.5492964360 | -0.0000349051 |
| Slice0 + Slice1 | 0.5687671093 | 0.5687891921 | +0.0000220828 |

冻结门槛要求：

- slice1 delta `>= 0`；
- slice0 + slice1 delta `>= +0.0003`。

两项均未通过。因此 `evaluation-report.json` 状态为
`rejected_before_slice_2`，并明确记录
`slice_2_metrics_read: false`。

## Artifacts

本地目录：

```text
result/dataset2_partial_listwise_expert_blend_20260728/
```

关键分数矩阵 SHA-256：

| Artifact | SHA-256 |
|---|---|
| champion probabilities | `0a39d5f4d2ba8eedb2966a91075afce0a9be469cea89913e6f2eb71078a85983` |
| listwise MLP probabilities | `f62575fcc79952b37c0deae60c95b6672897cc1fe3feb33cc258abd2d2eba06f` |
| listwise Two-Tower probabilities | `a0d4dbab28a6d7da9ee41754890c27e8a49a519b301fc60c206c13d158f67a58` |

主要报告：

- `preflight-report.json`
- `frozen-config.json`
- `score-report.json`
- `listwise-mlp-slice0-scan.json`
- `listwise-two-tower-slice0-scan.json`
- `listwise-two-tower-selection.json`
- `listwise-two-tower-forward-gate.json`
- `evaluation-report.json`

## Verification

```text
focused partial-blend tests: 7 passed (Windows and Linux)
related blend regressions: 17 passed locally
ruff: passed
champion full/slice reproduction: exact
local/remote score-array SHA-256: matched
slice2 candidate metrics: not read
checkpoint/package generation: skipped by gate
```

## Judgment

A2 已完成验证，但没有产生可晋升候选。不要放宽同一轮门槛，也不要在已经看到
slice0/slice1 后重新挑权重。若未来重开这条线，必须产生新的 OOF/长跨度校准
证据，或重新训练与当前 `short_none` 服务口径匹配的 listwise 专家；不能继续
消费本轮 slice。
