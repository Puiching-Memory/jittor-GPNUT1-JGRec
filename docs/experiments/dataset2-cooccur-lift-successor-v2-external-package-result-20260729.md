# Dataset2 Cooccur-Lift Gap-Aware V2 External 与提交包结果

## 结论

- 标准 external safety gate 已且仅执行一次，结果为 `accepted`。
- 七个近视界不降门全部通过，`failed_gates=[]`，
  `package_authorized=true`。
- external 只承担安全门角色；本报告不复述、不解释 raw delta，也不把它当作
  效应量。
- `cooccur_lift_gap_aware_v2` 提交包已生成并同步到本地，未自动上传。

提交包：

`result/dataset2_cooccur_lift_successor_v2_external_20260729/submission/result.zip`

SHA-256：

`ea18d7fd8383bc0e21c0a5b4e9f82de6448fbf5535bd4d1c605f05f2a3223bfd`

## 七门结果

| Gate | 结果 |
|---|---|
| `mrr_meets_minimum` | 通过 |
| `hit_at_1_meets_minimum` | 通过 |
| `hit_at_3_meets_minimum` | 通过 |
| `hit_at_10_meets_minimum` | 通过 |
| `ndcg_at_10_meets_minimum` | 通过 |
| `mean_rank_meets_maximum` | 通过 |
| `improved_minus_worsened_meets_minimum` | 通过 |

只含门状态的规范摘要：

`result/dataset2_cooccur_lift_successor_v2_external_20260729/external-gate-summary.json`

SHA-256：

`ab107216c2d7b17beab6e100b6657985d61e1925e9b27a5184964c1ba9cdcb7d`

标准 external report SHA-256：

`dddfe6ad03b4a6a74948bb0467a22cc175d434eefee808116046a29eba0f94ac`

selection lock SHA-256：

`b52b529534b717ef136c82b17a090889b5aa4d67aed8618605ecbfda828e7e30`

## Full-Origin 与 External 证据

- 候选：`cooccur_lift_gap_aware_v2`。
- config SHA-256：
  `2b4a0bf3c61e183f28ffbda7e601b94298e0a68acbdbec8fc691fb6efb325ed3`。
- full-origin seed：`33100`。
- 训练状态：一个 near copy 加三个 P75/P90/P100 collapsed copy，共
  `800000` 个有效训练行。
- 训练权重：
  `0.6002802763655326 + 3 × 0.13323990787815582`。
- CPU 独立双跑的 state、loss 和 external probability 均完全一致，最大误差
  都是 `0`；容差保持 `rtol=2e-5`、`atol=2e-6`，未放宽。
- external 严格评分行 `19981`，支持指示器全为 `1`，collapsed fraction 为
  `0`。
- materialization 阶段没有计算 external ranking metric；标准 evaluator
  创建 receipt 后才形成唯一 gate 判决。

模型 SHA-256：

`526a28c3eb642ee6c29c05f34e44a5a5b6bd3eeda67da24b1c7f5704886880cd`

## 在线支持度语义

训练混合权重与部署支持指示器是两个不同量：

- `61325 / 153420 = 39.9720%` 是审计观测到的 short-lift 全零行比例，只用于
  full-origin 训练的 near/collapsed 混合权重；
- `61109 / 153420 = 39.8312%` 是部署支持指示器按冻结公式
  `query_time - min(query_time, train_history_end) < w` 得出的真实 collapsed
  行数。

候选配置禁止用 `cooccur_lift_short != 0` 作为支持代理，因此最终在线物化采用：

- supported rows：`92311`；
- collapsed rows：`61109`；
- short window：`17038080` 秒；
- availability rule：`min(query_time, train_history_end)`。

在线 probability：

- shape：`[153420, 100]`；
- row-sum 最大误差：`9.992007221626409e-16`；
- SHA-256：
  `d5346e437c8aa78b70a9bdbf3c21367994ca936046e2052f8db8e83be97da3a4`。

## 提交包验证

最终公式保持冻结的 `0.50` 权重：

`candidate = 0.50 * bugfixed_v1_champion + 0.50 * gap_aware_v2`

| Member | Rows | Columns | SHA-256 |
|---|---:|---:|---|
| `dataset1.csv` | 61051 | 100 | `81285eddfba612b9c05e96075be29a15718d8a94839d143e3083bf5353644369` |
| `dataset2.csv` | 153420 | 100 | `080a06ddc8ee6d36f9f0acabdf4d86e0ebaf5cfc2fccbb3e3e2bb16e6b14547a` |

- Dataset1 与 bugfixed V1 新冠军 member 字节一致。
- Dataset2 baseline member SHA-256：
  `702f46d6a14b36e5330cac315ceefb130e54d4a68f9a173ce9d65c5a1d06192f`。
- bugfixed V1 baseline ZIP SHA-256：
  `b90960c3427f70e2745bcb381289fca4625c208ebfaefb43ecdbc7a7387ff2f0`。
- 最终 ZIP 只包含 `dataset1.csv` 和 `dataset2.csv`。
- 本地重新计算的 ZIP/member hash 与 package report 完全一致。

## TDD 证据

### RED

首次目标测试在服务器环境因
`ModuleNotFoundError: jgrec.cooccur_lift_successor_external` 失败，证明测试确实
覆盖尚未实现的 successor external 合同，而不是运行器伪失败。

### GREEN

实现后六个聚焦测试通过，覆盖：

- gap-aware 候选、selection lock、V1 baseline 和七门绑定；
- 三档 collapsed copy 权重；
- external 支持度全 `1`；
- 部署支持度来自 feature availability；
- 训练混合比例不得强行替代部署支持指示器；
- 任一 gate 失败时禁止 package。

标准协议回归合计 `22 passed`；相关源码与脚本通过 Ruff。

### REFACTOR

- 将 selection/config 校验、支持度计算、manifest 构造和 package 授权集中到
  `jgrec.cooccur_lift_successor_external`。
- package 授权只消费 `status`、精确七个 gate 布尔值和哈希绑定，不消费
  external raw delta。
- 保留了冻结候选、模型、权重和容差，没有为了通过 external 或在线物化修改
  任何选择规则。

## 执行异常与处置

执行中有两个在结果前被硬拦截的操作问题，均完整保留证据：

1. 首个 launcher 因服务器缺少 evaluator runner，以退出码 `2` 停止。由于
   receipt、external state 和 report 均不存在，这不构成 external 判决；
   后续通过单独冻结 evaluator/协议哈希后才执行真正且唯一的一次 external。
2. external accepted 后，首个在线尝试把 `61325` 的训练混合参考误当成部署
   指示器计数，在创建在线输出目录前被断言拦截。修正清单明确区分两个量，
   不改模型、不改权重、不重开 external。

最终 finish exit code 为 `0`。历史失败 marker 与日志保留在同一结果目录，避免
把操作失败伪装成实验失败或成功。

## 本地制品

- `external-gate-summary.json`：只含七门与授权状态。
- `external-evaluation/external-open-receipt.json`：唯一开门凭据。
- `external-evaluation/external-evaluation-report.json`：标准协议原始报告。
- `online-materialization/test-materialization-report.json`：在线支持度与概率哈希。
- `submission/successor-package-report.json`：候选、V1 baseline 与 ZIP 哈希链。
- `submission/result.zip`：待用户手工提交的最终包。
