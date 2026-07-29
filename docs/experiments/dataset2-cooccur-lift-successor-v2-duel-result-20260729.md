# Dataset2 Cooccur-Lift Successor V2 双候选预注册结果

## Verdict

预注册、协议补冻、三档 gapped cache、确定性 CPU 双跑和冻结 selector 均已完成。
唯一机器判决为 **`cooccur_lift_gap_aware_v2` selected**；
`cooccur_lift_full_only_v2` 因三个 gapped 折全部退化而 rejected。
external 未读取、未打开，仍只具备后续一次性 safety-gate 资格。

baseline 已绑定包含 v1 的 promoted champion checkpoint：
`796d8d21a0c706ad11f244385b314d471d522c3b807748a54fe4ac78722f5880`。
候选空间恰为两个，权重都固定为 `0.50`，v1 家族权重回扫仍被禁止。

## Final Internal Selection

| Candidate | Near MRR deltas | Gapped MRR deltas (P75/P90/P100) | Decision |
|---|---|---|---|
| `cooccur_lift_full_only_v2` | `+0.008496 / +0.010735 / +0.011904` | `-0.002409 / -0.000402 / -0.003893` | rejected：远视界 MRR/NDCG 门失败 |
| `cooccur_lift_gap_aware_v2` | `+0.012193 / +0.012395 / +0.014808` | `+0.006620 / +0.007734 / +0.004013` | selected：六折 MRR/NDCG 门全部通过 |

gap-aware 的平均 near MRR delta 为 `+0.013131963492500232`，平均 gapped
MRR delta 为 `+0.006122215617720824`，按冻结的 `39.971972%` collapsed
混合得到 deployment-mixture MRR delta `+0.010330029009246112`。
zero-short 仍仅为诊断臂，没有参与选择。

机器证据：

| Artifact | SHA-256 |
|---|---|
| Corrected rolling manifest | `2423cdf55329a66d904095f235777ef257ba555e917c6573cb61f90b443ac63d` |
| Selection report | `dc25b5b445e6b8f188f072bfec03e80b19e6f8fa1acfb34991c90d3cb9f25344` |
| Selection lock | `b52b529534b717ef136c82b17a090889b5aa4d67aed8618605ecbfda828e7e30` |

运行结束时未发现 external receipt、external evaluation、checkpoint、ZIP 或
submission 产物。

## Selector Wiring Repair

六折训练原始 manifest 漏写了 plan-v2 已冻结的顶层 `baseline_sha256`，标准
selector 因而正确拒绝。修复没有覆盖原始 manifest
（SHA-256 `bcf9a3ebf576d83d3724962c983b60a6b3d9e943a8742612f19604c1df95ce1e`），
而是生成可审计副本，仅加入冻结值
`db0f180b087ca2e5b758046f8b934ccb0419b458bfdb8d5b913321d842cb4993`。
修复报告明确记录：

- `changed_fields=["baseline_sha256"]`；
- `score_artifacts_modified=false`；
- `metrics_modified=false`；
- `thresholds_modified=false`；
- `external_scores_read=false`。

远端最初还缺少 selector 入口脚本；同步后脚本及协议模块均与本地逐字节同哈希。
流水线曾把 Python 的 file-not-found 退出码 `2` 误报为“all candidates
rejected”，该状态已由真实 selector report/lock 取代。

## Frozen Candidates

| Priority | Candidate | Frozen change | Config SHA-256 |
|---:|---|---|---|
| 0 | `cooccur_lift_full_only_v2` | 删除 short 及其 context 通道，64 raw → 192 context | `82395bbbb07c17b6f6adfe5629a01ef054e07c35da68f0f28d734d7c69fa9cc0` |
| 1 | `cooccur_lift_gap_aware_v2` | 保留 full/short，增加 row-level support，66 raw → 198 context | `2b4a0bf3c61e183f28ffbda7e601b94298e0a68acbdbec8fc691fb6efb325ed3` |

`short_window_supported` 冻结为：

```text
int(query_time - fold_training_time_max < 17038080)
```

它在候选轴上原样广播；边界使用严格小于；禁止用
`cooccur_lift_short != 0` 代替。gap-aware 的每个 outer-fold head 必须看到
support=1 的 near 因果样例和 support=0 的 gapped 因果样例，不能做 near-only
训练。

协议补冻把这个约束落实为相同训练行/候选的两份 view：near copy 权重
`0.6002802763655326`，fold-gap collapsed copy 权重
`0.39971972363446745`。full-only 和 v1 baseline 不做这项复制。

三个远视界折已逐行绑定：

| Fold | Train rows | Score rows | Gap |
|---|---:|---:|---:|
| P75 | 79,909 | 34,720 | 251 days |
| P90 | 118,816 | 42,040 | 308 days |
| P100 | 159,804 | 40,471 | 349 days |

三折共享 `rows[0:1137757]` 的无泄漏 encoder context；总 query 数
277,035，最晚 scoring time 固定为既有 internal reference end
`1255824000`。

## Frozen Decision

资格门是合取，不允许视界间补偿：

1. 每个 near fold：MRR/NDCG@10 delta `>= 0`；
2. 每个 P75/P90/P100 gapped fold：MRR delta `> 0` 且 NDCG@10 delta
   `>= 0`；
3. 两组都过才 eligible；
4. 都过时按平均 gapped MRR、最差 gapped MRR、平均 near MRR、预登记
   tie-break 排序；
5. full-only 同分优先。

deployment mixture 和可选 zero-short 仍输出诊断，但均不能改变资格结论。
external 只有在 selection lock 存在后才可一次性打开，角色固定为 safety gate；
raw delta 不作效应量估计，`19.5x` 只保留为校准折扣。

## Plan Evidence

| Artifact | SHA-256 |
|---|---|
| Validation plan | `07f0ac9a244077a3ad8e7e3cd76bd7c95c6b7c00d8a42766601785f069e95efd` |
| Plan-v2 lock | `3519a496a5807b18e4b6f0aefdfd9c92dce34cfdf188c0f719458785f2ed6d98` |
| Plan-v2 preflight | `7b1f67f99925a6028535f6c9106b4dba8e5d8a6ac6b809064c6f6f47eb0ce5af` |
| Promotion replay report | `a089c83766e49951888c6d1a752a5cba60a2ebce67d46869818be8ecbef014c7` |
| Promoted manifest | `95473458aee80e3cde3c4fabe4237e458b692ad005a9444ad2cea7d2b5eb7976` |

Preflight 明确：

- `candidate_count=2`；
- `selection_metrics_read=false`；
- `reserved_fold_metrics_read=false`；
- `external_holdout_read=false`；
- `package_authorized=false`。

旧 lock 没有删除，只作为补冻前审计历史；正式运行只接受 `plan-v2`。独立审计还
确认 successor 结果目录没有 selection lock、external receipt、checkpoint 或
package。

## Deterministic Execution Amendment

首次自动 duel 在任何 successor 指标前被原 replay gate 拒绝：

```text
fold-0 v1 deterministic replay drifted:
max_abs_error=0.10732766809698313
```

后续真实数据双跑已证明该误差来自 CUDA 训练非确定性，而 CPU 在 2k、20k 和
full-data V1 上均精确回放。没有放宽 `rtol=2e-5`、`atol=2e-6`，也没有删除
校验；执行补充合同改为：

- V1、full-only、gap-aware 及 gapped prior Setwise head 全部 CPU 独立双跑；
- state、loss、probability 三类证据必须同时通过原容差；
- 历史 CUDA V1 fold score 只报告 legacy drift，不参与 replay gate 或 selection；
- candidate、fold、权重、seed、容量和 selector 规则保持原 plan-v2 不变。

执行合同：
`docs/experiments/cooccur-lift-successor-v2-duel.execution.preregistered.json`，
SHA-256
`452687bee6e2770634c05d10169e673b93c65295886d92cc00a2125289495498`。

## Remaining Risk

当前审计只排除了候选流行度的边际分布差异，没有排除：

- source-conditioned 分布差异；
- source/candidate 联合结构差异。

如果某候选逐折通过远视界门、但线上再次缩水，这两项是下一条审计向量，不能把
该现象归因于已排除的 popularity marginal。

## Verification

- RED：旧协议 `4 failed, 11 passed`；baseline 传播另有 `1 failed`。
- 第二轮 GREEN：候选/materializer 定向回归 `19 passed`，连同标准协议为
  `36 passed`。
- Ruff：`All checks passed!`。
- 远端真实数据 dry-run 命中 251/308/349 天三档和全部冻结行号，没有读取
  external。
- 确定性执行补充合同 RED→GREEN 后，本地相关回归 `52 passed`；远端聚焦
  `12 passed`，Ruff 与 bash syntax 均通过。

运行期发现一个基础设施保留项：正式历史 encoder 预处理期间，主机与 22223
端口仍可达，但 sshd 无法及时返回协议 banner。后续重任务启动门必须在模型预计
峰值之外保留 `max(25% RAM, 8 GiB)` 给 OS/sshd，并限制 CPU/IO 优先级与单机
并发；“未 OOM”不再作为资源安全的充分条件。本次已启动进程不在线改参、不重复
启动，以保留首次 exact-plan 证据。

SSH 恢复后确认正式 cache PID `62698` 健康：第 13/68 批已完成
`53,248/277,035` 行，RSS 约 8.1 GiB、系统可用约 23 GiB，瓶颈为单线程
`structure` 特征而非 GPU。保守剩余时间约 5–6 小时。已挂载低优先级续跑器
PID `64371`，脚本 SHA-256 为
`18292c94306f8237a62ae25e6de5723cecf783d4629ca681b2f41c149e228ee7`：
它只会在原 cache PID 结束、report 完整且全部 artifact hash 复验通过后启动
duel，再运行冻结 selector；无论 selected/rejected 都硬停在 external 门前。

用户随后授权在不降质量的前提下提高服务器利用率。顺序 run/watcher 已分别在
65,536 行处用 TERM 正常停止，旧目录完整保留。新正式目录为
`gapped-cache-v2-parallel4`，cache PID `65298`；父进程保持原候选 RNG，只把
structure 查询稳定分给 4 个 Linux fork worker。真实首批完整 63 维 A/B：

- exact array/SHA parity passed；
- sequential `319.1316s`，parallel4 `192.5798s`；
- speedup `1.6571x`，4 个 worker 全部参与；
- 系统仍有约 19 GiB `MemAvailable`，SSH 正常。

replacement watcher PID `66254` 已绑定新 PID/目录；完成后仍只执行 artifact hash
复验、duel 和冻结 selector，不打开 external。

## Next Move

第 2 步本地判决已结束。`cooccur_lift_gap_aware_v2` 已获得打开 external
safety gate 的资格，但 external 是一次性动作，当前仍未授权、未打开。
下一步只能在用户明确授权后，使用上述 selection lock 执行一次 external；
不得回扫 v1 权重或把 external raw delta 当作效应量。
