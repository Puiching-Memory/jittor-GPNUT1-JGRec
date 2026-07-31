# Result: Dataset2 OOF Utility Router v3

## Verdict

**实现并运行完成；按冻结规则拒绝。** v3 显著提高了 routed 行的
RR-change 命中率，但 selection 的第二个连续时间片为负，因此没有读取
gate，也没有启动 change-only LambdaMRR Phase 2。冠军、提交和专家主干均未
改变。

## Implemented Model

- 纯 Jittor `217 → 128 → 128 → 4` hurdle MLP。
- 四个输出：
  - `P(change)`；
  - `P(gain | change)`；
  - 条件 gain magnitude；
  - 条件 loss magnitude。
- 路由分数：
  `P(change) × [P(gain|change) × gain
  − 2 × P(loss|change) × loss]`。
- 默认 short，只允许 frozen `top10/cap0.02` medium/long bounded action。
- 无 candidate ID、无 positive-column 输入，候选共同置换不改变行特征。
- 所有可训练模块仅使用 Jittor。

## Training Evidence

不再强制 medium/long 取共同 40k 区间：

| Action | OOF training rows |
|---|---:|
| medium | 64,322 |
| long | 23,334 |
| total action rows | 87,656 |

训练标签：

- gain：1,096
- no-change：84,644
- loss：1,916

先 warm-up 8 epochs，再挖出 6,296 个 hard negatives：

- 高分 no-change：4,380
- loss：1,916

最终模型从头训练 24 epochs，checkpoint replay error 为 `0.0`。

## Frozen Selection Result

固定 policy scan：

- route cap：`0.25% / 0.5% / 0.8% / 1%`
- minimum `P(change)`：`0.1–0.6`
- minimum expected utility：`0`
- 24 个 policy 配置，不搜索专家、top-k、cap 或 residual 参数。

扫描结论：

- aggregate delta 为正：4/24
- RR-change 命中率达到 12%：24/24
- 两个连续时间片均非负：0/24
- 全部门槛通过：0/24

aggregate 最好的策略为 `P(change) >= 0.6`：

| Metric | Value |
|---|---:|
| routed rows | 3 / 8,719 |
| actual coverage | 0.03441% |
| delta MRR | **+0.0000535230** |
| slice 1 | +0.0001147052 |
| slice 2 | **-0.0000076453** |
| gain / loss / routed no-op | 1 / 1 / 1 |
| routed RR-change hit rate | **66.67%** |

第二片负值违反冻结的“所有连续片非负”规则，所以 selection lock 没有生成。

## New Finding

v3 确实解决了一部分原瓶颈：

- v2 selection 的 RR-change 命中约为 `5/87 = 5.75%`；
- v3 最佳策略达到 `2/3 = 66.67%`；
- aggregate selection delta 也从 v2 的约 `+0.0000193` 提高到
  `+0.0000535`。

但模型收缩到只有 3 条路由，收益仍由单个 gain/loss 主导，尚未形成跨时间片
稳定性。当前问题已经从“大量 noop”转成“有效样本过少、方差过大”。

## Gating and Stop Rule

- `gate`: `not_read`
- Phase 2: `blocked_v3_failed`
- change-only LambdaMRR: 未实现、未训练
- online champion changed: false
- submission generated: false
- external evaluated: false

这遵守了 goudi 止损线：先证明路由精度和时间稳定性；selection 未通过时，
不允许继续增强 residual。

## Safety and Artifacts

- focused tests：5 passed
- neighboring regression tests：18 passed
- ruff：passed
- bounded route audits：selection scans 全部 passed
- trainable frameworks：`["jittor"]`
- non-Jittor trainable models：`[]`
- checkpoint SHA-256：
  `93a3834a2a517ded5a245834c855ecf64160da5e81b66ebe546ec730e3159e5a`
- 本地 checkpoint 与报告 hash 一致。

Artifacts:

- `result/dataset2_oof_utility_router_v3_20260727/evaluation-report.json`
- `result/dataset2_oof_utility_router_v3_20260727/frozen-config.json`
- `result/dataset2_oof_utility_router_v3_20260727/model.npz`
- `result/dataset2_oof_utility_router_v3_20260727.training.log`

## Decision

保留 v3 作为研究组件，但不进入 Phase 2。下一次若继续，应增加严格 OOF 的
正 change 样本量或做跨折 bagging/calibration；不能放宽本轮已经冻结的时间片
门槛，也不能事后读取 gate 来选阈值。
