# Goal Document: Dataset2 Cooccur-Lift Gap-Aware V2 External Safety Gate 与提交包

## Go / No-Go

- **Judgment**: Go
- **Reason**: 双视界 selector 已按冻结规则选中
  `cooccur_lift_gap_aware_v2`，selection lock 存在且 external 尚未打开；用户已明确
  授权一次性 external safety gate，并要求仅在七个不降门通过后生成提交包。

## Target Outcome

对已锁定的 gap-aware v2 做一次且仅一次 external 近视界安全检查。只读取七个门
的通过/失败状态，不把 external raw delta 当作效应量；通过后用同一锁定
full-origin 模型生成 Dataset2 测试概率，并与当前 bugfixed V1 新冠军按固定
`0.50` 权重生成提交包。

## Goal Definition

- **Type**: operational / quality / delivery
- **Boundary**: full-origin gap-aware v2 确定性训练、20k external 无指标物化、
  一次性标准 external gate、通过后的 test 物化与 ZIP 打包、哈希回传。
- **Non-goals**:
  - 不重做 near/gapped selection，不重新估计远视界效果。
  - 不扫描权重、窗口、seed、容量、gap 或候选。
  - 不使用 external raw delta 做效应量估计或候选排序。
  - 不修改现有 V1/V2 checkpoint、历史 manifest、selection report/lock。
- **Deferred work**:
  - 用户手工上传 ZIP 后的线上结果记录。
  - 若内部 gapped 通过但线上缩水，再审计 source-conditioned 与
    source/candidate joint structure。
- **Verification rule**: external manifest 必须绑定 selection lock、baseline、
  候选 config、holdout lineage 和全部 score 哈希；一次性 evaluator 只接受七门
  全过；package 必须绑定 accepted report、full-origin model、test probability
  与当前 V1 新冠军 ZIP。
- **Evidence source**: SHA-256、CPU 双跑 state/loss/probability、external
  open receipt、标准 evaluator report、test materialization report、package
  report 和 ZIP member hash。
- **Pass criteria**:
  - selected candidate 精确为 `cooccur_lift_gap_aware_v2`；
  - full-origin 训练 CPU 双跑在原 `rtol=2e-5`、`atol=2e-6` 下完全匹配；
  - external 20k 行 `short_window_supported=1`，collapsed fraction 为 `0`；
  - MRR 严格增加；Hit@1/3/10、NDCG@10 不下降；mean rank 不增加；
    improved-worsened 至少为 `1`；
  - evaluator `status=accepted` 后才允许 test 物化和 package；
  - Dataset1 member 与新 V1 包字节一致，Dataset2 行数 `153420`、候选数 `100`；
  - 无第二次 external receipt，无回扫或阈值修改。
- **Confidence note**: gapped 三折已经承担远视界证据责任；本轮 external 只有
  `0%` collapsed 的近视界安全门解释权。
- **Judgment owner**: 标准 external evaluator 决定 gate；package builder 只接受
  evaluator 的 accepted report；用户决定是否上传生成的 ZIP。

## Current State

- selection report SHA-256：
  `dc25b5b445e6b8f188f072bfec03e80b19e6f8fa1acfb34991c90d3cb9f25344`。
- selection lock SHA-256：
  `b52b529534b717ef136c82b17a090889b5aa4d67aed8618605ecbfda828e7e30`。
- selected candidate config SHA-256：
  `2b4a0bf3c61e183f28ffbda7e601b94298e0a68acbdbec8fc691fb6efb325ed3`。
- current bugfixed V1 package SHA-256：
  `b90960c3427f70e2745bcb381289fca4625c208ebfaefb43ecdbc7a7387ff2f0`。
- external 未读取、未打开；无 receipt 或 successor package。

## Plan Rewrite Notes

| Existing item | Decision | Reason |
|---|---|---|
| gapped 三折 | keep as completed evidence | 已承担远视界效果责任 |
| external raw delta | rewrite to gate-only | 只映射七个布尔门，不作效应量解释 |
| fold-specific gap-aware head | merge into one full-origin refit | package 需要唯一模型 |
| P75/P90/P100 collapsed states | equal-weight merge | 沿用冻结的 gapped 等权，不引入结果驱动权重 |
| 自动打包 | gate after accepted report | rejected 时必须停止 |

## Drift Diagnosis

- **Goal drift**: 再看 gapped 效应量或开新候选不能证明本轮近视界安全。
- **Phase drift**: 先打包后开 gate 会产生未授权提交物。
- **Validation drift**: 只看 MRR 或平均分会绕过其余六门。
- **Compatibility drift**: 旧 V1 external/materializer 使用旧 selection-lock
  协议，不能直接复用为 successor 证据。
- **Cleanup drift**: 不覆盖历史 V1/V2 结果目录或生产 checkpoint。

## Priority Rationale

- 先冻结 full-origin refit 与七门解释，再物化 external，避免看到结果后补规则。
- external manifest 先生成且不算排名指标；只有标准 evaluator 创建 receipt 后
  才算一次性开门。
- test 与 ZIP 严格依赖 accepted report，保证 rejected 路径无提交包。

## Assumptions and Open Decisions

| Item | Status | Impact | Owner / Next step |
|---|---|---|---|
| external 是近视界且 collapsed fraction `0` | confirmed by user | score support 固定为 1 | preflight 复核全部 20k 行 |
| full-origin collapsed 训练状态 | derived and frozen here | 影响唯一最终 head | 三档 gap 等分 collapsed 总权重 |
| full-origin seed | derived and frozen | 防止 seed 回扫 | `60 + 3*1009 + 30013 = 33100` |
| test support | confirmed by feature contract | 覆盖真实线上塌缩 | 严格按 `query_time-training_time_max<w` |
| package baseline | confirmed by latest accepted V1 | 决定最终混合 | 使用 SHA `b90960...` 的新 V1 ZIP |

## Phases

### Phase 1: 合同与资产 preflight

- **Purpose**: 在 external 指标前证明执行路径唯一且资产完整。
- **Entry condition**: 用户一次性授权已记录。
- **Phase rules**:
  - 只读 external 候选行、时间和特征资产，不计算排名指标。
  - 任一 hash、shape、lineage、selection binding 不符即停止。
- **Todos**:
  - [ ] 核验 selection lock、候选 config、V1 baseline、20k cache 和新 V1 ZIP。
    - **Surface**: remote assets
    - **Proof**: SHA/shape/absence audit
    - **Depends on**: none
  - [ ] 用 RED/GREEN 固化 successor external/test/package 合同。
    - **Surface**: code/tests
    - **Proof**: targeted pytest and Ruff
    - **Depends on**: asset audit
- **Exit proof**: dry-run 不创建 receipt、不读取外部排名指标。
- **Stop condition**: 无法从冻结规则唯一构造 full-origin 候选或资产漂移。

### Phase 2: Full-origin 模型与无指标 external manifest

- **Purpose**: 生成七门 evaluator 的唯一输入。
- **Entry condition**: Phase 1 通过。
- **Phase rules**:
  - near copy 权重 `0.6002802763655326`；
  - P75/P90/P100 collapsed copies各占
    `0.39971972363446745 / 3`；
  - 三种 collapsed copy 使用已冻结 gap 秒数和相同标签/候选；
  - CPU 独立双跑；external score support 全为 1；
  - 此阶段禁止 ranking metric。
- **Todos**:
  - [ ] 物化三档 full-origin stale lift 并复验。
    - **Surface**: external materialization cache
    - **Proof**: row/hash/availability report
    - **Depends on**: Phase 1
  - [ ] 双跑训练 gap-aware head，生成 baseline/candidate score 与 manifest。
    - **Surface**: successor external run
    - **Proof**: deterministic replay + standard manifest validation
    - **Depends on**: stale lift
- **Exit proof**: manifest 完整，ranking metrics 尚未读取。
- **Stop condition**: replay 漂移、external support 非全 1、任何 artifact hash 不符。

### Phase 3: 一次性七门 external

- **Purpose**: 只判断近视界安全。
- **Entry condition**: manifest 静态验证通过且 state dir 不存在。
- **Phase rules**:
  - evaluator 先写 one-shot receipt；
  - 只消费 `status` 与七个 gate 布尔值；
  - 不用 raw delta 排序、调参或估计线上效果。
- **Todos**:
  - [ ] 运行标准 evaluator 一次。
    - **Surface**: external state
    - **Proof**: receipt + evaluation report
    - **Depends on**: Phase 2
- **Exit proof**: 七门全过且 `package_authorized=true`，或 rejected 后停止。
- **Stop condition**: 任一门失败、receipt 已存在或 evaluator 合同错误。

### Phase 4: Test 物化与提交包

- **Purpose**: 将 accepted gap-aware 模型变成可提交 ZIP。
- **Entry condition**: Phase 3 accepted。
- **Phase rules**:
  - 复用同一个 accepted full-origin model；
  - test support 按冻结严格时间公式逐行计算；
  - 固定 `0.50` 混合新 V1 champion；不改生产 checkpoint。
- **Todos**:
  - [ ] 生成 Dataset2 test auxiliary probabilities。
    - **Surface**: online materialization
    - **Proof**: shape、finite、row-sum、hash、support coverage
    - **Depends on**: accepted report
  - [ ] 生成并复验 result.zip。
    - **Surface**: submission package
    - **Proof**: package report、member rows/hash、ZIP SHA
    - **Depends on**: test probabilities
- **Exit proof**: 本地得到可提交 ZIP 和完整 hash chain。
- **Stop condition**: Dataset1 member 变化、test order/shape/hash 不符。

### Phase 5: 回传与持久状态

- **Purpose**: 让一次性证据和提交包可在本地复核。
- **Entry condition**: Phase 3 terminal；Phase 4 在 accepted 时完成。
- **Phase rules**:
  - 只回传小型 JSON/日志与最终 ZIP；
  - 不自动上传竞赛平台。
- **Todos**:
  - [ ] 回传 receipt/report/contract/package。
    - **Surface**: local result/docs
    - **Proof**: local/remote SHA 相同
    - **Depends on**: terminal state
- **Exit proof**: 用户可从本地路径获取报告和 ZIP。
- **Stop condition**: 哈希回传不一致。

## Dry-Run Findings

- 旧 `materialize_dataset2_cooccur_lift_external.py` 只支持旧 V1
  `exact_integrated_weight_selection_lock_v1`，不能用于当前标准 selection lock。
- final refit 不能任选一个 gap；按 gapped 折既有等权合同，将 collapsed 总权重
  等分到 P75/P90/P100，是不读取结果的唯一对称延拓。
- external evaluator 会计算 gate 所需 raw deltas，但本轮只解释七个布尔门和最终
  accepted/rejected，不报告效应量。
- package 必须以最新 bugfixed V1 ZIP 为 baseline，不能退回旧冠军包。

## Final Validation

- 聚焦 pytest/Ruff 全绿；
- 远端源码与合同 SHA 匹配；
- external state 只有一个 receipt；
- 七门报告与 frozen lock 一致；
- accepted 后 test/package hash chain 完整；
- successor 目录无第二个 external state 或未绑定 package。

## First Execution Step

只读核验远端 selection lock、bugfixed V1 full-origin model/package、20k external
cache、训练 cache 和磁盘/内存，并确认本轮目标目录均不存在。

## Completion

- **Status**: Complete。
- external 已且仅打开一次，七门全部通过；解释角色保持
  `safety_gate_only`，未用 raw delta 估计效应量。
- CPU 双跑 state/loss/probability 最大误差均为 `0`，未放宽容差。
- 在线支持指示器按时间可用性计算为 `61109` 行 collapsed；审计的 `61325`
  全零行只保留为训练混合权重来源，不作为支持代理。
- 提交包已生成并同步到本地，SHA-256：
  `ea18d7fd8383bc0e21c0a5b4e9f82de6448fbf5535bd4d1c605f05f2a3223bfd`。
- 完整结果：
  `docs/experiments/dataset2-cooccur-lift-successor-v2-external-package-result-20260729.md`。
