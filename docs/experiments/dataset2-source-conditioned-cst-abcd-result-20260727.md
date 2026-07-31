# Dataset2 Source-Conditioned CST A/B/C/D Result

## 结论

**A/B/C/D 已全部完成三折 rolling-origin，共 12 个正式模型。**

- 前两折锁定 D；
- D 在第三折门禁通过；
- 按协议只对 D 做一次 200k 全量重训和外部 20k 评估；
- 外部 MRR 为 `0.4744714277`，显著低于冻结冠军
  `0.5478966506`；
- 实验最终状态：**rejected，不生成提交，线上冠军保持不变。**

这次最重要的发现不是 D 的绝对分数，而是：

1. candidate ID 是内部时间折最大的新增信号；
2. source history 有小幅但不稳定的增益；
3. 无约束的 candidate-ID 分支会压过已经很强的 raw-feature
   路径，在外部分布切换时严重失效。

## 固定结构

| 变体 | candidate ID | source history | candidate self-attention |
|---|---:|---:|---:|
| A | 否 | 否 | 是 |
| B | 是 | 否 | 是 |
| C | 是 | 是 | 否 |
| D | 是 | 是 | 是 |

所有变体使用：

- 同一份 `200k × 100 × 63` 特征；
- 同一批候选与正例；
- 同一组三个 timestamp-aligned expanding folds；
- 同一 listwise cross-entropy、训练预算和 early-stop 规则；
- 单 seed `60`；
- tie-neutral MRR；
- 纯 Jittor 可训练模块。

## 三折结果

| 变体 | Fold 0 | Fold 1 | Fold 2 gate | 三折均值 |
|---|---:|---:|---:|---:|
| A | 0.428568 | 0.425194 | 0.421688 | 0.425150 |
| B | 0.477867 | 0.474864 | 0.468289 | 0.473673 |
| C | 0.477550 | 0.468084 | **0.472115** | 0.472583 |
| D | **0.481184** | 0.474562 | 0.468394 | **0.474713** |

前两折均值：

- A：`0.4268811537`
- B：`0.4763653094`
- C：`0.4728172510`
- D：`0.4778726726`

因此在读取 Fold 2 前锁定 D。

## 可归因发现

### 1. candidate ID 是主增益

B 相对 A：

- Fold 0：`+0.049299`
- Fold 1：`+0.049670`
- Fold 2：`+0.046601`

三折方向完全一致，平均约 `+0.04852`。现有 63 维 raw features
没有完整表达 candidate identity；让模型直接学习 item embedding，
确实能打破大量仅靠关系特征难以区分的候选。

### 2. source history 增益小且不稳定

在 candidate self-attention 都开启时，D 相对 B：

- Fold 0：`+0.003317`
- Fold 1：`-0.000302`
- Fold 2：`+0.000105`

三折平均仅约 `+0.001040`。source history 有信息，但当前
cross-attention 路由没有形成稳定的大增益。

### 3. candidate self-attention 也随时间漂移

在 candidate ID 与 source history 都开启时，D 相对 C：

- Fold 0：`+0.003633`
- Fold 1：`+0.006477`
- Fold 2：`-0.003721`

前两折支持 D，第三折却由 C 明显胜出。候选之间的竞争关系不是固定收益，
后续若继续使用，应由只看 source/candidate 稳定特征的保守路由控制，
不能默认永远开启。

## 第三折门禁

D 相对 A：

- Fold 2 full delta：`+0.046706`
- 三折 mean delta：`+0.049563`
- activity Q1：`+0.031276`
- activity Q2：`+0.052964`
- activity Q3：`+0.054449`
- activity Q4：`+0.048135`

所有冻结条件通过，因此允许全量重训和一次外部评估。

## 外部 20k

D 使用前两折 best epoch 的中位数，固定训练 3 epoch：

| 指标 | D | 冻结冠军 | Delta |
|---|---:|---:|---:|
| full | 0.474471 | 0.547897 | -0.073425 |
| time slice 0 | 0.490911 | 0.586739 | -0.095828 |
| time slice 1 | 0.471382 | 0.549014 | -0.077632 |
| time slice 2 | 0.461124 | 0.507943 | -0.046819 |
| activity Q1 | 0.546930 | 0.626139 | -0.079210 |
| activity Q2 | 0.519290 | 0.587513 | -0.068224 |
| activity Q3 | 0.453990 | 0.532637 | -0.078646 |
| activity Q4 | 0.377676 | 0.445297 | -0.067621 |

失败是全局性的，不是单一时间片或 activity 分段造成。

## 原因审计

缓存契约均通过：

- feature/candidate/src/time/dst 行数一致；
- candidate 第 0 列逐行等于真实 dst；
- 训练与 score sidecar 都按时间递增；
- source history 严格使用 `event_time < query_time`；
- Fold score history 冻结在 origin，不读取 score 段真值；
- candidate/source item 共用同一 Jittor embedding；
- 12 个 fold logits 均无正负 exact tie；
- checkpoint 保存重载测试通过。

最强证据指向 **candidate-ID 分支过度主导**：

- `candidate_id_scale` 初值为 `0.1`；
- B 三折最终约为 `1.035 / 1.168 / 1.056`；
- D 三折最终约为 `1.017 / 1.184 / 1.459`；
- D 全量模型最终为 `1.407`；
- 相比之下 D 全量 `source_scale` 仅为 `0.162`。

也就是说，ID 路径的可训练幅度增长了约 14 倍。内部 folds 的候选采样和
item prior 让这种记忆获益，但外部 20k 的时间/context 分布变化后，
ID 先验压过 raw-feature 路径，导致整体排序退化。

另一个背景差异是：200k 训练 cache 的 raw features 来自较早的固定
context，而外部 validation features 来自更完整的 `train_end` context。
外部 raw signal 更强，继续让 ID 分支自由放大尤其危险。

## 下一步建议

最高价值的后续不是继续加大 D，而是做 **bounded ID residual**：

1. 用已验证的纯 Jittor A/CST checkpoint 作为冻结或低学习率 base；
2. candidate ID 与 source branch 只输出 residual；
3. residual gate 使用有上限的参数化，例如最大幅度 `0.02–0.10`，
   不允许从 `0.1` 漂到 `1.4`；
4. 对 item embedding 使用更强 weight decay、ID dropout 和
   frequency shrinkage；
5. 仍在前两折选择 residual 上限，第三折门禁；
6. 外部 20k 不做权重扫描。

如果继续研究 source history，优先比较：

- B + bounded source residual；
- C/D 的时间稳定路由；
- recent-16 / recent-64 两层 history；

而不是再次无约束地端到端联合训练全部路径。

## 产物

远端：

- `cache/source_conditioned/dataset2_abcd_recent200k_full100_20260727`
- `result/dataset2_source_conditioned_cst_abcd_20260727`

本地证据：

- `artifacts/dataset2_source_conditioned_cst_abcd_20260727`

合规：

- `trainable_frameworks = ["jittor"]`
- `non_jittor_trainable_models = []`
- `submission_generated = false`
