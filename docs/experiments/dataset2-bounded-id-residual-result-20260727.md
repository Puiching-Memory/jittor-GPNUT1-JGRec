# Dataset2 Frozen CST + Bounded ID Residual Result

## 结论

**绝对幅度 bounded ID residual 已完成三折与一次外部评估，但不生成提交。**

- residual 的数学与工程边界成立；
- 前两折锁定 `cap=0.10`；
- Fold2 门禁通过；
- 三折平均相对 frozen CST 提升 `+0.00033817`；
- 外部 residual 相对 full frozen CST 仅提升约 `+0.00000777`，基本中性；
- 外部 MRR 为 `0.54439935`，低于现冠军 `0.54789665`
  共 `-0.00349730`；
- 实验状态：**rejected，线上冠军与提交包保持不变。**

最重要的新发现是：绝对上限成功消除了无约束 ID 分支的灾难性覆盖，
但也暴露出当前 ID-only head 的真实可迁移增益很小。之前内部 folds 上
约 `+0.05` 的 ID 收益，主要来自允许 ID 路径取得与主干同量级的分数尺度。

## 正式边界

正式 v2 使用：

```text
score = frozen_base
      + cap * tanh(raw_id_logit - row_mean(raw_id_logit))
```

固定配置：

- cap：`0.02 / 0.05 / 0.10`；
- embedding dim：`32`；
- ID dropout：`0.10`；
- weight decay：`1e-3`；
- 训练：固定 3 epoch；
- 主干：冻结纯 Jittor A/CST；
- 正例位置：0；
- 指标：tie-neutral MRR；
- 单 seed：60。

normalized v1 曾把 cap 乘以 base row standard deviation，导致
`cap=0.10` 的外部 residual 实际可达 `0.484`。它不满足绝对幅度要求，
已从正式结论中排除；v1 和 v2 的结果/checkpoint 目录完全隔离。

## 三折结果

表中为相对各折 frozen A/CST 的 MRR delta：

| Cap | Fold 0 | Fold 1 | Fold 2 | 三折均值 |
|---:|---:|---:|---:|---:|
| 0.02 | -0.00000048 | 0.00000000 | 0.00000000 | -0.00000016 |
| 0.05 | -0.00000586 | +0.00000024 | 0.00000000 | -0.00000187 |
| **0.10** | **+0.00059289** | **+0.00035960** | **+0.00006202** | **+0.00033817** |

前两折 selection：

- cap 0.02 mean delta：`-0.00000024`，不合格；
- cap 0.05 mean delta：`-0.00000281`，不合格；
- cap 0.10 mean delta：`+0.00047625`，且两折均不退化；
- 因此在读取 Fold2 前锁定 cap 0.10。

## Fold2 门禁

cap 0.10：

- Fold2 full delta：`+0.00006202`；
- 三折 mean delta：`+0.00033817`；
- activity Q1：`+0.00001187`；
- activity Q2：`+0.00019465`；
- activity Q3：`-0.00003833`；
- activity Q4：`+0.00007989`。

最差 activity delta 高于冻结阈值 `-0.001`，因此允许一次 external
evaluation。cap 0.02/0.05 的 Fold2 只作为 lock 后诊断，不改变选择。

## 绝对边界审计

所有 9 个 fold report 的 `bound_audit.passed=true`。

| Cap | Fold0 max residual | Fold1 | Fold2 | 允许上限 |
|---:|---:|---:|---:|---:|
| 0.02 | 0.000639 | 0.000031 | 0.000001 | 0.02 |
| 0.05 | 0.006418 | 0.000422 | 0.000144 | 0.05 |
| 0.10 | 0.054133 | 0.041658 | 0.020749 | 0.10 |

三个 frozen A score replay 最大误差分别为：

- Fold0：`3.34e-6`；
- Fold1：`3.81e-6`；
- Fold2：`3.81e-6`。

因此内部变化不是主干重训、checkpoint 漂移或越界造成。

## 外部 20k

| 指标 | Bounded ID | 现冠军 | Delta |
|---|---:|---:|---:|
| full | 0.544399 | 0.547897 | -0.003497 |
| time slice 0 | 0.579253 | 0.586739 | -0.007486 |
| time slice 1 | 0.546475 | 0.549014 | -0.002539 |
| time slice 2 | 0.507476 | 0.507943 | -0.000468 |
| activity Q1 | 0.622283 | 0.626139 | -0.003856 |
| activity Q2 | 0.587475 | 0.587513 | -0.000039 |
| activity Q3 | 0.531595 | 0.532637 | -0.001042 |
| activity Q4 | 0.436244 | 0.445297 | -0.009053 |

相对同一个 full frozen CST base：

| 指标 | Frozen CST | Bounded ID | Residual delta |
|---|---:|---:|---:|
| full | 0.544392 | 0.544399 | +0.000008 |
| time slice 0 | 0.579200 | 0.579253 | +0.000053 |
| time slice 1 | 0.546502 | 0.546475 | -0.000027 |
| time slice 2 | 0.507478 | 0.507476 | -0.000003 |

也就是说，绝对 cap 确实把风险压到了近乎零，但没有产生足以追上冠军的
新增信号。外部实际最大 residual 只有 `0.003592 < 0.10`；
不是触顶后仍然过强，而是 full 训练下 ID head 本身收缩得很小。

## 新发现与后续判断

1. **bounded residual 是正确的安全结构，但不是当前提分主线。**
   它把无约束 D 的外部灾难性退化从约 `-0.0734` 收敛到对 frozen CST
   约 `+0.000008`，证明“冻结主干 + 硬上限”的防护有效。

2. **单纯 candidate ID prior 的可迁移收益接近零。**
   内部 cap 0.10 三折方向稳定，但增益只有万分之几；不值得围绕
   0.02–0.10 再做密集 cap 扫描。

3. **当前主要差距在 frozen base，而不是 residual 上限。**
   full frozen CST 本身约为 `0.544392`，落后冠军约 `0.003505`；
   bounded ID 无法弥补这一基线差距。

4. **若继续使用 ID，只应做置信路由后的稀疏纠错。**
   下一步应把 residual 限定在冠军/CST 分歧大、ID support 高且时间稳定的
   少数行或 top-k pair，而不是对全部 100 candidates 统一注入 ID prior。

## 产物

远端正式结果：

- `result/dataset2_bounded_id_residual_v2_20260727`
- `cache/bounded_id_residual/dataset2_frozen_a_20260727`

本地证据：

- `artifacts/dataset2_bounded_id_residual_v2_20260727`

实现与验证：

- `src/jgrec/rankers/hybrid/bounded_id_residual.py`
- `scripts/train_dataset2_bounded_id_residual.py`
- `tests/test_hybrid_bounded_id_residual.py`

合规：

- `trainable_frameworks = ["jittor"]`
- `non_jittor_trainable_models = []`
- `submission_generated = false`
