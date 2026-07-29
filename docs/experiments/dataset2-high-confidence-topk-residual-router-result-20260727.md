# Dataset2 默认 Short 的纯 Jittor Bounded Top-k Router 结果

## 结论

训练完成，内部时间 gate 按预先冻结规则通过，但属于**小幅、低样本量的
research accept**，还不能生成提交。

最终选择：

- default：short corrected logits
- alternative：medium/long residual 应用到 short 公共 anchor
- top-k：10
- 二次 switch cap：0.02
- selection 最大切换率：1%
- selection 置信阈值：0.0306763
- 路由模型：纯 Jittor 353 → 128 → 128 → 2 reward MLP
- candidate ID / positive-column feature：无

## 时间切分

共同覆盖区间 `[159804, 200000)` 共 40,196 行：

| Split | 局部行 | 行数 | 用途 |
|---|---:|---:|---|
| train | `[0,23334)` | 23,334 | 唯一训练数据 |
| selection | `[23334,32053)` | 8,719 | variant/coverage/threshold |
| router-unseen gate | `[32053,40196)` | 8,143 | lock 后一次评估 |

这里的 gate 对具体路由模型是不可见的；在训练前做过 model-free oracle
opportunity preflight，因此不把它描述成完全 label-blind。

## 实际结果

| Split | Default short MRR | Routed MRR | Delta | 切换率 |
|---|---:|---:|---:|---:|
| selection | 0.4188148523 | 0.4188341497 | **+0.00001930** | 0.9978% |
| gate | 0.4171508932 | 0.4172477726 | **+0.00009688** | 0.7982% |

Gate 三个连续时间子片：

- `-0.00002866`
- `+0.00013510`
- `+0.00018416`

Gate 共切换 65 行：

- medium：8
- long：57
- MRR gain：2 行
- MRR loss：5 行
- 不变：8,136 行

净增益为正，是因为两条 gain 的排名收益大于五条 loss；这也说明结果方差
仍然很高，不能据此进入外部提交。

## 最重要的新发现

### 1. 候选支持差有效，纯 disagreement summary 基本无效

第一版仅使用 residual magnitude/cosine/gap：

- selection `+0.0000123`
- gate `0.0`
- gate 切换 244 行但没有一行改变正例 MRR

v2 增加原始候选特征、alternative top1 支持差和 promoted/demoted 支持差后：

- selection 所有三个时间子片为正；
- gate 获得 `+0.0000969`；
- 路由覆盖从约 3% 自动收缩到约 0.8%。

这直接验证了“各专家 top1 对应原始特征/支持度差”比单纯模型分歧更有价值。

### 2. Oracle 空间很大，但当前路由只吃到约 7.8%

所选 `top10/cap0.02` 在 gate 的 5% oracle：

- oracle delta：`+0.00123919`
- 正机会行：约 1.98%
- 实际 router delta：`+0.00009688`

当前 MLP 只回收约 7.8% 的理论空间。下一次提升重点不应继续扫 cap，而应
提升 candidate-level gain 识别能力。

### 3. “默认 short”是必要结构

所选 variant 的两路 alternative 全量平均 reward 为
`-0.0000646`。如果全量融合 medium/long 会下降；只有低覆盖路由才可能
获得正收益。

## 安全与复现审计

- 未路由行：与 short 逐元素完全一致
- top10 外：逐元素完全不变
- 最大 switch：`0.0200001001`
- cap：0.02，审计通过
- 行内 switch delta 零均值：通过
- gate 最大允许 81 行，实际 65 行
- selected checkpoint SHA-256：匹配
- checkpoint prediction replay error：`0.0`
- feature contract：353 个有序名称写入 checkpoint
- trainable frameworks：`["jittor"]`
- non-Jittor trainable models：`[]`

## 产物

远端完整目录：

`/home/edu/workspace/jittor-GPNUT1-JGRec/result/dataset2_high_confidence_topk_residual_router_v2_20260727`

主要文件：

- `selection-lock.json`
- `gate-report.json`
- `evaluation-report.json`
- `variants/topk-10-cap-0.02/model.npz`
- `gate-scores.npy`
- `gate-route-index.npy`
- `gate-predicted-advantages.npy`

本地同步了报告与运行日志，模型和数组保留在远端。

## 最终判断

- 内部 bounded router：**通过**
- 是否替换 Dataset2 冠军：**否**
- 是否读取外部验证：**否**
- 是否生成提交：**否**

下一个值得做的模型是纯 Jittor candidate-set gain classifier：直接编码
short top10 内每个候选的 raw feature、三路 residual 和被提升/压低关系，
学习 medium/long 的正收益概率；继续保持本轮 top10/cap0.02/默认 short
不变。
