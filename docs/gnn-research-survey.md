# GNN 推荐论文调研

更新时间：2026-05-28

本文只覆盖和本项目主线相关的 GNN 推荐论文：user-item 二部图协同过滤、图对比学习、谱图过滤、动态图推荐、会话图推荐和大规模图推荐。不纳入 LLM、semantic ID、生成式检索、文本推荐等非 GNN 主线。

## 对当前项目的结论

当前赛题是给定每行 100 个候选目标节点后重排序，不是大规模召回。因此 GNN 的价值应体现在候选内排序分数，而不是替换成生成式检索。

优先方向：

1. **保留 LightGCN/XSimGCL 作为强图塔基线**。LightGCN 证明推荐场景里去掉复杂特征变换和非线性往往更稳；XSimGCL 代表近年的极简图对比学习路线。
2. **重点尝试谱图/低秩图协同特征**。SVD-GCN、LightGCL、ChebyCF、GSPRec 都说明图推荐正在从“堆 message passing”转向“理解图过滤和谱信号”。这和本项目可以低成本增加谱图候选分数的方向贴合，但必须先做隔离实验。
3. **负采样必须跟线上反馈一起看**。MixGCF、MixDec Sampling 这类工作说明 GNN-BPR 的收益高度依赖 hard negatives；但本项目已经出现 hard-negative 本地高分、线上低分的反例，因此负采样协议必须先通过线上锚点校准。
4. **动态图模型要谨慎**。JODIE、TGAT、TGN、DyGFormer 适合连续时间 link prediction，但本项目是 100 候选重排序；可先把时间信息做成多窗口图和时间衰减边权。item-transition 图已经离线高分但线上失败，不能作为默认方向。
5. **复杂 GNN 只有线上有效才保留**。近年论文普遍报告公开 benchmark 提升，但本项目已有强 stats+XSimGCL 基线；任何 GNN 改动必须先用第一版线上分 `1.1452` 做冠军基线。

## 论文地图

| 年份 | 论文                                                                                              | 路线                     | 核心思想                                                              | 对本项目的判断                                                                                     |
| ---: | ------------------------------------------------------------------------------------------------- | ------------------------ | --------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------- |
| 2017 | [Graph Convolutional Matrix Completion](https://arxiv.org/abs/1706.02263)                         | 二部图 link prediction   | 把矩阵补全视为 user-item 二部图上的消息传递。                         | GNN 推荐的早期基础；当前不必复现，作为建模视角参考。                                               |
| 2018 | [PinSage](https://arxiv.org/abs/1806.01973)                                                       | 工业级图推荐             | 用随机游走采样和图卷积做 web-scale item embedding。                   | 证明图采样、hard example 和离线 embedding 对生产推荐重要；本项目可借鉴采样和候选内 hard negative。 |
| 2019 | [NGCF](https://arxiv.org/abs/1905.08108)                                                          | 图协同过滤               | 在 user-item 图上传播 embedding，显式注入高阶协同信号。               | 已被 LightGCN 简化路线基本取代；不建议新增 NGCF。                                                  |
| 2019 | [SR-GNN](https://arxiv.org/abs/1811.00855)                                                        | 会话图推荐               | 将 session 序列转成图，捕捉复杂 item transition。                     | 可借鉴做 last-k item transition 图特征；不优先完整实现。                                           |
| 2019 | [JODIE](https://faculty.cc.gatech.edu/~skumar498/pubs/jodie-kdd2019.pdf)                          | 动态 user-item embedding | 为用户和物品学习随时间变化的 embedding trajectory。                   | 与动态推荐题面贴合，但工程跨度大；作为长期方向。                                                   |
| 2020 | [LightGCN](https://arxiv.org/abs/2002.02126)                                                      | 简化图协同过滤           | 去掉 GCN 中的特征变换和非线性，只保留邻居传播与层聚合。               | 当前主图塔的核心强基线，应继续作为第一对照。                                                       |
| 2020 | [DGCF](https://arxiv.org/abs/2007.01764)                                                          | 解耦图协同过滤           | 将 user-item 交互拆成多个 latent intent 图。                          | 可解释性强但实现复杂，当前优先级低于谱图和负采样。                                                 |
| 2020 | [TAGNN](https://arxiv.org/abs/2005.02844)                                                         | 目标感知会话图           | 针对不同候选目标激活不同 session interest。                           | 对候选重排序很相关；可落地为候选目标感知的历史转移/图分数。                                        |
| 2020 | [TGAT](https://arxiv.org/abs/2002.07962)                                                          | 时间图注意力             | 用时间编码聚合 temporal-topological neighborhood。                    | 可借鉴时间编码；完整 TGAT 推理成本高。                                                             |
| 2020 | [TGN](https://arxiv.org/abs/2006.10637)                                                           | memory-based 动态图      | 用节点 memory、message 和 graph module 处理 timed events。            | 如果多窗口静态 GNN 不够，再考虑；当前先做低成本时间图特征。                                        |
| 2021 | [SGL](https://arxiv.org/abs/2010.10783)                                                           | 图自监督推荐             | 对 user-item 图做 node/edge dropout、多视图对比。                     | 图增强能提升鲁棒性，但随机增强可能带来 seed 不稳；需严格复测。                                     |
| 2021 | [UltraGCN](https://arxiv.org/abs/2110.15114)                                                      | 超简化图推荐             | 用约束损失近似无限层图卷积，跳过显式 message passing。                | 若 JittorGeometric 图塔训练慢，可参考其约束式训练目标。                                            |
| 2021 | [MixGCF](https://ericdongyx.github.io/papers/KDD21-Huang-et-al-MixGCF.pdf)                        | GNN hard negative        | 在 GNN 不同传播层混合正负样本，合成更难负例。                         | 很适合本项目：先从候选/历史/popular/recent hard negatives 做轻量版。                               |
| 2021 | [GCE-GNN](https://arxiv.org/abs/2106.05081)                                                       | 全局+会话图              | 同时建 session graph 和 global transition graph。                     | 可用全局 item transition 图做候选辅助特征。                                                        |
| 2022 | [NCL](https://arxiv.org/abs/2202.06200)                                                           | 邻域增强对比学习         | 将结构邻居和语义邻居纳入 contrastive positives。                      | 对稀疏节点有吸引力，但聚类/EM 会增加不稳定性；暂不作为第一实现。                                   |
| 2022 | [XSimGCL](https://arxiv.org/abs/2209.02544)                                                       | 极简图对比学习           | 用 embedding 噪声替代复杂图增强。                                     | 当前默认图模型合理；调参应围绕窗口、负采样、融合选择。                                             |
| 2022 | [SVD-GCN](https://arxiv.org/abs/2208.12689)                                                       | 低秩谱图推荐             | 将 LightGCN 与 SVD/MF 联系起来，用截断 SVD 替代多层传播。             | 可作为下一轮隔离实验参考，不能直接并入默认链路。                                                   |
| 2022 | [BSPM](https://arxiv.org/abs/2211.09324)                                                          | 图过滤 CF                | 用 blurring-sharpening 过程解释和改进协同过滤。                       | 可启发图分数后处理和低成本图过滤；不先做完整模型。                                                 |
| 2022 | [HCCF](https://arxiv.org/abs/2204.12200)                                                          | 超图对比 CF              | 用 hypergraph 结构和自监督增强表示。                                  | 需要构造高阶超边，当前纯二部图场景优先级中低。                                                     |
| 2023 | [LightGCL](https://arxiv.org/abs/2302.08191)                                                      | SVD 图对比学习           | 用 SVD 生成全局协同视图，避免随机扰动破坏结构。                       | 非常贴合本项目：谱视图比随机 edge dropout 更可能稳定。                                             |
| 2023 | [AdaGCL](https://arxiv.org/abs/2305.10837)                                                        | 自适应图对比             | 用生成/去噪模型自动生成 contrastive views。                           | 方法先进但训练复杂；等轻量谱图实验确认收益后再考虑。                                               |
| 2023 | [How Expressive are GNNs in Recommendation?](https://arxiv.org/abs/2308.11127)                    | 理论/表达力              | 用 topological closeness 等推荐相关指标分析 GNN 表达力。              | 提醒不要只追求一般 GNN 表达力；应看候选节点距离/邻近性是否改善排序。                               |
| 2023 | [Topology-aware Analysis of Graph CF](https://arxiv.org/abs/2308.10778)                           | 图拓扑实证               | 分析数据拓扑与 LightGCN/DGCF/UltraGCN/SVD-GCN 效果关系。              | 建议先测本赛题图的度分布、重复边、连通性，再决定图模型复杂度。                                     |
| 2023 | [DyGFormer](https://arxiv.org/abs/2303.13047)                                                     | 动态图 Transformer       | 用历史一跳交互序列、邻居共现编码和 patching 做动态图学习。            | 动态图 SOTA 方向，但不是推荐专用；只有在图时间信号很强时再评估。                                   |
| 2024 | [LTGNN](https://arxiv.org/abs/2402.13973)                                                         | 线性时间图推荐           | 将 GNN 推荐扩展到接近 MF 的线性复杂度。                               | 如果图塔训练成为瓶颈，可参考其线性传播/缓存设计。                                                  |
| 2024 | [A Temporal Graph Network Framework for Dynamic Recommendation](https://arxiv.org/abs/2403.16066) | TGN 推荐化               | 将 TGN 直接用于动态推荐场景。                                         | 可作为长期重构参考；短期不应替代多窗口 LightGCN。                                                  |
| 2025 | [LightGNN](https://arxiv.org/abs/2501.03228)                                                      | 剪枝/蒸馏 GNN            | 自适应剪枝图边和 embedding 条目，用蒸馏保持效果。                     | 若 GNN 证明有效，再考虑用边权/剪枝降噪；当前先别加复杂蒸馏。                                       |
| 2025 | [ChebyCF](https://arxiv.org/abs/2505.00552)                                                       | Chebyshev 谱过滤         | 不依赖 learned embedding，直接对用户交互信号做灵活谱过滤。            | 很重要。可先做 Chebyshev/多阶图过滤的轻量特征版本。                                                |
| 2025 | [GSPRec](https://arxiv.org/abs/2505.11552)                                                        | 时间感知谱过滤           | 将 sequential transitions 融入图构造，并用 bandpass/low-pass 双过滤。 | 和赛题时间字段高度相关；可作为谱图特征稳定后的下一步。                                             |
| 2025 | [MixDec Sampling](https://arxiv.org/abs/2502.08161)                                               | soft-link 采样           | 用 mixup 和 decay sampling 生成软链接，改进 GNN 负采样。              | 可借鉴时间衰减和软负例思想，但要避免引入不可控随机性。                                             |
| 2026 | [DG-SA-GNN](https://arxiv.org/abs/2605.05238)                                                     | 动态相似图注意力         | 动态重建多种 user-user 相似图，并用 attention 融合。                  | 很新且只在小基准上报告；可作为 user similarity 图方向参考，不宜直接主线化。                        |

## 方向分解

### 1. 轻量图协同过滤

代表：NGCF、LightGCN、UltraGCN、LTGNN。

核心趋势是从复杂 GNN 往轻量传播、约束损失和线性复杂度收敛。对本项目而言，LightGCN/XSimGCL 比复杂 GAT/GraphSAGE 更像正确起点，因为输入只有 ID 交互，没有丰富节点特征。

落地动作：

- 保持 `stats`、`stats_gnn` 的验证选择层，防止图塔过拟合。
- 固定 `lightgcn` 与 `xsimgcl` 两条图塔基线，所有新图模型必须同时对比这两者。
- 记录 `gnn_full/gnn_recent/gnn_short` 的单特征贡献，确认到底哪个时间窗口有用。

### 2. 图对比学习与去噪

代表：SGL、NCL、XSimGCL、LightGCL、AdaGCL。

这条线解决的是稀疏交互、噪声边、长尾节点问题。早期 SGL 依赖随机图增强；XSimGCL 改用 embedding 噪声；LightGCL 和 AdaGCL 则转向更结构化或自适应的增强。

对本项目的判断：

- 随机 edge/node dropout 可能造成 seed 敏感，必须谨慎。
- LightGCL 的 SVD 视图更适合当前要求，因为它是确定性全局协同信号。
- AdaGCL/NCL 的复杂组件可能带来小样本不稳定，应放在谱图实验之后。

### 3. 谱图过滤与低秩图信号

代表：SVD-GCN、LightGCL、BSPM、ChebyCF、GSPRec。

这是近两三年最值得跟的 GNN 推荐方向之一。它把 LightGCN 的邻居传播解释成图过滤，强调不同频段的协同信号、低秩全局结构、时间转移图。

对本项目最直接：

- 谱图候选分数：用 user-item 归一化矩阵做截断 SVD 或图过滤，输出候选 dot score。
- 行内 rank 特征：对每个 query 的图分数做候选内 rank/percentile，降低不同窗口分数尺度问题。
- 时间图过滤：优先把近期边、时间衰减边作为不同图视图，而不是只用全量二部图；item-transition 边需要隔离重测。

保留条件：

- `stats_gnn` 必须在同一协议下超过第一版 GNN 基线。
- seed 42、seed 7 至少都不能退化。
- 若本地指标与线上锚点排序冲突，以线上反馈为准。

### 4. GNN 负采样

代表：PinSage curriculum hard negatives、MixGCF、MixDec Sampling。

GNN 推荐通常用 BPR 或 sampled softmax。如果负样本太容易，图 embedding 不会学到候选内细粒度边界；如果负样本太难或分布不一致，又会造成本地分数虚高。

本项目建议先做工程轻量版：

- 训练负例可以尝试 `random/popular/recent/history` 混合，但不能因为本地 MRR 上涨直接保留。
- 下一步加 graph-hard negatives 前，先建立能复现线上锚点排序的代理验证；否则只能做线上 A/B。
- 负例来源必须写入实验记录，因为采样策略改变会直接改变验证 MRR。

### 5. 动态图与会话图

代表：JODIE、TGAT、TGN、DyGFormer、SR-GNN、TAGNN、GCE-GNN、GSPRec。

这条线和“动态推荐”题面最像，但工程风险最大。动态 GNN 往往需要事件流 batch、节点 memory、时间编码和在线状态更新；而当前提交只需要对固定候选做概率重排序。

短期建议：

- 不先重写 TGN。
- 先做多窗口图塔和时间衰减边权；item-transition 图作为已失败实验，只能隔离重测。
- 如果这些轻量时间图特征稳定增益，再考虑 TGN/DyGFormer。

## 推荐实施路线

### P0：重新固定 GNN 基线

必须有这些对照：

- `stats`
- `stats + LightGCN`
- `stats + XSimGCL`
- `stats + XSimGCL + seq`，只作为旧默认对照

协议：

- 同一数据、同一 split、同一负采样。
- 至少 seed 42/7。
- 记录 `selected_fusion`，若最终仍选 `stats`，说明图特征没有真实增益。

### P1：谱图协同特征

参考：SVD-GCN、LightGCL、ChebyCF、GSPRec。

建议实现：

- 全量、近期、短期三个窗口的谱图候选分数
- 可配置实验开关
- 默认关闭，先做隔离分支
- 只通过 `stats_gnn` 验证选择进入最终模型

这是当前最符合“GNN 方向且不偏离主线”的先进改法。

当前工程状态：默认模型已恢复到第一版 GNN，谱图和 transition 分支不在主链路中。重新实现时必须先作为隔离实验，不修改默认 CLI。

### P2：图 hard negatives

参考：PinSage、MixGCF、MixDec Sampling。

实现顺序：

- 从 stats/GNN 高分候选中抽 hard negative。
- 控制比例，不替代全部 random negatives。
- 多 seed 不稳定则回退。

### P3：图分数后处理与融合

参考：LightGCN 层聚合、谱过滤、GSPRec 双过滤。

可加特征：

- `gnn_recent - gnn_full`
- `gnn_short - gnn_full`
- 候选行内图分数 rank
- 图分数与统计特征交叉项，如 `gnn_full * pair_strength`

### P4：时间图增强

参考：TAGNN、GCE-GNN、GSPRec、TGAT。

先做：

- 时间衰减 user-item 图
- 候选行内图分数 rank
- 图分数与统计特征交叉项

后做：

- item transition count / decay count
- source recent item 到候选 item 的转移图分数
- TGN/DyGFormer 事件流模型。

## 暂缓方向

- **KG-GNN 推荐**：KGAT、KGIN 等需要实体属性或知识图谱；本赛题当前只有 ID 交互，不优先。
- **大 GAT/Graph Transformer 直接替换 LightGCN**：没有节点特征时容易训练成本高、收益不稳。
- **复杂自适应图生成**：AdaGCL、DG-SA-GNN 这类思路先进，但会引入较多随机和超参；等轻量谱图路线跑通后再做。

## 下一步实验清单

1. 以第一版完整 GNN 提交为冠军基线，记录当前默认命令可复现的 run id。
2. 跑 `stats-only`、`stats+LightGCN`、`stats+XSimGCL` 的 dataset2 中样本严格基线。
3. 在隔离分支实现谱图特征，只测谱图特征是否让 `stats_gnn` 被选择。
4. 若 seed 42 提升，立刻复测 seed 7；不稳定则关闭。
5. 若谱图稳定，再加候选内 rank 特征。
6. 若图特征稳定，再做 graph-hard negative sampler。

所有结果写入 [性能基准](performance.md)，没有多 seed 稳定收益的改动不进入默认模型。
