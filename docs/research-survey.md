# 动态推荐与图推荐论文调研

本文档面向当前赛题工程：给定动态交互 `train.csv` 和每行 100 个候选目标节点，对候选做概率重排序。调研范围不是泛泛罗列推荐系统所有论文，而是覆盖和本工程最相关的路线：

- 隐式反馈排序与负采样。
- 序列推荐与会话推荐。
- 图协同过滤与图对比学习。
- 动态图/时间图表示学习。
- 工业级候选生成、排序融合与特征交互。
- 评估方法与可复现实证研究。

## 项目结论

对当前 `TemporalHybridRanker` 最有价值的不是马上换更大的动态图模型，而是先修正训练/验证分布，再补强低风险统计与融合特征。

优先落地顺序：

1. **Hard-negative 验证与训练**：当前监督样本负例来自全局随机 `dst_pool`，比正式 `test.csv` 的 100 候选更容易，容易导致本地 MRR 偏乐观。优先参考 BPR、sampled metrics、sampling-bias correction 相关论文，构造热门/近期/同源历史 hard negatives。
2. **时间窗口统计特征**：当前已有重复率、近因、目标热度。下一步应加最近窗口内的 pair/dst/source 计数、近期排名、时间间隔分桶。这类特征符合 Wide & Deep、DIN/DIEN 的“记忆 + 目标相关兴趣”思想，工程风险低。
3. **目标感知序列特征**：当前 SASRec 输出一个序列 embedding 与候选 dot。DIN/DIEN/TiSASRec 提示，应让候选目标参与历史行为注意力或时间间隔建模，而不是只做候选无关的序列摘要。
4. **图塔继续用 LightGCN/XSimGCL 做特征，不要先上 TGN**：动态图模型更适合事件级连续时间 link prediction；本赛题是给定 100 候选重排序，静态/多窗口图塔 + 统计特征更稳。
5. **融合头可试 DCN/AutoInt 风格**：当前 MLP 对 8-12 维特征做隐式交互。DCN/AutoInt/xDeepFM 的启发是显式建模交叉项，适合小维度强特征。

## 2023-2026 近期进展

截至 2026-05，推荐系统近年的论文热点明显转向大模型、生成式检索、语义 ID、Mamba/SSM 序列建模、扩散模型和更严格的复现实证。对当前比赛最重要的判断是：**近期热点多数面向大规模召回或通用推荐基础模型，而本项目是候选已给定的 100 路重排序**。因此可以借鉴思想，但不应直接把工程主线切到大模型生成式推荐。

| 年份 | 论文                                                                                                                  | 路线                              | 核心思想                                                                           | 对本项目的启发                                                                              |
| ---- | --------------------------------------------------------------------------------------------------------------------- | --------------------------------- | ---------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------- |
| 2023 | [Recommender Systems with Generative Retrieval / TIGER](https://arxiv.org/abs/2305.05065)                             | 生成式检索、语义 ID               | 用 RQ-VAE/离散 code 形成 item semantic ID，再自回归生成目标 item ID。              | 主要解决大规模召回；当前候选已给定，短期不优先。但可借鉴“item ID 离散语义化”做候选特征。    |
| 2023 | [Recformer: Text Is All You Need](https://arxiv.org/abs/2305.13731)                                                   | 文本表示序列推荐                  | 把 item 属性文本 flatten 成句子，用 Transformer 学 item sequence 的语言表示。      | 如果赛题没有 item 文本，直接收益有限；可提醒不要只依赖原始 ID，若有 side info 可引入。      |
| 2023 | [Scaling Law of Large Sequential Recommendation Models](https://arxiv.org/abs/2311.11351)                             | 大序列模型 scaling                | 研究纯 ID 序列推荐中模型/数据规模和效果的幂律关系。                                | 小比赛数据不宜盲目扩大模型；应先用小配置预测收益趋势。                                      |
| 2023 | [GiffCF: Graph Signal Diffusion Model for Collaborative Filtering](https://arxiv.org/abs/2311.08744)                  | 图信号扩散推荐                    | 用 item-item 图上的扩散/去噪过程重构用户交互信号。                                 | 可借鉴 item-item 图平滑思想；完整扩散推理成本高，不适合先上。                               |
| 2023 | [Graph Meets LLMs: Towards Large Graph Models](https://arxiv.org/abs/2308.14522)                                      | 大图模型综述                      | 总结图基础模型、图与语言对齐、图推理能力。                                         | 当前图是纯 ID 二部图，缺少文本/属性；更适合继续轻量图塔，而非上大图模型。                   |
| 2024 | [Actions Speak Louder than Words / HSTU](https://arxiv.org/abs/2402.17152)                                            | 生成式推荐、大规模序列 transducer | 将推荐重构为 generative sequential transduction，HSTU 面向超长行为流和工业级规模。 | 证明长行为序列很有价值；当前可借鉴目标感知/长短期分离，不适合直接移植。                     |
| 2024 | [Mamba4Rec](https://arxiv.org/abs/2403.03900)                                                                         | Mamba/SSM 序列推荐                | 用 selective state space model 降低长序列建模复杂度。                              | 若 SASRec 成本高，可考虑 Mamba 类序列塔；但当前序列塔未被选择，优先做轻量目标感知特征。     |
| 2024 | [MaTrRec](https://arxiv.org/abs/2407.19239)                                                                           | Mamba + Transformer               | 结合 Mamba 长期依赖和 Transformer 短期全局注意力。                                 | 对“长短期兴趣分离”有启发，可在统计特征里先做 recent/full 差分。                             |
| 2024 | [SIGMA](https://arxiv.org/abs/2408.11451)                                                                             | gated/bidirectional Mamba         | 解决 Mamba 在短序列和上下文建模中的不足。                                          | 说明 Mamba 不是无脑替代 Transformer；短序列下仍需专门处理近期模式。                         |
| 2024 | [LC-Rec](https://arxiv.org/abs/2311.09049)                                                                            | LLM + collaborative semantic ID   | 通过向量量化把协同语义编码成 LLM 可生成的 item index。                             | 对当前纯 ID 数据有思想价值；实现成本高，短期可借鉴为 item cluster/semantic bucket 特征。    |
| 2024 | [IDGenRec](https://arxiv.org/abs/2403.19021)                                                                          | LLM textual ID                    | 学习简洁、语义丰富、跨平台 textual ID，让 LLM 生成推荐 item。                      | 面向跨域/零样本；本赛题无文本语义时不优先。                                                 |
| 2024 | [EAGER](https://arxiv.org/abs/2406.14017)                                                                             | 行为语义双流生成式推荐            | 将行为信号和语义信号双流融合做生成推荐。                                           | 提醒融合行为/语义比单一路线更稳；当前可理解为 stats/GNN/sequence 多塔融合。                 |
| 2024 | [GenRec: Generative Sequential Recommendation with LLMs](https://arxiv.org/abs/2407.21191)                            | LLM 生成式序列推荐                | 将序列推荐变成 seq2seq 生成任务，强调轻量低资源训练。                              | 候选已给定时生成不是核心；可借鉴 masked item prediction 做预训练，但优先级低。              |
| 2024 | [DDRM: Denoising Diffusion Recommender Model](https://arxiv.org/abs/2401.06982)                                       | 扩散去噪推荐                      | 对 user/item embedding 做多步去噪，提高隐式反馈噪声鲁棒性。                        | 可借鉴“噪声鲁棒”思想；完整扩散模型需谨慎，推理成本和收益不确定。                            |
| 2025 | [Diffusion Recommender Models and the Illusion of Progress](https://arxiv.org/abs/2505.09364)                         | 复现实证、扩散推荐质疑            | 复现 2023-2024 扩散推荐论文，指出方法学问题和复杂模型未必优于简单强基线。          | 非常重要：近期论文不等于更适合比赛；每个复杂模型必须和强统计/LightGCN 基线同口径比较。      |
| 2025 | [Generative Recommendation with Semantic IDs: A Practitioner's Handbook / GRID](https://arxiv.org/abs/2507.22224)     | 语义 ID 实践框架                  | 系统 ablation 生成式推荐中的 semantic ID、架构和训练组件。                         | 若后续走生成式路线，应先用统一框架做 ablation；当前不作为第一优先级。                       |
| 2025 | [Semantic IDs for Joint Generative Search and Recommendation](https://arxiv.org/abs/2508.10478)                       | 搜索推荐统一语义 ID               | 比较 task-specific/cross-task 语义 ID 构造方式。                                   | 强调 ID 表示很关键；当前可先做 item 聚类、图 embedding bucket 这类低成本 semantic ID 特征。 |
| 2026 | [GenRec: A Preference-Oriented Generative Framework for Large-Scale Recommendation](https://arxiv.org/abs/2604.14878) | 工业生成式推荐、偏好对齐          | 面向生产分页推荐，使用 page-wise NTP、semantic ID 压缩和 GRPO-SR 偏好优化。        | 方向先进但工程跨度大；对本项目最可借鉴的是 page/list 级训练目标，而不是逐点 item 目标。     |

近期论文给当前工程的直接动作：

- **先改训练目标/候选构造**：HSTU、2026 GenRec 都强调 page/list 级序列输出；当前融合训练候选只有 `1 + num_negatives`，应更接近测试 100 候选。
- **用 semantic ID 思想做轻量特征**：不要直接上生成式检索，可先基于 LightGCN/XSimGCL embedding 聚类，给候选加 `dst_cluster`、`src_recent_cluster_hit`、`cluster_popularity`。
- **谨慎对待扩散推荐**：2025 复现工作已经提示扩散路线可能存在“复杂但不稳”的问题；除非强基线饱和，否则不优先。
- **Mamba 类模型只作为序列塔替换候选**：若 SASRec 推理/训练太慢，Mamba4Rec 值得看；但当前更缺目标感知序列特征。

## 评估与负采样

这些论文决定“怎么判断改动真的提分”，优先级最高。

| 论文                                                                                                                                                                                         | 核心思想                                                                    | 对本项目的启发                                                                                      |
| -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------- |
| [BPR: Bayesian Personalized Ranking from Implicit Feedback](https://arxiv.org/abs/1205.2618)                                                                                                 | 用 `(user, positive, negative)` pairwise 排序目标优化隐式反馈推荐。         | 当前图塔 BPR 合理；融合层也可增加 pairwise/listwise 混合损失，尤其针对 hard negatives。             |
| [On Sampled Metrics for Item Recommendation](https://research.google/pubs/on-sampled-metrics-for-item-recommendation/)                                                                       | 采样候选上计算 ranking metrics 会引入偏差；若必须采样，需要校正或谨慎解释。 | 当前本地验证用采样负例，不能直接等价线上 100 候选 MRR；需要建立 hard-negative 验证集。              |
| [Sampling-Bias-Corrected Neural Modeling for Large Corpus Item Recommendations](https://research.google/pubs/sampling-bias-corrected-neural-modeling-for-large-corpus-item-recommendations/) | 大规模 item 推荐中的 batch/sample softmax 存在抽样偏差，需要 logQ 等校正。  | 若训练候选来自非均匀热门/近期采样，融合 logits 需要记录采样策略，必要时加采样概率校正特征或校正项。 |
| [Are We Really Making Much Progress?](https://arxiv.org/abs/1907.06902)                                                                                                                      | 很多神经推荐方法在强基线和严谨复现下并不稳。                                | 每个复杂模型必须和强统计/LightGCN/iALS 风格基线比较，不能只看小样本 smoke。                         |
| [Revisiting the Performance of iALS on Item Recommendation Benchmarks](https://arxiv.org/abs/2110.14037)                                                                                     | 经典 iALS 经过认真调参仍很强。                                              | 可考虑补一个低成本 MF/iALS 特征，作为图塔之外的协同过滤基线。                                       |

建议实验：

- 新增 `--negative-strategy`：`random`、`popular`、`recent`、`mixed`、`test_like`。
- 对每个策略记录 `dataset1/dataset2` 的 `stats`、`stats_gnn`、`stats_gnn_seq` MRR。
- 用同一验证事件构造多种候选集合：随机负例、热门负例、近期负例、混合负例。

## 经典隐式反馈与序列推荐

这些论文解释为什么“重复 + 近因 + 序列”通常比复杂动态图更先见效。

| 论文                                                                                                                   | 路线             | 核心思想                                       | 对本项目的启发                                                                  |
| ---------------------------------------------------------------------------------------------------------------------- | ---------------- | ---------------------------------------------- | ------------------------------------------------------------------------------- |
| [Factorizing Personalized Markov Chains](https://archives.iw3c2.org/www2010/pub/pdfs/RendleFreudenthaler2010-FPMC.pdf) | MF + Markov      | 同时建模长期用户偏好和短期 item 转移。         | 当前统计特征可增加 last-dst 到候选 dst 的转移计数/转移热度。                    |
| [GRU4Rec / Improved RNNs for Session-based Recommendations](https://arxiv.org/abs/1606.08117)                          | RNN 序列         | 用 RNN 处理 session 行为并优化 top-k。         | 若序列塔继续做，先关注短 session/近期行为，而不是全历史。                       |
| [Caser](https://arxiv.org/abs/1809.07426)                                                                              | CNN 序列         | 在近期 item 序列上用卷积捕获局部组合模式。     | 可用轻量 n-gram/last-k co-occurrence 特征替代重 CNN。                           |
| [SASRec](https://arxiv.org/abs/1808.09781)                                                                             | 自注意力序列     | 自注意力在长短期偏好之间折中。                 | 当前已接入，但本地未选中序列特征；需要目标感知/时间感知改造后再评估。           |
| [TiSASRec](https://jiachengli1995.github.io/files/wsdm20.pdf)                                                          | 时间间隔自注意力 | 在 self-attention 中显式使用 item 间时间间隔。 | 本赛题有 `time` 字段，建议加入时间间隔 bucket 或时间衰减序列特征。              |
| [BERT4Rec](https://arxiv.org/abs/1904.06690)                                                                           | 双向 Transformer | 用 Cloze/masked item 任务预训练序列表示。      | 正式预测是未来候选，不能直接用双向泄漏；可借鉴 masked 预训练，但实现成本高。    |
| [S3-Rec](https://arxiv.org/abs/2008.07873)                                                                             | 自监督序列       | 用多种自监督任务缓解稀疏性。                   | 若序列数据稀疏，可考虑 item/subsequence 对比任务，但优先级低于 hard negatives。 |
| [SR-GNN](https://arxiv.org/abs/1811.00855)                                                                             | 会话图           | 将 session 序列转成图，捕获复杂 item 转移。    | 可做 last-k 目标转移图统计；完整 GNN session 模型暂不优先。                     |

## 图协同过滤与图对比学习

当前图塔使用 XSimGCL/LightGCN，这一选择和主流图推荐路线一致。

| 论文                                                                      | 路线                | 核心思想                                                | 对本项目的启发                                                     |
| ------------------------------------------------------------------------- | ------------------- | ------------------------------------------------------- | ------------------------------------------------------------------ |
| [Graph Convolutional Matrix Completion](https://arxiv.org/abs/1706.02263) | 二部图 auto-encoder | 将推荐矩阵补全看作 user-item 图上的 link prediction。   | 说明把交互数据建成二部图是合理基础。                               |
| [PinSage](https://ai.stanford.edu/~jure/pubs/pinsage-kdd18.pdf)           | 工业级图卷积        | 面向 web-scale 推荐的采样式图卷积。                     | 数据大时需要采样/窗口化；当前三窗口图塔符合这个方向。              |
| [NGCF](https://arxiv.org/abs/1905.08108)                                  | 图协同过滤          | 在 user-item 图上传播 embedding，显式注入高阶协同信号。 | 可作为 LightGCN 前身理解，不建议新增 NGCF，复杂度更高。            |
| [LightGCN](https://hexiangnan.github.io/papers/sigir20-LightGCN.pdf)      | 简化图推荐          | 去掉特征变换和非线性，只保留邻居传播。                  | 当前可继续作为强基线；比复杂 GNN 更稳。                            |
| [SGL](https://arxiv.org/abs/2010.10783)                                   | 图自监督            | 对 user-item 图做 node/edge dropout 等多视图对比。      | 可对 LightGCN 加简单 edge dropout ensemble 或图特征稳定性实验。    |
| [UltraGCN](https://arxiv.org/abs/2110.15114)                              | 超简化图推荐        | 用约束损失近似无限层传播，跳过显式 message passing。    | 如果图塔训练慢，可试 UltraGCN 思想：把图约束变成额外 loss/特征。   |
| [NCL](https://arxiv.org/abs/2202.06200)                                   | 邻域增强对比学习    | 将结构邻居和语义邻居纳入 contrastive positives。        | 对当前小数据/稀疏节点，可能比随机扰动更稳，但实现复杂。            |
| [XSimGCL](https://arxiv.org/abs/2209.02544)                               | 极简图对比学习      | 用 embedding 噪声替代复杂图增强，改善表示均匀性和长尾。 | 当前默认图塔选择合理；应重点调窗口、负采样和融合，而不是先换模型。 |
| [LightGCL](https://arxiv.org/abs/2302.08191)                              | 轻量图对比学习      | 简化 contrastive graph learning，提高效率和鲁棒性。     | 可作为后续图塔替换候选，但需要单独记录训练耗时和 MRR。             |

可落地实验：

- 图塔窗口从固定 `100%/35%/10%` 改为可配置，并记录 dataset2 对窗口的敏感性。
- 增加 graph score 的归一化/排名特征：每行候选内 `gnn_full_rank`、`gnn_recent_rank`、`gnn_short_rank`。
- 保留 `stats` 与 `stats_gnn` 选择层，避免图塔过拟合拖累提交。

## 动态图与时间图表示学习

这些论文和赛题名称“动态推荐”高度相关，但不一定是当前最优工程路线。它们更适合连续事件流 link prediction；本项目是给定候选集重排序。

| 论文                                                                                      | 路线                        | 核心思想                                                              | 对本项目的判断                                                                  |
| ----------------------------------------------------------------------------------------- | --------------------------- | --------------------------------------------------------------------- | ------------------------------------------------------------------------------- |
| [JODIE](https://arxiv.org/abs/1908.01207)                                                 | 用户/物品动态 embedding     | 每次交互后更新 user/item embedding，并预测未来 embedding trajectory。 | 与 user-item 事件流很贴合；可作为长期方向，但需要重写训练和 batching。          |
| [DyRep](https://research.google/pubs/dyrep-learning-representations-over-dynamic-graphs/) | 动态图点过程                | 同时建模网络拓扑演化和节点间活动。                                    | 表达力强但重，当前候选重排序未必需要。                                          |
| [TGAT](https://arxiv.org/abs/2002.07962)                                                  | 时间图注意力                | 用时间编码聚合 temporal-topological neighborhood。                    | 可借鉴时间编码；完整 TGAT 推理成本较高。                                        |
| [TGN](https://arxiv.org/abs/2006.10637)                                                   | memory-based temporal graph | 用 memory、message、embedding module 统一建模时间图事件。             | 适合动态图比赛路线，但实现复杂；只有在 hard-negative 验证证明图塔不足后再考虑。 |
| [EvolveGCN](https://arxiv.org/abs/1902.10191)                                             | snapshot 动态图             | 用 RNN 演化 GCN 参数，适合离散时间图快照。                            | 本数据是事件流，不天然是快照；需要先分桶，优先级低。                            |
| [Graph Neural Networks for temporal graphs: survey](https://arxiv.org/abs/2302.01018)     | 综述                        | 给出 temporal GNN 的任务和方法分类。                                  | 用于判断是否值得引入 TGN/TGAT 等完整动态图模型。                                |

当前建议：

- 不先上完整 TGN。
- 先把时间信息变成强统计特征和时间窗口图特征。
- 若后续尝试 JODIE/TGN，必须和 `stats_gnn` 在同一 hard-negative 验证集上比较训练耗时、推理耗时、MRR。

## 工业排序与特征交互

本赛题已经给定候选集，因此“排序阶段”的论文比“召回阶段”更直接。

| 论文                                                                                                   | 核心思想                                                   | 对本项目的启发                                                                          |
| ------------------------------------------------------------------------------------------------------ | ---------------------------------------------------------- | --------------------------------------------------------------------------------------- |
| [Deep Neural Networks for YouTube Recommendations](https://research.google.com/pubs/archive/45530.pdf) | 两阶段推荐：candidate generation + ranking。               | 本赛题跳过召回，重点应放在候选重排序特征和融合头。                                      |
| [Wide & Deep](https://arxiv.org/abs/1606.07792)                                                        | 线性记忆交叉 + 深度泛化。                                  | 当前统计特征属于 wide 强规则，GNN/序列属于 deep 表示；融合头可显式加交叉项。            |
| [DIN](https://arxiv.org/abs/1706.06978)                                                                | 对目标 item 做历史行为注意力，得到 target-aware interest。 | 最值得借鉴：对每个候选 dst，从 src 历史中做目标相关聚合，而不是候选无关序列 embedding。 |
| [DIEN](https://arxiv.org/abs/1809.03672)                                                               | 提取并演化用户兴趣，目标相关 attention 强化有效兴趣。      | 可用轻量版本：候选 dst 与最近 k 个历史 dst 的共现/embedding 相似度聚合。                |
| [DCN](https://arxiv.org/abs/1708.05123)                                                                | 显式 feature crossing，提高稀疏特征交互效率。              | 当前 8-12 维特征很适合试一层 CrossNet 替代纯 MLP。                                      |
| [DCN V2](https://arxiv.org/abs/2008.13535)                                                             | 改进 CrossNet，用于 web-scale learning-to-rank。           | 可作为融合头升级参考，但先做小型 CrossNet。                                             |
| [xDeepFM](https://arxiv.org/abs/1803.05170)                                                            | 同时建模显式和隐式高阶特征交互。                           | 可借鉴“显式交互”思想，不建议完整移植。                                                  |
| [AutoInt](https://arxiv.org/abs/1810.11921)                                                            | 用 self-attention 学习高阶特征交互。                       | 特征维度小，可以试轻量 field attention，但收益需验证。                                  |

可落地实验：

- 在 `FusionMLP` 前拼接手工交叉：`pair_strength * dst_popularity`、`pair_recency * dst_recency`、`repeat_rate * recent_hit`、`gnn_recent - gnn_full`。
- 新增 `FusionCrossNet`：保留原 MLP，并做 `stats`、`stats_gnn`、`stats_gnn_seq` 同样选择。
- 训练时从 32 候选改到更接近 100 候选，减少训练/测试 listwise 分布差异。

## 前沿生成式推荐

生成式推荐近年来发展很快，但大多数工作服务于大规模召回、跨域推荐或 LLM 推荐基础模型。当前比赛接口已经给定 100 个候选，所以更适合借鉴其中的表示学习和 list-wise 训练思想。

| 论文                                                                       | 核心思想                                                                            | 对本项目的启发                                                           |
| -------------------------------------------------------------------------- | ----------------------------------------------------------------------------------- | ------------------------------------------------------------------------ |
| [TIGER](https://arxiv.org/abs/2305.05065)                                  | 用 semantic ID 把 item 变成可生成 token，做生成式检索。                             | 可借鉴 semantic ID 做 item cluster 特征，不优先做生成式召回。            |
| [Actions Speak Louder than Words / HSTU](https://arxiv.org/abs/2402.17152) | 将推荐重构为大规模序列 transduction/generative recommendation，面向超大规模行为流。 | 证明“行为序列本身”很重要；本项目可借鉴目标感知序列建模，不适合直接移植。 |
| [LC-Rec](https://arxiv.org/abs/2311.09049)                                 | 用 collaborative semantic ID 让 LLM 生成推荐 item。                                 | 如果没有文本语义，先做协同聚类特征比 LLM 更现实。                        |
| [GenRec](https://arxiv.org/abs/2407.21191)                                 | 用 LLM 做生成式序列推荐。                                                           | 可借鉴低资源序列预训练思想；候选重排序不是生成式推荐的主战场。           |
| [GenRec 2026](https://arxiv.org/abs/2604.14878)                            | 面向工业推荐的偏好导向生成框架，包含 page-wise 训练和偏好优化。                     | 最值得借鉴的是 page/list 级目标：训练候选规模应更接近测试 100 候选。     |

## 推荐实现路线

### 第一阶段：验证可信化

目标：降低本地 MRR 与线上分数的偏差。

- 新增 `docs/model-experiments.md` 或在 `performance.md` 记录模型实验表。
- 实现 hard-negative 验证候选：
  - `random`: 当前全局随机。
  - `popular`: 从全局热门目标采样。
  - `recent`: 从最近时间窗口目标采样。
  - `mixed`: 随机、热门、近期混合。
  - `history`: src 历史目标中非当前正样本的一部分。
- 每个模型报告 `random_mrr` 与 `mixed_mrr`，并优先相信 `mixed_mrr`。

### 第二阶段：统计特征增强

目标：用低风险特征提升 `dataset2`。

候选特征：

- `dst_popularity_recent_1p/5p/20p`：最近 1%、5%、20% 训练事件内目标热度。
- `pair_count_recent_1p/5p/20p`：最近窗口内 src-dst 重复次数。
- `src_recent_activity_1p/5p/20p`：源节点近期活跃度。
- `pair_time_gap_log`：`log1p(query_time - pair_last_time)`。
- `dst_time_gap_log`：目标最近被交互时间间隔。
- `last_dst_to_candidate_count`：src 最近一个 dst 到候选 dst 的全局转移计数。

### 第三阶段：融合头增强

目标：让强统计、图塔、序列塔的交互更可学习。

- 加手工交叉特征，先不改模型结构。
- 若有效，再实现轻量 CrossNet。
- 保留验证选择层，继续在 `stats`、`stats_gnn`、`stats_gnn_seq`、`stats_gnn_cross` 中选择。

### 第四阶段：目标感知序列

目标：解决当前 SASRec 没被选择的问题。

- 先不训练新 Transformer。
- 用候选 dst 对 src 最近 k 个历史 dst 做相似/共现聚合：
  - 是否在最近 k 中出现。
  - 最近 k 中同候选共现强度最高的历史 dst。
  - 基于图 embedding 的 max/mean similarity。
- 若这些轻量目标感知特征有效，再考虑 DIN/DIEN 风格 attention。

## 论文优先级清单

优先阅读并落实：

1. [On Sampled Metrics for Item Recommendation](https://research.google/pubs/on-sampled-metrics-for-item-recommendation/)
2. [BPR](https://arxiv.org/abs/1205.2618)
3. [Diffusion Recommender Models and the Illusion of Progress](https://arxiv.org/abs/2505.09364)
4. [DIN](https://arxiv.org/abs/1706.06978)
5. [Wide & Deep](https://arxiv.org/abs/1606.07792)
6. [LightGCN](https://hexiangnan.github.io/papers/sigir20-LightGCN.pdf)
7. [XSimGCL](https://arxiv.org/abs/2209.02544)
8. [TIGER](https://arxiv.org/abs/2305.05065)
9. [Mamba4Rec](https://arxiv.org/abs/2403.03900)
10. [Actions Speak Louder than Words / HSTU](https://arxiv.org/abs/2402.17152)

暂不优先实现：

- 完整 TGN/TGAT/DyRep：实现成本高，且当前任务是候选重排序。
- BERT4Rec/S3-Rec 大规模预训练：数据和工程成本较高。
- 生成式推荐/HSTU/TIGER/LLMRec：方向先进，但短期只借鉴 semantic ID、list-wise 目标和长短期序列思想，不直接移植完整架构。
