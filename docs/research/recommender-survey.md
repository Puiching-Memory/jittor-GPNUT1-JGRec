# 推荐系统论文调研归档

本文档保留非 GNN 方向的论文线索，只作为背景资料。当前比赛主线是 GNN 候选重排序，GNN 相关研究和实施路线见 [GNN 推荐论文调研](gnn-survey.md)。

## 当前结论

- 当前任务是给定每行 100 个候选目标节点后的重排序，不是大规模召回。
- 生成式推荐、LLM 推荐、Semantic ID、扩散推荐、Mamba 序列推荐等近期热点多数面向召回、跨域或通用推荐基础模型，不进入当前默认链路。
- 已经尝试过的 Semantic ID 聚类塔因 seed 不稳定被拒绝；后续不要直接恢复为主线。
- 若继续研究非 GNN 方向，只能作为隔离实验，并且必须以 [实验与基准](../experiments/benchmarks.md) 中的第一版线上分 `1.1452` 为冠军基线。

## 评估方法

这些论文决定“怎么判断改动真的提分”，优先级最高。

| 论文                                                                                                                                                                                         | 核心思想                                                            | 对本项目的启发                                                  |
| -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------- | --------------------------------------------------------------- |
| [BPR: Bayesian Personalized Ranking from Implicit Feedback](https://arxiv.org/abs/1205.2618)                                                                                                 | 用 `(user, positive, negative)` pairwise 排序目标优化隐式反馈推荐。 | 当前图塔 BPR 合理；融合层可以研究 pairwise/listwise 混合损失。  |
| [On Sampled Metrics for Item Recommendation](https://research.google/pubs/on-sampled-metrics-for-item-recommendation/)                                                                       | 采样候选上计算 ranking metrics 会引入偏差。                         | 本项目已经出现本地 MRR 与线上得分严重 gap，不能只相信采样验证。 |
| [Sampling-Bias-Corrected Neural Modeling for Large Corpus Item Recommendations](https://research.google/pubs/sampling-bias-corrected-neural-modeling-for-large-corpus-item-recommendations/) | 大规模 item 推荐中的 sampled softmax 存在抽样偏差，需要校正。       | 若引入非均匀负采样，必须记录采样分布并校验线上效果。            |
| [Are We Really Making Much Progress?](https://arxiv.org/abs/1907.06902)                                                                                                                      | 很多神经推荐方法在强基线和严谨复现下并不稳。                        | 复杂模型必须和当前第一版 GNN 冠军基线比较。                     |
| [Revisiting the Performance of iALS on Item Recommendation Benchmarks](https://arxiv.org/abs/2110.14037)                                                                                     | 经典 iALS 经过认真调参仍很强。                                      | 可作为后续协同过滤特征参考，但不能替代线上验证。                |

## 序列与目标感知排序

这些方向可借鉴为轻量特征，但当前 SASRec 序列塔在第一版提交中没有进入最终选择。

| 论文                                                                                 | 路线               | 对本项目的启发                                                                                   |
| ------------------------------------------------------------------------------------ | ------------------ | ------------------------------------------------------------------------------------------------ |
| [FPMC](https://archives.iw3c2.org/www2010/pub/pdfs/RendleFreudenthaler2010-FPMC.pdf) | MF + Markov        | 可用 last-k 行为和候选之间的转移计数做轻量特征，但 transition 图已有线上失败记录，必须隔离重测。 |
| [GRU4Rec](https://arxiv.org/abs/1606.08117)                                          | RNN 序列           | 若重做序列塔，先关注短 session/近期行为。                                                        |
| [Caser](https://arxiv.org/abs/1809.07426)                                            | CNN 序列           | 可用轻量 n-gram/last-k co-occurrence 特征替代重模型。                                            |
| [SASRec](https://arxiv.org/abs/1808.09781)                                           | 自注意力序列       | 当前已接入；默认训练后由验证选择层决定是否采用。                                                 |
| [TiSASRec](https://jiachengli1995.github.io/files/wsdm20.pdf)                        | 时间间隔自注意力   | 可借鉴时间间隔 bucket 或时间衰减序列特征。                                                       |
| [DIN](https://arxiv.org/abs/1706.06978)                                              | 目标感知 attention | 最值得借鉴的是候选目标参与历史聚合，而不是只做候选无关序列摘要。                                 |
| [DIEN](https://arxiv.org/abs/1809.03672)                                             | 兴趣演化           | 可用轻量版本：候选 dst 与最近 k 个历史 dst 的共现/embedding 相似度聚合。                         |

## 工业排序与特征交互

本赛题已经给定候选集，因此排序阶段论文比召回阶段更直接。

| 论文                                                              | 核心思想                         | 对本项目的启发                                         |
| ----------------------------------------------------------------- | -------------------------------- | ------------------------------------------------------ |
| [YouTube DNN](https://research.google.com/pubs/archive/45530.pdf) | 两阶段推荐：召回 + 排序。        | 本赛题跳过召回，重点应放在候选重排序特征和融合头。     |
| [Wide & Deep](https://arxiv.org/abs/1606.07792)                   | 线性记忆交叉 + 深度泛化。        | 统计特征是强规则信号；GNN/序列是表示信号。             |
| [DCN](https://arxiv.org/abs/1708.05123)                           | 显式 feature crossing。          | 当前低维强特征可尝试手工交叉或轻量 CrossNet。          |
| [DCN V2](https://arxiv.org/abs/2008.13535)                        | 改进 CrossNet。                  | 作为融合头升级参考。                                   |
| [xDeepFM](https://arxiv.org/abs/1803.05170)                       | 显式和隐式高阶特征交互。         | 可借鉴交互思想，不建议完整移植。                       |
| [AutoInt](https://arxiv.org/abs/1810.11921)                       | 用 self-attention 学习特征交互。 | 特征维度小，可以试轻量 field attention，但收益需验证。 |

## 前沿生成式推荐

这些论文方向先进，但当前不进入主线。原因是当前比赛接口已经提供 100 个候选，不需要生成式召回。

| 论文                                                                                          | 路线                            | 对本项目的判断                                           |
| --------------------------------------------------------------------------------------------- | ------------------------------- | -------------------------------------------------------- |
| [TIGER](https://arxiv.org/abs/2305.05065)                                                     | 生成式检索、Semantic ID         | 主要解决大规模召回；当前不优先。                         |
| [HSTU](https://arxiv.org/abs/2402.17152)                                                      | 大规模序列 transducer           | 可借鉴 page/list 级训练思想，但不直接移植。              |
| [Mamba4Rec](https://arxiv.org/abs/2403.03900)                                                 | Mamba/SSM 序列推荐              | 只有当 SASRec 成本成为瓶颈且序列特征被证明有效时再考虑。 |
| [LC-Rec](https://arxiv.org/abs/2311.09049)                                                    | LLM + collaborative Semantic ID | 当前没有文本语义，工程跨度大。                           |
| [Diffusion Recommender Models and the Illusion of Progress](https://arxiv.org/abs/2505.09364) | 复现实证                        | 提醒复杂新模型未必强于简单强基线。                       |
| [GenRec 2026](https://arxiv.org/abs/2604.14878)                                               | 工业生成式推荐、偏好对齐        | 可借鉴 list-wise 目标，不直接作为实现路线。              |

## 非 GNN 方向使用规则

- 只能作为隔离实验，不修改默认 CLI。
- 必须同时记录本地 MRR、运行参数、产物路径和线上反馈。
- 没有线上收益或已校准代理验证收益时，不进入默认模型。
- Semantic ID、mixed hard negatives 和 transition 类实验已有失败记录，重新尝试必须从零评估。
