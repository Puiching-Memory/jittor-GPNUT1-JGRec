# 第六届计图（Jittor-7）赛道一竞争对手分析

> 更新时间：2026-07-15
> 分析范围：GitHub 公开仓库
> 本队仓库：
> - GitHub: https://github.com/Puiching-Memory/jittor-GPNUT1-JGRec
> - GitLink: https://gitlink.org.cn/Puiching-Memory/jittor-GPNUT1-JGRec
> - 本队 A 榜 MRR：1.2044

## 1. 已发现竞争对手清单

| 仓库                                                                                                                                        | 战队/作者  | 核心方案                                                                  | 已披露 A 榜 MRR |
| ------------------------------------------------------------------------------------------------------------------------------------------- | ---------- | ------------------------------------------------------------------------- | --------------- |
| [chaoheng666/jittor](https://github.com/chaoheng666/jittor)                                                                                 | 巴巴博弈   | dataset1 纯规则 / dataset2 Jittor 时序推荐 + 规则融合                     | 未披露          |
| [Zhulina985/jittor-track1-v6](https://github.com/Zhulina985/jittor-track1-v6)                                                               | —          | LinkModelV6：V3 骨干 + Transformer Self-Attn + SE + Cross-Attn + 轻量 GAN | ~1.110          |
| [Winston2003/jittor-handcrafted-ai-temporal-link-prediction](https://github.com/Winston2003/jittor-handcrafted-ai-temporal-link-prediction) | 手工智能   | 改进版 CRAFT：放大 hidden/layers/heads + Cosine LR                        | 1.084           |
| [escape9th/jittor-track1](https://github.com/escape9th/jittor-track1)                                                                       | escape9th  | 纯启发式：时序衰减 + 频次 + 热门度                                        | ~1.08           |
| [Flan246/jittor-Floatouble-track1](https://github.com/Flan246/jittor-Floatouble-track1)                                                     | floatouble | 基于 JittorGeometric 的 CRAFT baseline + warmup                           | 未披露          |

## 2. 各对手详细分析

### 2.1 巴巴博弈（chaoheng666/jittor）—— 最危险对手

- **整体策略**：按数据集特性分治。
  - `dataset1`：repeat-heavy 场景，使用保守的统计/规则 ranker（`base_intensity_v3 + manual_rule`），避免深度模型带来的过拟合和训练开销。
  - `dataset2`：new-pair 场景，使用 Jittor 时序推荐模型 + 规则融合。
- **dataset2 时序推荐模型**（`src/dataset2/temporal_recommender.py`）：
  - 用户状态：src embedding + 历史 dst 序列加权平均 + 时序 gap 编码 + 全局时间特征。
  - 打分：点积 + dst bias + 候选 pair MLP。
  - 训练：full/sampled softmax + BPR 辅助 loss + rerank loss。
  - 负采样：hard negatives（来源历史 + 全局热门）+ 采样修正 `D2_SAMPLED_CORRECTION`。
  - 推理：模型分与规则分先做 z-score 再融合；未知 dst 单独处理（demote / fallback / boost）。
- **工程亮点**：
  - 使用 `split` 列做时间序验证。
  - 按桶评估：overall / repeated / new_pair / cold_dst / cold_src / no_history。
  - 自动搜索规则与模型的融合权重。
- **威胁评估**：高。方法论和本队高度一致，但工程细节更细，且未披露分数，可能接近或超过本队。

### 2.2 Zhulina985（jittor-track1-v6）

- **模型**：在 V3 骨干（GRU + 手工特征 + 转移分 + 难负采样）上叠加：
  1. Transformer Self-Attention 残差。
  2. SE 通道注意力。
  3. Cross-Attention（候选 query × 历史 K/V）。
  4. 轻量 GAN 生成难负样本。
- **训练**：CrossEntropy + 难负采样（popularity alias + 历史负样本 + GAN 难负）。
- **推理**：规则分与模型分 z-score 融合，dataset1 blend=0.45，dataset2 blend=0.25。
- **分数**：V6 无 GAN 约 **1.110**，带 GAN 版同样约 ~1.110；V3 约 1.109。
- **结论**：注意力叠加收益有限，远低于本队 1.2044。README 自述纯 CRAFT / 纯 TriAttn 曾掉到 ~0.99。

### 2.3 手工智能（Winston2003/jittor-handcrafted-ai-temporal-link-prediction）

- **模型**：改进版 CRAFT（JittorGeometric）。
- **改进点**：
  - dataset1: hidden=128，layers=3，heads=4。
  - dataset2: hidden=96，layers=2，heads=2（适配 6GB 显存）。
  - 手动实现 Cosine 预热 + 退火学习率。
  - 断点续训支持。
- **分数**：A 榜 **1.084**，排名 110。
- **结论**：纯靠放大模型容量的 baseline 改进，上限明显，不构成直接威胁。

### 2.4 escape9th（jittor-track1）

- **方法**：纯启发式，无神经网络。
  - 时序衰减（Recency）。
  - 交互频次（Frequency）。
  - 全局热门度（Popularity）。
- **分数**：v1 Max-normalize ~1.08；v2 Softmax ~0.9。
- **结论**：无深度学习，分数已接近天花板，无威胁。

### 2.5 floatouble（Flan246/jittor-Floatouble-track1）

- **内容**：基于 JittorGeometric，包含 warmup 和 CRAFT baseline。
- **分数**：未披露。
- **结论**：信息不足，持续关注即可。

## 3. 横向对比

| 维度          | GPNUT1（本队）    | 巴巴博弈        | Zhulina985 | 手工智能 |
| ------------- | ----------------- | --------------- | ---------- | -------- |
| 已披露分数    | **1.2044**        | 未知            | ~1.110     | 1.084    |
| 核心架构      | hybrid 多特征融合 | 分场景定制      | 三重注意力 | 大 CRAFT |
| dataset1 策略 | hybrid 自动选择   | 纯规则          | 混合模型   | 大模型   |
| dataset2 策略 | hybrid 自动选择   | 时序推荐 + 规则 | 混合模型   | 小模型   |
| 工程成熟度    | 高                | 极高            | 中         | 中       |

## 4. 关键结论

1. **本队目前分数显著领先已披露对手**：1.2044 比 Zhulina985 的 ~1.110 高约 8.5%，比手工智能 1.084 高约 11%。
2. **最大潜在威胁是巴巴博弈**：其分场景策略和本队一致，工程细节更细，但尚未披露分数。
3. **单纯堆叠注意力收益有限**：Zhulina985 的尝试说明 V3 + 三重注意力无法突破 ~1.11，说明特征工程和分场景策略比单纯增加注意力模块更重要。
4. **对手普遍在做的事本队已经在做**：规则+模型融合、难负采样、历史序列建模、分场景策略。本队领先可能来自更广的特征融合（stats + candidate_prior + structure + two_tower + graph + sequence）。

## 5. 可借鉴的改进点

从巴巴博弈方案中可进一步验证或引入：

1. **dataset2 更细粒度的评估桶**：增加 repeated / new_pair / cold_dst / cold_src / no_history 的 MRR 监控，定位短板。
2. **候选级 pair MLP**：在模型打分中加入候选与 user_state 的 pair-wise 交互项。
3. **未知 dst 处理策略**：demote / fallback / boost 等策略对 cold 候选可能有效。
4. **自动融合权重搜索**：在验证集上搜索规则与模型分的最佳融合权重。

## 6. 跟踪计划

- 持续监控巴巴博弈仓库更新。
- 比赛结束前每周复查一次 GitHub 是否有新仓库出现。
- 如本队方案有重大更新，同步更新本文件。
