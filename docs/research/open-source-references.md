# 开源参考项目

记录日期：2026-05-27

本文档记录后续优化赛道一动态推荐模型时可参考的项目和本地代码。参考项目只用于理解方法和工程结构，比赛提交不得使用官方数据集之外的外部数据。

## 本地参考

### JittorGeometric 官方仓库

- 本地路径：`third_party/JittorGeometric`
- 上游链接：https://github.com/AlgRUC/JittorGeometric
- 相关性：高
- 说明：本仓库已通过 `pyproject.toml` 将 JittorGeometric 配置为本地 editable 依赖。
- 可参考内容：图数据结构、时序图 dataloader、动态图模型示例、推荐模型示例。

### 推荐系统示例

- 本地路径：`third_party/JittorGeometric/examples/recsys_example.py`
- 相关性：中
- 说明：展示 LightGCN、SimGCL、XSimGCL、DirectAU 等推荐模型的训练与评估方式。
- 可参考内容：用户物品图建模、BPR 类训练、Top-K 指标组织。
- 注意事项：示例面向静态推荐数据集，不能直接覆盖当前赛题的时序查询和候选重排序格式。

### 时序图示例

- 本地路径：
  - `third_party/JittorGeometric/examples/jodie_example.py`
  - `third_party/JittorGeometric/examples/dygformer_example.py`
  - `third_party/JittorGeometric/examples/graphmixer_example.py`
  - `third_party/JittorGeometric/examples/sasrec_example.py`
- 相关性：高
- 说明：这些示例覆盖 JODIE、DyGFormer、GraphMixer、SASRec 等时序交互建模思路。
- 可参考内容：时间编码、邻居采样、负采样、序列/动态图 embedding、链接预测头。
- 注意事项：示例数据接口与比赛 CSV 不一致，应优先复用模型思想，不直接复用数据加载逻辑。

## 方法参考方向

### 统计重排序

- 当前实现：`src/jgrec/rankers/temporal_graph/ranker.py`
- 适用阶段：MVP、快速提交、特征验证。
- 优化方向：
  - 用训练集尾部构造验证集。
  - 对当前 6 个特征学习权重。
  - 增加二跳共现、源节点活跃度、时间桶热度等特征。

### 序列推荐

- 可参考示例：`sasrec_example.py`
- 适用场景：同一源节点存在较长历史交互序列。
- 优化方向：
  - 按源节点聚合目标序列。
  - 对候选目标做序列条件打分。
  - 使用时间间隔或位置编码补充时序信息。

### 时序图神经网络

- 可参考示例：`jodie_example.py`、`graphmixer_example.py`、`dygformer_example.py`
- 适用场景：需要同时利用图结构演化和时间信息。
- 优化方向：
  - 统一比赛 CSV 到 `TemporalData` 风格结构。
  - 训练链接预测模型。
  - 测试时只对给定 100 个候选节点重排序。

### 静态图推荐

- 可参考示例：`recsys_example.py`
- 适用场景：重复边比例高、历史连接记忆较强的数据集。
- 优化方向：
  - 使用全量历史图训练 LightGCN 类模型。
  - 将静态图分数作为候选重排序特征。
  - 与时序统计特征做融合。

## 当前建议

短期不要直接上复杂动态图模型。更稳的顺序是：

1. 构造本地时间切分验证集并实现 MRR。
2. 学习当前统计特征权重。
3. 增加序列和二跳共现特征。
4. 再引入 JittorGeometric 时序模型。

这样可以避免模型复杂度上升后没有可靠验证闭环。

## 全网类似比赛开源调研（2026-07-24）

### 与本赛题最接近的基准：TGB

- 仓库：https://github.com/snap-stanford/tgb ；榜单：https://tgb.complexdatalab.com/docs/leader_linkprop/
- 任务格式与本赛题一致：给定 `(src, time)` 和固定数量候选负样本，按 MRR 评测动态链接预测。
- tgbl-wiki-v2 榜单（2026-01 快照）：TPNet 0.827 > Heuristic(LocalGlobal) 0.821 > HyperEvent 0.810 > DyGFormer 0.798 > NAT / Base3 / DyGMamba / TNCN / CAWN，EdgeBank(tw) 仅 0.571。
- 结论：榜单前列同时存在深度模型与纯启发式。本项目 hybrid（强统计记忆 + 融合）路线与榜首思路一致，应继续先强化统计与共现特征。

### 纯启发式 / 无训练方法（与 stats 塔直接相关）

1. On the Power of Heuristics in Temporal Graphs（King AI Labs，arXiv:2502.04910）
   - 四个启发式：LR（`(u,v)` 最近一次交互的近期度）、GR（`v` 最近交互时间）、LP（`(u,v)` 交互次数）、GP（`v` 总交互次数），以及组合 Combined；热度用于打破并列。
   - 成绩：tgbl-wiki 上 LR 单独 0.817，tgbl-coin 上 Combined 0.899，超过多数神经网络。
   - 可学习：把 LR/GR/LP/GP 各做成一个候选级特征，再加一个组合特征；核对当前 stats 是否覆盖 GR（目标节点最近活跃时间）。
2. Base3（Emma Kondrup, Mila，arXiv:2506.12764）
   - EdgeBank（边重现记忆）+ PopTrack（节点流行度）+ t-CoMem（时序共现与邻域活跃度记忆）插值融合，完全无训练。
   - 成绩：tgbl-coin 0.773、tgbl-flight 0.794，超过 DyGFormer/TGN；在更难的负采样协议下优势更明显。
   - 可学习：保留一个"无训练插值基线"作为融合层的兜底对照；t-CoMem 的邻域活跃度可做成特征。
3. ESANN 2024《Link prediction heuristics for temporal graph benchmark》
   - 在时间窗邻居上计算 PA（优先连接）等结构启发式，tgbl-wiki 上超过 TGN。
   - 可学习：候选与 src 在时间窗内的共同邻居数 / PA 分数是低成本结构特征。

### 深度时序图模型（若要新增模型塔）

1. TPNet（NeurIPS 2024，https://github.com/lxd99/TPNet ）：带时间衰减的 temporal walk matrix + 随机特征传播，tgbl-wiki-v2 第一，最高 33x 加速。可借鉴"带时间衰减的二跳游走计数"做成轻量特征。
2. TNCN（https://github.com/GraphPKU/TNCN ）：时序多跳共同邻居 pairwise 表示，比 GNN 基线快 6.4 倍。可借鉴"src 与候选的时间窗共同邻居计数"特征。
3. DyGFormer / DyGLib（https://github.com/yule-BUAA/DyGLib ）：邻居共现编码（src 与 dst 历史邻居重叠）+ 长序列 patching；DyGLib 统一了 8+ 模型的训练评测管线。JittorGeometric 本地已有 `dygformer_example.py` 可对照。
4. 备选：DyGMamba、HyperEvent（https://github.com/jianjianGJ/HyperEvent ）、NAT、GraphMixer。
5. 教训（DyGLib 论文）：不同实现下 baseline 结果漂移大，统一验证协议比堆模型更重要。

### 候选集重排序类比赛（召回 + 排序范式）

1. Kaggle OTTO – Multi-Objective Recommender System（2022-23）
   - 代表仓库：https://github.com/TheoViel/kaggle_otto_rs （第 3 名）、https://github.com/Datadote/otto23 、cdeotte 公开 kernel（Candidate ReRank Model LB 0.575）。
   - 套路：多种 item-item 共现矩阵（12-24h 时间窗、按行为类型加权）+ session 内位置/时间权重 + Word2Vec 相似度生成候选；LightGBM/XGB Ranker 用数百个聚合特征重排。
   - 细节：位置权重 `np.logspace(base=2)-1`；共现对距离越远权重越低；训练 ranker 时删除无正样本的行；负采样约 1:40。
   - 可学习：补充"候选与 src 历史序列各项的加权共现聚合分"特征；融合训练剔除无正例行。
2. KDD Cup 2020 Debiasing（美团冠军方案）
   - 方案文：https://tech.meituan.com/2020/08/20/KDD-Cup-Debiasing-Practice.html ；开源：https://github.com/LogicJake/2020_KDD_Debiasing_TOP13 、KDDCUP_2020_Debiasing_1st_Place。
   - item-CF 边权 = 共现频数 × 时间间隔因子 × 用户活跃度惩罚 × 商品流行度惩罚；正序权重大于逆序（"BC" > "BA"）；二跳 i2i 路径分 = 边权乘积求均值。
   - u2i2i 建模：把用户下一次点击转成 i2i 样本标签；流行度加权损失与排序后处理消偏；多路召回合并去重，删除召回未命中样本。
   - 可学习：结构共现特征加入时间/位置衰减与方向不对称；热门惩罚项；二跳路径聚合特征。
3. WSDM Cup 2022 XMRec Top1（https://github.com/bottergpt/wsdm2022-xmrec-top1-solution ）：itemCF baseline + LightGCN 的参考实现组合。

### 工程与评测经验

- TGB 的历史负采样（historical negatives）更接近线上候选分布；本地验证集应模仿线上 100 候选的构成（历史交互节点 + 随机节点混合），否则线上线下不齐。
- B 榜为百万节点 / 千万边：TPNet、TNCN 都把昂贵的 pairwise 计算变成可增量维护的字典/矩阵，与本项目 byte-budget LRU 缓存思路一致。
- 赛题提示重复边比例高，EdgeBank 类记忆基线是下限：任何新模型塔必须先在线上打过纯统计融合再保留。

## 落地记录（2026-07-24）

已把上述学习点落地为两个新模块：

1. `src/jgrec/rankers/hybrid/heuristic.py` —— `HeuristicTower`（14 维，接入 encoder 位于 structure 塔之后、source_profile 之前）：
   - 四象限启发式（LR/GR/LP/GP + Combined），Combined 用 LR 主导、热度打破并列；
   - 时间窗共同邻居（短/中/长窗 + 衰减加权 `cn_decay`），对应 TNCN/DyGFormer 的邻居共现信号；
   - 方向化时间衰减共现（`cooccur_fwd`/`cooccur_bwd`，借鉴美团 itemCF 的方向与衰减）与 2 跳路径分（`hop2_score`，TPNet 时序游走矩阵的 2 跳截断版）。
   - 复用 structure 塔的 `TemporalInteractionIndex`（shared_index 注入），不重复建图；确定性前缀缓存同步覆盖。
   - 注意：二部图（dataset1）上 `hop2_score` 自动退化为 0（无 2 跳路径），属预期；非二部图（dataset2）才有信号。
2. `src/jgrec/rankers/hybrid/interpolation.py` —— Base3 风格无训练插值兜底（EdgeBank + PopTrack + tCoMem，凸组合 + 行内归一），`fit_weights_on_validation` 在验证集网格搜索 (α,β,γ) 使 MRR 最大。不接入融合，作为提交前 sanity floor 与应急提交。

融合候选集 `_feature_masks` 与 `_config_for_selected_features` 已同步加入 heuristic 区间，可消融验证该塔增量。特征总维度 77（原 63 + heuristic 14）。

验证候选构造对齐线上协议：`test_candidate_negative_ratio` 机制已具备（负采样混入 test 候选），比例需按线上 100 候选真实构成标定后开启。

下一步：全量 `uv run jgrec-build` 训练融合，对比融合层对 heuristic 特征的权重学习与线上 MRR 锚点（当前最优 1.2044）。

## 环境修复（2026-07-24）

Jittor CUDA 编译此前在本容器失败（`nvcc` 返回 256）。根因：CUDA 11.8 的 nvcc 不识别 Ubuntu 24.04 默认 GCC 13 的 `__builtin_dynamic_object_size` builtin（`/usr/include/.../string_fortified.h`），导致 `nan_checker.cu` 等所有 `.cu` 编译失败。修复：host 编译器改用 CUDA 11.8 兼容的 `g++-11`（同时作为 nvcc `-ccbin`）。已在 `sitecustomize.py` 末尾自动设置 `cc_path=/usr/bin/g++-11`（若存在，否则 g++-12），无需手动环境变量。修复后 Jittor core 用 overlay nvcc(11.8.89) + g++-11 编译成功（sm_80），全量 195 个测试通过。旧 g++13 编译缓存需清除：`rm -rf .venv/jittor_home/.cache/jittor/*/g++13*`。

## 代码结构（2026-07-24 重构）

新模型代码已从 `hybrid` 独立为平行包 `src/jgrec/rankers/hybrid_heuristic/`：

- `hybrid` 包保持原状（线上 1.2044 基线对应的实现），不受新实验影响。
- `hybrid_heuristic` 为完整自包含副本（复制 hybrid 全部源码），额外集成 `HeuristicTower` 与 `interpolation` 兜底，与 hybrid 零依赖、可独立演化。
- 通过 `--model hybrid-heuristic` 选择（已注册到 `rankers/registry.py` 并在 `cli.py` 的 `_ranker_config` 中走 `hybrid_heuristic.config.TrainingConfig`）。
- 新塔单元测试在 `tests/test_hybrid_heuristic_tower.py`、`tests/test_hybrid_heuristic_interpolation.py`；原 `tests/test_hybrid_*.py` 未改动。
- 用法：`uv run jgrec-build --model hybrid-heuristic ...`（其余参数与 hybrid 相同，额外支持 `--disable-heuristic` 等 heuristic_* 配置）。
