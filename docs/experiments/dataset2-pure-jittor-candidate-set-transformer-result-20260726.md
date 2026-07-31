# Dataset2 纯 Jittor Candidate-Set Transformer 结果

## 结论

Candidate-Set Transformer 已经可以替代 Dataset2 的 Setwise + LightGBM
最终排序头。最佳方案不是单模型，而是两个纯 Jittor 结构的固定概率融合：

- 60%：64 维、4 头、2 层、`mean_max` 行内上下文；
- 40%：同一 Transformer，并增加 32 维 pointwise residual 分支；
- 两个专家都使用 200k×100 完整候选训练；
- 所有可训练参数都属于 `jt.nn.Module`；
- 最终融合是固定 NumPy 概率加权，不含可训练的非 Jittor 模块。

锁定验证集上，融合 MRR 为 `0.5496239795`，当前冠军基线为
`0.5478966506`，增益 `+0.0017273289`。三个时间片均非退化，因此通过离线门禁。
但线上得分为 `1.3524832772351396`，比当前冠军
`1.3545839690981516` 下降 `0.002100691863012`，最终判定为不晋升。

这次结果否证了当前离线门禁，而不是纯 Jittor 实现本身：模型训练和选模使用
`train_end` 因果 encoder 特征，生产推理使用 full-trained encoder。封装阶段已经观测到
两种 encoder 状态在同一历史样本上的最终概率最大差异约 `0.283086`，说明模型输入
存在显著分布迁移。

## 验证结果

| 模型 | Full MRR | 相对当前冠军 |
|---|---:|---:|
| 当前冠军比较基线 | 0.547896651 | — |
| 单个主 Transformer | 0.547617049 | -0.000279602 |
| 单个 residual Transformer | 0.544841944 | -0.003054707 |
| 60/40 固定概率融合 | **0.549623979** | **+0.001727329** |

融合的分片结果：

| 时间片 | 当前冠军 | 纯 Jittor 融合 | Delta |
|---|---:|---:|---:|
| Slice 0 | 0.586688134 | 0.586825089 | +0.000136956 |
| Slice 1 | 0.549013805 | 0.552506646 | +0.003492840 |
| Slice 2（forward gate） | 0.507982026 | 0.509534190 | +0.001552165 |

模型选择只读取前两个时间片；第三片仅做一次 forward gate。

## 新发现

1. Candidate-set attention 本身有效。最佳单模型比旧纯 Jittor Setwise
   `0.546273778` 高约 `+0.001343271`，说明显式候选交互优于仅拼
   `row_mean/row_max` 后的 MLP。
2. 单个模型仍未胜过包含 LightGBM 的当前冠军；真正的增益来自结构多样性。
   residual 专家单独较弱，但与主模型误差互补，固定概率融合后跨三片提升。
3. 随机种子不是这里需要的多样性；结构分支比同结构多种子更有价值。
4. percentile-rank 融合不可用。验证正样本固定在候选位置 0，精确 rank
   相加产生大量并列，而严格 `>` 的 MRR 会让位置 0 获得虚假优势。
   最终实现明确禁止 rank-average，只保留 probability blend。
5. 历史 causal validation cache 与 full-trained production encoder
   不是同一状态。门禁复现必须在锁定特征上比较；生产链只做 shape、有限性、
   概率归一化和无外部 ML 导入验证，不能错误要求逐值相等。

## 合规边界

- 新训练和推理模块不导入 LightGBM/sklearn。
- 输入 63 个特征来自 NumPy 确定性统计或 Jittor 专家，不含 LightGBM
  预测列。
- checkpoint provenance：
  `trainable_frameworks=["jittor"]`，
  `non_jittor_trainable_models=[]`。
- 当前冠军 validation scores 仅进入比较器，不进入模型输入、标签、蒸馏或融合。
- `TemporalHybridRanker` 安装新 head 后清空旧 MLP、LightGBM、Setwise、
  时间融合、窗口融合与路由状态；纯路径 hydrate/predict 不加载这些模块。
- 本次只替换 Dataset2。完整比赛 checkpoint 中 Dataset1 按当前冠军字节保留；
  Dataset1 是否满足同样的纯 Jittor 边界是独立迁移任务。

## 主要交付

- 模型实现：`src/jgrec/rankers/hybrid/candidate_set_transformer.py`
- 生产接入：`src/jgrec/rankers/hybrid/ranker.py`
- 训练入口：`scripts/train_dataset2_candidate_set_transformer.py`
- 融合选择：`scripts/scan_candidate_set_transformer_ensembles.py`
- 融合 checkpoint 构建：
  `scripts/build_candidate_set_transformer_ensemble_checkpoint.py`
- Dataset2 比赛 checkpoint/提交构建：
  `scripts/package_dataset2_candidate_set_transformer.py`
- 测试：`tests/test_hybrid_candidate_set_transformer.py` 和
  `tests/test_hybrid_checkpoint.py`

## 验证证据

- Linux/CUDA/Jittor 合并回归：`20 passed`。
- Ruff：`All checks passed!`
- 固定融合 checkpoint SHA-256：
  `7c6877d8653066d40eb85d059fc0cb51b21f9c90f656597a96931dfd48a0af16`。
- 20k×100 重载复算与选择报告的 MRR 完全一致。
- 在阻断 `lightgbm`、`sklearn` 和 legacy fusion 导入时，
  production ranker checkpoint hydrate/predict 测试通过。
- 真实生产 head 重载最大绝对误差 `2.3841858e-7`，100 候选排序完全一致；
  完整生产链概率和最大误差 `1.1920929e-7`。
- 生产 checkpoint：
  `/home/edu/workspace/jittor-GPNUT1-JGRec/checkpoints/d1_time_ramp_g050_d2_pure_jittor_cst_seed60_20260726.pkl`，
  `5,007,639,306` bytes，SHA-256
  `65f63cb0e66bfcbe5fadb85ede48e81d1ac27b5c2dfd54b5cac2af28021b7553`。
- 提交包含 Dataset1 `61,051` 行、Dataset2 `153,420` 行，`unzip -t`
  两个成员均为 `OK`；SHA-256
  `333b2fae190a2604627a8a0c14e8febd6d39143a01c720e233c713f54fa3baac`。

## 下一步

不晋升本候选，继续保留当前冠军。下一轮不再对同一 causal cache 调 Transformer
结构或融合权重；先解决训练/生产 encoder 状态不一致：

1. 对 63 个特征逐列测量 causal validation 与 full-trained encoder replay 的
   均值、方差、分位数和行内 rank 漂移。
2. 第一优先尝试只保留稳定的确定性特征，并把模型输入改为行内 percentile rank /
   robust z-score，降低 learned embedding 分数的尺度迁移。
3. 建立“同一 encoder 状态生成训练特征和验证特征”的 rolling-origin 协议，
   再决定 Candidate-Set Transformer 是否值得重新训练。
4. 新门禁必须同时通过 causal forward 指标和 encoder-state shift 稳健性检查，
   禁止再用当前单一 cache 的 `+0.0017` 直接授权提交。
