# Dataset2 Source-Conditioned CST A/B/C/D Goal

## 结论

**Go。** 在统一的纯 Jittor 协议下完整运行 A/B/C/D 四个结构消融，先回答“候选 ID、源序列、候选自注意力各自是否带来可迁移增益”，再决定是否生成新的 Dataset2 提交。

## 1. 可验证目标

在 Dataset2 的 `200k × 100` 完整候选训练集上，完成四种模型的三折 rolling-origin 评估：

| 变体 | 候选 item ID | source 历史序列 | candidate self-attention |
|---|---:|---:|---:|
| A | 否 | 否 | 是 |
| B | 是 | 否 | 是 |
| C | 是 | 是 | 否 |
| D | 是 | 是 | 是 |

成功必须同时满足：

1. A/B/C/D 使用相同数据、时间折、优化器、损失、训练预算和早停规则；
2. 候选 item ID 与 source 序列严格共享同一张 Jittor item embedding；
3. source 序列只包含 `event_time < query_time` 的事件，禁止同时间和未来事件；
4. 候选维没有位置编码，候选同步置换后 logits 必须同步置换；
5. 所有影响最终排序的可训练模块均为 `jt.nn.Module`，不依赖 sklearn/LightGBM；
6. 前两折选择出的胜者在第三折通过门禁后，才允许全量重训和读取外部 20k；
7. 输出逐折 tie-neutral MRR、分段指标、训练报告、checkpoint/hash 和最终判断。

## 2. 边界

### 包含

- 复用现有 Dataset2 `200k × 100` raw feature cache；
- 构建候选 item token 与严格因果的 source recent-64 序列缓存；
- 实现共享 item embedding、source sequence encoder/cross-attention，以及可开关的 candidate self-attention；
- 三个 timestamp-aligned expanding-window folds；
- A/B/C/D 共 12 次折内训练；
- 胜者的第三折门禁；
- 仅在门禁通过时，全量 200k 重训并对外部 20k 评估一次；
- 保存可复现的纯 Jittor 训练/推理产物。

### 不包含

- 不改 Dataset1；
- 不使用 OOF expert logits 作为输入；
- 不引入 LightGBM 或 sklearn 训练依赖；
- 不用随机种子集成掩盖结构差异；
- 不在第三折或外部 20k 上调参；
- 门禁失败时不生成提交包。

## 3. 当前状态

- 当前线上冠军：`1.3545839690981516`。
- Dataset2 当前离线冠军参考：tie-neutral MRR `0.5478966505694405`。
- 现有纯 Jittor CST 只使用候选 raw features 与行内相对特征，没有 candidate item ID，也没有 source 历史序列。
- 现有 OOF 实验暴露出大量正负候选完全同分；rank/percentile 压缩不能补回缺失的 item/source 信息。
- 已有 `GRUSequenceModel` 和 CST 基础组件可复用，但缺少“共享 item embedding + source-to-candidate 条件化”的统一模型。

## 4. 固定实验协议

### 时间折

先按 query timestamp 排序，并只在 timestamp 边界切分，目标比例为：

- Fold 0：前约 40% 训练，随后约 20% 验证；
- Fold 1：前约 60% 训练，随后约 20% 验证；
- Fold 2：前约 80% 训练，最后约 20% 作为不可见门禁。

若目标位置落在相同 timestamp 内，切点向后移动到下一 timestamp，确保同一时间不跨 train/validation。

### 训练

- positive 始终由候选集合标签确定，不从候选位置推断；
- loss：相同的 listwise cross-entropy；
- 结构外超参数固定：model dim、head 数、dropout、batch size、学习率、最大 epoch、patience；
- 以 tie-neutral validation MRR 早停；
- 不对四个变体分别调参；
- 只使用一个固定 seed；本实验测结构，不测 seed ensemble。

### 选择与门禁

1. 仅依据 Fold 0 与 Fold 1 的平均 MRR 选择 A/B/C/D 胜者；
2. Fold 2 结果在选择完成后才揭示；
3. 胜者必须在 Fold 2：
   - 不低于 A；
   - 相对 A 的增益方向与前两折一致；
   - source activity 分段没有不可接受的集中退化；
4. 门禁通过后，以前两折选出的固定 epoch 规则全量重训；
5. 外部 20k 只评估锁定胜者一次，并与冻结冠军比较。

## 5. 每阶段规则

### 阶段一：数据与契约

- 校验 feature/src/time/candidate sidecar 行数和顺序；
- item vocabulary 仅表示稳定 ID 映射，不携带频率或未来标签统计；
- 序列缓存必须通过严格因果测试和 hash/report 校验。

### 阶段二：模型实现

- A 必须退化为 raw/relative-feature CST；
- B 只增加 candidate ID；
- C 只在 B 上增加 source sequence，且关闭 candidate self-attention；
- D 在 C 上开启 candidate self-attention；
- 四者共享同一训练器，不得复制成四套易漂移实现。

### 阶段三：评估

- 先跑小样本 smoke；
- 再跑 12 个正式折实验；
- 结果文件在所有变体完成前不得宣布胜者；
- 报告均需记录代码版本、配置、数据 hash、seed 和运行环境。

### 阶段四：晋级

- 第三折失败：停止，不读取外部 20k；
- 第三折通过：只全量训练一个锁定胜者；
- 外部 20k 未超过冻结冠军：保留研究结论，不打包；
- 超过冻结冠军且合规检查通过：才生成候选提交。

## 6. 待办与证据

- [ ] 冻结三折切点与缓存清单  
  证据：`fold-manifest.json`、sidecar shape/hash。
- [ ] RED：模型/缓存契约测试先失败  
  证据：TDD 报告中的失败命令和失败原因。
- [ ] GREEN：实现共享 embedding、因果序列、四变体开关  
  证据：目标测试通过。
- [ ] Smoke：四变体均能在 Linux/CUDA 前向、反向、保存、加载  
  证据：smoke report。
- [ ] 正式运行 Fold 0/1  
  证据：8 份训练报告与预测 hash。
- [ ] 锁定胜者并运行 Fold 2  
  证据：选择清单在 Fold 2 指标之前落盘。
- [ ] 若通过门禁，全量重训并读取外部 20k 一次  
  证据：full-training report、external evaluation report。
- [ ] 汇总结论与下一步  
  证据：最终实验报告。

## 7. Dry-run

给定任一 batch：

1. raw candidate features 形状为 `[B, 100, 63]`；
2. candidate token 形状为 `[B, 100]`；
3. source recent-64 token/time bucket/mask 形状分别为 `[B, 64]`；
4. raw features 映射到 candidate hidden；
5. B/C/D 从共享 item embedding 读取 candidate 表示；
6. C/D 编码严格因果 source 序列，并让每个 candidate cross-attend source memory；
7. A/B/D 执行 candidate self-attention，C 跳过；
8. pointwise head 输出 `[B, 100]` logits；
9. 同步置换 candidate raw/token 后，输出只做相同置换；
10. listwise loss 更新模型，validation 用 tie-neutral MRR 选 epoch。

失败路径：

- sidecar 对不齐：停止构建；
- 历史包含 `>= query_time`：测试失败并停止；
- 置换不等价：模型不得进入训练；
- 任一变体 NaN/无法 reload：12-run 不启动；
- Fold 2 门禁失败：不读取外部 20k。

## 8. Go / No-Go

**Go。** 当前最大的不确定性不是再加一个融合器，而是缺少 candidate identity 和 source history 是否真能打破同分。A/B/C/D 是最小且可归因的完整证据链；若 D 不赢，也能明确判断收益来自 ID、序列还是 candidate interaction，避免继续堆不可解释模块。
