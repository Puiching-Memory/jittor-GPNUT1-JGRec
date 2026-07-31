# Goal Document: Dataset2 纯 Jittor Candidate-Set Transformer

## Go / No-Go

- **Judgment**: Go
- **Reason**: 当前 Setwise 已证明完整 100 候选联合训练有效，但最终冠军仍混入 LightGBM。先建立纯 Jittor 候选集建模与独立复现契约，可以同时验证候选交互的增益和比赛框架合规性。

## Target Outcome

建立一个只含 Jittor 可训练模块的 Candidate-Set Transformer，直接对每组 100 个候选输出排序分数，替代当前 `Setwise + LightGBM` 最终路径；训练、保存、加载和推理均不需要安装 `lightgbm` 或 `sklearn`。当前冠军只作为离线指标比较基线，不作为模型输入、训练标签或运行时依赖。

## Goal Definition

- **Type**: technical / learning / quality
- **Boundary**:
  - 包含纯 `jt.nn.Module` 候选集编码器、训练损失、checkpoint、推理接口和最小训练入口。
  - NumPy 只用于数据读取、固定特征计算、批处理和确定性指标/融合。
  - `sklearn` 只允许出现在离线指标工具中；`lightgbm` 不得进入本目标的训练、checkpoint 和推理调用链。
  - 当前冠军排序只允许作为独立基线 artifact 参与离线指标比较。
- **Non-goals**:
  - 本阶段不删除仓库内用于历史实验复现的 LightGBM 代码。
  - 本阶段不做冠军蒸馏，不把冠军 logits/rank 作为 Transformer 输入或监督。
  - 本阶段不同时引入 source-sequence decoder、多兴趣路由或新的 GNN。
- **Deferred work**:
  - rolling-origin OOF 多折训练、跨窗口模型集成和 source 级联合解码。
  - Dataset1 迁移。
- **Verification rule**: 从纯净进程屏蔽 `lightgbm`/`sklearn` 后，仍可构建模型、执行一次训练步、保存并加载 checkpoint、对 `[batch, 100, feature_dim]` 输入输出有限的 `[batch, 100]` 排序分数；同一模型对候选同步置换时输出必须等变。
- **Evidence source**: pytest 行为测试、依赖扫描、checkpoint 内容检查、离线 MRR/Recall 对比报告。
- **Pass criteria**:
  - 单元与集成测试全部通过。
  - 新路径生产代码不导入 `lightgbm`/`sklearn`。
  - checkpoint 元数据声明 `trainable_frameworks = ["jittor"]`、`non_jittor_trainable_models = []`。
  - 不安装 `lightgbm`/`sklearn` 的隔离验证通过。
  - 固定 Dataset2 验证切片上产出 Transformer 与当前冠军的同口径指标；是否晋升由第三时间片门禁决定。
- **Confidence note**: 代码与依赖测试能高置信证明框架边界；真实榜单收益只能由 rolling-origin 验证和线上提交证明。
- **Judgment owner**: 自动化测试负责代码合规与行为验收；固定验证集和线上榜单负责模型晋升。

## Current State

- Dataset2 当前最佳线上分数为 `1.3545839690981516`，其排序路径包含 Setwise 与 LightGBM，不能作为纯 Jittor 最终提交的合规终点。
- 现有 Setwise 以行内相对特征和 MLP 为主，没有显式的候选间注意力。
- 仓库已有 Jittor 训练、特征缓存和候选组数据管线，可复用数据协议，但必须切断新路径对 LightGBM 模型对象和 sklearn estimator 的依赖。
- 主要风险是 Jittor 注意力算子兼容性、100 候选下显存占用，以及重复调参造成验证集过拟合。

## Plan Rewrite Notes

| Existing item | Decision | Reason |
|---------------|----------|--------|
| `Setwise + LightGBM` 最终融合 | rewrite | 改为纯 Jittor Candidate-Set Transformer 单模型输出 |
| 当前冠军 | keep | 仅保留为同口径离线和线上比较基线 |
| full-100 候选训练协议 | keep | 这是当前最大且稳定的已知收益来源 |
| 多专家置信路由与高维手工门控 | remove | 已多次出现离线正、线上负，不属于本阶段关键假设 |
| rolling-origin 多折 | reorder | 在单折模型与合规契约成立后再扩大训练 |
| source-sequence decoder | defer | 先单独证明候选交互增益，避免归因混乱 |

## Drift Diagnosis

- **Goal drift**: 过去部分实验优化“怎么融合更多专家”，却没有解决最终可训练路径必须纯 Jittor 的目标。
- **Phase drift**: 候选交互、时间窗口、路由和残差曾同时变化，导致无法判断增益来源。
- **Validation drift**: 20k 验证集被反复使用，离线小增益不能直接作为晋升证据。
- **Compatibility drift**: 历史 LightGBM 路径继续保留用于复现，但不得为新路径增加兼容 shim 或隐式 fallback。
- **Cleanup drift**: 不借本目标清理所有历史实验代码和依赖声明。

## Priority Rationale

- 先证明候选同步置换等变和无 LightGBM 训练/加载，这是架构正确性与比赛合规性的最高风险点。
- 再接现有 full-100 数据协议和排序损失，最后才投入大规模 rolling-origin 训练，避免昂贵训练建立在错误接口上。

## Assumptions and Open Decisions

| Item | Status | Impact | Owner / Next step |
|------|--------|--------|-------------------|
| 每组候选数通常为 100，但模型接口应支持动态候选数 | assumed | 决定 attention mask 和 shape 契约 | 由现有数据管线与测试确认 |
| 输入特征不包含运行时生成的 LightGBM 预测 | confirmed | 决定纯 Jittor 边界 | 新训练入口显式校验 feature provenance |
| 采用 pre-norm 自注意力 + pointwise FFN + residual score head | assumed | 首版容量与稳定性 | 单元测试和小样本过拟合验证 |
| 首版损失采用 listwise softmax，可选 LambdaMRR 权重后置 | assumed | 控制实现风险 | 先用最小损失完成端到端闭环 |
| 最终晋升阈值 | unresolved | 决定是否替换当前冠军 | 第三时间片必须不退化，线上必须高于当前分 |

## Phases

### Phase 1: 固定纯 Jittor 行为与依赖契约

- **Purpose**: 在实现前定义模型必须满足的外部行为和合规边界。
- **Entry condition**: 目标文档完成。
- **Phase rules**:
  - 只允许修改测试和目标/TDD 文档；生产实现保持缺失以获得真实 RED。
  - 测试通过 public API 约束 shape、候选置换等变、padding mask、训练步和 checkpoint provenance。
  - RED 必须因目标模块/行为不存在而失败，不接受环境或测试语法错误。
- **Todos**:
  - [ ] 添加 Candidate-Set Transformer shape 与候选置换等变测试。
    - **Surface**: tests
    - **Proof**: 最小 pytest 命令按预期失败。
    - **Depends on**: none
  - [ ] 添加 listwise loss、masked candidate 和 checkpoint 合规测试。
    - **Surface**: tests
    - **Proof**: 失败原因指向缺失行为。
    - **Depends on**: none
- **Exit proof**: 所有目标行为都有正确失败的 RED 证据。
- **Stop condition**: 现有 Jittor 运行环境无法执行任何最小张量测试。

### Phase 2: 最小纯 Jittor 模型闭环

- **Purpose**: 用最小实现让架构、损失与 checkpoint 测试变绿。
- **Entry condition**: Phase 1 RED 已验证。
- **Phase rules**:
  - 所有参数化模块必须继承 `jt.nn.Module`。
  - 不导入或调用 LightGBM/sklearn；不添加历史路径 fallback。
  - 每新增一个行为先回到 RED。
- **Todos**:
  - [ ] 实现候选自注意力、pre-norm block、score head 和 mask。
    - **Surface**: `src/jgrec/rankers/hybrid`
    - **Proof**: shape、等变和 mask 测试通过。
    - **Depends on**: Phase 1
  - [ ] 实现纯 Jittor listwise loss 与单步训练 API。
    - **Surface**: model/training
    - **Proof**: 损失有限、优化后参数发生变化。
    - **Depends on**: attention model
  - [ ] 实现带 provenance 的保存/加载。
    - **Surface**: checkpoint
    - **Proof**: round-trip logits 一致且元数据合规。
    - **Depends on**: model config
- **Exit proof**: 定向测试全绿且生产调用链静态扫描无非 Jittor estimator。
- **Stop condition**: 为通过测试必须引入 Jittor 不支持的核心算子，且无法用基础张量操作等价实现。

### Phase 3: 接入 full-100 训练与比较基线

- **Purpose**: 让新模型可复用现有 Dataset2 候选组缓存训练，并与冠军同口径比较。
- **Entry condition**: Phase 2 全绿。
- **Phase rules**:
  - 数据 artifact 必须记录 feature provenance；发现 LightGBM 预测列立即停止。
  - 冠军预测只能进入 evaluator，不进入训练 dataset/model。
  - 先做小样本过拟合，再做完整训练。
- **Todos**:
  - [ ] 增加纯 Jittor训练 CLI/config 和现有 full-100 cache adapter。
    - **Surface**: scripts / training pipeline
    - **Proof**: 小样本可将训练损失显著压低并生成 checkpoint。
    - **Depends on**: Phase 2
  - [ ] 增加冠军 artifact 的只读比较报告。
    - **Surface**: evaluation
    - **Proof**: 同一 query 集输出 baseline/new/delta，训练调用链不读取 baseline。
    - **Depends on**: train/predict
- **Exit proof**: 无 LightGBM/sklearn 环境完成 train-save-load-predict-evaluate dry-run。
- **Stop condition**: 现有缓存无法证明不含 LightGBM 派生特征。

### Phase 4: rolling-origin 训练与晋升门禁

- **Purpose**: 用未反复调参的时间切片判断 Candidate-Set Transformer 是否真正替代冠军。
- **Entry condition**: Phase 3 dry-run 通过。
- **Phase rules**:
  - 前两时间片选择配置，第三时间片只做一次不可见门禁。
  - 第三片退化则不得提交；不以反复调第三片救回结果。
  - Dataset1 字节保持不变。
- **Todos**:
  - [ ] 训练 4 折 rolling-origin 纯 Jittor模型并记录每折指标。
    - **Surface**: remote training / experiment artifacts
    - **Proof**: fold-level MRR/Recall 与均值、方差。
    - **Depends on**: Phase 3
  - [ ] 通过门禁后生成提交包并与 `1.3545839690981516` 比较。
    - **Surface**: result package
    - **Proof**: SHA256、离线报告、线上分数。
    - **Depends on**: rolling-origin gate
- **Exit proof**: 纯 Jittor提交线上分数超过当前冠军，或留下明确的否证结论与可复现实验。
- **Stop condition**: 第三时间片明显退化，或新模型增益不足以覆盖折间方差。

## Dry-Run Findings

- 不能直接把现有冠军 logits 当输入，否则虽然运行时可卸载 LightGBM，训练因果链仍不纯。
- 必须先核验 full-100 cache 的列来源；复用分组与原始专家特征可以，复用 LightGBM 输出列不可以。
- Jittor 版本的 attention 应优先用基础矩阵乘、softmax 和 mask 实现，减少高级 API 版本差异。
- 昂贵训练必须后置到小样本过拟合与 checkpoint round-trip 之后。

## Final Validation

- `uv run pytest <candidate-set-transformer tests>`
- 在阻断 `lightgbm` 与 `sklearn` 导入的子进程内完成 checkpoint round-trip 和预测。
- 静态扫描新生产调用链无 `lightgbm`/`sklearn` import。
- rolling-origin 第三片不退化，且线上提交高于 `1.3545839690981516` 才晋升为新冠军。

## First Execution Step

读取现有 Setwise 数据 shape、Jittor 模型约定和 checkpoint 接口，随后先添加候选同步置换等变的失败测试。
