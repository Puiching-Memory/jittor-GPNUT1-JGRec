# Goal Document: Dataset2 多时间尺度 OOF 路由 + top-k LambdaMRR 联训

## Go / No-Go

- **Judgment**: Go
- **Reason**: 多时间尺度 OOF residual、纯 Jittor 路由器和独立 LambdaMRR
  组件均已存在；缺口集中在可测试的联合目标、联合 checkpoint 和严格时间门禁。

## Target Outcome

产出一个纯 Jittor 联训模型：以 short 为默认专家，使用 short/medium/long
多时间尺度 OOF 信号预测高置信路由，同时仅在冠军 top-k 内学习
LambdaMRR residual 排序；路由损失与排序损失共同更新共享主干。模型通过
selection 后才读取最终时间 gate，并生成可复现评估报告。

## Goal Definition

- **Type**: technical / learning / quality
- **Boundary**: Dataset2；复用现有多时间尺度 OOF 缓存；新增联合模型、损失、
  runner、测试、checkpoint 和内部时间门禁。所有影响最终排序的可训练模块必须使用
  `jt.nn.Module`。
- **Non-goals**:
  - 不改变 Dataset1 包或分数。
  - 不允许 LightGBM/sklearn 进入训练或推理。
  - 不在 selection 选型完成前读取最终 gate 指标。
  - 本阶段不因内部正增益自动替换线上冠军或生成正式提交。
- **Deferred work**:
  - 全量重训、测试集推理和候选提交包只在最终 gate 达到准入条件后进行。
  - 更大的 Transformer 主干和额外序列特征不属于本轮。
- **Verification rule**: 单元测试证明联合梯度、LambdaMRR 权重、bounded top-k
  安全契约和 checkpoint replay；runner 先锁定 selection 配置，再一次性读取 gate。
- **Evidence source**: pytest RED/GREEN 记录、训练日志、selection lock、gate report、
  evaluation report、SHA-256 checkpoint replay。
- **Pass criteria**:
  - 联合 loss 中 router loss 和 LambdaMRR loss 均非零且共同更新共享参数；
  - 未路由行逐值等于 short，top-k 外逐值等于 short，改分幅度不超过冻结 cap；
  - selection `delta_mrr > 0` 且每个 selection 时间子片最差增益不低于冻结容差；
  - 配置在读取 gate 前锁定，gate 不参与选型；
  - gate `delta_mrr > 0`、安全审计通过、checkpoint replay 最大误差不超过
    `2e-6`；
  - 训练框架清单仅包含 Jittor。
- **Confidence note**: 最终 gate 是最近、不可见时间段，能验证前向泛化；但内部
  MRR 仍是线上榜单的代理，因此即使通过也只产生候选，不宣称线上必涨。
- **Judgment owner**: 自动测试与冻结时间门禁共同判定技术完成；线上替换仍由外部
  评测结果判定。

## Current State

- 已有 short/medium/long 多时间跨度 OOF residual，整体离线增益分别约
  `+0.003548 / +0.002542 / +0.001802`。
- 已有纯 Jittor 高置信路由器，但其训练目标是加权 MSE；top-k 只是 bounded
  projection，不是 LambdaMRR 排序学习。
- 已有独立 Champion top-k LambdaMRR residual：top20 selection 最佳约
  `+0.000361`，未过旧门槛且未读取最终 gate。
- 风险是路由正样本稀疏、联合损失尺度失衡，以及 top-k 排序头覆盖过大时重新引入
  旧实验的负迁移。

## Priority Rationale

- 先以测试固定“真正联训”的梯度与安全语义，避免只把两个阶段串联却误称联训。
- 再做最小联合模型和 replay，最后才运行昂贵训练；这样最早暴露泄漏和契约问题。

## Assumptions and Open Decisions

| Item | Status | Impact | Owner / Next step |
|---|---|---|---|
| 默认专家固定为 short | confirmed | 保持最强单尺度基线 | runner 固化 |
| 联训共享主干，分别输出 2 路 advantage 与 top-k residual | assumed | 定义“联合反向传播” | 用梯度测试验证 |
| 初始 top-k 搜索为 10/20，cap 为 0.01/0.02 | assumed | 控制搜索量与破坏面 | 仅用 selection 选择 |
| 排序标签可从 OOF 正样本位置构造 | confirmed | LambdaMRR 不需要读取测试标签 | runner 审计对齐 |
| gate 最差子片允许的冻结容差 | assumed | 避免单个小子片噪声否决微增益 | 在读 gate 前固定为 `-5e-5` |

## Phases

### Phase 1: 用测试冻结联合训练契约

- **Purpose**: 证明模型确实联合优化路由与排序，而非两个互不相关的后处理器。
- **Entry condition**: 目标文档已提交到工作树。
- **Phase rules**:
  - 只新增测试；生产代码尚不存在时测试必须因缺少目标 API 或行为失败。
  - 测试覆盖联合梯度、LambdaMRR 对 top-ranked swap 的权重、bounded top-k
    投影和 checkpoint replay。
- **Todos**:
  - [ ] 新增联合 loss 与联合模型行为测试。
    - **Surface**: `tests/test_hybrid_joint_oof_lambdamrr.py`
    - **Proof**: 最小 pytest 命令出现预期 RED。
    - **Depends on**: none
- **Exit proof**: RED 失败原因仅为目标联合 API/行为缺失。
- **Stop condition**: 现有 OOF artifact 无法提供正样本位置或候选对齐信息。

### Phase 2: 最小纯 Jittor 联训实现

- **Purpose**: 让共享主干同时学习 route reward 与 top-k LambdaMRR residual。
- **Entry condition**: Phase 1 RED 已确认。
- **Phase rules**:
  - 所有可训练参数属于 `jt.nn.Module`；NumPy 仅负责固定数据与指标计算。
  - 默认保持 short；只有冻结阈值选中的行应用 bounded top-k residual。
  - 不做无关重构。
- **Todos**:
  - [ ] 实现联合模型、LambdaMRR loss、训练、预测和 checkpoint replay。
    - **Surface**: `src/jgrec/rankers/hybrid/`
    - **Proof**: Phase 1 测试 GREEN。
    - **Depends on**: Phase 1
  - [ ] 实现严格 timestamp split 的训练 runner 与报告。
    - **Surface**: `scripts/`
    - **Proof**: runner 参数/小样本契约测试和静态检查通过。
    - **Depends on**: 联合模型 GREEN
- **Exit proof**: 目标测试、相关 hybrid 测试、ruff 与 py_compile 全部通过。
- **Stop condition**: Jittor 无法对所需 pairwise gather 路径反向传播。

### Phase 3: 远端训练、selection 锁定与最终 gate

- **Purpose**: 获得严格前向验证结果，判断联合目标是否值得进入全量候选。
- **Entry condition**: Phase 2 全绿，远端 OOF 完整 artifact 可用。
- **Phase rules**:
  - selection 只使用 gate 之前的数据；写出 lock 后才读取 gate。
  - gate 只执行锁定配置一次，不回头调参。
  - 无论结果正负都保留完整报告，不用 gate 反向挑配置。
- **Todos**:
  - [ ] 同步代码并运行联合训练。
    - **Surface**: 远端 workspace/result
    - **Proof**: training log、variants、selection lock。
    - **Depends on**: Phase 2
  - [ ] 执行一次最终 gate 与 checkpoint replay。
    - **Surface**: evaluation report
    - **Proof**: gate report、安全审计、hash/replay。
    - **Depends on**: selection lock
- **Exit proof**: evaluation report 明确给出 accepted/rejected、完整指标和制品路径。
- **Stop condition**: selection 没有任何正增益候选时不读取 gate，直接记录 No-Go。

### Phase 4: 候选全量化（条件阶段）

- **Purpose**: 仅当 gate 通过时生成测试候选，供外部评测。
- **Entry condition**: Phase 3 gate 满足全部 pass criteria。
- **Phase rules**:
  - 使用锁定超参数在允许的全量训练区间重训。
  - checkpoint 在无 LightGBM/sklearn 环境中可推理复现。
- **Todos**:
  - [ ] 全量重训并生成 Dataset2 候选提交。
    - **Surface**: result/package
    - **Proof**: manifest、SHA-256、字节级 Dataset1 冻结审计。
    - **Depends on**: Phase 3 accepted
- **Exit proof**: 候选包和复现命令完整。
- **Stop condition**: gate rejected。

## Dry-Run Findings

- OOF residual 已覆盖各自可用时间段，joint runner 必须取三者共同覆盖区间，不能用
  缺失 horizon 的零填充制造伪信号。
- 路由 reward 极稀疏；必须保留样本加权，但 LambdaMRR loss 只在训练行 top-k 内计算。
- 旧 LambdaMRR 的 `+0.000361` 不是本模型基线；本轮比较基线必须是同一时间行上的
  bounded short decoder。
- selection 无正增益时跳过 gate 是必要的防泄漏规则，不能为“完成训练”而破例。

## Final Validation

- `uv run pytest tests/test_hybrid_joint_oof_lambdamrr.py`
- `uv run pytest tests/test_hybrid_high_confidence_topk_router.py tests/test_hybrid_champion_residual.py`
- `uv run ruff check <changed-python-files>`
- `uv run python -m py_compile <changed-python-files>`
- 远端 evaluation report + checkpoint replay + SHA-256 审计。

## First Execution Step

新增联合训练公共 API 的行为测试，并运行最小 pytest 获得预期 RED。
