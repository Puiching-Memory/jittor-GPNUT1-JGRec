# Goal Document: Dataset2 纯 Jittor OOF Stacking

## Go / No-Go

- **Judgment**: Go
- **Reason**: 上一轮 Candidate-Set Transformer 在单一 causal cache 上离线
  `+0.001727`、线上却 `-0.002101`，证明当前最大问题不是 head 容量，而是
  专家训练、meta 训练和生产 encoder 状态之间的分布迁移。rolling-origin OOF
  stacking 能同时消除同样本专家泄漏，并为稳定的行内 logits 表征提供可检验路径。

## Target Outcome

建立一个端到端纯 Jittor 的 Dataset2 OOF stacking 管线：

1. 多个 Jittor 专家按 rolling-origin 前缀训练，只预测紧随其后的不可见时间块；
2. 拼成逐行有 provenance 的 OOF logits；
3. 将 logits 转为对尺度漂移更稳健的行内 rank、robust margin、entropy、top1
   支持度和跨专家一致性特征；
4. 用纯 Jittor Candidate-Set Transformer 或小 MLP 训练 meta ranker；
5. 专家在全量 200k×100 数据上重训，生成 validation/test logits，再由同一
   transform 和 meta checkpoint 推理；
6. Dataset1 提交字节保持不变。

## Goal Definition

- **Type**: technical / learning / quality / delivery
- **Boundary**:
  - NumPy 仅负责 memmap、确定性行内变换、指标和固定 artifact 管理。
  - 所有可训练专家和 meta 模型必须是 `jt.nn.Module`。
  - LightGBM/sklearn 不得进入 OOF 生成、meta 训练、checkpoint 或 Dataset2 推理。
  - 当前冠军仅作为独立比较基线。
- **Non-goals**:
  - 不修改 Dataset1 模型。
  - 不自动提交线上排行榜。
  - 不把当前冠军或历史 LightGBM logits 作为专家输入。
  - 不用随机种子平均冒充专家多样性。
- **Deferred work**:
  - 更换底层 GNN/GRU/two-tower encoder。
  - 训练跨 Dataset1/Dataset2 的统一 stacker。
- **Verification rule**:
  - 每个 OOF 行必须能映射到唯一 fold，且专家训练行时间严格早于该 fold 预测行。
  - external 20k validation 不得进入专家或 meta 训练。
  - 同一 logits 经稳定 transform 在正仿射缩放后 rank/robust 特征保持一致或在规定
    容差内。
  - checkpoint 在阻断 LightGBM/sklearn 后完成 load/predict。
- **Evidence source**: pytest RED/GREEN、fold manifest、artifact SHA-256、OOF 覆盖率、
  外部时间片 MRR、encoder-state shift 诊断、生产 smoke 和线上分数。
- **Pass criteria**:
  - OOF 覆盖区间无泄漏、无重复、按时间连续，覆盖至少 75% 的 200k 训练行。
  - meta 模型只读取 OOF stable features；所有 provenance 为 Jittor 或
    NumPy deterministic。
  - external validation full MRR 高于当前冠军且三个时间片不退化。
  - full-trained encoder replay 下 stable feature shift 显著小于 raw logits shift；
    若不成立则不得打包。
  - 生产 Dataset2 推理无外部 ML 导入，CSV/zip/SHA 校验通过。
- **Confidence note**: rolling-origin 可以证明 meta-train 无同段泄漏；离线分数仍是
  代理证据，最终晋升必须由线上高于 `1.3545839690981516` 决定。
- **Judgment owner**: 自动化泄漏检查和测试负责实现验收；external forward gate
  负责离线晋级；线上排行榜负责冠军晋升。

## Current State

- 已有 200k×100 train cache、20k×100 external validation cache和 63 维基础特征。
- 已有两个纯 Jittor CST 配置：主 attention 专家与 pointwise residual 专家。
- 仓库已有 Setwise Jittor MLP，可作为第三种结构专家。
- 上一轮仅保存 full-trained 模型的 validation logits，没有 CST OOF logits。
- 已有一套 GNN-short-listwise OOF artifact，但其后续融合验证下降
  `-0.002556`，且不属于本轮 CST OOF 契约，不能直接复用为成功证据。
- 生产 encoder 与 causal validation encoder 在相同历史 query 上产生显著输出漂移；
  stable logits transform 必须成为新门禁的一部分。

## Plan Rewrite Notes

| Existing item | Decision | Reason |
|---|---|---|
| rolling-origin 训练多个专家 | keep | OOF 无泄漏的核心 |
| 每折只预测下一时间块 | keep | 建立严格时间边界 |
| rank/margin/entropy 特征 | rewrite | 增加 tie-safe percentile rank、MAD/IQR robust margin、top1 vote |
| CST/小 MLP meta 模型 | keep | 两者均为纯 Jittor，按外部门禁选择 |
| 全量专家重训 | keep | 与生产测试推理一致 |
| 继续扫描两个 full 模型融合权重 | remove | 已被线上结果否证 |
| 直接复用历史 GNN OOF logits | remove | provenance 和 fold 契约不同，且已有负验证 |

## Drift Diagnosis

- **Goal drift**: 继续调单模型层数或 0.6/0.4 权重不能解决 OOF 缺失。
- **Phase drift**: 必须先验证 fold/transform 契约，再启动多折训练。
- **Validation drift**: 单一 external cache 的小幅提升不能再单独授权生产。
- **Compatibility drift**: 不保留 meta 模型回退到 LightGBM/旧 Setwise 融合的路径。
- **Cleanup drift**: 不清理历史实验和旧依赖声明。

## Priority Rationale

- 最高风险是隐性泄漏和 fold/test 特征契约不一致，必须在昂贵训练前用测试锁死。
- 第二风险是 full-trained encoder shift，因此 stable transform 和 shift 诊断先于模型
  调参。
- 只有 OOF artifact 完整且可信后，meta 模型容量比较才有意义。

## Assumptions and Open Decisions

| Item | Status | Impact | Owner / Next step |
|---|---|---|---|
| 使用约 40k warmup + 4 个 timestamp-aligned forward folds | confirmed | 精确 40k 边界均切开同一时间戳，必须左移到时间组首行 | manifest 强制 `train_time_max < score_time_min` |
| 专家为 CST-main、CST-residual、Setwise-MLP | assumed | 提供结构而非种子多样性 | 先做小规模折训练确认接口 |
| external 20k 只做最终门禁 | confirmed | 避免 meta early-stop 泄漏 | meta early-stop 使用最后 OOF fold |
| stable transform 不读取正样本位置 | confirmed | 防止候选 0 偏置 | tie/置换测试证明 |
| 最终全量 CST 可复用已有 checkpoint | unresolved | 节省训练但需配置/缓存 SHA 一致 | 运行前校验 provenance |

## Phases

### Phase 1: 冻结 OOF 与稳定特征契约

- **Purpose**: 在训练前证明 fold 边界、覆盖率、tie 行为和变换不泄漏标签。
- **Entry condition**: 本目标文档完成。
- **Phase rules**:
  - 只修改测试和新 OOF 核心模块；不启动正式训练。
  - external validation 不得参与任何 fixture 的 meta-train 切分。
- **Todos**:
  - [ ] 增加 rolling-origin manifest 的连续性、唯一覆盖和前缀边界测试。
    - **Surface**: tests / OOF split API
    - **Proof**: RED 原因为目标 API 缺失，GREEN 后覆盖 160k 行且无泄漏。
    - **Depends on**: none
  - [ ] 增加 tie-safe rank、robust margin、entropy 和候选置换等变测试。
    - **Surface**: stable logits transform
    - **Proof**: 正仿射缩放与候选置换测试通过，ties 不偏爱候选 0。
    - **Depends on**: none
- **Exit proof**: 目标测试全部 GREEN，transform 不读取 labels/positive indices。
- **Stop condition**: 现有 cache 行序不是严格时间递增，无法建立 rolling-origin。

### Phase 2: 纯 Jittor 专家与 OOF artifact

- **Purpose**: 生成可审计的多个专家 OOF logits。
- **Entry condition**: Phase 1 GREEN。
- **Phase rules**:
  - 每折独立构建/加载模型；不得把上一折预测块加入该折训练以外的未来数据。
  - 每个 artifact 记录 config、训练区间、预测区间、cache SHA 和模型 SHA。
- **Todos**:
  - [ ] 为 CST-main、CST-residual、Setwise-MLP 建立统一 expert adapter。
    - **Surface**: Jittor training/checkpoint API
    - **Proof**: tiny fold 可 train-save-load-predict。
    - **Depends on**: Phase 1
  - [ ] 运行四个 rolling folds 并写入 OOF logits memmap。
    - **Surface**: remote training artifacts
    - **Proof**: manifest 覆盖率、有限性、SHA 和逐折 MRR 报告。
    - **Depends on**: expert adapter
- **Exit proof**: 三专家 OOF logits 完整覆盖同一 160k 行且 provenance 合规。
- **Stop condition**: 任一折时间边界或 artifact hash 不匹配。

### Phase 3: 纯 Jittor meta ranker

- **Purpose**: 用 OOF stable features 学习专家组合。
- **Entry condition**: Phase 2 artifact 完整。
- **Phase rules**:
  - 前三个 OOF fold 训练，第四 fold early-stop；external validation 不参与训练。
  - 比较小 MLP 与小 CST，但不扫描大量结构。
- **Todos**:
  - [ ] 实现 on-the-fly stable feature view 和 Jittor meta MLP/CST。
    - **Surface**: stacker model/trainer
    - **Proof**: tiny ranking任务收敛、checkpoint round-trip。
    - **Depends on**: Phase 2
  - [ ] 训练并锁定 meta 配置。
    - **Surface**: remote experiment
    - **Proof**: 第四 OOF fold MRR 和模型选择报告。
    - **Depends on**: meta trainer
- **Exit proof**: 锁定一个纯 Jittor meta checkpoint，未读取 external validation labels。
- **Stop condition**: 第四 OOF fold 不优于最佳单专家。

### Phase 4: 全量专家、external gate 与生产

- **Purpose**: 验证 OOF 学到的融合能迁移到全量专家和生产 encoder。
- **Entry condition**: Phase 3 锁定。
- **Phase rules**:
  - 专家全量重训配置必须与 OOF 对应专家一致。
  - external validation 只运行一次正式门禁。
  - shift 稳健性失败时不得通过调 external 权重救回。
- **Todos**:
  - [ ] 全量重训/校验专家并生成 external validation logits。
    - **Surface**: full expert checkpoints
    - **Proof**: config/cache SHA 与 OOF manifest 一致。
    - **Depends on**: Phase 3
  - [ ] 执行 external 三时间片门禁和 encoder-state shift 诊断。
    - **Surface**: evaluation
    - **Proof**: full/slices delta 与 stable-vs-raw shift 报告。
    - **Depends on**: full experts
  - [ ] 通过后生成 Dataset2 test logits、checkpoint 和提交包。
    - **Surface**: production packaging
    - **Proof**: blocked-import smoke、CSV/zip/SHA。
    - **Depends on**: gate passed
- **Exit proof**: 合规提交包生成，或留下明确拒绝报告。
- **Stop condition**: external 任一时间片退化、shift 门禁失败或生产依赖不纯。

## Dry-Run Findings

- 200k cache 已确认时间与原始行索引单调；但精确 40k 边界全部切开相同时间戳，
  因此 fold 切点必须左移到该时间戳首行。
- meta early-stop 不能使用 external 20k，需要预留最后一个 OOF fold。
- raw logits 不能直接跨 fold 拼接；不同专家/折的温度和尺度必须在每行内消除。
- ties 必须使用平均秩，不能使用稳定排序顺序，否则候选 0 再次获得虚假 MRR 优势。
- full-trained encoder 与 causal cache 无法逐值比较；shift 门禁比较的是 raw/stable
  特征漂移比例和生产输出有效性，不是假设二者相同。

## Final Validation

- `uv run --no-sync pytest tests/test_hybrid_oof_stacking.py -q`
- Linux/CUDA 上运行 OOF、meta checkpoint 和 ranker 集成回归。
- Ruff、`git diff --check`、artifact SHA-256、blocked-import smoke。
- external full/slice MRR 与当前冠军同口径比较。
- 只有线上高于 `1.3545839690981516` 才晋升。

## First Execution Step

读取 200k train cache 的 row/time sidecar 和现有 rolling-origin/Setwise/CST API，
确认行时间严格递增并冻结 40k warmup + 4×40k forward fold manifest。
