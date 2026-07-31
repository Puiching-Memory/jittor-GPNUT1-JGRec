# Goal Document: Dataset2 Setwise 行内相对特征 v2

## Go / No-Go

- **Initial judgment**: Go
- **Final judgment**: No-Go for packaging
- **Reason**: percentile rank 与 robust z-score 已在 `SetwiseFeatureView` 中按 batch 动态生成，未重建 4.7GB full-100 缓存；但 v2 在两个可见时间片均低于 v1，均匀融合也因 slice0 下降而失去资格。前缀锁最终选择原 v1 冠军，第三片门禁因此 full 增益为 0，未达到 `+0.001`。

## Target Outcome

保持 Setwise v1 与当前冠军行为不变，新增无候选位置偏置的 context transform v2，并训练一个 Dataset2 `200k × 100 × seed60` v2 模型。只用前两片从 `v1 / v2 / v1+v2` 中锁定一个候选，第三片只作门禁；不过门不改包。

## Goal Definition

- **Type**: learning / quality
- **Boundary**:
  - v2 动态通道固定为 `raw`、`raw-row_mean`、`raw-row_max`、行内 tie-neutral percentile rank、median/MAD robust z-score；
  - 复用 Dataset2 200k full-100 原始特征缓存；
  - 只训练一个 v2 seed60 模型；
  - 固定 `0.80 Setwise + 0.20 LightGBM` 外层融合；
  - 比较 v1、v2、v1/v2 均匀概率融合。
- **Non-goals**:
  - 不重建特征缓存；
  - 不新增随机种子、时间窗口或半衰期；
  - 不搜索外层权重；
  - 不修改原始 63 个特征；
  - 不用第三片选模型或调 transform。
- **Deferred work**:
  - 仅当门禁通过后实现 v2 或 v1+v2 的生产 checkpoint 推理；
  - 其他 robust scaling、rank 方向或 clipping 变体留到独立实验。
- **Verification rule**:
  - v1 默认输出与现有实现逐值相等；
  - percentile ties 使用相同 midrank，不能借 candidate index 区分；
  - robust z-score 使用冻结公式；
  - 选择报告先写 SHA-256 锁，随后独立 gate 才计算 full / slice2。
- **Evidence source**: 单元测试、Linux/Jittor 回归、frozen-config、selection-report、evaluation-report。
- **Pass criteria**:
  - 本地纯 NumPy与 Linux/Jittor 测试通过；
  - 当前 v1 冠军四项指标精确复现；
  - 锁定候选 full MRR 相对冠军 `>= +0.001`；
  - slice0、slice1、slice2 均不下降；
  - 若打包，Dataset1 CSV SHA-256 保持 `6760fd966fde3a8de4693d06f27ec0f2458e49c44e2ea48fc79e1393434e6e2b`。
- **Confidence note**: 20k full-100 验证集与线上成功的 Dataset2 Setwise 使用同一已验证缓存；第三片隔离降低选择泄漏，但仍是单次时间切分。
- **Judgment owner**: 冻结门禁脚本。

## Current State

- v1 仅生成 `raw`、`raw-row_mean`、`raw-row_max`，63 维扩展为 189 维。
- 当前 Dataset2 冠军 full MRR `0.5469178184464882`，slice0 / slice1 / slice2 分别为 `0.5863014322270679` / `0.5482466913826506` / `0.5061992242459902`。
- 时间窗口实验证明 Setwise 多样性可带来 full `+0.00139612` 与 slice2 `+0.00242598`，但因 slice1 `-0.00028430` 被拒绝。
- 当前所有调用都隐式依赖 v1；新增版本参数必须默认 v1，模型文件必须显式记录 transform version。
- percentile rank 的主要风险是 ties：普通双 `argsort` 会用候选顺序拆开并列值，而正样本固定在 candidate0，可能形成位置泄漏。
- robust z-score 的主要风险是 MAD=0；必须输出 0 而非无穷或极大值。

## Plan Rewrite Notes

| Existing item | Decision | Reason |
|---|---|---|
| 50k / 100k / 200k 窗口训练 | remove | 本轮只归因 context transform，不再改变训练分布 |
| 200k full-100 缓存 | keep | 无需重建，直接提供 raw batch |
| seed60 与现有超参 | keep | 隔离 transform 变量 |
| 前两片合并 MRR 直接选最高 | rewrite | 先要求 slice0、slice1 均不低于冠军，再最大化合并 MRR |
| 第三片门禁 | keep | 防止用未来片调 transform |
| 固定 0.80/0.20 外层融合 | keep | 与当前冠军可比 |

## Drift Diagnosis

- **Goal drift**: 不把本轮扩成窗口、权重或随机种子的联合搜索。
- **Phase drift**: transform 正确性与昂贵训练分开，先证明 ties / MAD 边界。
- **Validation drift**: 改动文件不算成功，必须有 RED/GREEN、前缀锁和第三片门禁。
- **Compatibility drift**: 不让所有旧模型自动切到 v2；默认值与旧 checkpoint 始终是 v1。
- **Cleanup drift**: 不顺带重构其他重复的 Setwise 模型 I/O。

## Priority Rationale

- tie-neutral rank 是最高风险点：若处理错误，candidate0 位置泄漏可制造虚假高分，必须先用测试杀死。
- v1 兼容性先于训练，否则无法证明增益来自新增通道。
- 生产 checkpoint 支持放到门禁后，避免为失败实验扩大改动面。

## Assumptions and Open Decisions

| Item | Status | Impact | Owner / Next step |
|---|---|---|---|
| percentile 方向为低值 0、高值 1 | confirmed | 保持“值越大 percentile 越大”，原始方向仍由 MLP 学习 | 冻结公式 |
| ties 使用 ascending average rank / `(C-1)` | confirmed | 消除 candidate index 偏置 | 单元测试 |
| robust z 使用 `(x-median)/(1.4826*MAD)` | confirmed | 标准 MAD 尺度 | 单元测试 |
| MAD `<= 1e-6` 时整列输出 0 | confirmed | 避免不稳定极值 | 单元测试 |
| v2 维度为 `5 × 63 = 315` | confirmed | 增加 CPU transform 与 MLP 输入成本 | preflight / 训练日志 |
| 第三片之前只选满足 slice0、slice1 不退化的候选 | confirmed | 防止 slice0 掩盖 slice1 | 选择器测试 |

## Phases

### Phase 1: Transform 契约 RED

- **Purpose**: 用失败测试冻结 v1 兼容、tie-neutral percentile 和 robust z 边界。
- **Entry condition**: v1 当前行为已读并有现有回归测试。
- **Phase rules**:
  - 只改测试，不改生产代码；
  - ties 测试必须打乱候选顺序并保持并列 percentile 相同；
  - MAD=0 必须被覆盖。
- **Todos**:
  - [x] 添加 v2 通道值与形状测试
    - **Surface**: `tests/test_hybrid_setwise.py`
    - **Proof**: 因缺少 version=2 行为而 RED
    - **Depends on**: none
  - [x] 添加 v1 逐值兼容测试
    - **Surface**: `tests/test_hybrid_setwise.py`
    - **Proof**: v1 期望数组
    - **Depends on**: none
- **Exit proof**: 目标测试因缺少 v2 API/行为正确失败。
- **Stop condition**: 无法定义不依赖候选位置的 ties 行为。

### Phase 2: GREEN 与兼容回归

- **Purpose**: 最小实现 transform version 2，并保持默认 v1。
- **Entry condition**: RED 原因正确。
- **Phase rules**:
  - `transform_version=1` 为默认；
  - 不引入新第三方依赖；
  - 只在 batch 访问时动态计算。
- **Todos**:
  - [x] 实现 tie-neutral percentile midrank
    - **Surface**: `src/jgrec/rankers/hybrid/setwise.py`
    - **Proof**: ties / permutation 测试通过
    - **Depends on**: Phase 1
  - [x] 实现 robust z-score
    - **Surface**: 同上
    - **Proof**: median/MAD/zero-MAD 测试通过
    - **Depends on**: Phase 1
  - [x] 为 lazy view 增加显式 version 与正确 shape
    - **Surface**: 同上
    - **Proof**: v1/v2 view 测试通过
    - **Depends on**: 前两项
- **Exit proof**: 本地纯 NumPy测试、Ruff、Linux/Jittor 回归通过。
- **Stop condition**: v1 默认输出发生任何数值变化。

### Phase 3: 200k v2 训练与前缀锁定

- **Purpose**: 在不读取第三片指标的情况下训练并选择 v2 候选。
- **Entry condition**: Phase 2 通过；远端缓存和冠军哈希匹配。
- **Phase rules**:
  - 只训练一个 200k seed60 v2；
  - early stopping 只看 `[0,13334)`；
  - 候选固定为 v1、v2、v1/v2 uniform；
  - 先筛 slice0、slice1 均不退化，再按合并前缀 MRR、较少模型、v1 优先平分。
- **Todos**:
  - [x] 写入 frozen-config
    - **Surface**: 实验脚本 / result
    - **Proof**: 输入哈希、公式、候选、选择规则完整
    - **Depends on**: Phase 2
  - [x] 训练 v2 并保存 20k 概率
    - **Surface**: 远端 GPU
    - **Proof**: 模型、history、概率 SHA-256
    - **Depends on**: frozen-config
  - [x] 锁定前缀选择
    - **Surface**: selection-report
    - **Proof**: 报告与 SHA-256 sidecar；无 full / slice2 指标
    - **Depends on**: v2 概率
- **Exit proof**: 唯一候选在打开第三片前锁定。
- **Stop condition**: baseline 前缀指标无法精确复现或无候选满足前两片约束。

### Phase 4: 第三片门禁与条件打包

- **Purpose**: 验证锁定候选的时间稳健性。
- **Entry condition**: selection-report SHA-256 锁匹配。
- **Phase rules**:
  - 只评估锁定候选；
  - 不在 gate 后改 version、候选、选择规则或权重；
  - rejected 不生成包。
- **Todos**:
  - [x] 计算 full 与三片 delta
    - **Surface**: evaluation-report
    - **Proof**: 所有冻结 gate 布尔值
    - **Depends on**: Phase 3
  - [x] accepted 时实现生产推理并保持 Dataset1 字节
    - **Surface**: checkpoint / package
    - **Proof**: gate rejected，条件不成立，明确未生成 checkpoint/package；冻结 Dataset1 哈希仍匹配
    - **Depends on**: gate passed
- **Exit proof**: accepted 包或明确 rejected 报告。
- **Stop condition**: 任一 slice 下降或 full delta 小于 `+0.001`。

## Dry-Run Findings

- 原始 memmap 仍是 63 维；v2 只在 `SetwiseFeatureView.__getitem__` 生成 315 维 batch，不产生新 4.7GB cache。
- 直接双 argsort 不满足 ties 中性，必须在 sorted 轴上识别相等分组并回填 midrank。
- robust z 的 MAD=0 在稀疏/二值特征上很常见，冻结为 0 输出是必需边界。
- 当前 ranker 和历史脚本依赖无参数调用；默认 v1 可避免批量迁移。
- 若 v1/v2 融合胜出，生产支持是条件工作，不应在 gate 前实现。

## Final Validation

- `uv run --no-sync pytest tests/test_hybrid_setwise.py tests/test_hybrid_fusion_listwise.py -q`
- `uv run --no-sync ruff check src/jgrec/rankers/hybrid/setwise.py tests/test_hybrid_setwise.py`
- Linux/Jittor 同组测试。
- `selection-report.sha256` 与 gate 输入一致。
- accepted 时 Dataset1 SHA-256 保持冻结值。

## Execution Result

- v2 在 epoch 3 取得最佳 Setwise 前缀 MRR `0.5616044917365675`，epoch 5 触发 early stop；总训练选择耗时 `850.827s`。
- 固定 `0.80 Setwise + 0.20 LightGBM` 后：
  - v1：slice0 `0.5863014322270679`，slice1 `0.5482466913826506`，前缀 `0.5672740618048593`；
  - v2：slice0 `0.5833335086129465`，slice1 `0.5441074631570938`，前缀 `0.5637204858850202`，不合格；
  - v1/v2 uniform：slice0 `0.5859838878003787`，slice1 `0.5483494708945199`，前缀 `0.5671666793474492`，因 slice0 下降而不合格。
- SHA 锁定候选为 `v1_champion`；selection report SHA-256 为 `44d9a244b77c209062ad7e5d6093d3839bb1947497db13f09cb4a4b57f03d87a`。
- 独立 gate 精确复现冠军 full / slice0 / slice1 / slice2；所有 delta 为 `0`，full 增益门槛失败。
- `gate_passed=false`、`package_authorized=false`、`package_generated=false`；Dataset1 冻结文件 SHA-256 仍为 `6760fd966fde3a8de4693d06f27ec0f2458e49c44e2ea48fc79e1393434e6e2b`。
- 结论：保留通用 v2 transform 实现和测试，但本次模型不进入冠军、不打包。

## First Execution Step

新增只描述 v2 行为的失败测试，先确认 transform version 2 缺失、ties 与 MAD 边界尚未实现。
