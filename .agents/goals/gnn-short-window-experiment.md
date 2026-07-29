# Goal Document: GNN 短窗强化受控实验

## Go / No-Go

- **Judgment**: Go
- **Reason**: 已有消融、短窗重训和独立 listwise MRR 三类正面证据；首轮只调整现有配置，不需要改生产代码，风险可控。

## Target Outcome

在完全一致的数据、切分、seed、候选集和融合设置下，量化提高
`gnn_epochs` 与 `gnn_max_train_edges` 对 GNN 独立指标及最终融合指标的影响，
选出一个可复现、显存可承受的候选配置，或用受控结果否定“当前 GNN 明显欠拟合”。

## Goal Definition

- **Type**: learning / quality
- **Boundary**: 定位并复现当前 GNN 基线；只扫描 `gnn_epochs` 和
  `gnn_max_train_edges`；记录独立塔与融合后的验证指标、耗时和峰值显存。
- **Non-goals**:
  - 本轮不修改 GNN 网络结构、损失函数、特征或融合权重。
  - 本轮不直接覆盖冠军配置或提交线上包。
  - 本轮不同时处理 listwise 混合、full-refit 漂移及其他 A/B/C 项。
- **Deferred work**:
  - 将胜出 GNN 配置接入 listwise/部分混合权重搜索。
  - 多 seed 复验、线上提交和默认配置更新。
- **Verification rule**: 使用同一评估命令比较匹配基线与实验组，保存逐组配置、
  指标、日志和资源消耗；任何胜出配置必须再运行一次复验。
- **Evidence source**: 仓库生成的验证指标与实验日志。
- **Pass criteria**: 四个受控配置至少完成三个；若存在胜出配置，其 GNN 独立 MRR
  与最终选择指标均不得低于匹配基线，且至少一个主要指标严格提高；复验方向一致。
  若没有配置满足该条件，则以“欠拟合假设未被支持”的可复现实验结论完成本目标。
- **Confidence note**: 单一切分只能支持本地候选筛选，不能替代长跨度外部留出或线上结果。
- **Judgment owner**: 固定评估脚本产出的指标；是否进入下一阶段由实验结果决定。

## Current State

- 生效配置据现有分析为 `gnn_epochs=10`、`gnn_max_train_edges=40000`。
- GNN 塔设计容量据现有分析为 50 epochs、200000 edges。
- 已有证据表明短窗/近期训练和 listwise GNN 可能提供独立信号，但尚未完成受控接入。
- 已知风险包括训练耗时、24 GB 显存上限、旧实验与当前代码/数据不完全同源，
  以及单切分本地指标对线上效果的代理偏差。

## Plan Rewrite Notes

| Existing item | Decision | Reason |
|---------------|----------|--------|
| A1 零代码提高两个 GNN 参数 | rewrite | 改为匹配基线、单变量对照、满容量组合和复验，避免把联合变化误判为单一参数贡献 |
| A2 listwise 部分混合 | defer | 需要先确定强化后的 GNN 候选，避免同时改变塔与融合 |
| A3 full-refit 漂移 | defer | 是独立工程问题，会破坏本轮唯一变量原则 |
| A4 提交验证包 | remove | 与 GNN 容量假设无直接关系 |
| B/C/D 其余方向 | remove | 不服务于本轮因果判断 |

## Drift Diagnosis

- **Goal drift**: 原计划同时包含多个独立方向；本轮只验证 GNN 容量假设。
- **Phase drift**: “提高两个参数”缺少基线、单变量归因和复验阶段。
- **Validation drift**: 不能以训练成功或生成包为完成，必须比较匹配指标。
- **Compatibility drift**: 本轮不引入兼容路径。
- **Cleanup drift**: 默认值、防呆和遗留代码清理全部排除。

## Priority Rationale

- 先确认入口、实际生效值和可复用基线，避免昂贵训练跑错配置。
- 然后分别扩大 epochs 与 edges，最后才跑 50/200k 组合，保留归因能力。
- 只对胜出组复验，控制总 GPU 成本。

## Assumptions and Open Decisions

| Item | Status | Impact | Owner / Next step |
|------|--------|--------|-------------------|
| 当前机器可访问项目所需 GPU | assumed | 决定是否能执行容量扫描 | Phase 1 通过环境探测确认 |
| 仓库已有固定 seed 与评估入口 | assumed | 决定实验是否可比 | Phase 1 从代码和历史命令确认 |
| 旧基线可复用 | unresolved | 可节省一次训练，但可能与当前版本不匹配 | 仅在 commit、数据指纹、配置和 seed 一致时复用 |
| 主要选择指标 | unresolved | 决定胜出配置 | 沿用当前冠军实验的显式指标，不采用有争议的默认值 |

## Phases

### Phase 1: 冻结实验口径

- **Purpose**: 找到真实入口、配置覆盖方式、数据切分、seed、评估指标和输出位置。
- **Entry condition**: 目标文档已写入仓库。
- **Phase rules**:
  - 只读检查，不修改训练代码或默认配置。
  - 明确记录最终解析后的参数，不能只相信配置文件静态值。
  - 若冠军口径无法从仓库和历史产物确定，则暂停长训练。
- **Todos**:
  - [ ] 定位 GNN 配置定义、CLI 参数与训练调用链。
    - **Surface**: config / CLI / ranker / scripts
    - **Proof**: 文件位置与可运行命令。
    - **Depends on**: none
  - [ ] 识别已有实验产物和可比较的基线。
    - **Surface**: logs / outputs / docs
    - **Proof**: commit、数据、seed、完整配置和指标均可核对，或明确判定不可复用。
    - **Depends on**: none
  - [ ] 探测 uv 环境、GPU 和磁盘余量。
    - **Surface**: runtime
    - **Proof**: `uv`、GPU 型号/显存和可用空间输出。
    - **Depends on**: none
- **Exit proof**: 一条基线命令和三个实验命令可在不猜参数的情况下执行。
- **Stop condition**: 数据、必要模型资产、冠军评估口径或 GPU 不可用。

### Phase 2: 建立匹配基线

- **Purpose**: 获得当前代码与当前数据上的 10 epochs / 40000 edges 基线。
- **Entry condition**: Phase 1 口径冻结。
- **Phase rules**:
  - 固定 seed、数据、切分、候选集、融合与所有非目标参数。
  - 可复用旧基线仅限全部指纹一致，否则必须重跑。
  - 保存最终解析配置、日志、指标、耗时和资源用量。
- **Todos**:
  - [ ] 运行或严格复用 10/40000 基线。
    - **Surface**: experiment output
    - **Proof**: 完整配置与独立塔/融合指标。
    - **Depends on**: Phase 1
- **Exit proof**: 可作为后续三组唯一对照的匹配基线存在。
- **Stop condition**: 基线不能稳定完成或指标管线异常。

### Phase 3: 容量扫描

- **Purpose**: 分离 epochs 与 edges 的贡献，并测试设计满容量。
- **Entry condition**: 匹配基线有效。
- **Phase rules**:
  - 依次运行 50/40000、10/200000、50/200000。
  - 除两个目标参数外不得改变任何实验变量。
  - 单组发生 OOM 时只记录失败；不得临时缩 batch 后继续声称是严格对照。
  - 若预计总耗时或磁盘明显超出可用预算，先完成两个单变量组，再决定组合组。
- **Todos**:
  - [ ] 运行 50 epochs / 40000 edges。
    - **Surface**: experiment output
    - **Proof**: 完整日志、指标、耗时与资源用量。
    - **Depends on**: Phase 2
  - [ ] 运行 10 epochs / 200000 edges。
    - **Surface**: experiment output
    - **Proof**: 完整日志、指标、耗时与资源用量。
    - **Depends on**: Phase 2
  - [ ] 运行 50 epochs / 200000 edges。
    - **Surface**: experiment output
    - **Proof**: 完整日志、指标、耗时与资源用量。
    - **Depends on**: 两个单变量组
- **Exit proof**: 至少三个配置有可比较结果，或失败原因与资源证据完整。
- **Stop condition**: 连续两组 OOM、指标文件不可比较或检测到非目标变量变化。

### Phase 4: 复验与结论

- **Purpose**: 排除偶然波动，给出接入或止损判断。
- **Entry condition**: Phase 3 至少有一个候选不劣于基线。
- **Phase rules**:
  - 只复验最佳候选；保持原 seed 时验证运行确定性，必要时再用第二 seed 检查方向。
  - 本阶段不改默认配置。
- **Todos**:
  - [ ] 重跑最佳候选并生成对比表。
    - **Surface**: experiment output / report
    - **Proof**: 两次结果方向一致，且满足本目标 pass criteria。
    - **Depends on**: Phase 3
- **Exit proof**: 给出“进入融合实验”或“停止容量扩张”的明确结论。
- **Stop condition**: 复验反转、指标波动大于观察到的收益或资源成本不可接受。

## Dry-Run Findings

- 旧分析给出了参数位置和容量上限，但没有给出当前可执行命令、seed、数据指纹和
  主要选择指标；必须在任何长训练前补齐。
- 直接从 10/40000 跳到 50/200000 无法判断收益来自更多 epoch 还是更多 edges，
  因此增加两个单变量组。
- 单一切分上涨不能授权线上提交；胜出后仍需 A1 的融合接入和更强验证。

## Final Validation

生成包含配置、数据/代码指纹、独立 GNN 指标、融合指标、相对基线差值、耗时、
峰值显存和复验结果的对比表，并按 pass criteria 给出明确判断。

## First Execution Step

只读定位训练入口、配置覆盖方式、当前实验产物与运行环境，冻结可执行命令。

## Execution Outcome

- Phase 1 发现原假设中的 `10/40000` 不是冠军实验的实际运行口径：
  `TrainingConfig` 静态默认是 10，但生产 CLI/checkpoint 已解析为
  `50/40000`。因此无需再把 epochs 从 10 提到 50，未验证变量只剩
  `max_train_edges: 40000 -> 200000`。
- Phase 2 复用 SHA 完全匹配的 `short_none 50/40000` 受控结果：
  fixed-blend full MRR `0.5484923183`。
- Phase 3 在远端 RTX 4090 完成 `short_none 50/200000`：
  fixed-blend full MRR `0.5475115740`，相对匹配对照
  `-0.0009807443`；三个时间切片均下降。
- 200k 候选仅比旧冠军高 `+0.0005937556`，没有达到 `+0.001`，
  且 slice 1 下降，因此 gate 失败。
- Phase 4 的进入条件不成立，未做复验，也未修改默认配置或生成提交包。
- 最终判断：容量扩张假设在 edges 轴上被否定；保留 `50/40000`，
  后续若继续 A1，应从该短窗对照做生产集成，而不是继续加边。
