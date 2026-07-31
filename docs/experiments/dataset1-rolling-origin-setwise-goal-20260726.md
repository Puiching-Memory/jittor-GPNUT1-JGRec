# Goal Document: Dataset1 Rolling-Origin 多时间折

## Go / No-Go

- **Judgment**: Level-1 已完成；Level-2 No-Go。
- **Reason**: rolling-origin 基础设施、manifest 和前三折 selection 已建立，
  但三个时间递增候选都在 fold1、fold2 回退，没有候选满足“每折非负且均值
  至少 `+0.0002`”。因此不读取 fold3，不投入 full-pipeline rolling-origin。

## Target Outcome

建立一个可复用、可审计的 Dataset1 rolling-origin 协议：

1. 每折模型只能读取该折 origin 之前的训练行；
2. 每个 query 只被一个时间折评分；
3. 前三折选择配置，第四折在 selection report hash 锁定后独立门禁；
4. 首轮用相同 100k 历史窗口分别训练 raw listwise MLP 与 Setwise MLP，
   检验时间递增融合是否跨 origin 稳定；
5. 输出 manifest、每折模型/预测、selection/gate report 和复现命令。

## Goal Definition

- **Type**: learning / quality / delivery
- **Boundary**:
  - Dataset1 only；
  - 数据固定为现有 recent-200k full-100 cache，SHA-256
    `a8f4b5d71dedd1b5aa89a9f0c40e1501afc882929d036ddaaa73076e5be6a6ef`；
  - cache 200k 行按时间顺序切成 4 个 sliding-origin 折；
  - 每折 train window `100,000`，score horizon `25,000`，step
    `25,000`；
  - 精确范围：
    - fold0 train `[0,100000)`，score `[100000,125000)`；
    - fold1 train `[25000,125000)`，score `[125000,150000)`；
    - fold2 train `[50000,150000)`，score `[150000,175000)`；
    - fold3 train `[75000,175000)`，score `[175000,200000)`；
  - fold0/1/2 为 selection，fold3 为独立 gate；
  - control 为 raw 63-feature listwise MLP，candidate 为
    Setwise v1 189-feature MLP；
  - 两个 head 均使用同一 fold 训练窗口、seed 60、hidden 32、
    learning rate `0.001`、batch 256、固定 4 epochs；
  - 固定 epochs，不用 score fold early-stop；
  - 时间融合候选固定为 `gamma ∈ {0.5,1.0,2.0}`；
  - 每折 `progress` 使用该 score horizon 的全局时间 min/max；
  - selection 资格：三个 selection fold delta 均 `>=0`，mean delta
    `>=+0.0002`；
  - 选择顺序：worst-fold delta、mean delta、较大 gamma；
  - gate：fold3 delta `>=0` 且四折 mean delta `>=+0.0002`。
- **Non-goals**:
  - 不直接用最终冠军 head 回放历史折；
  - 不读取 Dataset1 test；
  - 不调整 epochs、hidden dim、学习率或 gamma grid；
  - 本轮结果不自动替换 `1.3540333477186608` 线上冠军；
  - gate 前不生成新的提交包。
- **Deferred work**:
  - 每折重建 stats/graph/sequence/two-tower encoder 的 Level-2
    full-pipeline rolling-origin；
  - rolling-origin 超参搜索；
  - Dataset2 多时间折。
- **Verification rule**:
  - 单测证明折覆盖、无行泄漏、时间单调、selection/gate 隔离；
  - manifest 锁定 cache/sidecar/checkpoint hash；
  - 固定 epoch 训练日志证明 score fold 未参与训练或 early stopping；
  - selection report SHA 在 gate 读取前生成；
  - gate 只运行已锁定 gamma。
- **Evidence source**: RED/GREEN tests、manifest、模型/预测 SHA、
  selection report/hash、independent gate report。
- **Pass criteria**:
  - 4 折均成功完成，输入 cache SHA 前后不变；
  - 每折 `train.stop <= score.start`；
  - score ranges 不重叠且覆盖 cache 最后 100k；
  - 有 eligible gamma；
  - locked gamma 的 fold3 delta 非负；
  - 四折 mean delta 至少 `+0.0002`。
- **Confidence note**: Level-1 对 head 和时间融合是严格因果的，但 encoder
  固定在所有折之前，没有模拟 encoder 随 origin 更新；它能否稳定是进入
  Level-2 的筛选证据，不是完整线上泛化证明。
- **Judgment owner**: selection helper 锁定 gamma；独立 gate helper 决定
  Level-2 Go/No-Go；线上 leaderboard 才能声明新冠军。

## Current State

- 新线上冠军总分 `1.3540333477186608`，来自 Dataset1
  `gamma=0.5` 时间递增融合。
- 当前唯一 20k validation 上，full delta `+0.0023190998`，三片均正。
- Dataset1 已有 200k×100×63 chronological full-candidate cache、
  candidates/src/dst/time/row-index sidecar 和完整 SHA。
- 已有 `fit_fusion_mlp_listwise_fixed`，可以固定 epoch 训练而不让 score
  fold 参与 early stopping。
- 现有 `expanding_oof_folds` 面向 GNN OOF，没有 selection/gate role、
  sliding train window 或独立 gate contract。

## Priority Rationale

1. 先在纯 NumPy 边界层锁死 folds 和 gate 隔离，避免跑完后改折法。
2. 用现有 cache 做 Level-1，数分钟获得 head 稳定性，而非先花约一小时重建
   多套 encoder cache。
3. control/candidate 都按折重训，避免用最终冠军参数污染历史 origin。
4. 最后一折只有在 selection SHA 锁定后读取，保留真正的 forward falsifier。

## Assumptions and Open Decisions

| Item | Status | Impact | Owner / Next step |
|---|---|---|---|
| 200k cache/time/row indices严格递增 | manifest 已复核 | folds 可按行和时间解释 | complete |
| cache encoder context 早于全部 200k query rows | confirmed by cache report | Level-1 无 feature future leakage | manifest 记录限制 |
| fixed 4 epochs 可替代原 early-stop epoch4 | chosen | 避免 score-label 泄漏 | 冻结，不调参 |
| fold timestamp 边界可能出现相同 timestamp | 四个边界均不相等 | 每个 score horizon 严格晚于训练窗口 | complete |
| Level-1 pass 后是否立即跑 Level-2 | selection 未通过 | 省去无依据的 full-pipeline 重建 | No-Go |

## Phases

### Phase 1: Fold Contract RED

- **Purpose**: 在任何模型训练前锁定 sliding-origin 和 gate 隔离。
- **Entry condition**: 4 折范围、训练协议和阈值已冻结。
- **Phase rules**:
  - 纯 NumPy；
  - 先 RED 后实现；
  - selection API 不接收 gate fold metric。
- **Todos**:
  - [x] 新增折覆盖和无泄漏 RED
    - **Surface**: `tests/test_hybrid_rolling_origin.py`
    - **Proof**: target module/API missing
    - **Depends on**: none
  - [x] 新增 selection/gate 隔离 RED
    - **Surface**: 同上
    - **Proof**: negative fold rejection、forward-only gate
    - **Depends on**: none
- **Exit proof**: focused test 因目标实现缺失正确失败。
- **Stop condition**: 冻结范围无法覆盖最后 100k cache rows。

### Phase 2: Core GREEN and Manifest

- **Purpose**: 实现通用 rolling-origin contract 并绑定真实 artifact。
- **Entry condition**: RED 原因正确。
- **Phase rules**:
  - 不导入 Jittor；
  - manifest 在训练前写出；
  - 输入 hash 不匹配立即停止。
- **Todos**:
  - [x] 实现 fold/time/selection/gate helpers
    - **Surface**: `src/jgrec/rankers/hybrid/rolling_origin.py`
    - **Proof**: focused GREEN
    - **Depends on**: Phase 1
  - [x] 构建 Dataset1 manifest
    - **Surface**: manifest builder
    - **Proof**: row/time boundaries、hash、roles
    - **Depends on**: core GREEN
- **Exit proof**: tests、Ruff、manifest preflight 全部通过。
- **Stop condition**: cache schema、sidecar或报告不一致。

### Phase 3: Three-Fold Selection

- **Purpose**: 只用前三个 origin 选择 gamma。
- **Entry condition**: manifest SHA 锁定。
- **Phase rules**:
  - raw/Setwise 均每折重训；
  - 固定 4 epochs；
  - fold3 features/labels/metrics 不读取；
  - 相同 cache 不允许写入。
- **Todos**:
  - [x] 训练 fold0/1/2 control 与 Setwise
    - **Surface**: selection runner
    - **Proof**: 每折 model/prediction SHA、epoch losses
    - **Depends on**: Phase 2
  - [x] 锁定 gamma（结果：无 eligible gamma，报告已锁定）
    - **Surface**: selection report + SHA sidecar
    - **Proof**: 三折 deltas、eligibility、forward_metrics_read=false
    - **Depends on**: selection predictions
- **Exit proof**: eligible locked gamma 或 evidence-backed stop。
- **Stop condition**: 任一输入 hash 改变、训练非有限或无 eligible gamma。

### Phase 4: Independent Fourth-Fold Gate

- **Purpose**: 用最后 origin 检验锁定配置迁移。
- **Entry condition**: selection pass 且 report/hash 匹配。
- **Phase rules**:
  - 只评估 locked gamma；
  - 不重新选择；
  - gate fail 时 Level-2 No-Go。
- **Todos**:
  - [ ] 训练 fold3 control/Setwise 并门禁（按 Phase 3 stop condition 跳过）
    - **Surface**: independent gate runner
    - **Proof**: fold3 delta、四折 mean、source hashes unchanged
    - **Depends on**: Phase 3 pass
- **Exit proof**: Level-2 Go/No-Go report。
- **Stop condition**: selection hash 不匹配或 gate 阈值失败。

## Dry-Run Findings

- 直接使用新冠军 checkpoint head 会让早期 fold 引入未来训练参数，已从方案中删除。
- 现有 200k cache 足够形成 4 个互不重叠的 25k score horizon，最后
  100k query 每行恰好评分一次。
- 每折两个 100k fixed-epoch head，无需新建数 GB cache；主要产物只是
  模型和约 20–40MB 预测。
- Level-1 的固定 encoder 是显式限制；只有 gate pass 才值得构建
  Level-2 全 pipeline caches。

## Final Validation

- `uv run --no-sync pytest tests/test_hybrid_rolling_origin.py -q`
- `uv run --no-sync ruff check src/jgrec/rankers/hybrid/rolling_origin.py tests/test_hybrid_rolling_origin.py`
- Linux 重复 focused tests。
- manifest/source cache SHA 前后相同。
- selection report SHA 已锁定为
  `1a6ea48d62ea25b3c179b513a9c4e7235b82eb557b78c310340426c690282363`。
- selection 无 eligible gamma；fold3 gate 保持未读取，不生成提交包。

## Execution Outcome

- manifest SHA:
  `3a5cc87efbee49c52367b19f16d41e548a4e47fa932d2cc92449b9cc638572af`。
- 三个 selection 折均使用 100k 训练行、25k score 行、固定 4 epochs；
  输入 source hashes 训练前后不变。
- `gamma=0.5` fold deltas:
  `(+0.00087314, -0.00027292, -0.00016395)`，mean `+0.00014543`。
- `gamma=1.0` fold deltas:
  `(+0.00105828, -0.00024774, -0.00022269)`，mean `+0.00019595`。
- `gamma=2.0` fold deltas:
  `(+0.00054065, -0.00009108, -0.00020514)`，mean `+0.00008148`。
- 结论：时间递增融合在最早 selection 折有效，但未随 origin 稳定迁移；
  线上冠军 `1.3540333477186608` 保持不变，不能据此追加 Level-2 实验。

## First Execution Step

新增 exact fold layout、无行泄漏、selection 不读取 gate metric、独立 gate
判定的 RED 测试。
