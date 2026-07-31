# Dataset2 Setwise 时间窗口多样性实验目标（2026-07-26）

## 0. Plan Rewrite Notes

- **替换项**：停止继续扩展 Dataset2 随机种子集成；将多样性来源改为训练时间窗口与固定时间衰减。
- **保留项**：复用 Dataset2 `200k × 100` full-candidate Setwise 成功协议、seed60、当前 `0.80 Setwise + 0.20 LightGBM` 外层融合权重、20k 时间验证集及三时间片定义。
- **删除项**：不再训练新随机种子，不再用三种子均匀 rank ensemble。
- **新增项**：训练最近 50k、最近 100k 两个窗口模型；复用最近 200k 冠军模型；增加一个固定半衰期的 200k 时间衰减模型；只在前两片选择单模型或跨窗口均匀融合子集，最后一片仅作不可见门禁。
- **证据**：三种子均匀 rank ensemble 相对当前冠军下降 `-0.0016566946675822258`，且 slice0 / slice1 分别下降 `-0.003629105733156557` / `-0.0024193924754468688`，见 `result/dataset2_setwise_three_seed_rank_ensemble_20260726/evaluation-report.json`。

## 1. Goal

在 Dataset1 预测字节完全不变的前提下，为 Dataset2 训练并选择由不同近期训练窗口产生的 Setwise 专家；只有当预先冻结的第三片门禁通过时，才把选中的多窗口融合写入新提交包。

目标不是证明“小窗口一定更好”，而是用一次可归因、可复现、无末片泄漏的实验回答：

1. 50k / 100k / 200k 时间窗口是否提供有效且互补的排序信号；
2. 固定时间衰减能否替代硬窗口切分；
3. 前两片选出的融合能否在不可见第三片及全量验证上稳健超过当前 Dataset2 冠军。

## 2. Scope

### In scope

- Dataset2 现有 `200k × 100` 训练缓存和 `20k × 100` 验证缓存。
- 最近 50k、100k、200k 三个硬窗口。
- 一个固定时间衰减对照：200k 窗口、半衰期 100k 行。
- 单一随机种子 seed60。
- Setwise 专家之间的均匀概率融合。
- 当前固定外层融合：`0.80 × Setwise 专家概率 + 0.20 × 当前 LightGBM 概率`。
- 前两时间片选择、第三时间片门禁。
- 通过门禁后才实现/写入多模型推理状态和新提交包。

### Out of scope

- 新随机种子或更多随机种子集成。
- 重建 Dataset2 full-100 特征缓存。
- 重新搜索 Setwise / LightGBM 外层权重。
- 用 slice2、全量指标或线上结果调参。
- 修改 Dataset1 模型、预测或 CSV 字节。
- 在本实验中搜索多个时间衰减半衰期。

## 3. Current State

### 当前 Dataset2 冠军

- Setwise 模型：最近 200k、完整 100 候选、seed60。
- 验证特征：`cache/supervised_features/dataset2_joint_recent200k_full100_val_seed60_20260725.val.npy`
- 验证形状：`(20000, 100, 63)`。
- 外层融合：`0.80 Setwise + 0.20 LightGBM`。
- full MRR：`0.5469178184464882`。
- slice0：`0.5863014322270679`。
- slice1：`0.5482466913826506`。
- slice2：`0.5061992242459902`。
- 选择前缀：验证行 `[0, 13334)`，对应前两个时间片。
- 证据：`result/d1_champion_d2_setwise_w080_seed60_20260725/evaluation-report.json`。

### 已否定方向

- seed17 / seed41 均弱于 seed60。
- 三种子均匀 rank ensemble full MRR 为 `0.545261123778906`，低于当前冠军 `0.0016566946675822258`。
- 结论：随机初始化没有产生可用多样性；本轮禁止继续增加种子。

### 可复用资产

- 200k full-100 训练 memmap；50k / 100k 直接取其尾部连续行，不复制、不重建。
- 20k full-100 验证 memmap。
- 当前 200k seed60 Setwise 模型及预测。
- 当前 Dataset2 LightGBM 验证分数。
- 当前冠军包中的 Dataset1 CSV；若最终打包，其 SHA-256 必须保持不变。

## 4. Frozen Protocol

### 4.1 专家定义

| 专家 ID | 训练行 | 行权重 | seed | 处理 |
|---|---:|---|---:|---|
| `recent50k` | `train[-50_000:]` | 全 1 | 60 | 新训练 |
| `recent100k` | `train[-100_000:]` | 全 1 | 60 | 新训练 |
| `recent200k` | `train[-200_000:]` | 全 1 | 60 | 复用当前冠军 Setwise |
| `recent200k_decay100k` | `train[-200_000:]` | 指数时间衰减 | 60 | 新训练 |

时间衰减在开始训练前固定为：

```text
raw_weight[i] = 2 ** (-(N - 1 - i) / 100000)
weight = raw_weight / mean(raw_weight)
```

其中 `i=0` 为窗口中最旧行，`i=N-1` 为最新行。归一化只保持 loss 尺度可比，不改变相对权重。不得根据验证结果改变半衰期。

### 4.2 训练契约

- 所有新模型继承当前 200k seed60 模型的 Setwise 配置、特征列、初始化、batch size、epoch 与 early-stopping 规则。
- 唯一允许变化：
  - 硬窗口模型：输入训练行数；
  - 衰减模型：每行固定 loss 权重。
- `train_row_weights=None` 必须保持现有训练行为。
- 每个新模型只用验证行 `[0, 13334)` 做 early stopping，并在冻结模型后才对同一 20k 验证集生成概率；slice2 不参与 epoch 或方案选择。既有 `recent200k` 冠军模型只作为不可更改基线复用。

### 4.3 候选融合与选择

1. 形成四个 Setwise 专家概率。
2. 枚举四个专家的全部非空子集，共 15 个候选。
3. 每个子集内部对专家概率做均匀算术平均。
4. 每个候选统一使用：

   ```text
   final_probability = 0.80 * mean(setwise_probabilities)
                     + 0.20 * champion_lightgbm_probability
   ```

5. 只在 `[0, 13334)` 上按 MRR 选择最高候选。
6. 平分规则依次为：
   - 更少的专家；
   - 按冻结专家顺序
     `recent50k, recent100k, recent200k, recent200k_decay100k`
     生成的子集字典序。
7. `recent200k` 单模型候选就是当前冠军，确保候选集包含零变化基线。
8. 在打开 `[13334, 20000)` 之前，必须持久化：
   - 候选清单；
   - 每个候选的前缀 MRR；
   - 唯一选中子集；
   - 所有输入文件 SHA-256；
   - 冻结协议与代码版本。

### 4.4 不可见门禁

打开第三片后，只评估已锁定的唯一候选。相对当前 Dataset2 冠军必须同时满足：

- full MRR 增量 `>= +0.0010000000000000`；
- slice0 增量 `>= 0`；
- slice1 增量 `>= 0`；
- slice2 增量 `>= 0`；
- 运行复现误差 `<= 1e-12`；
- 若打包，Dataset1 CSV SHA-256 与当前冠军完全一致。

任一条件失败即 `rejected`，不替换冠军、不打包。不得在看到第三片结果后改窗口、半衰期、融合方式、权重或平分规则。

## 5. Phased Route

### Phase 0 — 输入与基线审计

- 验证 200k / 20k memmap 形状、dtype、SHA-256。
- 验证当前 200k Setwise 和 LightGBM 分数能精确复现冠军四项指标。
- 验证 Dataset1 冻结 CSV 的路径、大小与 SHA-256。

**Proof**：生成 preflight JSON；基线误差每项 `<= 1e-12`。

### Phase 1 — RED

- 先写失败测试，锁住：
  - 50k / 100k 必须取 200k 缓存尾部连续行；
  - 衰减权重为正、单调不减、均值为 1、半衰期比例正确；
  - weighted Setwise loss 的 batch 权重跟随 shuffle 后行索引；
  - 前缀选择不读取或依赖 slice2；
  - 平分规则确定且可复现。

**Proof**：记录目标测试在实现前因缺失行为而失败。

### Phase 2 — GREEN / REFACTOR

- 为 streaming listwise trainer 增加可选训练行权重。
- 实现时间窗口视图、固定衰减权重和前缀子集选择器。
- 保持无权重训练路径行为不变。

**Proof**：目标测试及相关回归测试通过；格式与静态检查通过。

### Phase 3 — 训练与盲选

- 复用 200k 模型。
- 依次训练 `recent50k`、`recent100k`、`recent200k_decay100k`。
- 生成四专家验证概率。
- 只计算并持久化前缀选择报告，锁定唯一子集。

**Proof**：选择报告中不包含 slice2 / full 指标，且文件哈希在门禁前固定。

### Phase 4 — 第三片门禁

- 读取已锁定选择报告。
- 计算 full 与三个 slice 指标。
- 与当前冠军逐项比较，输出 accepted / rejected。

**Proof**：evaluation report 包含绝对值、delta、门禁布尔值和输入哈希。

### Phase 5 — 条件打包

- 仅 accepted 时：
  - 为选中多专家子集写入可复现推理状态；
  - 生成 Dataset2 新预测；
  - 原样复制当前冠军 Dataset1 CSV；
  - 复算本地 checkpoint、CSV、离线指标和 SHA-256；
  - 输出新提交包及 manifest。
- rejected 时：只保留实验报告和模型产物，不生成候选提交包。

**Proof**：manifest、checkpoint 重放结果、Dataset1 字节相等检查。

## 6. Per-phase Rules

- 第三片是门禁，不是调参集。
- 任何选择报告写入后不得覆盖；门禁报告另存。
- 所有产物路径必须带 `20260726` 实验 ID。
- 所有训练命令必须通过项目 `uv run` 执行。
- 不覆盖当前冠军 checkpoint、CSV 或报告。
- 不重新训练 200k seed60 基线。
- 不新增随机种子。
- 超过一次训练失败只允许修复实现/环境错误；不得借机改变实验协议。

## 7. Todo With Proof

| Todo | 完成证据 | 状态 |
|---|---|---|
| 审计三种子失败和当前冠军 | 两份 evaluation report 的指标与哈希 | done |
| 冻结窗口/衰减/选择/门禁协议 | 本目标文档 | done |
| RED：新增行为测试 | RED 命令与正确失败原因 | done |
| GREEN：实现 weighted Setwise 与选择器 | 目标测试通过 | done |
| 回归：无权重路径兼容 | 相关 pytest 通过 | done |
| 远端 preflight | frozen-config.json 与资产哈希 | done |
| 训练 50k / 100k / decay100k | 三份模型与训练报告 | done |
| 前缀盲选并锁定 | selection-report.json | done |
| 第三片门禁 | evaluation-report.json | done |
| 条件打包或明确拒绝 | `rejected`，未打包 | done |

## 8. Dry Run

1. 加载 200k 训练 memmap；确认 `recent50k` 与 `recent100k` 分别是其最后 50k / 100k 行的只读视图。
2. 加载当前 200k Setwise 模型与验证分数，确认固定 `0.80/0.20` 融合复现基线。
3. 训练两个硬窗口模型；训练一个固定衰减模型。
4. 保存四个模型的 20k 概率，但选择进程先只把 `[0,13334)` 交给选择器。
5. 枚举 15 个候选，锁定前缀 MRR 最高者，写 selection report。
6. 新进程读取 selection report 与输入哈希，拒绝任何不一致。
7. 只对锁定候选打开 slice2，计算门禁。
8. 若门禁通过，再实现/验证多模型 checkpoint 与打包；否则停止。

## 9. Go / No-Go

**Go**：缓存、当前模型和基线可精确复现；目标测试能锁住末片隔离与加权 loss；远端 GPU / 磁盘满足三次训练。

**No-Go**：

- 200k / 20k 缓存哈希或形状不匹配；
- 当前冠军无法以 `<= 1e-12` 误差复现；
- 无法证明选择过程隔离 slice2；
- Dataset1 冻结 CSV 无法定位或哈希不一致；
- 训练实现改变无权重路径；
- 门禁未同时满足所有冻结条件。

## 10. Stop Rule

- 本轮最多新增三个模型：50k、100k、200k-decay100k。
- 本轮只测试一个固定半衰期：100k 行。
- 门禁失败即停止，不追加 25k / 150k 窗口，不搜索衰减参数，不改融合权重。
- 门禁通过且打包复现完成，目标才算完成。

## 11. Result

- 锁定子集：`recent100k + recent200k + recent200k_decay100k`。
- 前两片合并选择 MRR：`0.5681553233702769`；当前冠军为 `0.5672740618048593`。
- candidate full MRR：`0.5483139338179761`；相对冠军 `+0.0013961153714878716`。
- slice delta：
  - slice0：`+0.0020468235138509927`；
  - slice1：`-0.00028430038301574534`；
  - slice2：`+0.002425977455217332`。
- 结论：full 增益与第三片门禁均为正，但 slice1 退化，违反冻结的三片不退化条件；状态 `rejected`，未生成包，当前冠军保持不变。
- 证据：`result/dataset2_setwise_window_diversity_20260726/selection-report.json` 与 `evaluation-report.json`。
