# Dataset2 Bounded Source Decoder 多时间跨度 OOF Residual Goal

## Go / No-Go

**Go。** 复用已经完成门禁的 `cap=0.10` bounded source decoder 与对应 rolling-origin 冻结 CST-A 主干，不重新训练模型；只对各 origin 之后的多个时间段做严格前向推理，生成可重放、带有效掩码和时间跨度审计的 OOF residual 专家库。

## 1. Target

生成一个独立的 Dataset2 多时间跨度 OOF residual 产物，供后续纯 Jittor 路由器或 Set Transformer 使用：

- `short`：每个 origin 后紧邻的下一时间片；
- `medium`：跳过一个时间片后预测；
- `long`：跳过两个时间片后预测；
- 每条 residual 都来自该样本时间之前已冻结的 CST-A 与 bounded decoder；
- 无覆盖位置显式为零并由 `valid-mask` 标为无效；
- 保存 base logits、corrected logits、residual、origin、gap day 与完整 cap/replay 审计。

## 2. Goal Definition

### 2.1 Verifiable objective

在远端 Jittor 环境生成：

`result/dataset2_bounded_source_multi_horizon_oof_20260727`

核心数组契约：

| 文件 | 形状 | 契约 |
|---|---:|---|
| `residuals.npy` | `[3, 200000, 100]` | `corrected - base`，无效位置严格为 0 |
| `base-logits.npy` | `[3, 200000, 100]` | 对应 origin 冻结 CST-A 的输出 |
| `corrected-logits.npy` | `[3, 200000, 100]` | bounded decoder 纠正后的输出 |
| `valid-mask.npy` | `[3, 200000]` | 唯一合法覆盖定义 |
| `origin-index.npy` | `[3, 200000]` | 使用的 rolling origin，未覆盖为 -1 |
| `gap-days.npy` | `[3, 200000]` | 查询时间距 origin 的天数，未覆盖为 NaN |
| `manifest.json` | - | 输入、切片、模型、哈希与形状 |
| `audit.json` | - | 因果边界、cap、有限值、回放误差 |
| `metrics.json` | - | 每个 horizon / origin 的 base 与 corrected MRR |

成功条件：

1. 任意有效行满足 `residual == corrected - base`，最大回放误差不超过 `1e-6`。
2. 任意 residual 的绝对值不超过 checkpoint 的硬 cap `0.10 + 1e-6`。
3. 任意切片满足 `train_stop <= score_start` 且 `origin_time < min(score_time)`。
4. 同一 horizon 内切片不重叠；无效行的三个 logits/residual 数组严格为零。
5. 产物只依赖 NumPy 数据编排与 Jittor 模型推理，不引入 sklearn/LightGBM。
6. 单测、静态检查和 CUDA 全量生成全部通过。

### 2.2 Canonical rolling lattice

沿用现有三个 bounded decoder origin：

| origin | 训练前缀 | short | medium | long |
|---:|---:|---:|---:|---:|
| 0 | `[0, 79909)` | `[79909, 118816)` | `[118816, 159804)` | `[159804, 200000)` |
| 1 | `[0, 118816)` | `[118816, 159804)` | `[159804, 200000)` | - |
| 2 | `[0, 159804)` | `[159804, 200000)` | - | - |

因此：

- `short` 覆盖 `[79909, 200000)`；
- `medium` 覆盖 `[118816, 200000)`；
- `long` 覆盖 `[159804, 200000)`；
- 最后一个时间片同时拥有 short / medium / long 三种严格 OOF residual，可直接训练分歧路由。

这里的 horizon 是“距模型冻结 origin 的真实时间跨度”，不是随机种子。实际 gap 范围由时间戳计算并写入审计；当前 200k rolling lattice 只能覆盖约 1–103 天，不能伪称模拟 468 天线上时间外推。

### 2.3 Boundary

本轮包含：

- 多 horizon 切片契约与审计模块；
- 使用已有冻结 checkpoint 批量推理；
- 对每个 target slice 按 origin 冻结 source history；
- 生成训练可消费的 OOF residual 专家库；
- 单测、远端 CUDA 运行和结果文档。

本轮不包含：

- 不重训 CST-A 或 bounded decoder；
- 不训练新的路由器/stacker；
- 不读取外部验证标签做选择；
- 不生成线上提交；
- 不把 103 天以内的结果外推成 468 天稳定性结论。

## 3. Current State

- 已有 rolling-origin CST-A checkpoints：
  `result/dataset2_source_conditioned_cst_abcd_20260727/folds/variant-A/fold-{0,1,2}/model.npz`
- 已有选定 bounded decoder checkpoints：
  `result/dataset2_bounded_source_decoder_20260727/folds/cap-0.10/fold-{0,1,2}/model.npz`
- 已有每个 origin 的 train base logits：
  `cache/bounded_id_residual/dataset2_frozen_a_20260727/fold-{0,1,2}/train-base-logits.npy`
- 已有 200k × 100 特征、候选、source、time、label 和 causal source sequence 缓存。
- 现有 bounded decoder 只对紧邻下一片做过验证；尚未把同一冻结 decoder 推向更远的未来切片，也没有统一 residual tensor。

## 4. Priority

1. P0：证明每条 residual 的训练边界早于查询行，杜绝伪 OOF。
2. P0：保证 cap、零填充、mask 与 replay 契约精确。
3. P1：复用现有 checkpoint，保证产物能由固定输入重放。
4. P1：提供每个 origin × horizon 的衰减曲线，为后续路由选择提供证据。
5. P2：压缩或分块写盘，避免无必要内存峰值。

## 5. Assumptions

- fold manifest 与三个现有 checkpoint 一一对应，且 checkpoint 训练数据严格止于 `train_stop`。
- CST-A 不依赖 source sequence，但 bounded decoder 必须使用截止 origin 的冻结 source history；生成器会重新构建目标片的 origin-frozen sequence。
- `cap=0.10` 是现有离线 fold 门禁选定值；本轮不再次用目标片标签选择 cap。
- 候选正例仍位于列 0，MRR 只用于产物诊断，不参与本轮模型选择。

## 6. Phased Route

### Phase 1 — Contract RED

- 为切片 lattice、严格时间边界、无覆盖零值、同 horizon 不重叠、cap 与 replay 写失败测试。
- 证据：测试因缺少模块/API 按预期失败。

### Phase 2 — Minimal GREEN

- 实现纯 NumPy 的切片/组装/审计模块。
- 证据：契约测试通过，未触碰模型逻辑。

### Phase 3 — Jittor generation

- 加载每个 origin 的 CST-A 与 bounded decoder；
- 生成更远目标片的 base logits；
- 构建 origin-frozen source sequence 与 support；
- 推理 corrected logits 并写入 canonical horizon tensor。
- 证据：远端 CUDA 日志、manifest、数组与逐片指标。

### Phase 4 — Verification and handoff

- 运行相关测试、ruff/py_compile；
- 核对文件哈希、形状、coverage、cap、replay、时间边界；
- 写 TDD 与结果文档，明确可用范围和下一步消费接口。

## 7. Todos with Proof

- [x] RED 测试能捕获 score 回看 train、horizon 重叠与无效行非零。
- [x] GREEN 模块输出固定 `[short, medium, long]` 轴顺序。
- [x] 生成器只加载 `cap-0.10` checkpoint，且记录 checkpoint SHA-256。
- [x] 三个 horizon 覆盖数分别为 120091、81184、40196。
- [x] 有效 residual 最大绝对值 `<= 0.100001`。
- [x] 无效位置 base/corrected/residual 全零。
- [x] `corrected == base + residual` 最大误差 `<= 1e-6`。
- [x] 每个切片 source history cutoff 等于其 origin，查询时间严格更晚。
- [x] 远端全量 CUDA 生成退出码为 0。

## 8. Dry Run

以最后时间片 `[159804, 200000)` 为例：

1. `short` 加载 origin 2 的 CST-A 与 bounded decoder，模型只见过 `[0,159804)`。
2. `medium` 加载 origin 1，模型只见过 `[0,118816)`。
3. `long` 加载 origin 0，模型只见过 `[0,79909)`。
4. 三路均用各自 origin 之前的 source history 编码同一批候选。
5. 三路输出各自的 `base`、`corrected` 与 `residual`。
6. 写入相同行坐标、不同 horizon 轴；路由器可直接比较三路 disagreement，同时通过 `origin-index` 和 `gap-days` 知道它们的时间新鲜度。
7. 若任何一路越过 `0.10` cap、出现 NaN、历史 cutoff 晚于 origin 或写入重复位置，则生成失败而不是静默产出。

## 9. Stop / No-Go Rules

- checkpoint 或 fold manifest 对不上时停止，不猜测 origin。
- 无法重建 origin-frozen source history 时停止，不退化为 query-time causal history。
- 任意有效 residual 越 cap、回放误差超阈值或无效位置非零时停止发布产物。
- 若只有 short 覆盖成功，则不称为 multi-horizon OOF。
