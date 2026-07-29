# 各塔优化器与双塔 In-Batch Negatives 结果 — 2026-07-28

> **2026-07-30 勘误**：本报告保留为历史执行记录，但不再支持模型效应结论。
> 原筛选器对 exact ties 偏向正例列，in-batch destination 又把真实 bucket 0
> 误作中性 context。两处均已在本地修复；修复证据见
> `tower-optimization-inbatch-negatives-bugfix-tdd-20260730.md`。四臂需要重新
> 生成后才能重新判决。

## 结论

实现完成，但本轮没有候选获得晋级权：

- 四个可学习塔均已支持独立 learning rate、`constant/cosine` scheduler、
  minimum LR ratio 和 weight decay；
- Two-Tower 已支持默认关闭的 multi-positive in-batch auxiliary loss；
- 新字段默认不改变旧 checkpoint；
- Dataset2 Two-Tower 2×2 筛选的三个候选全部被预注册硬门禁拒绝；
- 不进入最终集成 rolling-origin，不打开 external，不生成提交包，也不改冠军默认值。

## 为什么必须使用 Matched Control

首次报告复用了 2026-07-24 的历史 control checkpoint。其分数可以精确回放，但
今天用同代码、同 seed、同数据、同结构重新训练的 control 与历史模型差异很大：

| Metric | 历史 control | matched control | Delta |
| --- | ---: | ---: | ---: |
| MRR | 0.4641248061 | 0.4411252910 | -0.0229995150 |
| Hit@1 | 0.35680 | 0.32695 | -0.02985 |
| Hit@3 | 0.52875 | 0.48585 | -0.04290 |
| Hit@10 | 0.67295 | 0.66260 | -0.01035 |
| NDCG@10 | 0.5103898961 | 0.4828223440 | -0.0275675521 |
| Mean rank | 25.27125 | 12.61095 | -12.66030 |

因此，各臂内置的“相对历史 control”结果只保留为诊断；最终结论全部来自
same-code/same-seed matched report。这一步避免把训练代码漂移或训练随机性误记为
scheduler 收益。

## 冻结配置

- 数据：Dataset2；
- 训练前缀：1,441,568 events；
- 训练抽样：200,000 positive events；
- 每行显式候选：1 positive + 99 negatives；
- objective：listwise；
- early stop：完整候选组 MRR，patience 3；
- batch size：512；
- seed：60；
- control：constant LR、weight decay 0、in-batch off；
- optimizer-only：cosine，minimum LR ratio 0.1，weight decay `1e-4`；
- inbatch-only：constant LR、weight decay 0、in-batch weight 1.0、
  temperature 1.0；
- combined：同时打开上述 optimizer 和 in-batch 设置；
- 验证：20,000 × 100，三个等长时间片。

没有根据中途结果扫描 scheduler、weight decay、in-batch weight 或 temperature。

## Same-Code Matched 结果

| Arm | MRR | Hit@1 | Hit@3 | Hit@10 | NDCG@10 | Mean rank | Gate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| control | 0.4411252910 | 0.32695 | 0.48585 | 0.66260 | 0.4828223440 | 12.61095 | baseline |
| optimizer-only | 0.4913865342 | 0.36735 | 0.55260 | 0.75980 | 0.5503651070 | 17.01970 | reject |
| inbatch-only | 0.3276585510 | 0.31740 | 0.31740 | 0.31740 | 0.3174000000 | 46.68435 | reject |
| combined | 0.4878573286 | 0.36955 | 0.55335 | 0.72780 | 0.5402002747 | 19.04755 | reject |

### Delta vs Matched Control

| Arm | ΔMRR | ΔHit@1 | ΔHit@3 | ΔHit@10 | ΔNDCG@10 | ΔMean rank |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| optimizer-only | +0.0502612431 | +0.04040 | +0.06675 | +0.09720 | +0.0675427631 | **+4.40875** |
| inbatch-only | -0.1134667401 | -0.00955 | -0.16845 | -0.34520 | -0.1654223440 | **+34.07340** |
| combined | +0.0467320376 | +0.04260 | +0.06750 | +0.06520 | +0.0573779308 | **+6.43660** |

平均排名越低越好；表中的正 delta 是恶化。

### 时间片稳定性

| Arm | ΔMRR slice 0 / 1 / 2 | ΔNDCG@10 slice 0 / 1 / 2 | ΔMean rank slice 0 / 1 / 2 |
| --- | --- | --- | --- |
| optimizer-only | +0.02266 / +0.06384 / +0.06428 | +0.02860 / +0.08077 / +0.09326 | +7.85914 / +3.40753 / +1.96010 |
| inbatch-only | -0.31044 / -0.08078 / +0.05079 | -0.37848 / -0.13267 / +0.01484 | +46.32973 / +32.53907 / +23.35323 |
| combined | +0.00730 / +0.06440 / +0.06848 | +0.00301 / +0.07709 / +0.09203 | +11.42259 / +5.02265 / +2.86531 |

optimizer-only 和 combined 的 top-heavy 指标在每片都上涨，但每片平均排名也都恶化，
因此不是单一旧时间段造成的偶然失败。

### Query Movement

| Arm | Improved | Worsened | Unchanged | Net |
| --- | ---: | ---: | ---: | ---: |
| optimizer-only | 8,304 | 4,850 | 6,846 | +3,454 |
| inbatch-only | 3,634 | 13,203 | 3,163 | -9,569 |
| combined | 8,033 | 5,436 | 6,531 | +2,597 |

optimizer-only 和 combined 把更多 query 推到前排，但少数被伤害的 query 下坠更深，
最终表现为 MRR/Hit/NDCG 上涨而 mean rank 恶化。按预注册规则，mean rank 是硬门禁，
不能因为 MRR 涨幅大而事后删除。

## 训练稳定性与成本

- control：387.64 秒；
- optimizer-only：383.27 秒；
- inbatch-only：401.44 秒；
- combined：393.78 秒；
- 四臂总计：1,566.13 秒（约 26.1 分钟）；
- 四个训练进程 exit 均为 0；
- 无 OOM、NaN、Inf 或 traceback；
- 训练以 nice 10 串行执行，未与其他 GPU compute 进程并发。

## 决策

1. 保留代码能力和关闭默认值，不写入冠军配置。
2. 拒绝 in-batch weight 1.0；它产生明显的两极化排名，不能进入集成。
3. optimizer-only 和 combined 虽有强 top-of-list 信号，但违反冻结的平均排名门禁，
   本轮不得进入 rolling/external。
4. 不根据本结果继续扫描 `in_batch_negative_weight=0.1/0.25/0.5`；如果未来重开，
   必须先预注册新的 tail-risk 约束或辅助权重，而不是用本次验证集反向调参。
5. 本次 Two-Tower 结果不证明 GNN、GRU 或 SourceProfile 的 scheduler 一定无效；
   它只禁止把塔级 optimizer 设置作为全局默认值直接推广。

## Evidence

- 权威 matched report：
  `artifacts/tower_optimization/matched-screen-report.json`
- matched report SHA-256：
  `f7a9a8e4ad8e6932350e5769f90407c948b9a8e7bd5bfe3ad5dd030b4ec1dae2`
- matched control report：
  `artifacts/tower_optimization/matched-control-report.json`
- 四臂日志：
  `artifacts/tower_optimization/{matched_control,optimizer_only,inbatch_only,combined}.log`
- 服务器权威目录：
  `result/dataset2_two_tower_opt_inbatch_seed60_20260728_matched_screen/`
