# Dataset2 Bounded Source Decoder 多时间跨度 OOF Residual 结果

## 结论

多时间跨度 OOF residual 已完整生成，可直接作为后续纯 Jittor 路由器的
专家输入。六个 origin × target 切片全部为正收益；decoder 的增益随模型
陈旧度稳定衰减，但在本次可观测的 104 天以内没有翻负。

这不是新提交，也没有替换线上冠军。它是严格 OOF 训练资产。

## 产物

远端完整目录：

`/home/edu/workspace/jittor-GPNUT1-JGRec/result/dataset2_bounded_source_multi_horizon_oof_20260727`

| 文件 | 形状 | 大小 |
|---|---:|---:|
| `residuals.npy` | `[3, 200000, 100]` | 240,000,128 bytes |
| `base-logits.npy` | `[3, 200000, 100]` | 240,000,128 bytes |
| `corrected-logits.npy` | `[3, 200000, 100]` | 240,000,128 bytes |
| `valid-mask.npy` | `[3, 200000]` | 600,128 bytes |
| `origin-index.npy` | `[3, 200000]` | 1,200,128 bytes |
| `gap-days.npy` | `[3, 200000]` | 2,400,128 bytes |

本地保留了 manifest、audit、metrics 与运行日志；大数组留在远端，避免无
意义复制约 724 MB。

## Horizon 结果

| Horizon | gap days | OOF rows | Base MRR | Corrected MRR | Delta |
|---|---:|---:|---:|---:|---:|
| short | 1–37 | 120,091 | 0.425114 | 0.428662 | **+0.003548** |
| medium | 33–72 | 81,184 | 0.422559 | 0.425101 | **+0.002542** |
| long | 68–104 | 40,196 | 0.422481 | 0.424282 | **+0.001802** |

相对 short，medium 增益衰减约 28.4%，long 衰减约 49.2%。这是“时间越
远越不可信”的直接证据，也说明后续路由不能给旧 residual 固定大权重。

六个独立切片的 delta：

| Horizon | Origin | Target fold | Delta MRR |
|---|---:|---:|---:|
| short | 0 | 0 | +0.003703 |
| short | 1 | 1 | +0.003967 |
| short | 2 | 2 | +0.002972 |
| medium | 0 | 1 | +0.002467 |
| medium | 1 | 2 | +0.002618 |
| long | 0 | 2 | +0.001802 |

## 新发现

1. **时间衰减是连续的，不是突然失效。**
   1–37、33–72、68–104 天三档均为正，但增益单调下降。
2. **多 horizon 有真实但有限的多样性。**
   short/long residual 相关系数为 `0.8657`，最大 residual 候选一致率为
   `77.88%`；short/medium 相关系数更高，为 `0.9347`。因此它比随机种子
   更有结构性差异，但不适合无约束平均。
3. **陈旧模型仍能纠正一小部分高价值行。**
   long 的逐行 gain 比例为 `8.82%`，loss 比例为 `6.60%`，其余
   `84.58%` 不改变 MRR。最合理的用途是低覆盖、高置信路由，不是全量融合。
4. **当前证据不能覆盖线上时间跨度。**
   训练 lattice 最长只有 104 天，而外部验证距完整训练尾部约 468 天。
   该产物证明了“短中期衰减规律”，不能证明 468 天 residual 仍为正。

## 审计

- hard cap：通过；最大绝对 residual `0.1000003815`
- cap 浮点超量：`3.815e-7`，低于 `2e-6` 容差
- 行内零均值：最大误差 `3.406e-7`
- `base + residual == corrected`：最大误差 `3.725e-9`
- invalid 行零填充：通过
- origin/gap metadata：通过
- short source sequence 与原缓存：三个 origin 全部逐字节一致
- short logits 与原结果：最大 GPU 重放误差不超过 `3.815e-6`
- 训练框架：`["jittor"]`
- 非 Jittor 可训练模型：`[]`
- 新训练：无，只复用已有 checkpoint 做 Jittor 推理

## 后续最有价值的使用方式

在最后时间片 `[159804, 200000)` 上，三路 residual 同时有效。下一步应：

1. 把该片再按时间切成前两段训练、最后一段不可见门禁；
2. 路由输入使用 residual disagreement、各路 top1/margin、gap days、
   source support/last-hit 差；
3. 默认保持最新 short 路，只有预测收益超过高阈值时才切换 medium/long
   或启用纠正；
4. gate 输出必须是 bounded top-k correction，不能自由重排 100 个候选；
5. 外部 468 天仍单独门禁，失败时线上冠军保持不变。

## 验证命令

```bash
.venv/bin/python -m pytest \
  tests/test_hybrid_multi_horizon_oof.py \
  tests/test_hybrid_bounded_source_decoder.py \
  tests/test_hybrid_source_sequence_cache.py -q

bash scripts/run_dataset2_bounded_source_multi_horizon_oof_20260727.sh
```

