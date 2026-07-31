# Hai TDD: Dataset2 Confidence-Routed Top-k ID Correction

## Target Behavior

冻结 CST logits，只允许 candidate-ID correction 改变 frozen-base top-k，
再由纯 Jittor router 在固定行预算内选择少量 query。目标结构必须保证：

- top-k 外候选逐元素不变；
- 未路由行逐元素不变；
- correction 绝对幅度不超过 `0.10`；
- 实际 route rate 不超过 5%/10% 预算；
- router inference features 不接收 positive/label；
- correction/router checkpoint 重载一致；
- 时间留出段没有改善样本时安全退回零路由。

## RED

- **Test added**:
  `tests/test_hybrid_confidence_routed_topk_id.py`
- **Behavior asserted**:
  top-k mask、hard row quota、候选置换、零初始化、correction checkpoint、
  router checkpoint。
- **Command**:

```bash
.venv/bin/python -m pytest -q \
  tests/test_hybrid_confidence_routed_topk_id.py
```

- **Observed failure**:

```text
ModuleNotFoundError:
  No module named 'jgrec.rankers.hybrid.confidence_routed_topk_id'
```

- **Failure is correct because**:
  测试在 production module 创建前运行，失败原因是新增行为完全不存在，
  不是语法或环境错误。

正式 Fold2 暴露了第二个边界：top5 router holdout 没有任何改善标签。
在修改实现前新增：

```text
test_router_with_no_positive_supervision_safely_routes_nothing
```

当前实现按预期 RED：

```text
ValueError: confidence router requires both label classes
```

这证明单类别时间窗口的安全 fallback 是真实缺失行为。

## GREEN

- **Minimal implementation**:
  - `TopKIDCorrection`：Jittor item embedding + zero-init linear head；
  - mask 内 centered `tanh` absolute-cap residual；
  - `ConfidenceRouter`：12 个 label-free descriptors 的小型 Jittor MLP；
  - hard quota：稳定 probability 排序，且要求 `p>=0.5`；
  - `sparse_correction_audit`：边界、top-k 外、未路由行和配额联合审计；
  - correction/router 的 NumPy 容器 + Jittor state checkpoint；
  - 无正例时 output weight 清零、bias 固定为 `-20`，形成可保存的纯 Jittor
    no-route router。
- **Command**:

```bash
.venv/bin/python -m pytest -q \
  tests/test_hybrid_confidence_routed_topk_id.py
```

- **Observed pass**:

```text
7 passed in 2.71s
```

## REFACTOR

- **Refactor done**: yes
- **Change**:
  - correction proposal、router features、hard route 和 audit 分成独立纯函数；
  - top5/top10 共用同一个模型 pipeline，5%/10% route budget 只改变确定性路由；
  - router label 生成函数与 inference feature 函数分离，防止真值进入推理；
  - 单类别 fallback 仍使用同一 checkpoint contract，不引入 sklearn 或特殊
    非模型旁路。
- **Command after refactor**:

```bash
.venv/bin/ruff check \
  src/jgrec/rankers/hybrid/confidence_routed_topk_id.py \
  tests/test_hybrid_confidence_routed_topk_id.py \
  scripts/train_dataset2_confidence_routed_topk_id.py
```

- **Observed result**: `All checks passed!`

完整相关回归：

```text
30 passed, 2 NumPy subnormal warnings in 2.91s
external_not_read
```

## CUDA Smoke

真实 frozen base 取 1,536 行，前 1,024 行训练 correction，后 512 行生成
router supervision：

```text
topk-id-correction epoch=1 train_loss=2.377287
confidence-router epoch=1 train_loss=1.409128
status=complete
```

Smoke 的 sparse audit 通过；指标不参与正式 selection。

## Next Behavior

本轮结构行为已完成。实验结果表明 correction opportunity 随时间快速消失；
后续如果继续，应更换信号而不是放宽当前 top-k/route 安全边界。
