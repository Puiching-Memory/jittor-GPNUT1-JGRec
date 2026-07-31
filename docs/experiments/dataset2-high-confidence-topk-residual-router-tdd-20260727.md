# Hai TDD: 默认 Short 的纯 Jittor Bounded Top-k Router

## Target Behavior

默认逐元素返回 short corrected logits；仅对高置信行选择 medium/long
residual，并且只改 short 当前 top-k、行内改变量和为零、最大幅度受第二层
cap 限制。路由特征不能使用 candidate ID 或正例列，候选共同置换后结果应
保持不变。

## RED

- **Test added**:
  `tests/test_hybrid_high_confidence_topk_router.py`
- **Behavior asserted**:
  exact short fallback、top-k 外零改动、cap、行内零均值、稀疏 hard route、
  候选置换不变与严格 timestamp split。
- **Command**:
  `.venv/bin/python -m pytest
  tests/test_hybrid_high_confidence_topk_router.py -q`
- **Observed failure**:
  第一轮测试收集报
  `ModuleNotFoundError: high_confidence_topk_router`。
- **Failure is correct because**:
  bounded alternative 与 hard route 公共契约尚未实现。

## GREEN

- **Minimal implementation**:
  只实现 pure NumPy 的 bounded top-k alternative、hard route、
  permutation-invariant summary 与 60/20/20 timestamp split。
- **Command**:
  `.venv/bin/python -m pytest
  tests/test_hybrid_high_confidence_topk_router.py -q`
- **Observed pass**:
  `5 passed`。

## RED 2

- **Test added**:
  reward target、bounded route audit、Jittor reward MLP checkpoint replay。
- **Observed failure**:
  `ImportError: cannot import ResidualAdvantageRouterConfig`。
- **Failure is correct because**:
  可训练模块和 checkpoint 协议尚不存在。

## GREEN 2

- **Minimal implementation**:
  新增两层 `jt.nn.Module` reward router、train-only normalizer、固定 epoch
  加权 reward regression，以及纯 Jittor checkpoint save/load/predict。
- **Observed pass**:
  `7 passed`，checkpoint 重放误差在单测中 `<= 1e-6`。

## RED 3

- **Test added**:
  candidate raw feature、alternative top1 与 promoted/demoted support 差的
  permutation-invariance，以及 checkpoint 中的精确 feature-name contract。
- **Observed failure**:
  `ImportError: cannot import router_candidate_support_features`。
- **Failure is correct because**:
  第一版只有行级 residual summary，尚未具备候选支持差。

## GREEN 3

- **Minimal implementation**:
  每路增加：
  - default top1 的 63 维原始特征；
  - alternative top1 - default top1；
  - 最大提升候选 - 最大压低候选；
  并把完整 353 维 feature-name contract 存入 checkpoint。
- **Observed pass**:
  router 单测 `8 passed`。

## REFACTOR

- **Refactor done**: yes
- **Change**:
  runner 固定六个 `top_k × cap` variant，按 origin 公共数据一次构造
  alternative；selection lock 落盘后才计算实际 gate 指标。第一版纯 summary
  gate 为 0 后，保留其结果作为 ablation，v2 仅扩充候选支持特征，没有放松
  fallback/top-k/cap。
- **Command after refactor**:
  `.venv/bin/python -m ruff check
  src/jgrec/rankers/hybrid/high_confidence_topk_router.py
  scripts/train_dataset2_high_confidence_topk_router.py
  tests/test_hybrid_high_confidence_topk_router.py`

  `.venv/bin/python -m pytest
  tests/test_hybrid_high_confidence_topk_router.py
  tests/test_hybrid_multi_horizon_oof.py
  tests/test_hybrid_bounded_source_decoder.py -q`
- **Observed result**:
  ruff/py_compile 通过，相关测试 `18 passed`；v2 selected checkpoint 独立
  SHA-256 匹配，实际 prediction replay error 为 `0.0`。

## Next Behavior

done。若继续优化，应把 reward regression 换成 candidate-set gain
classification，并保留本轮 v2 的 frozen gate 作为比较基线。
