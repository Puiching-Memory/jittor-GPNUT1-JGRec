# Hai TDD: Partial-Listwise Online Candidate Packaging

## Target Behavior

在不修改当前冠军的前提下，把一个已验证的 Dataset2 完整 reranker 提交概率以
固定 `0.20` 混入冠军，生成可直接提交的双数据集 ZIP；Dataset1 必须原字节
复用，Dataset2 必须满足固定公式和标准提交结构。

## RED 1

- **Test added**:
  `tests/test_partial_listwise_submission.py`
- **Behavior asserted**:
  - 报告 Hit@K 和排名分布，不只提供 MRR；
  - 固定 residual blend 公式；
  - 8 位 CSV 的全量舍入误差受控；
  - Dataset1 ZIP member 原字节复制；
  - ZIP 只有两个目标 member；
  - 已有输出目录拒绝覆盖；
  - 非归一专家矩阵被拒。
- **Command**:
  `uv run --no-sync pytest tests/test_partial_listwise_submission.py -q`
- **Observed failure**:
  `ModuleNotFoundError: No module named 'jgrec.partial_listwise_submission'`
- **Failure is correct because**:
  生产打包 API 尚不存在。

## GREEN 1

- **Minimal implementation**:
  新增 `jgrec.partial_listwise_submission`，只实现 multi-metric panel、冻结混合、
  CSV/ZIP 生成、哈希和结构校验。
- **Command**:
  `uv run --no-sync pytest tests/test_partial_listwise_submission.py -q`
- **Observed pass**:
  `3 passed`。

## RED 2

- **Test added**:
  `test_materialize_submission_member_scores_verifies_source_and_shape`
- **Behavior asserted**:
  从精确 ZIP/member 哈希恢复 Dataset2 概率矩阵，验证 shape、概率、行归一，
  并拒绝覆盖已有 `.npy`。
- **Command**:
  `uv run --no-sync pytest tests/test_partial_listwise_submission.py -q`
- **Observed failure**:
  `ImportError: cannot import name 'materialize_submission_member_scores'`
- **Failure is correct because**:
  完整 reranker ZIP 被发现后，生产代码还没有受测试保护的 member 恢复接口。

## GREEN 2

- **Minimal implementation**:
  新增原子化 member materialization；以来源 ZIP/member SHA-256、shape、
  finite/range 和 `1e-6` 行和容差为合同。
- **Command**:
  `uv run --no-sync pytest tests/test_partial_listwise_submission.py -q`
- **Observed pass**:
  `4 passed`。

## REFACTOR

- **Refactor done**: yes
- **Change**:
  发现历史完整 Two-Tower reranker 包后，删除不再需要的 GPU test scorer，
  把流程收敛为“核验并恢复完整直接候选 → 固定 0.20 混合”。这减少了重复推理
  和 standalone/完整集成语义漂移。
- **Command after refactor**:
  `uv run --no-sync pytest tests/test_partial_listwise_submission.py
  tests/test_hybrid_partial_listwise_blend.py tests/test_submission.py -q`
- **Observed result**:
  `18 passed`；相关 Ruff 检查通过。

## Artifact Evidence

- Candidate ZIP:
  `result/d1_time_ramp_g050_d2_short_none50k_setwise_w080_twotower_full_w020_20260728/result.zip`
- ZIP SHA-256:
  `10fe35d73d7981e29a33a3bab45e8e7737fdc9686f5c48c5a76679e0e263a1c6`
- Dataset1 byte identity: passed。
- Dataset2 full formula replay: maximum error
  `4.0000002199391815e-09`。
- Standard row/column/finite/range validation: passed。

## Next Behavior

用户提交后回传线上分数。只有严格高于 `1.3557002251184347`，才为这个完整
reranker partial blend 增加正式 checkpoint 持久化和标准双回放。
