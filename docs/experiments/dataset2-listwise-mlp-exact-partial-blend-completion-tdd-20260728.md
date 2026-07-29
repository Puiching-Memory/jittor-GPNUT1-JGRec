# Hai TDD: Dataset2 listwise-MLP 精确部分混合完成

## Target Behavior

用不可事后修改的三折 rolling-origin 协议物化当前冠军与基础 listwise-MLP 的
精确部分混合候选；只有跨折稳定通过才生成 selection lock，并要求后续 external
候选的 MRR 严格提升。

## RED

- **Test added**:
  `tests/test_listwise_mlp_exact_blend.py`
- **Behavior asserted**:
  固定三折边界、权重必须精确匹配旧 frozen config、候选必须执行
  `(1-w)*baseline + w*auxiliary`、manifest 必须保留 integration identity 和
  candidate fingerprint。
- **Command**:
  `uv run --no-sync pytest tests/test_listwise_mlp_exact_blend.py -q`
- **Observed failure**:
  `ModuleNotFoundError: No module named 'jgrec.listwise_mlp_exact_blend'`
- **Failure is correct because**:
  目标契约模块尚不存在，失败发生在待实现行为而非测试夹具。

第二个 RED：

- **Test added**:
  `tests/test_robust_weight_selection.py::test_external_requires_strict_mrr_improvement`
- **Behavior asserted**:
  external candidate 与 baseline MRR 相等时必须拒绝，并显式报告
  `mrr_strictly_increasing=false`。
- **Command**:
  `uv run --no-sync pytest tests/test_robust_weight_selection.py::test_external_requires_strict_mrr_improvement -q`
- **Observed failure**:
  `KeyError: 'mrr_strictly_increasing'`
- **Failure is correct because**:
  原 evaluator 只实现了 MRR 不下降，尚未实现目标要求的严格提升门禁。

## GREEN

- **Minimal implementation**:
  新增 `src/jgrec/listwise_mlp_exact_blend.py`，实现固定折定义、冻结权重校验、
  精确候选物化、哈希/fingerprint 和 rolling manifest；新增远端 runner，按折训练
  fixed-4 Setwise 与 fixed-5 auxiliary；把 external MRR 门禁改为严格大于容差。
- **Command**:
  `uv run --no-sync pytest tests/test_listwise_mlp_exact_blend.py tests/test_robust_weight_selection.py -q`
- **Observed pass**:
  本地 `12 passed`。

远端 Jittor 回归：

- **Command**:
  `uv run --no-sync pytest tests/test_listwise_mlp_exact_blend.py tests/test_robust_weight_selection.py tests/test_hybrid_fusion_listwise.py -q`
- **Observed pass**:
  Linux/Jittor 环境 `23 passed`。

## REFACTOR

- **Refactor done**: yes
- **Change**:
  将 LightGBM batch 的列选择拆成“先切 batch、再按最后一维取列”，避免 NumPy
  advanced indexing 改变轴顺序；统一格式并保持 runner 不接受 external 参数。
- **Command after refactor**:
  `uv run --no-sync ruff check src/jgrec/listwise_mlp_exact_blend.py src/jgrec/robust_weight_selection.py scripts/train_dataset2_listwise_mlp_exact_rolling.py scripts/select_robust_integrated_weight.py tests/test_listwise_mlp_exact_blend.py tests/test_robust_weight_selection.py`
- **Observed result**:
  `All checks passed!`

## Next Behavior

Done。三折真实训练已经完成，selector 正确拒绝全部六个权重；未生成 lock、
未打开 external、未生成提交包。
