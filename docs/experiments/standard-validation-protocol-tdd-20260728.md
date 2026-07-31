# Hai TDD: 标准 rolling-origin 选择与长跨度 external gate

## Target Behavior

新的特征组和 ensemble 权重候选必须先预登记完整精确组合，再用至少三折
rolling-origin 的等权平均和跨折稳定性选出唯一候选；锁定后才能一次打开覆盖真实
部署 horizon 的 external holdout，且不能用结果反向扫描候选。

## RED

- **Test added**:
  `tests/test_standard_validation_protocol.py`，6 个协议测试。
- **Behavior asserted**:
  plan 先冻结；候选空间、selection policy、external policy 分别哈希绑定；单折峰值
  候选被跨折门禁拒绝；多折使用等权平均；external 校验 468 天终点 horizon；receipt
  先于分数读取且只能写一次；完整指标面板和禁止反向扫描标志必须存在。
- **Command**:
  `nvcc_path= JITTOR_HOME=/var/tmp/jittor-standard-validation-cache
  .venv-wsl/bin/python -m pytest
  tests/test_standard_validation_protocol.py -q`
- **Observed failure**:
  `ModuleNotFoundError: No module named
  'jgrec.standard_validation_protocol'`。
- **Failure is correct because**:
  标准协议模块尚不存在；失败不是语法、fixture 或环境问题。

## GREEN

- **Minimal implementation**:
  新增 `standard_validation_protocol.py`，实现 plan freeze、候选无关的等权多折选择、
  多指标硬门禁、selection lock、external 468 天 horizon 校验、one-shot receipt 和
  external report；复用现有 `ranking_metrics()`，不改旧协议产物。
- **Command**:
  同上。
- **Observed pass**:
  `6 passed`。

## REFACTOR

- **Refactor done**: yes
- **Change**:
  将选择优先级从旧协议的“最差折优先”调整为本任务明确要求的“多折平均 MRR
  优先、最差折 MRR 次级”；跨折稳定性仍是进入排序前的硬门禁。候选 ID 不再限定为
  数值权重，因此 feature group、ensemble weight 和它们的精确组合共用同一协议。
- **Command after refactor**:
  `.venv-wsl/bin/python -m pytest
  tests/test_standard_validation_protocol.py -q`
- **Observed result**:
  `6 passed`。

## Next Behavior

正式 rolling runner 必须能固定 feature mask 和 ensemble 权重，不能在每折的单一
tune split 内重新搜索。

---

## Target Behavior

Hybrid 每次正式 rolling 训练可接收一个冻结 feature candidate 和一个冻结
MLP/LGBM 权重，并完全绕过折内单切分候选扫描。

## RED

- **Test added**:
  `test_frozen_ensemble_weight_bypasses_single_split_weight_search` 和
  `test_frozen_feature_candidate_disables_single_split_mask_search`。
- **Behavior asserted**:
  exact feature mask 只训练一个候选；exact ensemble weight 原样返回；未知 mask 和
  非法权重失败。
- **Command**:
  `.venv-wsl/bin/python -m pytest
  tests/test_hybrid_fusion_lgbm.py::test_frozen_ensemble_weight_bypasses_single_split_weight_search
  tests/test_hybrid_fusion_lgbm.py::test_frozen_feature_candidate_disables_single_split_mask_search
  -q`
- **Observed failure**:
  `TrainingConfig.__init__()` 不接受
  `frozen_fusion_feature_candidate` 和
  `frozen_ensemble_mlp_weight`。
- **Failure is correct because**:
  Hybrid 尚无冻结正式候选的配置入口，当前只能执行单切分扫描。

## GREEN

- **Minimal implementation**:
  `TrainingConfig` 新增两个可选冻结字段；`_feature_masks()` 在冻结时只返回唯一精确
  mask；`_find_ensemble_weight()` 在冻结时校验并直接返回指定权重。
- **Command**:
  同上。
- **Observed pass**:
  `2 passed`。

## REFACTOR

- **Refactor done**: yes
- **Change**:
  使用 `getattr` 保持旧 checkpoint/config 的缺字段兼容；保留 `None` 作为探索模式，
  正式协议 runner 显式传冻结值。
- **Command after refactor**:
  见最终相关回归命令。
- **Observed result**:
  相关回归保持通过。

## Next Behavior

CLI 必须把两个冻结字段接到 Hybrid config，供服务器 runner 直接使用。

---

## Target Behavior

主 CLI 能传递标准验证所需的冻结 feature candidate 和 ensemble 权重。

## RED

- **Test added**:
  `tests/test_cli.py::test_cli_passes_frozen_standard_validation_candidates`。
- **Behavior asserted**:
  CLI 参数无损进入 `TrainingConfig`。
- **Command**:
  `.venv-wsl/bin/python -m pytest
  tests/test_cli.py::test_cli_passes_frozen_standard_validation_candidates -q`
- **Observed failure**:
  `CLIConfig.__init__()` 不接受两个冻结参数。
- **Failure is correct because**:
  配置层已有能力，但服务器入口尚未接线。

## GREEN

- **Minimal implementation**:
  `CLIConfig` 新增字段并由 `_ranker_config()` 传入 Hybrid。
- **Command**:
  同上。
- **Observed pass**:
  `1 passed`。

## REFACTOR

- **Refactor done**: no
- **Change**:
  无额外重构；保持 CLI 与 `TrainingConfig` 同名，减少映射歧义。
- **Command after refactor**:
  not needed
- **Observed result**:
  not needed

## Next Behavior

done；真实分数生成需等服务器恢复。
