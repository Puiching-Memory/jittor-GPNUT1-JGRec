# Hai TDD: Dataset2 OOF 路由与 top-k LambdaMRR 联训

## Target Behavior

纯 Jittor 联合模型必须让 route reward loss 和 top-k LambdaMRR loss 同时反向传播
到共享主干；推理只在默认 short 的 top-k 内应用 route-specific residual，并能从
checkpoint 精确重放 route advantage 与 candidate residual 两个输出。

## RED

- **Test added**:
  `test_joint_loss_backpropagates_to_shared_router_and_rank_heads`
- **Behavior asserted**: route head、candidate head 和共享行级主干都收到非零联合梯度。
- **Command**:
  `.venv/bin/python -m pytest tests/test_hybrid_joint_oof_lambdamrr.py -q`
- **Observed failure**:
  `ModuleNotFoundError: jgrec.rankers.hybrid.joint_oof_lambdamrr`
- **Failure is correct because**: 联合模型与联合 loss API 尚不存在；不是测试语法或
  Jittor 环境错误。

## GREEN

- **Minimal implementation**: 新增共享行级主干、2 路 route advantage head、2 路
  candidate residual head，以及 route MSE + LambdaMRR pairwise softplus 联合 loss。
- **Command**:
  `.venv/bin/python -m pytest tests/test_hybrid_joint_oof_lambdamrr.py -q`
- **Observed pass**: `1 passed`。

## REFACTOR

- **Refactor done**: no
- **Change**: 第一个行为切片只建立最小联合反向传播路径。
- **Command after refactor**: not needed
- **Observed result**: 保持 GREEN。

## Next Behavior

总 horizon correction 与 learned residual 必须共同满足 bounded top-k 安全契约。

---

## RED

- **Test added**:
  `test_joint_alternatives_bound_total_horizon_and_lambda_correction`
- **Behavior asserted**: top-k 外逐值保持、每行 residual 和为零、总改分不超过 cap。
- **Command**:
  `.venv/bin/python -m pytest tests/test_hybrid_joint_oof_lambdamrr.py -q`
- **Observed failure**:
  `ImportError: cannot import name 'bounded_joint_topk_alternatives'`
- **Failure is correct because**: horizon 与 Lambda residual 的整体投影 API 尚不存在。

## GREEN

- **Minimal implementation**: 新增 `bounded_joint_topk_alternatives`，先合并
  horizon delta 与 route-specific residual，再居中和按最大幅度投影。
- **Command**:
  `.venv/bin/python -m pytest tests/test_hybrid_joint_oof_lambdamrr.py -q`
- **Observed pass**: `2 passed`。

## REFACTOR

- **Refactor done**: no
- **Change**: 无额外重构。
- **Command after refactor**: not needed
- **Observed result**: 保持 GREEN。

## Next Behavior

训练、推理和 checkpoint 必须闭环重放两个输出。

---

## RED

- **Test added**: `test_joint_fit_checkpoint_replays_both_outputs`
- **Behavior asserted**: toy 数据上 route/rank loss 均非零，保存和加载后两个输出在
  `1e-6` 内一致，provenance 仅为 Jittor。
- **Command**:
  `.venv/bin/python -m pytest tests/test_hybrid_joint_oof_lambdamrr.py -q`
- **Observed failure**:
  `ImportError: cannot import name 'JointOOFLambdaMRRTrainingConfig'`
- **Failure is correct because**: fit/predict/checkpoint 公共契约尚不存在。

## GREEN

- **Minimal implementation**: 新增训练配置、流式候选 normalizer、top-k hard
  negatives、route-specific `ΔMRR` pair weights、联合训练循环、双输出推理和
  NPZ checkpoint。
- **Command**:
  `.venv/bin/python -m pytest tests/test_hybrid_joint_oof_lambdamrr.py -q`
- **Observed pass**: 初次实现暴露旧 Champion 工具拒绝负 logits；新模块保留相同
  稳定排序和 `ΔMRR` 公式但只要求 finite logits 后，`3 passed`。

## REFACTOR

- **Refactor done**: yes
- **Change**: 移除对只接受非负概率分数的旧 hard-negative helper 依赖，使实现与
  多尺度 `corrected-logits` 契约一致。
- **Command after refactor**:
  `.venv/bin/python -m pytest tests/test_hybrid_joint_oof_lambdamrr.py -q`
- **Observed result**: `3 passed`，checkpoint replay 通过。

## Next Behavior

runner 的候选特征必须在候选置换下等变，不能隐式使用 candidate ID 或正样本列。

---

## RED

- **Test added**:
  `test_joint_candidate_features_follow_candidate_permutation`
- **Behavior asserted**: 同步置换 raw features、分数和 residual 后，增强候选特征只
  做相同候选维置换。
- **Command**:
  `.venv/bin/python -m pytest tests/test_hybrid_joint_oof_lambdamrr.py -q`
- **Observed failure**:
  `ModuleNotFoundError: scripts.train_dataset2_joint_oof_lambdamrr`
- **Failure is correct because**: 严格时间 runner 与 lazy candidate view 尚不存在。

## GREEN

- **Minimal implementation**: 新增 lazy candidate feature view、12 变体联合训练、
  selection lock、一次性 gate、安全审计、hash 和 evaluation report。
- **Command**:
  `.venv/bin/python -m pytest tests/test_hybrid_joint_oof_lambdamrr.py -q`
- **Observed pass**: `4 passed`。

## REFACTOR

- **Refactor done**: yes
- **Change**: 修复最终目录从 `.building` 原子重命名后报告 checkpoint 路径仍指向
  临时目录的问题；模型字节不变。
- **Command after refactor**:
  `.venv/bin/python -m pytest tests/test_hybrid_joint_oof_lambdamrr.py tests/test_hybrid_high_confidence_topk_router.py tests/test_hybrid_champion_residual.py -q`
- **Observed result**: `16 passed`；ruff 与 py_compile 均通过；selected
  checkpoint replay error `0.0`，SHA-256 与报告一致。

## Next Behavior

done。模型实现完成；最终 gate rejected，因此不进入全量重训和提交阶段。
