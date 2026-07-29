# TDD Evidence: Dataset1 K256 Static Setwise Dual-Horizon

## Target Behavior

- K256 同时覆盖 structure 与 source-profile 两个预测历史限制，其他配置不变。
- 静态 Setwise 权重网格精确为 `0.05..0.80`、步长 `0.05`。
- near 三折 MRR/NDCG 均不退化，gapped 三折 MRR 严格改善且 NDCG 不退化，
  才能锁定权重。
- external 只执行七个方向性安全门，不读取效应量，也不允许 MRR 仅持平。

## RED

### Static grid / K256

- **Test**: `tests/test_dataset1_static_setwise_dual_horizon.py`
- **Observed failure**:
  `ModuleNotFoundError: jgrec.rankers.hybrid.static_setwise`
- **Why correct**: 目标模块和 K256/static blend contract 尚不存在。

### Dual-horizon selection

- **Test**: 同文件
- **Observed failure**:
  `ImportError: cannot import name 'select_dual_horizon_static_weight'`
- **Why correct**: near/gapped 逐折门与选择顺序尚未实现。

### External safety-only gate

- **Test**: 同文件
- **Observed failure**:
  `ImportError: cannot import name 'evaluate_external_safety_deltas'`
- **Why correct**: 七门和 MRR strict-increase contract 尚未实现。

## GREEN

- 新增 `src/jgrec/rankers/hybrid/static_setwise.py`：
  - 固定 16 权重；
  - query-invariant convex blend；
  - 双 K 覆盖；
  - near/gapped selector；
  - external 七门。
- cache builder 新增 `--prediction-history-limit`，报告同时保留 checkpoint
  原始 limits 和实际 limits。
- 新增无标签 plan freezer、6 折训练/选择 runner、一次性 external runner 和
  可恢复自动 shell 链。
- focused tests:
  - Windows：`5 passed`
  - Linux：`5 passed`
- `py_compile`：通过。
- Ruff：全部通过。

## REFACTOR

- 将静态网格、选择状态机和 external gates 留在纯 NumPy 模块，训练 runner
  只负责模型/数组编排。
- 复用现有 base-context runner 的 head 训练、预测和 artifact 序列化函数，
  没有复制第二套模型定义。
- cache、internal、external 三阶段分别写 PID/exit/status；internal 不通过时
  自动链不会创建 external receipt。

## Production Evidence

- plan SHA-256:
  `352de592c65ea329d354c8e9c8acc53ed7cb7f7912304a6f16ccf1645fa55623`
- K256 cache：
  - 200k train + 20k untouched external joint build 完成；
  - 8-worker exact parity trial 通过；
  - cache exit `0`。
- internal 已自动启动；最后人工稳定性检查时位于 `near-0` base head epoch 9，
  训练 loss/validation MRR 正常，之后按用户要求不再持续轮询。
