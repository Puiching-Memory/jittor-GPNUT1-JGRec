# Hai TDD: 精确集成候选稳健选权重协议

## Target Behavior

对至少三个时间递进折上的最终集成分数做完整 rank 指标评估，只允许跨折稳定
候选锁定权重；锁定后 external holdout 只能开启一次，且必须绑定同一
`integration_id`、权重、候选顺序和 selection lock。

## RED

- **Test added**: `tests/test_robust_weight_selection.py`
- **Behavior asserted**:
  - 报告 MRR、Hit@1/3/10、NDCG@10、平均排名和 query movement；
  - 拒绝单折峰值、选择跨折稳定权重；
  - 拒绝少于三折、时间泄漏和跨专家 `integration_id`；
  - selection lock 不匹配时不消耗 external；
  - external candidate 的权重必须与 lock 相同；
  - external 首次读取后第二次必定失败。
- **Command**:
  `uv run --no-sync pytest tests/test_robust_weight_selection.py -q`
- **Observed failure**:
  首次收集失败：
  `ModuleNotFoundError: No module named 'jgrec.robust_weight_selection'`。
  增补 external candidate weight 契约时，目标测试以
  `DID NOT RAISE ValueError` 再次进入 RED。
- **Failure is correct because**: 首次失败证明目标模块尚不存在；第二次失败
  证明旧 GREEN 尚未约束 external 分数矩阵所声明的权重，而不是测试语法或环境
  错误。

## GREEN

- **Minimal implementation**:
  - 新增 `jgrec.robust_weight_selection`，实现严格排名、多指标面板、三折
    manifest 校验、同 `integration_id`/candidate fingerprint 校验；
  - 用“最差折 MRR → 折中位 MRR → pooled MRR → 更小权重”选择通过硬门禁的
    候选；
  - 生成 selection report/lock；
  - external 在读取分数前独占生成 receipt，并校验 lock SHA、integration、
    selected weight、candidate weight、时间跨度和 artifact hash；
  - 新增 selection 与 external 两条独立 CLI。
- **Command**:
  `uv run --no-sync pytest tests/test_robust_weight_selection.py -q`
- **Observed pass**: `7 passed`。

## REFACTOR

- **Refactor done**: yes
- **Change**: pooled baseline 指标移到权重循环外只计算一次；移除对 manifest fold
  字典写入临时 shape 的副作用；保持 selection 与 external 两条调用路径分离。
- **Command after refactor**:
  - `uv run --no-sync pytest tests/test_robust_weight_selection.py
    tests/test_partial_listwise_submission.py
    tests/test_hybrid_partial_listwise_blend.py
    tests/test_hybrid_rolling_origin.py tests/test_submission.py -q`
  - `uv run --no-sync ruff check src/jgrec/robust_weight_selection.py
    scripts/select_robust_integrated_weight.py
    scripts/evaluate_locked_weight_external.py
    tests/test_robust_weight_selection.py`
- **Observed result**: `30 passed`；Ruff `All checks passed!`。
  Windows `sitecustomize` 仍输出既有 CUDA symlink 权限警告，但命令退出码为 `0`，
  未影响测试或 lint。

## Next Behavior

框架行为完成。下一行为属于新的昂贵实验：为一个**新的预登记 integration
hypothesis** 生成三折最终集成分数；已线上拒绝的
`two_tower_full_reranker_partial_v1` 不得借此回扫邻近权重。
