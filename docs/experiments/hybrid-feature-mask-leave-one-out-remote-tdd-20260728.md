# Hai TDD: Hybrid 特征掩码远端验证协议

## Target Behavior

从一个完整且无重复的特征 schema 生成“完整集 + 每组精确补集”候选；每个候选必须绑定
稳定配置哈希和唯一 tie-break 优先级，使远端 rolling runner 无法在看到指标后改变候选。

## RED

- **Test added**: `tests/test_feature_mask_validation.py`
- **Behavior asserted**: 完整候选、逐组补集、唯一索引、配置哈希、保守 tie-break，以及组
  schema 不得遗漏或重复特征。
- **Command**:

  ```bash
  .venv-wsl/bin/python -m pytest -q \
    tests/test_feature_mask_validation.py
  ```

- **Observed failure**:

  ```text
  ModuleNotFoundError: No module named 'jgrec.feature_mask_validation'
  ```

- **Failure is correct because**: 生产代码尚无独立于训练器的冻结候选协议模块。

## GREEN

- **Minimal implementation**: 新增 `build_feature_mask_candidates()` 和不可变候选记录；使用
  canonical JSON SHA-256 绑定候选配置，并拒绝不完整、重复或移除后为空的组定义。
- **Command**:

  ```bash
  .venv-wsl/bin/python -m pytest -q \
    tests/test_feature_mask_validation.py
  ```

- **Observed pass**: `2 passed`

## REFACTOR

- **Refactor done**: yes
- **Change**: 远端训练编排复用已验证的 Dataset1 exact-integrated base-context helper；候选
  构造留在纯 Python 模块，GPU runner 只负责训练、物化分数和调用标准 selector。
- **Command after refactor**:

  ```bash
  pytest -q \
    tests/test_feature_mask_validation.py \
    tests/test_standard_validation_protocol.py \
    tests/test_hybrid_base_context_gate.py
  ```

- **Observed result**: 本地 `13 passed`；同步到远端后再次执行为 `13 passed`。Ruff、
  `py_compile`、runner `--help` 和资源门控 shell `bash -n` 均通过。

## Next Behavior

候选协议已完成。下一行为是等待 GPU 连续空闲五分钟，写入真实 checkpoint 绑定的 plan
lock，再执行三折 rolling；该行为由远端资源门控调度器负责。
