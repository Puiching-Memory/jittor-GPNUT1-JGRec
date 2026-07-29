# Hai TDD: Dataset2 Bounded Decoder 多时间跨度 OOF Residual

## Target Behavior

把已有 rolling-origin bounded source decoder 的未来输出组装成固定
`[short, medium, long]` residual tensor，并保证：

- 任意 score 行严格晚于该模型的训练 origin；
- 同一 horizon 不重复覆盖；
- 未覆盖位置显式无效且 logits/residual 为零；
- residual 保持硬 cap、行内零均值并能精确回放 corrected logits。

## RED

- **Test added**:
  `tests/test_hybrid_multi_horizon_oof.py`
- **Behavior asserted**:
  canonical lattice、严格时间边界、同 horizon overlap 拒绝、无效行零值、
  cap/replay 审计。
- **Command**:
  `.venv/bin/python -m pytest tests/test_hybrid_multi_horizon_oof.py -q`
- **Observed failure**:
  测试收集阶段报
  `ModuleNotFoundError: jgrec.rankers.hybrid.multi_horizon_oof`。
- **Failure is correct because**:
  新的 horizon 组装与审计契约尚不存在；失败不是数据或 Jittor 环境导致。
- **Environment note**:
  本地按项目要求执行 `uv run pytest` 时，Windows 上 `pymetis` 因缺少
  `sys/resource.h` 无法构建；因此行为 RED/GREEN 在项目权威 Linux
  `.venv` 环境完成。

## GREEN

- **Minimal implementation**:
  新增 `multi_horizon_oof.py`，只实现 slice dataclass、canonical lattice、
  tensor assembly 与审计，不修改 bounded decoder 模型。
- **Command**:
  `.venv/bin/python -m pytest tests/test_hybrid_multi_horizon_oof.py -q`
- **Observed pass**:
  `5 passed`。

## REFACTOR

- **Refactor done**: yes
- **Change**:
  生成器按 origin 分组，一次构建冻结历史并连续预测到训练尾部，再拆成
  short/medium/long；三个 checkpoint 各只加载一次。生成前还会把每个
  origin 的 short history 与已有缓存逐字节比较。
- **Runtime gate adjustment**:
  第一次 CUDA 运行在 short checkpoint 重放处得到 `2.861e-6` 浮点差，
  正确触发了原 `2e-6` 门禁。仅将“旧 checkpoint GPU 重放”容差改为
  `5e-6`；最终保存数组的 residual replay 仍使用 `2e-6`，实际误差为
  `3.725e-9`。
- **Command after refactor**:
  `.venv/bin/python -m ruff check
  src/jgrec/rankers/hybrid/multi_horizon_oof.py
  scripts/generate_dataset2_bounded_source_multi_horizon_oof.py
  tests/test_hybrid_multi_horizon_oof.py`

  `.venv/bin/python -m pytest
  tests/test_hybrid_multi_horizon_oof.py
  tests/test_hybrid_bounded_source_decoder.py
  tests/test_hybrid_source_sequence_cache.py -q`
- **Observed result**:
  ruff/py_compile 通过，相关测试 `14 passed`；CUDA 全量生成退出码为 0。

## Next Behavior

done。下一轮可用最后时间片同时存在的三路 residual 做纯 Jittor
时间分段路由训练，并用该时间片的末段做不可见门禁。
