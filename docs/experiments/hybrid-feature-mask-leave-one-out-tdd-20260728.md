# Hybrid 特征掩码留一法：TDD 证据

## Target Behavior

- 原 12 个递进加法候选名称和顺序保持不变。
- 默认配置建立 63 维完整掩码。
- 对 8 个已启用特征组分别建立 `full - group` 候选。
- 默认扫描不训练索引重复的候选。
- 所有留一候选都能以 `loo_without_<group>` 精确冻结。
- 禁用组不进入留一目录。

## RED

先新增 `tests/test_hybrid_feature_mask_leave_one_out.py`，再运行：

```bash
nvcc_path= JITTOR_HOME=/var/tmp/jittor-codex-feature-loo-red \
  .venv-wsl/bin/python -m pytest -q \
  tests/test_hybrid_feature_mask_leave_one_out.py
```

结果：测试收集失败，报错为：

```text
ImportError: cannot import name '_feature_mask_catalog'
```

失败原因与目标一致：生产代码尚无可审计的特征掩码目录，也没有留一映射。

## GREEN

最小实现：

- 新增 `_FeatureMaskCatalog` 与 `_LeaveOneOutFeatureMask`。
- `_feature_mask_catalog()` 先保留原递进候选，再建立完整掩码与逐组补集。
- 以特征索引元组去重；若留一补集已存在，只登记别名映射。
- `_feature_masks()` 支持冻结默认候选名或 `loo_without_<group>` 别名。

运行同一测试文件，结果：

```text
3 passed
```

## Refactor Decision

保留候选目录在 `ranker.py` 内部，因为特征组边界目前由该模块中的融合特征拼接契约定义。
未新增配置开关，也未改变融合模型、训练损失或选择指标。

原测试对 12 个候选做全列表相等断言；现改为断言这 12 个仍是稳定前缀，新增留一行为由
专门测试逐项验证，避免把“不能增加候选”误当成兼容契约。

## Verification

结构预检：

```bash
.venv-wsl/bin/python \
  scripts/preflight_hybrid_feature_mask_leave_one_out.py
```

输出：

```text
result/hybrid_feature_mask_leave_one_out_local_20260728/preflight-report.json
```

预检状态为 `ready_for_remote_rolling`；测试与静态检查的最终回归结果记录在结果文档。

最终验证：

```text
67 passed, 1 warning
3 passed  # Ruff 整理导入后的聚焦复跑
All checks passed!  # Ruff
```
