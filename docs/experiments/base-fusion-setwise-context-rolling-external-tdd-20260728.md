# TDD Evidence: Base Fusion Context Rolling / External Gate

## Target behavior

- 在 Dataset1 最终集成路径比较基础 MLP v0 与 v1，而不是比较 standalone MRR。
- 每个 rolling fold 共享同一 LGBM、同一 Setwise 专家、同一 `γ=0.5`；
  v0/v1 唯一允许的差异是 `context_transform_version`。
- 同时输出 MRR、Hit@1/3/10、NDCG@10、平均排名和 query movement。
- 至少三折全部满足稳定性硬门禁后才写 selection lock。
- selection 阶段不能接触 external；external 只允许打开一次，且在读取数组前写 receipt。
- external 通过前不授权生成提交包。

## RED

新增 `tests/test_hybrid_base_context_gate.py` 后，首次收集以
`ModuleNotFoundError: jgrec.rankers.hybrid.base_context_gate` 失败。该失败由目标
集成模块缺失直接造成，不是夹具或随机性问题。

Windows `uv run pytest` 另被 `pymetis` 在 Windows 下编译
`sys/resource.h` 失败阻断；这不是目标 RED，因此改用已由 uv 准备好的 WSL
项目环境复现上述正确 RED。

## GREEN

- 新增 `base_context_gate.py`，把共享 LGBM 融合和共享 time-ramp Setwise
  组合成一个不可绕过的最终分数函数。
- `validate_context_only_difference()` 强制 control=v0、candidate=v1，并拒绝
  seed、epoch、hidden dim、feature mask 等任何其他漂移。
- rolling runner 使用三个选择折；每折在 100k 窗口内留最后 20k 做因果 tune，
  基础 MLP/LGBM fit 区再遵守 source checkpoint 的 `max_train_events`；
  共享 Setwise 使用 tune 前完整 80k，随后写最终 control/candidate 分数。
- 现有 robust selection 状态机负责多指标稳定门禁与 selection lock。
- external runner 只能接受该 lock；候选头训练完成后、读取 external 数组前
  先独占写入 score-open receipt，再交给 one-shot external gate。

## Refactor decision

最终分数组合和“唯一差异”校验下沉到纯 NumPy 模块，GPU 训练与 artifact I/O
留在两个脚本中。这样测试不需要导入 Jittor，也避免把一次性实验状态机塞进
线上 ranker。

## Verification evidence

- 正确 RED：目标模块缺失，测试收集失败。
- Package authorization RED：
  `authorize_base_context_package` 缺失导致 import error；GREEN 后拒绝 external
  未通过、候选 head 哈希不匹配和任何 post-holdout tuning 授权。
- Head artifact RED：
  `base_context_head` 模块缺失导致 collection error；GREEN 后 round-trip 保存
  完整 v1 schema，并拒绝错误三通道宽度和 transform version。
- GREEN：`15 passed`：
  `tests/test_hybrid_base_context_gate.py` +
  `tests/test_hybrid_base_context_head_artifact.py` +
  `tests/test_robust_weight_selection.py`。
- 相关 fusion、LGBM、checkpoint、service normalizer、submission 回归：
  `84 passed`。
- Ruff：目标模块、六个执行/打包脚本和测试全部通过。
- 真实 rolling/external 尚未执行：CUDA 远端持续在 SSH banner 阶段断开；
  因而当前 `package_authorized=false`，没有生成提交包。
