# Dataset2 Cooccur-Lift Successor V2 双候选预注册 TDD

## Target Behavior

标准验证要能冻结并执行以下合取资格门，同时保留既有
`deployment_mixture` 模式：

- 每个 near fold 的 MRR/NDCG@10 delta 均 `>= 0`；
- 每个 gapped fold 的 MRR delta `> 0`、NDCG@10 delta `>= 0`；
- 只有两组门都通过，候选才 eligible；
- plan 声明 baseline 时，rolling 与 external manifest 必须声明同一
  `baseline_sha256`。

## RED

首次加入 baseline freeze、near-rejected、gapped-rejected 和 eligible 四个测试：

```text
uv run --no-sync pytest tests/test_standard_validation_protocol.py -q
4 failed, 11 passed
```

失败原因符合预期：

- plan lock 没有保存 baseline；
- far-horizon validator 只接受 deployment-mixture 的 selection order。

收口审查随后增加 baseline 传播测试：

```text
uv run --no-sync pytest \
  tests/test_standard_validation_protocol.py::test_rolling_selection_rejects_declared_baseline_mutation \
  -q
1 failed
```

旧 selector 对错误 baseline hash 没有抛错，证明仅在 plan 中记录 baseline 还不够。

## GREEN

最小实现包括：

- 新增
  `near_non_decreasing_and_gapped_strict_improvement` eligibility mode；
- near 逐折使用 inclusive MRR/NDCG@10 门；
- gapped 逐折使用 strict MRR 与 inclusive NDCG@10 门；
- eligible 候选按平均 gapped MRR、最差 gapped MRR、平均 near MRR、
  tie-break 排序；
- freeze/selection lock 保存 baseline 及其 canonical JSON hash；
- rolling/external manifest 在 baseline 存在时必须匹配该 hash。

旧 plan 未声明 eligibility mode 时继续解释为 `deployment_mixture`，历史行为不变。

## REFACTOR

- 用独立常量表示两种 far-horizon selection order 和 eligibility mode。
- 抽出 `_validate_manifest_baseline_binding`，rolling 与 external 共用同一检查。
- deployment mixture 仍计算和报告，但在本次 duel 中不参与 eligibility。
- zero-short 始终固定为诊断臂，不进入选择。

## Final Verification

```text
uv run --no-sync -- C:\Users\75556\anaconda3\python.exe -m ruff check \
  src/jgrec/standard_validation_protocol.py \
  tests/test_standard_validation_protocol.py
All checks passed!
```

```text
uv run --no-sync pytest \
  tests/test_standard_validation_protocol.py \
  tests/test_cooccur_lift_transport_audit.py \
  tests/test_cooccur_lift_promotion.py \
  tests/test_cooccur_lift_external.py -q
34 passed
```

没有实现候选模型、生成分数、创建 selection lock 或打开 external。

## 第二轮 RED：远视界物化

远端资产审计发现旧 20 万行 cache 只覆盖 179 天，短于最小 gapped 间隔
251 天；因此不能靠切旧 cache 合法构造远视界折。先增加以下失败测试：

```text
uv run --no-sync pytest \
  tests/test_cooccur_lift.py \
  tests/test_cooccur_lift_successor.py -q

ModuleNotFoundError:
  jgrec.rankers.hybrid.cooccur_lift_successor
```

RED 覆盖：

- observation time 与 feature availability time 必须分离；
- `gap == short_window` 时 short 必须严格塌缩为零、full 仍保留截止日前历史；
- native materializer 与 Python reference 必须逐值一致；
- full-only 只能产生 64 raw 列；
- gap-aware 必须产生 66 raw 列并按 row 广播 support；
- near/stale 两份训练 view 的 lazy concat 必须保持 NumPy 索引顺序。

## 第二轮 GREEN

最小实现：

- `cooccur_lift_scores(..., availability_time=...)`；
- native C++ materializer 增加逐行 availability time；
- `CooccurLiftFullOnlyView`、`CooccurLiftGapAwareView`、
  `ConcatenatedFeatureView`；
- exact historical cache builder；
- 固定 `0.50`、固定 seed/capacity、绑定 v1 baseline 的 near+gapped duel
  runner。

```text
uv run --no-sync pytest \
  tests/test_cooccur_lift.py \
  tests/test_cooccur_lift_successor.py -q
19 passed
```

```text
uv run --no-sync pytest \
  tests/test_cooccur_lift.py \
  tests/test_cooccur_lift_successor.py \
  tests/test_standard_validation_protocol.py -q
36 passed
```

远端 Ruff 对新增/变更实现全部通过；远端协议套件先得到
`35 passed, 1 failed`，唯一失败是未同步 example JSON，补齐该只读测试资产后该
测试单独通过。失败与模型、fold、selector 或数值逻辑无关。

本轮在任何 v2 指标前补冻了精确行边界与 support 混合合同；旧 plan lock 保留为
历史记录，新正式证据为：

- validation plan:
  `07f0ac9a244077a3ad8e7e3cd76bd7c95c6b7c00d8a42766601785f069e95efd`
- plan-v2 lock:
  `3519a496a5807b18e4b6f0aefdfd9c92dce34cfdf188c0f719458785f2ed6d98`
- full-only config:
  `82395bbbb07c17b6f6adfe5629a01ef054e07c35da68f0f28d734d7c69fa9cc0`
- gap-aware config:
  `2b4a0bf3c61e183f28ffbda7e601b94298e0a68acbdbec8fc691fb6efb325ed3`

截至正式 gapped cache 启动前，仍未读取或产生 successor 指标，也未打开
external。

## 第三轮 RED：不降质量的多核 structure 加速

顺序正式运行显示 `structure` 占约 93% feature wall time，但只使用一个 CPU
核。用户授权提高服务器利用率，条件是不能降低质量。先加入：

- Linux fork 下 source-stable 分片、原行号回填后必须与顺序结构特征
  `array_equal`；
- full-feature 首批 A/B 必须同时验证 shape、dtype、逐值相等和数组 SHA；
- 实际活跃 worker 少于 2 或实测速度小于 `1.5x` 必须失败。

```text
uv run --no-sync pytest tests/test_parallel_structure.py -q

ModuleNotFoundError:
  jgrec.rankers.hybrid.parallel_structure
```

失败发生在新并行模块缺失，符合目标行为尚未实现的 RED，而不是测试或环境错误。

## 第三轮 GREEN

最小实现：

- 父进程保持原 candidate RNG 顺序；
- 只把纯 NumPy `StructureFeatureTower` 按 `src % worker_count` 稳定分给
  Linux fork worker；
- worker 继承只读 index，并保留原 byte-bounded LRU；
- 每批结果按冻结行号回填；
- 正式 builder 首批先顺序、再 parallel4；精确相等且 `>=1.5x` 才继续。

本地：

```text
uv run --no-sync pytest \
  tests/test_parallel_structure.py \
  tests/test_cooccur_lift.py \
  tests/test_cooccur_lift_successor.py \
  tests/test_standard_validation_protocol.py -q

40 passed, 1 skipped
```

跳过项仅为 Windows 不支持 formal Linux `fork`。远端 Linux：

```text
.venv/bin/python -m pytest \
  tests/test_parallel_structure.py \
  tests/test_cooccur_lift.py \
  tests/test_cooccur_lift_successor.py \
  tests/test_standard_validation_protocol.py -q

41 passed
```

```text
.venv/bin/ruff check \
  src/jgrec/rankers/hybrid/parallel_structure.py \
  scripts/build_dataset2_cooccur_lift_gapped_cache.py \
  tests/test_parallel_structure.py

All checks passed!
```

远端真实首批 full-feature A/B parity/speed/resource gate 已通过：

- shape: `[4096, 100, 63]`，dtype: `float32`；
- sequential/parallel SHA-256 均为
  `c373e81aa58400315d4b9852650b639cfabeb4fc32083a1359e9db7314da1de4`；
- sequential `319.1316s`，parallel4 `192.5798s`，实测 `1.6571x`；
- 4 个 worker 全部实际参与；
- PSS 约 10.6 GiB，`MemAvailable` 约 19 GiB，SSH 正常。

因此加速路径完成真实 GREEN。旧 `gapped-cache-v1` 在 65,536 行处只作
superseded 证据保留，没有删除。

## 第四轮 RED：确定性 V1 Baseline 与执行接线

首次 cache 后自动 duel 在任何 successor 指标前以历史原值失败：

```text
RuntimeError: fold-0 v1 deterministic replay drifted:
max_abs_error=0.10732766809698313
```

这与后续 CUDA/CPU 双跑审计一致：旧 CUDA V1 不能继续作为修复后 CPU V1 的
authoritative replay 对象。先新增执行合同测试，要求 GPU training、单跑、容差
漂移、legacy authoritative 或 external authority 均被拒绝；初次 RED 为：

```text
ModuleNotFoundError:
  jgrec.cooccur_lift_successor_execution
```

GREEN 最小实现：

- 新增冻结执行合同 validator；
- 所有可训练 head 使用 CPU 独立双跑；
- state/loss/probability 继续使用 `rtol=2e-5`、`atol=2e-6`；
- CPU 两跑不一致仍硬失败；
- near V1 baseline 使用新 CPU auxiliary，旧 CUDA V1 只报告 diagnostic drift。

相关回归：

```text
52 passed
```

首次正式接线 smoke 随后在任何训练前暴露
`expected_cache_report_sha256` 误传给 near 函数。新增 AST wiring RED 后得到：

```text
1 failed:
expected_cache_report_sha256 unexpectedly present on _train_near_folds
```

修复后本地 `12 passed`，远端 `12 passed`，Ruff
`All checks passed!`，bash syntax 通过。失败目录保留为审计证据；正式
`v5-cpu-replay-wiringfix` 使用独立目录启动。
