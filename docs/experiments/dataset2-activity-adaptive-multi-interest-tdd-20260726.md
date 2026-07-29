# TDD Evidence: Dataset2 Activity-Adaptive Multi-Interest

## Target Behavior

- 指数衰减中心对时间整体平移不敏感，并提高近期事件的影响。
- recent-16/recent-64/full 中心具有固定顺序和完整历史语义。
- source activity 越高，event half-life 越短，旧 cluster 的 last-hit 与
  routing weight 越低。
- support/age/last-hit 聚合不依赖 cluster 编号。
- v2 proxy 保留旧 9 维并追加 10 维；cold ids 仍全零，candidate permutation
  只置换 candidate 轴。

## RED

- **Command**:
  `uv run --no-sync pytest tests/test_hybrid_multi_interest_proxy.py -q`
- **Observed failure**: `ImportError: cannot import name
  'ACTIVITY_ADAPTIVE_FEATURE_NAMES'`，测试在 collection/import 阶段失败。
- **Why this is the right failure**: 旧 proxy 测试仍在同一文件中；失败直接证明
  缺少本轮目标 contract，而不是数据、Jittor 或远端环境故障。

## GREEN

- **Implementation**:
  - `exponential_interest_center`
  - `hierarchical_interest_centers`
  - `activity_adaptive_cluster_interests`
  - `adaptive_interest_affinity_features`
  - v1/v2 自动 schema 的 query proxy
- **Local command**:
  `uv run --no-sync pytest tests/test_hybrid_multi_interest_proxy.py
  tests/test_hybrid_setwise.py -q`
- **Local result**: `14 passed`。
- **Linux command**:
  `uv run --no-sync pytest tests/test_hybrid_multi_interest_proxy.py
  tests/test_hybrid_setwise.py tests/test_hybrid_fusion_listwise.py -q`
- **Linux result**: `25 passed`。
- **Note**: 本地 Jittor suite 被既有 Windows MSVC/CUDA 编译错误拦截；
  同一 Jittor listwise 回归已在 Linux 通过。

## Refactor

- 把 ten-channel batch 计算抽成
  `activity_adaptive_features_for_candidate_batch`，query inference 与离线
  proxy 生成共用同一数值路径。
- 旧 9 维函数和旧 state schema 保持不变；adaptive keys 只有完整出现时才
  打开 v2，部分 state 会显式报错。
- **Ruff**: 本地和 Linux 均为 `All checks passed!`。
- **Experiment**: slice1 未过门，按协议未读取 slice2。
