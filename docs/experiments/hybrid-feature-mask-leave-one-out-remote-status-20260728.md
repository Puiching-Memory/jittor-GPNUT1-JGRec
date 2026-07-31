# Hybrid 特征掩码留一法：远端运行状态

## Current Status

`QUEUED_FOR_SAFE_GPU_WINDOW`

服务器连接、数据、checkpoint、rolling manifest 和依赖均已核对。远端 CPU 协议测试
`13 passed`，runner `--help` 通过。

当前 GPU 0 正由其他训练使用，因此 JGRec 尚未启动。资源门控调度器已经排队：

- launcher PID：`13668`
- 要求连续 10 次、每次间隔 30 秒均满足：
  - 无任何现有 compute process
  - GPU utilization ≤ 10%
  - GPU free memory ≥ 44,000 MiB
  - system available memory ≥ 16,000 MiB
  - 1-minute load average < 8
- JGRec 使用单任务串行、`nice -n 10`、最低 IO 优先级和 4 个 CPU 线程。
- 若运行中出现其他 GPU session，watchdog 只终止 JGRec 自己的进程组并写
  `collision.txt`，不操作其他进程。

## Frozen Experiment Shape

- Dataset：Dataset1
- Baseline：当前冠军 causal retrain，经共享 LGBM、共享 Setwise 和 `gamma=0.5` 最终集成
- Candidates：`full_enabled` + 8 个 `loo_without_<group>`
- Selection：3 个 rolling-origin selection folds
- Reserved：第 4 fold 只登记时间边界，不提前读取指标
- Metrics：MRR、Hit@1/3/10、NDCG@10、mean rank、improved/unchanged/worsened
- Gate：每折 MRR/NDCG 不回归，所有多折均值不回归，improved > worsened
- External：selection lock 前禁止打开
- Package：未授权

## Remote Artifacts

```text
result/dataset1_feature_mask_loo_rolling_20260728_launcher/
logs/dataset1_feature_mask_loo_rolling_20260728_launcher.log
logs/dataset1_feature_mask_loo_rolling_20260728.log
result/dataset1_feature_mask_loo_rolling_20260728/
```

最后一项只有资源门禁通过后才会创建。
