# Hybrid 特征掩码留一法：本地结果

## Verdict

本地工程部分完成，可进入服务器 rolling-origin 多折验证；当前没有任何效果结论，也没有
打开 external holdout 或生成提交包。

## Implemented

- 保留原 12 个递进加法候选及其名称顺序。
- 从 63 个基础融合特征建立完整候选。
- 为 `stats/prior/target/structure/profile/tower/gnn/seq` 建立 8 个稳定留一别名。
- 新增 7 个索引唯一的默认候选；`loo_without_seq` 与既有
  `stats_prior_target_structure_profile_tower_gnn` 共用索引，默认只训练一次。
- `frozen_fusion_feature_candidate=loo_without_<group>` 可精确冻结任意留一候选。
- 配置禁用或不存在的特征组不会进入留一目录。

## Local Evidence

本地预检结果：

- 状态：`ready_for_remote_rolling`
- 特征组：8
- 完整特征数：63
- 原候选数：12
- 去重后默认候选数：19
- 新增唯一候选数：7
- 留一覆盖：8/8
- 重复默认掩码：0
- 7 个可配置特征组的禁用检查：全部通过
- 训练数据、选择指标、external、线上分数：均未读取
- 提交包：未生成

机器可读证据：
`result/hybrid_feature_mask_leave_one_out_local_20260728/preflight-report.json`

验证结果：

- 新增留一法测试：3 passed
- Hybrid 相关回归：67 passed，1 个第三方 Jittor deprecation warning
- Ruff：All checks passed

## Validation Rule

后续不能让基础融合在单个切分内自由挑这 19 个候选。正式比较应冻结以下 9 个精确候选：

- 完整候选 `stats_prior_target_structure_profile_tower_gnn_seq`
- 8 个 `loo_without_<group>` 候选

先用 rolling-origin 多折平均和跨折稳定性淘汰不稳候选，再锁定唯一候选并只打开一次
长跨度 external holdout。只有通过该 gate 后才允许生成提交包。

## Remaining Remote Work

1. 为完整候选和 8 个留一候选分别运行相同训练预算的 rolling-origin 多折。
2. 同时报 MRR、Hit@1/3/10、NDCG@10、平均排名、改善/恶化 query 数。
3. 以跨折稳定性为硬门禁，不按单折峰值选择。
4. 锁定唯一候选后打开一次 external holdout。
