# Prediction Contract Safety and Rank Fusion — Result

## Verdict

**PASS：B 组三项已完成并进入默认执行路径。**

## Delivered

1. 输出边界确定性消除精确并列，使用 candidate prior → candidate id → 原列号的稳定次序。
2. `selection_metric` 默认 MRR，`fusion_mode` 默认 ensemble，异构专家默认 RRF。
3. `hybrid --no-refit-full` 已真实跳过最终全量 refit，并在训练报告记录 `refit_full=0`。
4. 提供 probability、temperature、RRF 三种专家融合；温度仅在训练内 validation 上按 NLL 标定。
5. 新 checkpoint 保存 mode、temperature、RRF k、MLP weight；旧 checkpoint 保持 probability。
6. 提供 `scripts/diagnose_prediction_ties.py` 做逐行精确并列审计。

## Safety Properties

- 非并列 pair 的相对排名不变。
- ULP 空间足够时只改精确并列分数。
- ULP 空间耗尽时才对该行做严格保序秩映射。
- 所有输出仍位于 `[0, 1]`。
- 不根据 external/线上分数反扫融合 mode、温度或权重。

## Verification

- 纯 NumPy / CLI：通过。
- Linux/Jittor 定向与 checkpoint 兼容：通过。
- Linux/Jittor 全回归：491 passed。
- Ruff、`compileall`、`git diff --check`：通过。

## Submission Artifact

- 本地路径：`artifacts/b_prediction_contract_tiesafe_20260728/result.zip`
- 字节数：`183718466`
- SHA-256：`085da277f6f20429a2f9e4872438de2f7dca672eea41ba5a8e7fe1d99fb50730`
- ZIP 成员：`dataset1.csv`、`dataset2.csv`
- dataset1：61,051 行，精确并列 0。
- dataset2：153,420 行，精确并列 0。
- 随包验证报告：`artifacts/b_prediction_contract_tiesafe_20260728/verification-report.json`

该包只启用新的输出契约；旧 checkpoint 的专家融合仍按 legacy probability 执行。RRF / temperature 会用于后续新训练 checkpoint，不把旧包伪装成 RRF 线上验证。
