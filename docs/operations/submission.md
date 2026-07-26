# 提交说明

本文档用于提交前复核：确认使用哪个包、如何复现、如何拼包，以及哪些参数会影响质量。

## 当前提交包

当前工作区已验证提交包：

```text
result/hybrid_perfcheck_d1d2_50k20k_mrr_r100_ch32_seed60/result.zip
```

线上反馈：

```text
1.2044345219596662
```

来源：

| 数据集     | CSV 来源                                                               |
| ---------- | ---------------------------------------------------------------------- |
| `dataset1` | `result/hybrid_perfcheck_d1d2_50k20k_mrr_r100_ch32_seed60/csv/dataset1.csv` |
| `dataset2` | `result/hybrid_perfcheck_d1d2_50k20k_mrr_r100_ch32_seed60/csv/dataset2.csv` |

关键设置：

- 全量输出 `dataset1` 和 `dataset2`，不使用 `--dataset` 或 `--limit-rows`。
- `max_fit_events=240000`、`max_train_events=50000`、`max_val_events=20000`。
- `selection_metric=mrr`、`test_candidate_negative_ratio=1.00`。
- `structure_cooccur_history_limit=32`。

提交时只上传 `result.zip`。压缩包根目录应直接包含 `dataset1.csv` 和 `dataset2.csv`。

## 比赛 checkpoint

完整训练并自动保存双数据集 checkpoint：

```bash
uv run jgrec-build --save-checkpoint checkpoints/checkpoint1.pkl
```

分开训练两个数据集时，两次命令必须使用同一路径：

```bash
uv run jgrec-build --dataset dataset2 --save-checkpoint checkpoints/checkpoint1.pkl
uv run jgrec-build --dataset dataset1 --save-checkpoint checkpoints/checkpoint1.pkl
```

第一轮只产生 `checkpoint1.pkl.tmp`；两个数据集状态齐全后才原子发布 `checkpoint1.pkl`。保存发生在各数据集训练结束、测试集推理开始之前，因此长时间推理失败不会丢失已经训练好的模型。

加载复核：

```bash
uv run jgrec-build --load-checkpoint checkpoints/checkpoint1.pkl --run-name checkpoint1_verify
```

加载模式跳过训练，只恢复 checkpoint 中对应数据集的 Hybrid 状态并生成 CSV。复核产物必须与原训练 run 的 CSV 做逐行概率或至少候选排序一致性比较。

## 提交前检查

确认 zip 内容：

```bash
python - <<'PY'
import zipfile

path = "result/hybrid_perfcheck_d1d2_50k20k_mrr_r100_ch32_seed60/result.zip"
with zipfile.ZipFile(path) as zf:
    print(zf.namelist())
PY
```

确认行列数：

```bash
python - <<'PY'
from pathlib import Path

expected = {
    "dataset1": 61051,
    "dataset2": 153420,
}
base = Path("result/hybrid_perfcheck_d1d2_50k20k_mrr_r100_ch32_seed60/csv")
for name, rows in expected.items():
    path = base / f"{name}.csv"
    actual_rows = 0
    bad_cols = 0
    for line in path.open("r", encoding="utf-8"):
        actual_rows += 1
        if len(line.rstrip("\n").split(",")) != 100:
            bad_cols += 1
    print(name, "rows", actual_rows, "expected", rows, "bad_cols", bad_cols)
PY
```

确认概率范围和格式：

```bash
python - <<'PY'
from pathlib import Path

base = Path("result/hybrid_perfcheck_d1d2_50k20k_mrr_r100_ch32_seed60/csv")
for path in sorted(base.glob("*.csv")):
    checked = 0
    for line in path.open("r", encoding="utf-8"):
        values = line.rstrip("\n").split(",")
        for value in values:
            x = float(value)
            assert 0.0 <= x <= 1.0, (path, value)
            assert "." in value and len(value.rsplit(".", 1)[1]) == 8, (path, value)
        checked += 1
        if checked >= 100:
            break
    print(path.name, "first_rows_ok", checked)
PY
```

## 复现环境

当前主要运行环境为 WSL 本地 Jittor：

```bash
cd /mnt/d/tmp/jittor-GPNUT1-JGRec-latest
export PYTHONPATH=src
export HOME=/home/user75556/jittor-cache-local
export PYTHONPYCACHEPREFIX=/home/user75556/jittor-cache-local/pycache
export PATH=/mnt/d/work/jittor-local/env/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/lib/wsl/lib
export LD_LIBRARY_PATH=/mnt/d/work/jittor-local/env/lib:/mnt/d/work/jittor-local/env/lib/gcc/x86_64-conda-linux-gnu/12.4.0
export cc_path=/mnt/d/work/jittor-local/env/bin/x86_64-conda-linux-gnu-g++
```

检查 CLI：

```bash
/mnt/d/work/jittor-local/env/bin/python -m jgrec.cli --help
```

## Dataset2 质量优先命令

当前 dataset2 质量版使用全量历史做最终 encoder，监督融合训练使用抽样事件。该命令保留
`candidate_prior`、`structure`、`two_tower`、`gnn` 和 `sequence`，不通过关特征换速度。

```bash
/mnt/d/work/jittor-local/env/bin/python -m jgrec.cli \
  --model hybrid \
  --dataset dataset2 \
  --run-name hybrid_d2_quality_stream_v8_seed60_fast_structure \
  --seed 60 \
  --batch-size 256 \
  --max-fit-events 0 \
  --max-train-events 50000 \
  --max-val-events 10000 \
  --num-negatives 63 \
  --epochs 6 \
  --train-batch-size 512 \
  --fusion-hidden-dim 128 \
  --gnn-model xsimgcl \
  --gnn-edge-weighting none \
  --gnn-embedding-dim 128 \
  --gnn-layers 2 \
  --gnn-epochs 2 \
  --gnn-batch-size 8192 \
  --gnn-max-graph-edges 120000 \
  --gnn-max-train-edges 60000 \
  --seq-epochs 2 \
  --seq-batch-size 256 \
  --seq-score-batch-size 128 \
  --seq-max-samples 80000 \
  --seq-max-len 64 \
  --seq-hidden-size 128 \
  --seq-layers 2 \
  --seq-heads 4 \
  --two-tower-epochs 4 \
  --two-tower-embedding-dim 128 \
  --two-tower-hidden-dim 128 \
  --two-tower-batch-size 512 \
  --two-tower-score-batch-size 512 \
  --two-tower-max-samples 150000 \
  --negative-sampling-workers 0 \
  --supervised-feature-memmap \
  --supervised-feature-batch-size 256
```

第一条 `feature-profile rows=10240` 是速度健康检查。旧 v6 推理阶段曾出现
`structure=866.7s`，修复 future-only 共现预聚合后，应该明显下降。

## 调参对比建议

冲分时优先只重跑 `dataset2`，用当前稳定 `dataset1.csv` 拼包。建议先用小输出对比训练/验证结果：

```bash
COMMON="--model hybrid --dataset dataset2 --seed 60 --batch-size 256 \
  --max-fit-events 0 --max-train-events 100000 --max-val-events 20000 \
  --num-negatives 63 --epochs 6 --train-batch-size 512 --fusion-hidden-dim 128 \
  --gnn-model xsimgcl --gnn-edge-weighting none --gnn-embedding-dim 128 --gnn-layers 2 \
  --gnn-epochs 2 --gnn-batch-size 8192 --gnn-max-graph-edges 120000 --gnn-max-train-edges 60000 \
  --seq-epochs 2 --seq-batch-size 256 --seq-score-batch-size 128 --seq-max-samples 80000 \
  --seq-max-len 64 --seq-hidden-size 128 --seq-layers 2 --seq-heads 4 \
  --two-tower-epochs 4 --two-tower-embedding-dim 128 --two-tower-hidden-dim 128 \
  --two-tower-batch-size 512 --two-tower-score-batch-size 512 --two-tower-max-samples 150000 \
  --negative-sampling-workers 0 --supervised-feature-memmap --supervised-feature-batch-size 256 \
  --limit-rows 2 --skip-validate"
```

AP 选择：

```bash
/mnt/d/work/jittor-local/env/bin/python -m jgrec.cli $COMMON \
  --run-name d2_tr100k_val20k_ap_r060 \
  --selection-metric ap \
  --test-candidate-negative-ratio 0.60
```

MRR 选择：

```bash
/mnt/d/work/jittor-local/env/bin/python -m jgrec.cli $COMMON \
  --run-name d2_tr100k_val20k_mrr_r060 \
  --selection-metric mrr \
  --test-candidate-negative-ratio 0.60
```

更偏 test-candidate 分布的负采样：

```bash
/mnt/d/work/jittor-local/env/bin/python -m jgrec.cli $COMMON \
  --run-name d2_tr100k_val20k_mrr_r075 \
  --selection-metric mrr \
  --test-candidate-negative-ratio 0.75
```

选中组合后，删除 `--limit-rows 2 --skip-validate` 跑全量。

## 参数含义

| 参数                              | 含义                                                                  |
| --------------------------------- | --------------------------------------------------------------------- |
| `--max-fit-events 0`              | 不截断训练历史，最终 encoder 使用完整 `train.csv`。                   |
| `--max-train-events`              | 融合器监督训练使用的正样本事件上限。                                  |
| `--max-val-events`                | 融合器特征组选择、早停和本地 AP/MRR 验证的事件上限。                  |
| `--num-negatives`                 | 每个正样本配多少个负候选。                                            |
| `--selection-metric`              | 融合器早停和特征 mask 选择使用 `ap` 或 `mrr`。                        |
| `--test-candidate-negative-ratio` | 负样本中来自 test 候选分布的比例；冷启动/新链接数据通常需要更高比例。 |
| `--supervised-feature-memmap`     | 监督特征落盘分块，降低常驻内存。                                      |

## 拼包原则

如果只重跑了一个数据集，需要与另一个数据集的稳定 CSV 拼成新提交包。拼包必须满足：

- `result/<run>/csv/dataset1.csv` 和 `dataset2.csv` 都存在。
- 每个 CSV 行数等于对应 `test.csv` 行数。
- zip 根目录只有 CSV 文件。
- 不使用 `--limit-rows` 产物提交。
