# 运行手册

## 前置条件

- Python 版本由 `.python-version` 固定为 `3.11`。
- 依赖由 `uv.lock` 锁定。
- Jittor 与 JittorGeometric 通过 `third_party/` 本地路径安装。
- 正式赛数据放在 `data/` 下。`data/` 已在 `.gitignore` 中，不应提交到仓库。

## 数据目录

```text
data/
├── dataset1/
│   ├── train.csv
│   └── test.csv
└── dataset2/
    ├── train.csv
    └── test.csv
```

程序会扫描 `data/` 下所有同时包含 `train.csv` 和 `test.csv` 的子目录。子目录名就是输出文件名，例如 `dataset1` 对应 `result/<run_id>/csv/dataset1.csv`。

## 环境同步

```bash
uv sync
```

如需构建文档：

```bash
uv sync --group dev
```

## 生成提交文件

```bash
uv run jgrec-build
```

默认会为每个数据集做一次时间切分训练，训练 XSimGCL 图塔、SASRec 序列塔和融合 MLP；验证协议对齐官方 baseline，使用 train.csv 尾部 15% 做验证，融合早停和特征组选择默认看 AP，同时打印 MRR 作为诊断指标。
CLI 使用 Rich 输出运行面板、训练进度和结果表格；提交文件仍只写入 `result/`，终端样式不会影响 CSV/ZIP 内容。

模型后端通过 `--model` 选择：

```bash
uv run jgrec-build --model hybrid
uv run jgrec-build --model craft
uv run jgrec-build --model third_party
```

默认输出：

```text
result/<run_id>/
├── csv/
│   ├── dataset1.csv
│   └── dataset2.csv
└── result.zip
```

`result.zip` 是固定文件名，始终位于本次运行目录内。`<run_id>` 使用人类可读短名，最后追加 8 位配置哈希避免不同隐藏参数写入同一目录。

默认规则：

```text
{model}_{full|sample-N-rows}_{cpu|cuda}_seed-{seed}_{hash}
```

`hybrid` 后端会额外追加图塔和序列塔状态：

```text
hybrid_full_cuda_seed-42_gnn-xsimgcl_sequence-on_1a2b3c4d
hybrid_sample-2-rows_cpu_seed-42_gnn-off_sequence-off_1a2b3c4d
third-party_full_cuda_seed-42_1a2b3c4d
```

## 冒烟测试

快速验证完整混合模型路径：

```bash
uv run jgrec-build --limit-rows 2 --max-fit-events 512 --max-train-events 32 --max-val-events 16 --num-negatives 3 --epochs 1 --gnn-epochs 1 --gnn-embedding-dim 16 --gnn-layers 1 --gnn-max-graph-edges 256 --gnn-max-train-edges 128 --seq-epochs 1 --seq-max-samples 128 --seq-max-len 16 --seq-hidden-size 16 --fusion-hidden-dim 16 --quiet-ranker
```

该命令适合验证环境、Jittor/JittorGeometric 初始化、CSV 读写和压缩包生成。

只验证提交链路、不训练图塔和序列塔：

```bash
uv run jgrec-build --limit-rows 100 --max-fit-events 2048 --max-train-events 256 --max-val-events 128 --epochs 1 --disable-gnn --disable-seq --quiet-ranker
```

验证第三方统计/结构重排器：

```bash
uv run jgrec-build --model third_party --limit-rows 2 --max-train-events 32 --max-val-events 16 --num-negatives 3 --epochs 1 --train-batch-size 8 --fusion-hidden-dim 8 --batch-size 2 --quiet-ranker --cpu
```

并行调参时必须使用不同参数组合，使生成的 `<run_id>` 不同；相同参数组合会写入同一个运行目录。

## CPU 运行

```bash
uv run jgrec-build --cpu
```

没有 GPU 时可以使用 `--cpu`。完整混合模型在 CPU 上会明显变慢，建议先用冒烟参数确认流程。

## 常用参数

| 参数                    |    默认值 | 说明                             |
| ----------------------- | --------: | -------------------------------- |
| `--model`               |  `hybrid` | 模型后端：`hybrid`、`craft`、`third_party` |
| `--data-dir`            |    `data` | 数据根目录                       |
| `--recent-window`       |      `32` | 每个源节点保留的最近目标节点数量 |
| `--batch-size`          |    `2048` | 每次送入 Jittor 的测试查询数     |
| `--limit-rows`          |        无 | 每个数据集最多输出的测试行数     |
| `--val-ratio`           |    `0.15` | `train.csv` 尾部验证比例         |
| `--context-ratio`       |    `0.75` | 用于训练监督样本的因果上下文比例 |
| `--max-train-events`    |   `20000` | 每个数据集最多训练监督事件       |
| `--max-val-events`      |    `5000` | 每个数据集最多验证事件           |
| `--num-negatives`       |      `31` | 每个正样本采样负候选数           |
| `--max-fit-events`      |       `0` | 模型训练历史尾部截断，0 表示全量 |
| `--epochs`              |       `5` | 融合 MLP 训练轮数                |
| `--train-batch-size`    |     `512` | 融合 MLP 训练 batch size         |
| `--lr`                  |   `0.001` | SASRec 和融合 MLP 学习率         |
| `--selection-metric`    |      `ap` | 融合早停和特征组选择指标         |
| `--early-stop`          |      `10` | 融合早停 patience，0 表示关闭    |
| `--gnn-model`           | `xsimgcl` | 图塔模型，可选 `lightgcn`        |
| `--gnn-epochs`          |       `3` | 每个图窗口训练轮数               |
| `--gnn-max-train-edges` |   `40000` | 每轮图训练采样边数               |
| `--seq-epochs`          |       `3` | SASRec 训练轮数                  |
| `--seq-max-samples`     |   `50000` | SASRec 最多训练样本              |
| `--disable-gnn`         |      关闭 | 禁用图塔                         |
| `--disable-seq`         |      关闭 | 禁用序列塔                       |
| `--seed`                |      `42` | 采样随机种子                     |
| `--quiet-ranker`        |      关闭 | 隐藏训练 epoch 进度和细节日志    |
| `--cpu`                 |      关闭 | 禁用 CUDA                        |
| `--skip-validate`       |      关闭 | 跳过输出格式校验                 |

## 输出校验

默认运行会校验：

- 每个输出 CSV 行数等于对应 `test.csv` 数据行数。
- 每行正好 100 列。
- 每个概率在 `[0, 1]` 范围内。

手工验证压缩包内容：

```bash
uv run python - <<'PY'
import zipfile

with zipfile.ZipFile("result/<run_id>/result.zip") as zf:
    print(zf.namelist())
PY
```

## 已验证数据规模

当前本地数据：

| 数据集     | 训练行数 | 测试行数 |
| ---------- | -------: | -------: |
| `dataset1` |   690848 |    61051 |
| `dataset2` |  2261283 |   153420 |

完整生成后：

| 文件                               |   行数 |
| ---------------------------------- | -----: |
| `result/<run_id>/csv/dataset1.csv` |  61051 |
| `result/<run_id>/csv/dataset2.csv` | 153420 |

## 常见问题

### Jittor 找不到 `python3.11-config`

仓库根目录的 `sitecustomize.py` 会设置 Jittor 所需的 Python/CUDA 环境变量。正常通过 `uv run jgrec-build` 运行即可加载。

### ZIP 工具不可用

生成压缩包不依赖系统 `zip` 或 `unzip` 命令，代码使用 Python 标准库 `zipfile`。

### 结果文件很大

正式赛要求每个测试查询输出 100 个 8 位小数概率。当前本地输出约为：

- `result/<run_id>/csv/dataset1.csv`: 65 MB
- `result/<run_id>/csv/dataset2.csv`: 161 MB
- `result/<run_id>/result.zip`: 55 MB
