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

程序会扫描 `data/` 下所有同时包含 `train.csv` 和 `test.csv` 的子目录。子目录名就是输出文件名，例如 `dataset1` 对应 `result/dataset1.csv`。

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

默认输出：

```text
result/
├── dataset1.csv
└── dataset2.csv
result.zip
```

## 冒烟测试

限制每个数据集只输出前 100 行：

```bash
uv run jgrec-build --limit-rows 100 --zip-path result-smoke.zip
```

该命令适合验证环境、Jittor 初始化、CSV 读写和压缩包生成。

## CPU 运行

```bash
uv run jgrec-build --cpu
```

当前 MVP 的图统计主要在 CPU 上完成，Jittor 只负责候选批量张量打分。没有 GPU 时可以使用 `--cpu`。

## 常用参数

| 参数              |       默认值 | 说明                             |
| ----------------- | -----------: | -------------------------------- |
| `--data-dir`      |       `data` | 数据根目录                       |
| `--output-dir`    |     `result` | 单数据集 CSV 输出目录            |
| `--zip-path`      | `result.zip` | 最终提交压缩包路径               |
| `--recent-window` |         `32` | 每个源节点保留的最近目标节点数量 |
| `--batch-size`    |       `2048` | 每次送入 Jittor 的测试查询数     |
| `--limit-rows`    |           无 | 每个数据集最多输出的测试行数     |
| `--cpu`           |         关闭 | 禁用 CUDA                        |
| `--skip-validate` |         关闭 | 跳过输出格式校验                 |

## 输出校验

默认运行会校验：

- 每个输出 CSV 行数等于对应 `test.csv` 数据行数。
- 每行正好 100 列。
- 每个概率在 `[0, 1]` 范围内。

手工验证压缩包内容：

```bash
uv run python - <<'PY'
import zipfile

with zipfile.ZipFile("result.zip") as zf:
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

| 文件                  |   行数 |
| --------------------- | -----: |
| `result/dataset1.csv` |  61051 |
| `result/dataset2.csv` | 153420 |

## 常见问题

### Jittor 找不到 `python3.11-config`

仓库根目录的 `sitecustomize.py` 会设置 Jittor 所需的 Python/CUDA 环境变量。正常通过 `uv run jgrec-build` 运行即可加载。

### ZIP 工具不可用

生成压缩包不依赖系统 `zip` 或 `unzip` 命令，代码使用 Python 标准库 `zipfile`。

### 结果文件很大

正式赛要求每个测试查询输出 100 个 8 位小数概率。当前本地输出约为：

- `result/dataset1.csv`: 65 MB
- `result/dataset2.csv`: 161 MB
- `result.zip`: 55 MB
