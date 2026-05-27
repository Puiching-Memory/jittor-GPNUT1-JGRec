# jittor-GPNUT1-JGRec

第六届计图人工智能挑战赛赛道一动态推荐项目。当前代码提供一个可复现、可提交的 MVP 管线：读取 `data/dataset*/train.csv` 和 `test.csv`，对每个测试查询的 100 个候选目标节点生成概率分布，并打包为运行目录内固定命名的 `result.zip`。

## Quick Start

```bash
uv sync
uv run jgrec-build
```

输出文件：

```text
result/<run_id>/
├── csv/
│   ├── dataset1.csv
│   └── dataset2.csv
└── result.zip
```

冒烟测试：

```bash
uv run jgrec-build --limit-rows 100
```

CPU 环境：

```bash
uv run jgrec-build --cpu
```

## Documentation

工程文档从 [docs/index.md](docs/index.md) 开始：

- [运行手册](docs/runbook.md)
- [系统架构](docs/architecture.md)
- [数据契约](docs/data-contract.md)
- [模型方案](docs/modeling.md)
- [性能基准](docs/performance.md)
- [开发规范](docs/development.md)

构建本地文档站点：

```bash
uv sync --group dev
uv run zensical build
```

预览：

```bash
uv run zensical serve
```

## Current Model

当前模型位于 `src/jgrec/model.py`。它使用因果时间切分训练 JittorGeometric XSimGCL/LightGCN 图塔、SASRec 序列塔和时序统计特征，并用 Jittor MLP 在候选集内做 softmax 重排序。

CLI 使用 Rich 展示运行配置、训练进度和结果表格；`--quiet-ranker` 可隐藏训练细节。
