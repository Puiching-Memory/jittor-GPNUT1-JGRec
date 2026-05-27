# jittor-GPNUT1-JGRec

第六届计图人工智能挑战赛赛道一动态推荐项目。当前代码提供一个可复现、可提交的 MVP 管线：读取 `data/dataset*/train.csv` 和 `test.csv`，对每个测试查询的 100 个候选目标节点生成概率分布，并打包为 `result.zip`。

## Quick Start

```bash
uv sync
uv run jgrec-build
```

输出文件：

```text
result/
├── dataset1.csv
└── dataset2.csv
result.zip
```

冒烟测试：

```bash
uv run jgrec-build --limit-rows 100 --zip-path result-smoke.zip
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

## Current Baseline

MVP 模型位于 `src/jgrec/model.py`。它使用历史边统计、源节点重复交互、目标节点热度、时序近因和最近邻命中等特征，通过 Jittor 张量批量融合并 softmax 归一化。该方案优先保证端到端可运行、格式正确、内存可控，后续可替换为可训练的 Jittor/JittorGeometric 时序图模型。
