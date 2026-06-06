# jittor-GPNUT1-JGRec

第六届计图人工智能挑战赛赛道一动态推荐项目。当前代码提供一个可复现、可提交的 MVP 管线：读取 `data/dataset*/train.csv` 和 `test.csv`，对每个测试查询的 100 个候选目标节点生成概率分布，并打包为运行目录内固定命名的 `result.zip`。

## Quick Start

```bash
uv sync
uv run jgrec-build
```

项目默认使用 `.python-version` 固定的 Python 3.12；如本机尚未安装，`uv sync` 会自动获取兼容解释器。

选择模型后端：

```bash
uv run jgrec-build --model temporal-graph   # 当前默认模型
uv run jgrec-build --model craft            # 官方 CRAFT baseline 适配器
```

输出文件：

```text
result/<run_id>/
├── csv/
│   ├── dataset1.csv
│   └── dataset2.csv
└── result.zip
```

`<run_id>` 使用可读短名，例如 `temporal-graph_full_cuda_seed-42_hist-64_candhist-32_dim-128_<hash>`。

冒烟测试：

```bash
uv run jgrec-build --limit-rows 100
```

运行单元测试：

```bash
uv run --group dev pytest
```

运行 Ruff 检查：

```bash
uv run --group dev ruff check .
```

Optuna 调参：

```bash
uv run jgrec-tune-temporal-graph --n-trials 32 --n-jobs 1 --gpu-id 0 --quiet
```

多 GPU 调参使用多个进程共享同一个 study，细节见 [模型优化](docs/experiments/model-optimization.md)。

CPU 环境：

```bash
uv run jgrec-build --cpu
```

## Documentation

工程文档从 [docs/index.md](docs/index.md) 开始：

- 任务与数据：[赛题说明](docs/task/competition.md)、[数据契约](docs/task/data-contract.md)、[当前数据画像](docs/task/data-profile.md)
- 系统与模型：[系统架构](docs/system/architecture.md)、[模型设计](docs/system/modeling.md)
- 运行与开发：[运行手册](docs/operations/runbook.md)、[开发规范](docs/operations/development.md)
- 实验与研究：[架构优化](docs/experiments/architecture-optimization.md)、[模型优化](docs/experiments/model-optimization.md)、[研究问题综述](docs/research/problem-overview.md)、[GNN 推荐论文调研](docs/research/gnn-survey.md)

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

当前默认模型位于 `src/jgrec/rankers/temporal_graph/`。它是端到端训练的动态图候选重排序模型：使用 JittorGeometric `TemporalData` 与 temporal neighbor sampler 构造因果历史邻域，复用 CRAFT cross-attention 模块做候选-历史交互建模，并用同一个候选集 softmax loss 更新节点 embedding、temporal memory update、attention 和 scorer。

统一入口还支持 `src/jgrec/rankers/craft/` 中的 CRAFT baseline 适配器。

CLI 使用 Rich 展示运行配置、训练进度和结果表格；`--quiet-ranker` 可隐藏训练细节。
