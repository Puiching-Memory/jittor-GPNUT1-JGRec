# jittor-GPNUT1-JGRec

第六届计图人工智能挑战赛赛道一动态推荐项目。当前代码提供一个可复现、可提交的 MVP 管线：读取 `data/dataset*/train.csv` 和 `test.csv`，对每个测试查询的 100 个候选目标节点生成概率分布，并打包为运行目录内固定命名的 `result.zip`。

## Quick Start

```bash
uv sync
uv run jgrec-build
```

选择模型后端：

```bash
uv run jgrec-build --model hybrid       # 当前默认模型
uv run jgrec-build --model craft        # 官方 CRAFT baseline 适配器
uv run jgrec-build --model third_party  # 多尺度统计/结构特征重排器
```

输出文件：

```text
result/<run_id>/
├── csv/
│   ├── dataset1.csv
│   └── dataset2.csv
└── result.zip
```

`<run_id>` 使用可读短名，例如 `hybrid_full_cuda_seed-42_gnn-xsimgcl_sequence-on_<hash>`。

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

当前默认模型位于 `src/jgrec/rankers/hybrid/`。它使用因果时间切分训练 JittorGeometric XSimGCL/LightGCN 图塔、SASRec 序列塔和时序统计特征，并用 Jittor MLP 在候选集内做 softmax 重排序。

统一入口还支持 `src/jgrec/rankers/craft/` 中的 CRAFT baseline 适配器，以及 `src/jgrec/rankers/third_party/` 中根据 `architecture.md` 落地的多尺度统计/结构特征重排器。

CLI 使用 Rich 展示运行配置、训练进度和结果表格；`--quiet-ranker` 可隐藏训练细节。
