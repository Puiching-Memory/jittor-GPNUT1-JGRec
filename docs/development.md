# 开发规范

## 本地命令

同步依赖：

```bash
uv sync
```

编译检查：

```bash
uv run python -m compileall -q src
```

冒烟测试：

```bash
uv run jgrec-build --limit-rows 2 --max-fit-events 512 --max-train-events 32 --max-val-events 16 --num-negatives 3 --epochs 1 --gnn-epochs 1 --gnn-embedding-dim 16 --gnn-layers 1 --gnn-max-graph-edges 256 --gnn-max-train-edges 128 --seq-epochs 1 --seq-max-samples 128 --seq-max-len 16 --seq-hidden-size 16 --fusion-hidden-dim 16 --quiet-ranker
```

完整生成：

```bash
uv run jgrec-build
```

文档构建：

```bash
uv sync --group dev
uv run zensical build
```

## 依赖策略

当前核心管线不依赖 pandas。新增依赖前需要确认：

- 是否能显著降低复杂度或提升性能。
- 是否会增加比赛环境部署风险。
- 是否会显著增加内存占用。
- 是否已经写入 `pyproject.toml` 并更新 `uv.lock`。

当前 CLI 依赖 Rich 做运行面板、进度条和结果表格；它只影响终端输出，不参与提交文件内容生成。

## 代码边界

修改建议：

- 数据格式变化优先改 `jgrec.data`。
- 模型或特征变化优先改 `jgrec.model`。
- 输出格式、打包和校验变化优先改 `jgrec.submission`。
- 命令行参数和主流程变化优先改 `jgrec.cli`。

避免把数据读取、模型打分、文件写出混在同一个函数中；这样后续替换模型时不需要重写提交逻辑。

## 输出文件管理

以下路径不提交：

- `data/`
- `result/`
- `site/`
- `.venv/`

## 提交前检查

建议至少运行：

```bash
uv run python -m compileall -q src
uv run jgrec-build --limit-rows 2 --max-fit-events 512 --max-train-events 32 --max-val-events 16 --num-negatives 3 --epochs 1 --gnn-epochs 1 --gnn-embedding-dim 16 --gnn-layers 1 --gnn-max-graph-edges 256 --gnn-max-train-edges 128 --seq-epochs 1 --seq-max-samples 128 --seq-max-len 16 --seq-hidden-size 16 --fusion-hidden-dim 16 --quiet-ranker
uv lock --check
```

如果改了文档：

```bash
uv sync --group dev
uv run zensical build
```

性能优化需要记录前后基准，统一写入 [性能基准](performance.md)。

## 当前已知限制

- `fit()` 会把单个数据集训练交互加载到内存中。
- 负采样还是随机采样，没有 hard negatives。
- 当前图塔是静态窗口图，不是严格事件级 TGN memory。
- 没有模型权重持久化，每次运行都会重新训练。

这些限制不影响生成合规提交文件，但会影响榜单分数。后续优化应优先补本地验证闭环。
