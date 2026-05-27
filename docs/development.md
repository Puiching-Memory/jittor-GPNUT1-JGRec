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
uv run jgrec-build --limit-rows 100 --zip-path result-smoke.zip
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
- `result.zip`
- `result-smoke.zip`
- `site/`
- `.venv/`

## 提交前检查

建议至少运行：

```bash
uv run python -m compileall -q src
uv run jgrec-build --limit-rows 100 --zip-path result-smoke.zip
uv lock --check
```

如果改了文档：

```bash
uv sync --group dev
uv run zensical build
```

## 当前已知限制

- 当前 baseline 不是训练型模型，性能上限有限。
- `fit()` 会把单个数据集训练交互加载到内存中。
- 没有内置本地验证集切分和 MRR 评估。
- 没有模型权重搜索或自动调参。

这些限制不影响生成合规提交文件，但会影响榜单分数。后续优化应优先补本地验证闭环。
