# jittor-GPNUT1-JGRec 文档

本文档按问题域组织，而不是按文件产生顺序平铺。每个主题只回答一类问题，避免运行、数据、模型、实验和研究资料互相混杂。

## 阅读路径

| 目标 | 入口 |
| --- | --- |
| 先把提交文件跑出来 | [运行手册](operations/runbook.md) |
| 确认输入输出格式 | [数据契约](task/data-contract.md) |
| 理解赛题到底是什么 | [赛题说明](task/competition.md) 和 [研究问题综述](research/problem-overview.md) |
| 判断数据适合什么模型 | [当前数据画像](task/data-profile.md) |
| 理解代码如何串起来 | [系统架构](system/architecture.md) |
| 理解当前模型为什么这样设计 | [模型设计](system/modeling.md) |
| 判断一次实验能不能保留 | [实验与基准](experiments/benchmarks.md) |
| 查找论文和外部实现线索 | [研究资料](research/gnn-survey.md) 与 [开源参考](research/open-source-references.md) |
| 修改代码前看工程约束 | [开发规范](operations/development.md) |

## 文档分组

### 任务与数据

- [赛题说明](task/competition.md)：比赛原文整理、评测指标、提交格式。
- [数据契约](task/data-contract.md)：本工程接受的 `train.csv`、`test.csv` 和输出 CSV/ZIP 约束。
- [当前数据画像](task/data-profile.md)：本地数据统计、候选分布、特征区分度和建模含义。

### 系统与模型

- [系统架构](system/architecture.md)：包结构、统一接口、数据流和扩展边界。
- [模型设计](system/modeling.md)：当前 hybrid/craft/third_party 后端、训练流程、特征和融合方式。

### 运行与开发

- [运行手册](operations/runbook.md)：环境、命令、常用参数、输出校验和常见问题。
- [开发规范](operations/development.md)：本地检查、依赖策略、代码边界和提交前检查。

### 实验与研究

- [实验与基准](experiments/benchmarks.md)：冠军基线、实验门禁、性能优化和复测命令。
- [研究问题综述](research/problem-overview.md)：将赛题抽象成动态图候选重排序研究问题。
- [GNN 推荐论文调研](research/gnn-survey.md)：图协同过滤、图对比学习、谱图和动态图方向。
- [推荐系统论文调研归档](research/recommender-survey.md)：非 GNN 推荐、序列、排序和生成式推荐背景。
- [开源参考](research/open-source-references.md)：本地 third_party 示例和可参考实现。

## 最短路径

```bash
uv sync
uv run jgrec-build
```

完成后检查：

```text
result/<run_id>/
├── csv/
│   ├── dataset1.csv
│   └── dataset2.csv
└── result.zip
```

每个 CSV 无表头，每行 100 个保留 8 位小数的概率。
