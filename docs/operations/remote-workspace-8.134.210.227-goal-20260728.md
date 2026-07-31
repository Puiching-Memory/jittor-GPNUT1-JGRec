# Goal Document: 新服务器 JGRec 隔离工作区

## Go / No-Go

- **Judgment**: Go
- **Reason**: 用户已明确授权连接指定服务器、选择工作区、上传当前代码并配置独立环境；所有变更可限制在远端 `edu` 账号的新目录内。

## Target Outcome

在 `8.134.210.227:22222` 的 `edu` 账号下建立一个不复用、不修改其他项目环境的 JGRec
工作区，上传当前本地代码，使用项目级 uv/Python/Jittor 缓存完成依赖同步，并通过 CPU/GPU、
项目导入和聚焦测试验证。完成前不报告“已配置好”。

## Goal Definition

- **Type**: operational / delivery
- **Boundary**: SSH 连通性、远端资源探测、独立目录选择、代码传输、项目级 uv 环境、最小运行验证。
- **Non-goals**:
  - 不修改系统 Python、系统 CUDA、全局 pip/conda 环境或其他用户/项目目录。
  - 不启动正式训练、rolling-origin、external gate 或提交包生成。
  - 不迁移本地数据集、checkpoint、结果目录和缓存，除非项目运行契约明确需要小型代码资产。
- **Deferred work**:
  - 正式 GPU 实验与长时间训练。
  - 数据集和大型 checkpoint 的定向同步。
- **Verification rule**: 远端独立目录存在；源代码完整；项目级 `uv sync` 成功；远端解释器来自该目录；Jittor/GPU 探测与聚焦测试成功。
- **Evidence source**: SSH 命令输出、路径/磁盘/GPU探测、`uv sync`、Python 导入、pytest。
- **Pass criteria**:
  - 新工作区的规范路径不与现有目录重合，且归 `edu` 所有。
  - `.venv`、uv 缓存、uv Python 安装目录、Jittor 缓存都位于新工作区。
  - 不写入 `.bashrc`、系统 site-packages 或其他项目环境。
  - 项目导入与至少一个不依赖数据的聚焦测试通过。
- **Confidence note**: 路径与解释器位置可直接检查；GPU 可用性以 `nvidia-smi` 和项目解释器内探测为准。
- **Judgment owner**: 远端验证命令。

## Current State

- 这是新服务器，尚未确认 SSH、主目录、可用磁盘、GPU、CUDA、uv 或 Python 状态。
- 本地工作树包含尚未提交的有效代码，不能只上传 Git HEAD。
- 项目要求使用 uv；Jittor 及 JittorGeometric 来自 `third_party/` 可编辑依赖。
- 密码只用于本次连接，不写入仓库、远端配置或日志文件。

## Priority Rationale

- 先只读探测，避免把项目放到容量不足或已有环境密集的目录。
- 再冻结独立路径和缓存边界，之后才传输和安装。
- 最后以实际解释器路径和测试证明隔离与可运行性。

## Assumptions and Open Decisions

| Item | Status | Impact | Owner / Next step |
|---|---|---|---|
| 远端为 Linux x86_64 | assumed | 决定 uv/依赖安装方式 | SSH 探测确认 |
| `edu` 对其主目录或数据盘有写权限 | assumed | 决定工作区位置 | 对候选目录做只读/写权限检查 |
| 服务器可访问 Python 包源 | unresolved | 影响 `uv sync` | 先检查现有 uv，再实际同步 |
| 正式数据无需本轮上传 | confirmed by boundary | 避免大文件和跨实验污染 | 仅传代码与必要 third_party |

## Phases

### Phase 1: 只读连接与资源探测

- **Purpose**: 确定是否可连接以及哪个根目录适合隔离工作区。
- **Entry condition**: 用户提供主机、端口和账号凭据。
- **Phase rules**:
  - 只运行身份、目录、磁盘、GPU、Python、uv 和进程/目录概览命令。
  - 不创建目录、不安装依赖。
- **Todos**:
  - [ ] 验证 SSH 和账号身份
    - **Surface**: 远端会话
    - **Proof**: `id`、`hostname`、`pwd`
    - **Depends on**: none
  - [ ] 比较可写候选盘容量与已有目录
    - **Surface**: 远端文件系统
    - **Proof**: `df`、候选目录元数据
    - **Depends on**: SSH
  - [ ] 探测 GPU/CUDA/Python/uv
    - **Surface**: 远端工具链
    - **Proof**: 版本命令
    - **Depends on**: SSH
- **Exit proof**: 选出一个明确、可写、空间足够且不与现有项目重合的工作区路径。
- **Stop condition**: SSH 不通、凭据错误、无安全可写目录或磁盘明显不足。

### Phase 2: 隔离工作区与代码传输

- **Purpose**: 建立唯一目标目录并上传当前工作树。
- **Entry condition**: Phase 1 选址完成。
- **Phase rules**:
  - 只创建所选新目录。
  - 排除 `.git`、本地虚拟环境、缓存、数据、结果、checkpoint 和临时文件。
  - 保留当前未提交源码与必要 `third_party/`。
- **Todos**:
  - [ ] 创建并验证目标目录所有权
    - **Surface**: 远端文件系统
    - **Proof**: 规范路径、owner、mode
    - **Depends on**: Phase 1
  - [ ] 生成受控源码归档并上传解包
    - **Surface**: 本地工作树 / 远端工作区
    - **Proof**: 关键文件、文件数、归档校验
    - **Depends on**: 目标目录
- **Exit proof**: 远端 `pyproject.toml`、`uv.lock`、`src/`、`tests/`、必要 third_party 均存在。
- **Stop condition**: 目标目录已存在且不是本轮创建、传输校验不一致或出现超范围大文件。

### Phase 3: 项目级环境配置

- **Purpose**: 在项目目录内部建立 uv、Python、venv 与编译缓存。
- **Entry condition**: 代码传输校验通过。
- **Phase rules**:
  - 不使用 `sudo`，不调用全局 pip，不修改 shell 启动文件。
  - 所有缓存和解释器路径显式指向工作区。
  - 优先使用远端已有 uv；没有则安装到工作区 `.tools/`。
- **Todos**:
  - [ ] 建立项目环境入口脚本
    - **Surface**: 工作区配置
    - **Proof**: 环境变量均解析到工作区内
    - **Depends on**: Phase 2
  - [ ] 执行 `uv sync`
    - **Surface**: `.venv` / 项目级缓存
    - **Proof**: 同步命令成功、解释器路径正确
    - **Depends on**: uv 可用
- **Exit proof**: `.venv` 和所有声明的缓存均位于新工作区，依赖同步成功。
- **Stop condition**: 需要系统级变更、会覆盖其他环境、依赖源不可达且无可用离线替代。

### Phase 4: 可运行性验证

- **Purpose**: 证明环境不是“安装完成但不可用”。
- **Entry condition**: `uv sync` 成功。
- **Phase rules**:
  - 只运行短时、无数据、无正式训练的 smoke test。
  - Jittor 缓存继续限定在工作区。
- **Todos**:
  - [ ] 验证 Python、项目与 Jittor 导入
    - **Surface**: 项目解释器
    - **Proof**: 模块路径和版本输出
    - **Depends on**: Phase 3
  - [ ] 验证 GPU 可见性
    - **Surface**: CUDA/Jittor
    - **Proof**: GPU/驱动探测
    - **Depends on**: 工具链
  - [ ] 运行聚焦测试
    - **Surface**: pytest
    - **Proof**: 留一法本地测试通过
    - **Depends on**: 项目导入
- **Exit proof**: 导入、GPU 探测和聚焦测试均成功。
- **Stop condition**: 验证要求修改系统环境或会启动长时间编译/训练且无法安全限制。

## Dry-Run Findings

- 直接 `git clone` 或 `git archive HEAD` 会丢失本地未提交实现，因此必须传受控的当前工作树。
- 直接复用远端 conda/venv 会破坏隔离目标，因此解释器、缓存和依赖均需工作区内落地。
- Jittor 首次导入可能触发编译；需要显式 `JITTOR_HOME`，否则会污染账号级缓存。
- 选址必须先看磁盘与已有目录，不能预设 `/home/edu` 一定合适。

## Final Validation

在远端环境入口生效后执行：

```bash
uv run python -c "import sys, jgrec, jittor; print(sys.executable); print(jgrec.__file__); print(jittor.__version__)"
uv run python -m pytest -q tests/test_hybrid_feature_mask_leave_one_out.py
```

并检查上述解释器、模块与 `JITTOR_HOME/UV_CACHE_DIR/UV_PYTHON_INSTALL_DIR` 全部位于新工作区。

## First Execution Step

通过 SSH 运行只读身份、文件系统、GPU和工具链探测，确认工作区选址。
