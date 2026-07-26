# 环境安装指南（项目级 CUDA 11.8 + Jittor 1.3.11.0）

## 适用范围

本文档记录在非 CUDA 11.8 系统上（例如本机默认使用 CUDA 13.2 / gcc 13）为项目搭建 CUDA 11.8 编译环境的完整步骤。目标是在 `uv` 管理的虚拟环境中让 Jittor 使用 GPU 运行测试和训练。

## 环境信息

- 容器/系统：Ubuntu 24.04.4 LTS
- 项目 Python 版本：3.12（由 `.python-version` 固定）
- 包管理器：uv 0.11.28
- 项目默认 CUDA 工具包：系统 `/usr/local/cuda` 为 13.2，gcc 13.3.0
- 项目目标 CUDA 工具包：CUDA 11.8，安装于 `third_party/cuda-toolkit/cuda-11.8`
- Jittor 版本：1.3.11.0（`third_party/jittor` 子模块，commit `5e412c63`）
- JittorGeometric 版本：`third_party/JittorGeometric` 子模块，commit `ff7d8ff`

## 安装步骤

### 1. 同步项目依赖

```bash
cd /root/workspace/jittor-GPNUT1-JGRec
uv sync
```

`uv sync` 会按 `pyproject.toml` + `uv.lock` 创建 `.venv` 并安装所有依赖，包括本地可编辑安装的 `jittor` 和 `jittor-geometric`。

### 2. 安装项目级 CUDA 11.8

如果系统默认 CUDA 高于 11.8（例如 13.2），需要单独下载 CUDA 11.8 工具包到项目目录，避免与系统 CUDA 冲突。

```bash
mkdir -p third_party/cuda-toolkit
cd third_party/cuda-toolkit
wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run
chmod +x cuda_11.8.0_520.61.05_linux.run
./cuda_11.8.0_520.61.05_linux.run --toolkit --silent --installpath=$(pwd)/cuda-11.8
```

验证安装：

```bash
./cuda-11.8/bin/nvcc --version
# 预期：release 11.8, V11.8.89
```

### 3. 安装与 CUDA 11.8 兼容的 C++ 编译器

CUDA 11.8 不支持 gcc 13，需要安装 gcc-11 / g++-11：

```bash
apt-get update
apt-get install -y gcc-11 g++-11
```

验证：

```bash
g++-11 --version
# 预期：g++-11 (Ubuntu 11.5.0-1ubuntu1~24.04.1) 11.5.0
```

### 4. 安装与 CUDA 11.8 兼容的 cuDNN

`pyproject.toml` 默认锁定的是 `nvidia-cudnn-cu12==8.9.7.29`，与 CUDA 11.8 不兼容。需要安装 CUDA 11 版本的 cuDNN 8：

```bash
uv pip uninstall nvidia-cudnn-cu12
uv pip install 'nvidia-cudnn-cu11>=8.9,<9'
```

验证：

```bash
uv pip show nvidia-cudnn-cu11
# 预期版本：8.9.6.50
ls .venv/lib/python3.12/site-packages/nvidia/cudnn/lib/libcudnn.so.8
```

同时，CUDA 11.8 需要 cuBLAS 11：

```bash
uv pip install nvidia-cublas-cu11
```

### 5. 清理 Jittor 旧编译缓存

如果之前用其他 CUDA 版本或 gcc 编译过，必须删除旧缓存和 overlay：

```bash
rm -rf .venv/jittor_nv126_overlay
rm -rf .venv/jittor_home/.cache/jittor
```

`sitecustomize.py` 会在运行时自动重建 `jittor_nv126_overlay`，将 CUDA 11.8 的 `include` 和 `lib64` 与 cuDNN 包链接在一起。

### 6. 验证 GPU 初始化

```bash
JITTOR_CUDA_HOME=/root/workspace/jittor-GPNUT1-JGRec/third_party/cuda-toolkit/cuda-11.8 \
cc_path=/usr/bin/g++-11 \
uv run python -c "import jittor as jt; jt.flags.use_cuda=1; print('has_cuda:', jt.has_cuda)"
```

成功时输出 `has_cuda: 1`，且不会报 `helper_cuda.h` 或 `host_config.h` 的编译错误。

## 运行测试

### 权重保存/加载功能测试（hybrid checkpoint）

```bash
JITTOR_CUDA_HOME=/root/workspace/jittor-GPNUT1-JGRec/third_party/cuda-toolkit/cuda-11.8 \
cc_path=/usr/bin/g++-11 \
uv run pytest tests/test_hybrid_checkpoint.py -v
```

预期结果：`9 passed`。

### 完整测试套件

```bash
JITTOR_CUDA_HOME=/root/workspace/jittor-GPNUT1-JGRec/third_party/cuda-toolkit/cuda-11.8 \
cc_path=/usr/bin/g++-11 \
uv run pytest tests/
```

预期结果：`174 passed, 1 warning`。

## 环境变量说明

每次运行 Jittor 相关命令时都需要设置以下变量：

| 变量               | 作用                                                       | 推荐值                                                                   |
| ------------------ | ---------------------------------------------------------- | ------------------------------------------------------------------------ |
| `JITTOR_CUDA_HOME` | 告诉 Jittor 和 `sitecustomize.py` 使用哪个 CUDA 工具包     | `/root/workspace/jittor-GPNUT1-JGRec/third_party/cuda-toolkit/cuda-11.8` |
| `cc_path`          | 告诉 Jittor 使用哪个 C++ 编译器（CUDA 11.8 不支持 gcc 13） | `/usr/bin/g++-11`                                                        |

`sitecustomize.py` 会在启动时基于 `JITTOR_CUDA_HOME` 自动创建 `jittor_nv126_overlay`，并将 `nvcc_path`、`CUDA_HOME`、`CUDA_PATH` 指向该 overlay。不要手动把系统 CUDA 13.2 的路径加到 `PATH` 或 `LD_LIBRARY_PATH`，否则会被 overlay 覆盖。

## 常见问题

### `helper_cuda.h` 报错 `computeMode` 和 `clockRate` 不是成员

原因：Jittor 的 `helper_cuda.h` 与 CUDA 13.2 的 `cudaDeviceProp` 结构不兼容。CUDA 13.2 已移除这两个字段。

解决：使用 CUDA 11.8，让 Jittor 使用兼容的 CUDA 头文件。

### `unsupported GNU version! gcc versions later than 11 are not supported`

原因：CUDA 11.8 的 `host_config.h` 拒绝 gcc 13。

解决：使用 `g++-11` 并通过 `cc_path=/usr/bin/g++-11` 传递给 Jittor。不要手动注释 `host_config.h` 的 `#error`，因为后续还会遇到更多 STL 不兼容错误。

### `undefined symbol: cudnnGetRNNParamsSize` 或 `cudnn is not loaded`

原因：安装的是 cuDNN 9，而 Jittor 1.3.11.0 的预编译 cuDNN 封装基于 cuDNN 8。

解决：安装 cuDNN 8 for CUDA 11：

```bash
uv pip install 'nvidia-cudnn-cu11>=8.9,<9'
```

### `Could not load library libcublasLt.so.12`

原因：系统中没有 CUDA 12 的 cuBLAS，或者 `nvidia-cublas-cu12` 与 CUDA 11.8 的 cudart 不匹配。

解决：安装 `nvidia-cublas-cu11`，让 Jittor overlay 链接到 cuBLAS 11。

## 版本对应关系（已验证）

| 组件            | 版本                                           |
| --------------- | ---------------------------------------------- |
| Python          | 3.12.13                                        |
| uv              | 0.11.28                                        |
| Jittor          | 1.3.11.0 (`5e412c63`)                          |
| JittorGeometric | `ff7d8ff`                                      |
| CUDA 工具包     | 11.8.89 (`third_party/cuda-toolkit/cuda-11.8`) |
| C++ 编译器      | g++-11 11.5.0                                  |
| cuDNN           | `nvidia-cudnn-cu11==8.9.6.50`                  |
| cuBLAS          | `nvidia-cublas-cu11==11.11.3.6`                |

## 单次测试命令（可直接复制）

```bash
cd /root/workspace/jittor-GPNUT1-JGRec
export JITTOR_CUDA_HOME=/root/workspace/jittor-GPNUT1-JGRec/third_party/cuda-toolkit/cuda-11.8
export cc_path=/usr/bin/g++-11
uv run pytest tests/
```

## 单次训练/提交命令（可直接复制）

```bash
cd /root/workspace/jittor-GPNUT1-JGRec
export JITTOR_CUDA_HOME=/root/workspace/jittor-GPNUT1-JGRec/third_party/cuda-toolkit/cuda-11.8
export cc_path=/usr/bin/g++-11
uv run jgrec-build
```
