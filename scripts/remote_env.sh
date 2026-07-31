#!/usr/bin/env bash

_jgrec_remote_script_dir="$(
    cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1
    pwd
)"
JGREC_PROJECT_ROOT="$(
    cd -- "${_jgrec_remote_script_dir}/.." >/dev/null 2>&1
    pwd
)"
export JGREC_PROJECT_ROOT
unset _jgrec_remote_script_dir

export UV_PROJECT_ENVIRONMENT="${JGREC_PROJECT_ROOT}/.venv"
export UV_CACHE_DIR="${JGREC_PROJECT_ROOT}/.remote-state/uv-cache"
export UV_PYTHON_INSTALL_DIR="${JGREC_PROJECT_ROOT}/.remote-state/uv-python"
export JITTOR_HOME="${JGREC_PROJECT_ROOT}/.remote-state/jittor"
export XDG_CACHE_HOME="${JGREC_PROJECT_ROOT}/.remote-state/xdg-cache"
export TMPDIR="${JGREC_PROJECT_ROOT}/.remote-state/tmp"

_jgrec_jtcuda_root="${JITTOR_HOME}/.cache/jittor/jtcuda/cuda12.2_cudnn8_linux"
if [[ -x "${_jgrec_jtcuda_root}/bin/nvcc" ]]; then
    export CUDA_HOME="${_jgrec_jtcuda_root}"
else
    export CUDA_HOME="/usr/local/cuda-11.8"
fi
unset _jgrec_jtcuda_root

export nvcc_path="${CUDA_HOME}/bin/nvcc"
export PATH="${JGREC_PROJECT_ROOT}/.tools:${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
