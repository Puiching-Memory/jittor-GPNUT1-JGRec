import os
import shutil
import stat
import sys
from pathlib import Path


os.environ.setdefault("JITTOR_HOME", str(Path(sys.prefix) / "jittor_home"))


def _prepend_path_env(name, *paths):
    values = [str(path) for path in paths if path and Path(path).exists()]
    if not values:
        return
    current = os.environ.get(name, "")
    current_values = [value for value in current.split(os.pathsep) if value]
    os.environ[name] = os.pathsep.join(values + [value for value in current_values if value not in values])


def _drop_cuda_path_env(name):
    current_values = [value for value in os.environ.get(name, "").split(os.pathsep) if value]
    os.environ[name] = os.pathsep.join(value for value in current_values if "cuda" not in value.lower())


def _find_package_dir(*parts):
    for entry in sys.path:
        candidate = Path(entry, *parts)
        if candidate.exists():
            return candidate
    candidate = Path(sys.prefix, "lib", f"python{sys.version_info.major}.{sys.version_info.minor}", "site-packages", *parts)
    if candidate.exists():
        return candidate
    return None


def _link_child(src, dst, replace_symlink=False):
    if dst.exists() or dst.is_symlink():
        if dst.is_symlink() and replace_symlink:
            dst.unlink()
        else:
            return
    dst.symlink_to(src, target_is_directory=src.is_dir())


def _link_children(src_dir, dst_dir, replace_symlinks=False):
    if not src_dir or not src_dir.exists():
        return
    dst_dir.mkdir(parents=True, exist_ok=True)
    for src in src_dir.iterdir():
        _link_child(src, dst_dir / src.name, replace_symlinks)


def _create_cuda_overlay(nvcc, cudnn_dir):
    real_nvcc = Path(nvcc).resolve()
    cuda_home = real_nvcc.parents[1]
    overlay = Path(sys.prefix) / "jittor_nv126_overlay"
    overlay_bin = overlay / "bin"
    overlay_include = overlay / "include"
    overlay_lib = overlay / "lib64"

    overlay_bin.mkdir(parents=True, exist_ok=True)
    _link_children(cuda_home / "include", overlay_include)
    _link_children(cuda_home / "lib64", overlay_lib)

    cudnn_include = cudnn_dir / "include"
    cudnn_lib = cudnn_dir / "lib"
    _link_children(cudnn_include, overlay_include, replace_symlinks=True)
    _link_children(cudnn_lib, overlay_lib, replace_symlinks=True)

    for lib in cudnn_lib.glob("libcudnn*.so.*"):
        alias = overlay_lib / f"{lib.name.split('.so.')[0]}.so"
        _link_child(lib, alias, replace_symlink=True)

    wrapper = overlay_bin / "nvcc"
    wrapper.write_text(f"#!/bin/sh\nexec {real_nvcc} \"$@\"\n")
    wrapper.chmod(wrapper.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)

    os.environ["nvcc_path"] = str(wrapper)
    os.environ.setdefault("CUDA_HOME", str(overlay))
    os.environ.setdefault("CUDA_PATH", str(overlay))
    os.environ.setdefault("JITTOR_CUDA126_OVERLAY", str(overlay))
    for name in ("PATH", "CPATH", "LIBRARY_PATH", "LD_LIBRARY_PATH"):
        _drop_cuda_path_env(name)

    _prepend_path_env("PATH", overlay_bin)
    _prepend_path_env("CPATH", overlay_include)
    _prepend_path_env("LIBRARY_PATH", overlay_lib)
    _prepend_path_env("LD_LIBRARY_PATH", overlay_lib)


if "python_config_path" not in os.environ:
    version = f"{sys.version_info.major}.{sys.version_info.minor}"
    names = (f"python{version}-config", "python3-config")
    dirs = [Path(sys.executable).resolve().parent]

    for prefix in (sys.base_prefix, sys.base_exec_prefix, sys.exec_prefix, sys.prefix):
        if prefix:
            path = Path(prefix) / "bin"
            if path not in dirs:
                dirs.append(path)

    for directory in dirs:
        for name in names:
            candidate = directory / name
            if candidate.exists():
                os.environ["python_config_path"] = str(candidate)
                break
        if "python_config_path" in os.environ:
            break

if "nvcc_path" not in os.environ:
    nvcc = None
    for candidate in (
        Path(os.environ.get("JITTOR_CUDA_HOME", "")) / "bin" / "nvcc",
        Path(os.environ.get("CUDA_HOME", "")) / "bin" / "nvcc",
        Path("/usr/local/cuda-12.6/bin/nvcc"),
        Path("/usr/local/cuda/bin/nvcc"),
        Path("/usr/bin/nvcc"),
        Path("/opt/cuda/bin/nvcc"),
    ):
        if candidate.exists():
            nvcc = str(candidate)
            break
    if nvcc is None:
        nvcc = shutil.which("nvcc")

    if nvcc is not None:
        cuda_root = Path(nvcc).resolve().parents[1]
        cudnn_dir = _find_package_dir("nvidia", "cudnn")
        cudnn_candidates = (
            cuda_root / "include" / "cudnn.h",
            cuda_root / "targets" / "x86_64-linux" / "include" / "cudnn.h",
            Path("/usr/include/cudnn.h"),
        )
        if cudnn_dir and (cudnn_dir / "include" / "cudnn.h").exists():
            _create_cuda_overlay(nvcc, cudnn_dir)
        elif not any(candidate.exists() for candidate in cudnn_candidates):
            os.environ["nvcc_path"] = ""
