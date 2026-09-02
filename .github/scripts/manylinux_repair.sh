#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.

# 对构建产物执行 auditwheel repair, 生成 manylinux wheel (仅 release 流程使用)
# 用法: manylinux_repair.sh [输入目录=dist] [输出目录=dist-manylinux]

# DTK 运行环境 (HIP/ROCm 库路径); env.sh 引用了未初始化的变量,
# 必须在 set -u 之前加载, 否则报 unbound variable
if [ -f /opt/dtk/env.sh ]; then
  # shellcheck disable=SC1091
  source /opt/dtk/env.sh
fi

set -euo pipefail

IN_DIR="${1:-dist}"
OUT_DIR="${2:-dist-manylinux}"

pip install --no-cache-dir auditwheel patchelf

# 依赖库搜索路径, 供 auditwheel 解析 wheel 内 .so 的依赖
TORCH_PATH=$(python3 -c "import torch, os; print(os.path.dirname(torch.__file__))" 2>/dev/null || true)
export LD_LIBRARY_PATH="${TORCH_PATH}/lib:${LD_LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="/opt/intel/mkl/lib/intel64_lin/:${LD_LIBRARY_PATH:-}"
# 定位 lightop 自带库目录; import 推导依赖自带 .so 且纯净环境下 import 可能
# 失败 (见 __extendhfsf2), 这里用 pip Location 免 import
LIGHTOP_PATH=""
LIGHTOP_LOC=$(python3 -m pip show lightop 2>/dev/null | sed -n 's/^Location: //p' || true)
if [ -n "${LIGHTOP_LOC}" ]; then
  LIGHTOP_PATH="${LIGHTOP_LOC}/lightop"
else
  # 兜底: 非 pip 管理的安装
  LIGHTOP_PATH=$(python3 -c "import lightop, os; print(os.path.dirname(lightop.__file__))" 2>/dev/null || true)
fi
echo "LIGHTOP_PATH=${LIGHTOP_PATH}"
export LD_LIBRARY_PATH="${LIGHTOP_PATH}/lib:${LD_LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="${LIGHTOP_PATH}/librarygemmbiasf/:${LD_LIBRARY_PATH:-}"
# auditwheel 用 --ldpaths 的调用路径只认这个专用变量, 一并导出
export AUDITWHEEL_LD_LIBRARY_PATH="${LIGHTOP_PATH}/lib:${LIGHTOP_PATH}/librarygemmbiasf/:${AUDITWHEEL_LD_LIBRARY_PATH:-}"

# glibc 版本 -> manylinux 平台标签, 如 manylinux_2_17_x86_64
GLIBC_VER=$(ldd --version | awk '{print $NF}' | head -n1 | tr -d '\n' | tr '.' '_')
PLAT="manylinux_${GLIBC_VER}_x86_64"

# HIP/ROCm/torch/ffmpeg 等运行时库由 DCU 驱动与 torch 环境提供, 不打进 wheel
excludes=(
  libgalaxyhip.so.5
  libMIOpen.so.1
  librccl.so.1
  libhipblas.so.0
  libhipfft.so
  libhiprand.so.1
  libhipsolver.so.0
  libhipsparse.so.0
  libhipnn.so
  librocblas.so.0
  librocsolver.so.0
  librocfft.so.0
  librocrand.so.1
  librocsparse.so.0
  librocm_smi64.so.2
  librocfft-device-0.so.0
  librocfft-device-1.so.0
  librocfft-device-2.so.0
  librocfft-device-3.so.0
  libc10.so
  libc10_hip.so
  libtorch_cpu.so
  libtorch_hip.so
  libtorch_python.so
  libtorch.so
  libtorchaudio_ffmpeg4.so
  libtorchaudio_ffmpeg5.so
  libtorchaudio_ffmpeg6.so
  libtorchaudio.so
  libtorchaudio_sox.so
  _torchaudio_ffmpeg4.so
  _torchaudio_ffmpeg5.so
  _torchaudio_ffmpeg6.so
  _torchaudio.so
  _torchaudio_sox.so
  libavcodec.so.58
  libavcodec.so.59
  libavcodec.so.60
  libavdevice.so.58
  libavdevice.so.59
  libavdevice.so.60
  libavfilter.so.7
  libavfilter.so.8
  libavfilter.so.9
  libavformat.so.58
  libavformat.so.59
  libavformat.so.60
  libavutil.so.56
  libavutil.so.57
  libavutil.so.58
  libsox.so
  libomp.so
  libhipblaslt.so.0
  libhipblas.so.2
  libhipfft.so.0
  libhipsparse.so.1
  librocblas.so.4
  librocsparse.so.1
)

exclude_args=()
for lib in "${excludes[@]}"; do
  exclude_args+=(--exclude "${lib}")
done

mkdir -p "${OUT_DIR}"
auditwheel repair --plat "${PLAT}" --strip \
  "${exclude_args[@]}" \
  -w "${OUT_DIR}" "${IN_DIR}"/*.whl
