#!/usr/bin/env bash
set -euo pipefail

source /opt/dtk/env.sh
cd /lightop

rm -rf /dist build dist
mkdir -p /dist

torch_version="$(python3 -c 'import torch; print(torch.__version__.split("+", 1)[0])')"
torch_tag="${torch_version//./}"
build_time="$(date +%y%m%d%H%M)"
git_commit="${LIGHTOP_SOURCE_COMMIT:?LIGHTOP_SOURCE_COMMIT is required}"
git_commit="${git_commit:0:6}"
nightly_suffix=".torch${torch_tag}.${build_time}.g${git_commit}"

# Add the standard Torch-aware nightly build suffix without editing source state.
python3 - "${nightly_suffix}" <<'PY'
from pathlib import Path
import sys

path = Path("setup.py")
text = path.read_text(encoding="utf-8")
needle = "version=get_version(ROCM_HOME),"
replacement = f'version=get_version(ROCM_HOME)+"{sys.argv[1]}",'
if text.count(needle) != 1:
    raise SystemExit(f"Expected one setup.py version declaration, found {text.count(needle)}")
path.write_text(text.replace(needle, replacement), encoding="utf-8")
PY

export PYTORCH_VERSION="${torch_version}"
python3 setup.py bdist_wheel
mkdir -p /tmp/lightop-wheelhouse
mv dist/*.whl /tmp/lightop-wheelhouse/
# The raw wheel is installed only in the disposable build container so
# auditwheel can resolve the package's bundled library directories. Tests run
# later in a pristine container and install only the repaired wheel.
python3 -m pip install /tmp/lightop-wheelhouse/*.whl

platform="manylinux_$(ldd --version | awk '{print $NF}' | sed -n '1p' | tr '.' '_')_x86_64"
exclude_libraries=(
    libgalaxyhip.so.5 libMIOpen.so.1 librccl.so.1 libhipblas.so.0 libhipfft.so
    libhiprand.so.1 libhipsolver.so.0 libhipsparse.so.0 libhipnn.so
    librocblas.so.0 librocsolver.so.0 librocfft.so.0 librocrand.so.1
    librocsparse.so.0 librocm_smi64.so.2 librocfft-device-0.so.0
    librocfft-device-1.so.0 librocfft-device-2.so.0 librocfft-device-3.so.0
    libc10.so libc10_hip.so libtorch_cpu.so libtorch_hip.so libtorch_python.so
    libtorch.so libtorchaudio_ffmpeg4.so libtorchaudio_ffmpeg5.so
    libtorchaudio_ffmpeg6.so libtorchaudio.so libtorchaudio_sox.so
    _torchaudio_ffmpeg4.so _torchaudio_ffmpeg5.so _torchaudio_ffmpeg6.so
    _torchaudio.so _torchaudio_sox.so libavcodec.so.58 libavcodec.so.59
    libavcodec.so.60 libavdevice.so.58 libavdevice.so.59 libavdevice.so.60
    libavfilter.so.7 libavfilter.so.8 libavfilter.so.9 libavformat.so.58
    libavformat.so.59 libavformat.so.60 libavutil.so.56 libavutil.so.57
    libavutil.so.58 libsox.so libomp.so libhipblaslt.so.0 libhipblas.so.2
    libhipfft.so.0 libhipsparse.so.1 librocblas.so.4 librocsparse.so.1
)
auditwheel_args=(repair --plat "${platform}" --strip)
for library in "${exclude_libraries[@]}"; do
    auditwheel_args+=(--exclude "${library}")
done

TORCH_PATH="$(python3 -c 'import torch, os; print(os.path.dirname(torch.__file__))')"
LIGHTOP_PATH="$(python3 -c 'import lightop, os; print(os.path.dirname(lightop.__file__))')"
export LD_LIBRARY_PATH="${TORCH_PATH}/lib:/opt/intel/mkl/lib/intel64_lin:${LIGHTOP_PATH}/lib:${LIGHTOP_PATH}/librarygemmbiasf${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
auditwheel "${auditwheel_args[@]}" -w /dist/ /tmp/lightop-wheelhouse/*.whl

mapfile -t repaired_wheels < <(find /dist -maxdepth 1 -type f -name '*.whl' -print)
if [[ "${#repaired_wheels[@]}" -ne 1 ]]; then
    echo "Expected exactly one repaired wheel, found ${#repaired_wheels[@]}" >&2
    exit 1
fi
