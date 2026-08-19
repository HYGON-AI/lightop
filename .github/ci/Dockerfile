ARG PYTHON_VERSION=3.10
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG PYTHON_VERSION=3.10
ARG INSTALL_TORCH_VERSION
ARG DTK_INDEX_VERSION=2604
ARG DTK_PKG_NAME
ARG DTK_PKG_VERSION
ARG DTK_VERSION
ARG AICC_VERSION
ARG LIGHTOP_SOURCE_BRANCH
ARG LIGHTOP_SOURCE_COMMIT

ENV ROCM_PATH=/opt/dtk \
    DTKROOT=/opt/dtk \
    COMPILE_ENV=lightop \
    TOOLKIT_DIR=/lightop \
    RESOURCE_SERVER=http://42.228.13.241:18000 \
    INSTALL_TORCH_VERSION=${INSTALL_TORCH_VERSION} \
    DTK_PKG_NAME=${DTK_PKG_NAME} \
    DTK_PKG_VERSION=${DTK_PKG_VERSION} \
    DTK_VERSION=${DTK_VERSION} \
    AICC_VERSION=${AICC_VERSION} \
    LIGHTOP_SOURCE_BRANCH=${LIGHTOP_SOURCE_BRANCH} \
    LIGHTOP_SOURCE_COMMIT=${LIGHTOP_SOURCE_COMMIT}

# The build-context script inserts the ENV declarations generated from the
# selected DTK package here.
# config dtk env

COPY dtk /opt/dtk
COPY dtk_llvm_*.run /
COPY lightop /lightop
COPY build-lightop-wheel.sh /usr/local/bin/build-lightop-wheel

# Register the base image's shared CPython directory with the runtime linker.
# Generated toolkit variables can replace LD_LIBRARY_PATH later in the build.
RUN python_lib="$(find /usr/local -maxdepth 3 -name "libpython${PYTHON_VERSION}.so.1.0" -print -quit 2>/dev/null)" && \
    test -n "${python_lib}" && \
    dirname "${python_lib}" > /etc/ld.so.conf.d/python.conf && \
    ldconfig && \
    python3 --version

# vLLM is only a numerical baseline for test/ci_tests. Each DTK build is bound
# to one Torch release through its local version label (`+das.dtk2604.torch2100`),
# so the version must be selected per matrix lane: an unpinned install resolves
# to the newest label, which belongs to the other lane and drags in an upstream
# Torch that replaces the DTK build.
#
# The PyPI fallback stays for llguidance: vllm 0.25.1 requires >=1.7.0, whose
# only cp310 x86_64 wheels are manylinux_2_31 and so are unusable on this
# glibc 2.28 base image. pip therefore builds it from sdist through maturin,
# which fetches puccinialin. vllm 0.21.0 needs llguidance 1.3.0, which does
# ship a manylinux_2_17 wheel and resolves entirely from the internal index.
RUN mkdir -p /root/.pip && \
    test -n "${INSTALL_TORCH_VERSION}" && \
    printf '%s\n' \
      '[global]' \
      "index-url = http://10.16.1.201:9929/nightly/dtk${DTK_INDEX_VERSION}/+simple/" \
      'trusted-host = 10.16.1.201' \
      > /root/.pip/pip.conf && \
    python3 -m pip install \
      pyyaml hypothesis Ninja MonkeyType "numpy==1.24.3" wheel packaging \
      einops pytest matplotlib auditwheel patchelf tqdm triton && \
    python3 -m pip install amdsmi && \
    python3 -m pip install cupy && \
    python3 -m pip install "torch==${INSTALL_TORCH_VERSION}" && \
    python3 -m pip uninstall -y cupy-cuda12x && \
    case "${INSTALL_TORCH_VERSION}" in \
      2.10.0) VLLM_VERSION="0.21.0" ;; \
      2.11.0) VLLM_VERSION="0.25.1" ;; \
      *) echo "No DTK vLLM baseline is mapped for torch ${INSTALL_TORCH_VERSION}." >&2; exit 1 ;; \
    esac && \
    python3 -m pip install "vllm==${VLLM_VERSION}" \
      --extra-index-url https://pypi.org/simple/ \
      --trusted-host pypi.org \
      --trusted-host files.pythonhosted.org && \
    chmod +x /dtk_llvm_*.run /usr/local/bin/build-lightop-wheel && \
    yes | /dtk_llvm_*.run

# Keep the build image focused on compiling and validating the wheel.
RUN python3 - <<'PY'
import importlib.metadata
import os
import torch

expected_torch = os.environ["INSTALL_TORCH_VERSION"]
base_torch, _, local_torch = torch.__version__.partition("+")
if base_torch != expected_torch:
    raise SystemExit(
        f"Torch version mismatch: expected {expected_torch}, got {torch.__version__}"
    )
# The DTK build carries a `+das...dtk...` local label. An upstream wheel shares
# the same base version, so only the label distinguishes them: without this
# check a dependency that replaces Torch stays invisible until hipcc fails.
if "dtk" not in local_torch:
    raise SystemExit(
        f"Torch {torch.__version__} is not a DTK build. A preceding pip install "
        f"replaced it with an upstream wheel."
    )

print("Resolved LightOp build runtime:")
for package in ("torch", "numpy", "cupy", "vllm"):
    try:
        print(f"  {package}={importlib.metadata.version(package)}")
    except importlib.metadata.PackageNotFoundError:
        print(f"  {package}=not installed")
PY

WORKDIR /lightop
CMD ["tail", "-f", "/dev/null"]
