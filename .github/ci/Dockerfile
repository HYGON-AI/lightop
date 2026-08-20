ARG PYTHON_VERSION=3.10
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG PYTHON_VERSION=3.10
ARG INSTALL_TORCH_VERSION
ARG DTK_INDEX_VERSION=2604
ARG FALLBACK_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple/
ARG FALLBACK_INDEX_HOST=pypi.tuna.tsinghua.edu.cn
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
# vLLM 0.25.1 requires llguidance>=1.7, but its cp310 x86_64 wheel requires
# glibc 2.31 while this Rocky 8 image provides glibc 2.28. LightOp's CI tests
# only use vLLM kernels and layer baselines, not structured output, so install
# vLLM without dependencies and then install its metadata requirements except
# llguidance and the lane-pinned Torch, Triton, and torchvision packages. This
# prevents a multi-hour Rust source build and protects the DTK packages from
# dependency replacement.
#
# NumPy is deliberately unpinned. Neither LightOp nor test/ci_tests imports it;
# it reaches the image only through vLLM, which constrains it from both sides:
# `numba==0.65.0` requires >=1.22,<2.5 and `opencv-python-headless>=4.13.0`
# requires >=2 on Python 3.9+. The former 1.24.3 pin satisfied neither and was
# a leftover from the aiter/deepgemm/deep_ep dependency set that this image no
# longer installs, so pip silently overrode it while installing vLLM.
#
# CuPy was a leftover from that same set and is gone for the same reason: no
# source file in this repository or in test/ci_tests imports it. The only
# build available on the internal index is 12.3.0, which requires numpy<1.29
# and therefore cannot coexist with vLLM's numpy>=2 — the intersection is
# empty, so pip resolved in vLLM's favour and left CuPy importable but broken.
RUN mkdir -p /root/.pip && \
    test -n "${INSTALL_TORCH_VERSION}" && \
    printf '%s\n' \
      '[global]' \
      "index-url = http://10.16.1.201:9929/nightly/dtk${DTK_INDEX_VERSION}/+simple/" \
      'trusted-host = 10.16.1.201' \
      > /root/.pip/pip.conf && \
    python3 -m pip install \
      pyyaml hypothesis Ninja MonkeyType wheel packaging \
      einops pytest matplotlib auditwheel patchelf tqdm && \
    python3 -m pip install amdsmi && \
    python3 -m pip install "torch==${INSTALL_TORCH_VERSION}" && \
    case "${INSTALL_TORCH_VERSION}" in \
      2.10.0) \
        VLLM_VERSION="0.21.0"; \
        TRITON_VERSION="3.8.0+dtk2604.torch2100.2608031654.g07864e"; \
        TORCHVISION_VERSION="0.25.0+das185.dtk2604.torch2100.2608130858.gdcd044" \
        ;; \
      2.11.0) \
        VLLM_VERSION="0.25.1"; \
        TRITON_VERSION="3.6.0+das185.dtk2604.torch2110.2608200031.g3efdea"; \
        TORCHVISION_VERSION="0.26.0+das.opt1.dtk2604.torch2110.2608071236.gc5a783" \
        ;; \
      *) echo "No DTK vLLM baseline is mapped for torch ${INSTALL_TORCH_VERSION}." >&2; exit 1 ;; \
    esac && \
    python3 -m pip install --no-deps "triton==${TRITON_VERSION}" && \
    python3 -m pip install --no-deps "torchvision==${TORCHVISION_VERSION}" && \
    python3 -m pip install --no-deps "vllm==${VLLM_VERSION}" \
      --extra-index-url "${FALLBACK_INDEX_URL}" \
      --trusted-host "${FALLBACK_INDEX_HOST}" && \
    python3 /lightop/.github/ci/install-vllm-dependencies.py \
      --extra-index-url "${FALLBACK_INDEX_URL}" \
      --trusted-host "${FALLBACK_INDEX_HOST}" && \
    chmod +x /dtk_llvm_*.run /usr/local/bin/build-lightop-wheel && \
    yes | /dtk_llvm_*.run

# Keep the build image focused on compiling and validating the wheel.
RUN python3 - <<'PY'
import importlib.metadata
import os

expected_torch = os.environ["INSTALL_TORCH_VERSION"]
torch_version = importlib.metadata.version("torch")
base_torch, _, local_torch = torch_version.partition("+")
if base_torch != expected_torch:
    raise SystemExit(
        f"Torch version mismatch: expected {expected_torch}, got {torch_version}"
    )
# The DTK build carries a `+das...dtk...` local label. An upstream wheel shares
# the same base version, so only the label distinguishes them: without this
# check a dependency that replaces Torch stays invisible until hipcc fails.
if "dtk" not in local_torch:
    raise SystemExit(
        f"Torch {torch_version} is not a DTK build. A preceding pip install "
        f"replaced it with an upstream wheel."
    )

# Triton is published as `+dtk<ver>.torch<tag>` per lane. If pip resolved the
# other lane's wheel (unpinned triton picks the newest label, which belongs to
# whichever lane was released last), the wheel is compiled against a different
# torch ABI and will fail at first kernel launch — pip only warns about it.
# Fail the build image now so the mismatch cannot silently ship.
try:
    triton_version = importlib.metadata.version("triton")
except importlib.metadata.PackageNotFoundError:
    raise SystemExit("Triton is required for the build image but is not installed.")
expected_triton_tag = f"torch{expected_torch.replace('.', '')}"
if expected_triton_tag not in triton_version:
    raise SystemExit(
        f"Triton {triton_version} is not built for torch {expected_torch} "
        f"(expected local label to contain '{expected_triton_tag}'). A preceding "
        f"pip install resolved the wrong lane's wheel; pin triton per torch "
        f"lane in the Dockerfile."
    )

print("Resolved LightOp build runtime:")
for package in ("torch", "triton", "numpy", "vllm"):
    try:
        print(f"  {package}={importlib.metadata.version(package)}")
    except importlib.metadata.PackageNotFoundError:
        print(f"  {package}=not installed")

# Runtime imports require libraries from /opt/hyhal, which is mounted only when
# the compile container starts. The compile job performs those checks there.
PY

WORKDIR /lightop
CMD ["tail", "-f", "/dev/null"]
