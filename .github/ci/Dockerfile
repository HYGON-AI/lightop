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

# The six functional tests use only Torch, Triton, tqdm, LightOp, and vLLM's
# `_custom_ops` baseline. Installing vLLM's complete server dependency graph
# pulls unrelated model-serving packages into the compile image and makes the
# build depend on continuously changing upstream constraints. Install only the
# small import chain required by `_custom_ops`, then install the lane-specific
# DTK packages without dependency resolution.
#
# NumPy 1.26.4 is new enough for the vLLM import chain while remaining on the
# NumPy 1.x ABI required by the DTK Torch builds.
RUN mkdir -p /root/.pip && \
    test -n "${INSTALL_TORCH_VERSION}" && \
    printf '%s\n' \
      '[global]' \
      "index-url = http://10.16.1.201:9929/nightly/dtk${DTK_INDEX_VERSION}/+simple/" \
      'trusted-host = 10.16.1.201' \
      > /root/.pip/pip.conf && \
    python3 -m pip install \
      pyyaml hypothesis Ninja MonkeyType wheel packaging "numpy==1.26.4" \
      einops pytest matplotlib auditwheel patchelf tqdm \
      "regex==2026.7.10" "requests==2.32.5" "typing_extensions==4.16.0" && \
    python3 -m pip install amdsmi && \
    python3 -m pip install "torch==${INSTALL_TORCH_VERSION}" && \
    case "${INSTALL_TORCH_VERSION}" in \
      2.10.0) \
        VLLM_VERSION="0.21.0"; \
        TRITON_VERSION="3.8.0+dtk2604.torch2100.2608031654.g07864e" \
        ;; \
      2.11.0) \
        VLLM_VERSION="0.25.1"; \
        TRITON_VERSION="3.6.0+das185.dtk2604.torch2110.2608200031.g3efdea" \
        ;; \
      *) echo "No DTK vLLM baseline is mapped for torch ${INSTALL_TORCH_VERSION}." >&2; exit 1 ;; \
    esac && \
    python3 -m pip install --no-deps "triton==${TRITON_VERSION}" && \
    python3 -m pip install --no-deps "vllm==${VLLM_VERSION}" && \
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

expected_vllm = {"2.10.0": "0.21.0", "2.11.0": "0.25.1"}[expected_torch]
vllm_version = importlib.metadata.version("vllm")
vllm_base, _, vllm_local = vllm_version.partition("+")
if vllm_base != expected_vllm or expected_triton_tag not in vllm_local:
    raise SystemExit(
        f"vLLM {vllm_version} does not match torch {expected_torch}; expected "
        f"vLLM {expected_vllm} with local label containing {expected_triton_tag}."
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
