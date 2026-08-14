ARG PYTHON_VERSION=3.10
FROM harbor.sourcefind.cn:5443/dcu/admin/base/dev:rockylinux8.6-mpi5.0-gcc10.3-cmake3.29-py${PYTHON_VERSION}-mkl2020.4.304

ARG PYTHON_VERSION=3.10
ARG INSTALL_TORCH_VERSION=2.10.0
ARG DTK_INDEX_VERSION=2604
ARG DTK_PKG_NAME
ARG DTK_PKG_VERSION
ARG DTK_VERSION
ARG AICC_VERSION

ENV ROCM_PATH=/opt/dtk \
    DTKROOT=/opt/dtk \
    COMPILE_ENV=lightop \
    TOOLKIT_DIR=/lightop \
    RESOURCE_SERVER=http://42.228.13.241:18000 \
    INSTALL_TORCH_VERSION=${INSTALL_TORCH_VERSION} \
    DTK_PKG_NAME=${DTK_PKG_NAME} \
    DTK_PKG_VERSION=${DTK_PKG_VERSION} \
    DTK_VERSION=${DTK_VERSION} \
    AICC_VERSION=${AICC_VERSION}

# The build-context script inserts the ENV declarations generated from the
# selected DTK package here, matching CIctl docker/build.sh.
# config dtk env

COPY dtk /opt/dtk
COPY dtk_llvm_*.run /
COPY lightop /lightop
COPY build-lightop-wheel.sh /usr/local/bin/build-lightop-wheel

RUN mkdir -p /root/.pip && \
    printf '%s\n' \
      '[global]' \
      "index-url = http://10.16.1.201:9929/nightly/dtk${DTK_INDEX_VERSION}/+simple/" \
      'trusted-host = 10.16.1.201' \
      > /root/.pip/pip.conf && \
    python3 -m pip install \
      pyyaml hypothesis Ninja MonkeyType "numpy==1.24.3" wheel packaging \
      einops pytest matplotlib auditwheel patchelf && \
    python3 -m pip install amdsmi && \
    python3 -m pip install cupy && \
    python3 -m pip install "torch==${INSTALL_TORCH_VERSION}" && \
    python3 -m pip uninstall -y cupy-cuda12x && \
    chmod +x /dtk_llvm_*.run /usr/local/bin/build-lightop-wheel && \
    yes | /dtk_llvm_*.run

# Keep the build image focused on compiling LightOp. vLLM and OpTest are added
# only in the separate test container, matching CIctl's phase boundary.
RUN python3 - <<'PY'
import importlib.metadata
import os
import torch

expected_torch = os.environ["INSTALL_TORCH_VERSION"]
actual_torch = torch.__version__.split("+", 1)[0]
if actual_torch != expected_torch:
    raise SystemExit(
        f"Torch version mismatch: expected {expected_torch}, got {torch.__version__}"
    )

print("Resolved LightOp build runtime:")
for package in ("torch", "numpy", "cupy"):
    try:
        print(f"  {package}={importlib.metadata.version(package)}")
    except importlib.metadata.PackageNotFoundError:
        print(f"  {package}=not installed")
PY

WORKDIR /lightop
CMD ["tail", "-f", "/dev/null"]
