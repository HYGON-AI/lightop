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
      pyyaml hypothesis Ninja MonkeyType numpy==1.24.3 wheel packaging \
      einops pytest matplotlib auditwheel patchelf && \
    python3 -m pip install amdsmi && \
    python3 -m pip install cupy && \
    python3 -m pip install vllm==0.15.1 && \
    python3 -m pip install numpy==1.25.0 && \
    python3 -m pip install \
      torch==2.9.0 aiter deepgemm deep_ep triton==3.5.1 && \
    python3 -m pip uninstall -y cupy-cuda12x && \
    python3 -m pip install "torch==${INSTALL_TORCH_VERSION}" && \
    python3 -m pip install numpy==1.24.3 && \
    chmod +x /dtk_llvm_*.run /usr/local/bin/build-lightop-wheel && \
    yes | /dtk_llvm_*.run

WORKDIR /lightop
CMD ["tail", "-f", "/dev/null"]
