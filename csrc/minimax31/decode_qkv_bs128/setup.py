import os
from setuptools import setup
os.environ.setdefault("PYTORCH_NVCC", "aicc")
os.environ.setdefault("PYTORCH_ROCM_ARCH", "gfx938")
from torch.utils.cpp_extension import CUDAExtension, BuildExtension
setup(name="m31_qkv_followup", ext_modules=[CUDAExtension(
    name="m31_qkv_followup", sources=["qk.cu"],
    extra_compile_args={"cxx":["-O3"],"nvcc":["-O3","--offload-arch=gfx938","-std=c++17"]},
)], cmdclass={"build_ext":BuildExtension})
