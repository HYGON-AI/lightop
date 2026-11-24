import os
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="prefill_shared_oai_native",
    ext_modules=[
        CUDAExtension(
            "prefill_shared_oai_native",
            ["bindings.cpp"],
            extra_objects=[os.environ["PREFILL_SHARED_OAI_LIGHTOP_OBJECT"]],
            extra_compile_args={"cxx": ["-O2", "-g0", "-std=c++20"]},
            extra_link_args=["-Wl,-Bsymbolic"],
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)
