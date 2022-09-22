from setuptools import setup
import os
import glob
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
from setuptools import find_packages, setup

def get_version():
    return "0.1"

def get_extensions():
    extensions = []
    include_dirs = []
    define_macros = []
    extra_compile_args = {'cxx': []}
    op_files = glob.glob('./fuse/csrc/*.cu') + \
        glob.glob('./fuse/csrc/*.cpp')
    extension = CUDAExtension
    include_dirs.append(os.path.abspath('./fuse/csrc/'))
    ext_ops = extension(
            name="fuseop",
            sources=op_files,
            include_dirs=include_dirs,
            define_macros=define_macros,
            extra_compile_args=extra_compile_args)
    extensions.append(ext_ops)
    return extensions

setup(
    name='fuse',
    version=get_version(),
    description='Torch fuseop Computer Vision Foundation',
    keywords='computer vision',
    packages=find_packages(),
    include_package_data=False,
    package_data = {
        'fuse':[
           "csrc/*.cuh",
           "csrc/*.cu",
           "csrc/*.hip",
           "csrc/*.cpp"
        ]
    },
    ext_modules=get_extensions(),
    cmdclass={
        'build_ext': BuildExtension
    },
    zip_safe=False
)