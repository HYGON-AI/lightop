import torch
from typing import List, Optional, Union
import glob
import os
import shlex
import subprocess
import sys
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
from setuptools import find_packages, setup
from setuptools.command.build_ext import build_ext
from pkg_resources import packaging  # type: ignore[attr-defined]


def _find_rocm_home() -> Optional[str]:
    rocm_home = os.environ.get('ROCM_HOME') or os.environ.get('ROCM_PATH')
    if rocm_home is None:
        try:
            pipe_hipcc = subprocess.Popen(
                ["which hipcc | xargs readlink -f"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
            hipcc, _ = pipe_hipcc.communicate()
            rocm_home = os.path.dirname(os.path.dirname(hipcc.decode(*()).rstrip('\r\n')))
            if os.path.basename(rocm_home) == 'hip':
                rocm_home = os.path.dirname(rocm_home)
        except Exception:
            rocm_home = '/opt/rocm'
            if not os.path.exists(rocm_home):
                rocm_home = None
    if rocm_home and torch.version.hip is None:
        print(f"No ROCm runtime is found, using ROCM_HOME='{rocm_home}'")
    return rocm_home


def _get_rocm_arch_flags(cflags: Optional[List[str]] = None) -> List[str]:
    if cflags is not None:
        for flag in cflags:
            if 'amdgpu-target' in flag:
                return ['-fno-gpu-rdc']
    archs = os.environ.get('PYTORCH_ROCM_ARCH', 'gfx900;gfx906')
    flags = ['--amdgpu-target=%s' % arch for arch in archs.split(';')]
    flags += ['-fno-gpu-rdc']
    return flags


ROCM_HOME = _find_rocm_home()
IS_HIP_EXTENSION = True if ((ROCM_HOME is not None) and (torch.version.hip is not None)) else False
COMMON_HIP_FLAGS = [
    '-fPIC',
    '-D__HIP_PLATFORM_HCC__=1',
]

COMMON_HIPCC_FLAGS = [
    '-DCUDA_HAS_FP16=1',
    '-D__HIP_NO_HALF_OPERATORS__=1',
    '-D__HIP_NO_HALF_CONVERSIONS__=1',
]


def is_ninja_available():
    try:
        subprocess.check_output('ninja --version'.split())
    except Exception:
        return False
    else:
        return True


def verify_ninja_availability():
    if not is_ninja_available():
        raise RuntimeError("Ninja is required to load C++ extensions")


def _is_cuda_file(path: str) -> bool:
    valid_ext = ['.cu', '.cuh']
    if IS_HIP_EXTENSION:
        valid_ext.append('.hip')
    return os.path.splitext(path)[1] in valid_ext


def _join_rocm_home(*paths) -> str:
    if ROCM_HOME is None:
        raise EnvironmentError('ROCM_HOME environment variable is not set. ')
    return os.path.join(ROCM_HOME, *paths)


def _write_ninja_file(path, cflags, post_cflags, cuda_cflags, cuda_post_cflags, sources,
                      objects, ldflags, library_target, with_cuda) -> None:
    def sanitize_flags(flags):
        if flags is None:
            return []
        else:
            return [flag.strip() for flag in flags]

    cflags = sanitize_flags(cflags)
    post_cflags = sanitize_flags(post_cflags)
    cuda_cflags = sanitize_flags(cuda_cflags)
    cuda_post_cflags = sanitize_flags(cuda_post_cflags)
    ldflags = sanitize_flags(ldflags)
    assert len(sources) == len(objects)
    assert len(sources) > 0
    compiler = os.environ.get('CXX', 'c++')
    config = ['ninja_required_version = 1.3']
    config.append(f'cxx = {compiler}')
    if with_cuda:
        if IS_HIP_EXTENSION:
            nvcc = _join_rocm_home('bin', 'hipcc')
        config.append(f'nvcc = {nvcc}')
    flags = [f'cflags = {" ".join(cflags)}']
    flags.append(f'post_cflags = {" ".join(post_cflags)}')
    if with_cuda:
        flags.append(f'cuda_cflags = {" ".join(cuda_cflags)}')
        flags.append(f'cuda_post_cflags = {" ".join(cuda_post_cflags)}')
    flags.append(f'ldflags = {" ".join(ldflags)}')
    sources = [os.path.abspath(file) for file in sources]
    compile_rule = ['rule compile']
    compile_rule.append('  command = $cxx -MMD -MF $out.d $cflags -c $in -o $out $post_cflags')
    compile_rule.append('  depfile = $out.d')
    compile_rule.append('  deps = gcc')
    if with_cuda:
        cuda_compile_rule = ['rule cuda_compile']
        nvcc_gendeps = ''
        required_cuda_version = packaging.version.parse('10.2')
        has_cuda_version = torch.version.cuda is not None
        if has_cuda_version and packaging.version.parse(torch.version.cuda) >= required_cuda_version:
            cuda_compile_rule.append('  depfile = $out.d')
            cuda_compile_rule.append('  deps = gcc')
        cuda_compile_rule.append(
            f'  command = $nvcc {nvcc_gendeps} $cuda_cflags -c $in -o $out $cuda_post_cflags')
    build = []
    for source_file, object_file in zip(sources, objects):
        is_cuda_source = _is_cuda_file(source_file) and with_cuda
        rule = 'cuda_compile' if is_cuda_source else 'compile'
        source_file = source_file.replace(" ", "$ ")
        object_file = object_file.replace(" ", "$ ")
        build.append(f'build {object_file}: {rule} {source_file}')
    if library_target is not None:
        link_rule = ['rule link']
        link_rule.append('  command = $cxx $in $ldflags -o $out')
        link = [f'build {library_target}: link {" ".join(objects)}']
        default = [f'default {library_target}']
    else:
        link_rule, link, default = [], [], []
    blocks = [config, flags, compile_rule]
    if with_cuda:
        blocks.append(cuda_compile_rule)
    blocks += [link_rule, build, link, default]
    with open(path, 'w') as build_file:
        for block in blocks:
            lines = '\n'.join(block)
            build_file.write(f'{lines}\n\n')


def _get_num_workers(verbose: bool) -> Optional[int]:
    max_jobs = os.environ.get('MAX_JOBS')
    if max_jobs is not None and max_jobs.isdigit():
        if verbose:
            print(f'Using envvar MAX_JOBS ({max_jobs}) as the number of workers...')
        return int(max_jobs)
    if verbose:
        print('Allowing ninja to set a default number of workers... ')
    return None


def _run_ninja_build(build_directory: str, verbose: bool, error_prefix: str) -> None:
    command = ['ninja', '-v']
    num_workers = _get_num_workers(verbose)
    if num_workers is not None:
        command.extend(['-j', str(num_workers)])
    env = os.environ.copy()
    try:
        sys.stdout.flush()
        sys.stderr.flush()
        stdout_fileno = 1
        subprocess.run(command, stdout=stdout_fileno if verbose else subprocess.PIPE, stderr=subprocess.STDOUT,
                       cwd=build_directory, check=True, env=env)
    except subprocess.CalledProcessError as e:
        _, error, _ = sys.exc_info()
        message = error_prefix
        if hasattr(error, 'output') and error.output:  # type: ignore[union-attr]
            message += f": {error.output.decode(*SUBPROCESS_DECODE_ARGS)}"  # type: ignore[union-attr]
        raise RuntimeError(message) from e


def _write_ninja_file_and_compile_objects(sources: List[str], objects, cflags, post_cflags, cuda_cflags,
                                          cuda_post_cflags, build_directory: str, verbose: bool,
                                          with_cuda: Optional[bool]) -> None:
    verify_ninja_availability()
    compiler = os.environ.get('CXX', 'c++')
    if with_cuda is None:
        with_cuda = any(map(_is_cuda_file, sources))
    build_file_path = os.path.join(build_directory, 'build.ninja')
    if verbose:
        print(f'Emitting ninja build file {build_file_path}...')
    _write_ninja_file(path=build_file_path, cflags=cflags, post_cflags=post_cflags, cuda_cflags=cuda_cflags,
                      cuda_post_cflags=cuda_post_cflags, sources=sources, objects=objects, ldflags=None,
                      library_target=None, with_cuda=with_cuda)
    if verbose:
        print('Compiling objects...')
    _run_ninja_build(
        build_directory,
        verbose,
        error_prefix='Error compiling objects for extension')


class BuildReleaseExtension(BuildExtension):
    def __init__(self, *args, **kwargs) -> None:
        super(BuildReleaseExtension, self).__init__(*args, **kwargs)

    def build_extensions(self) -> None:
        self._check_abi()
        cuda_ext = False
        extension_iter = iter(self.extensions)
        extension = next(extension_iter, None)
        while not cuda_ext and extension:
            for source in extension.sources:
                _, ext = os.path.splitext(source)
                if ext == '.cu':
                    cuda_ext = True
                    break
            extension = next(extension_iter, None)
        for extension in self.extensions:
            if isinstance(extension.extra_compile_args, dict):
                for ext in ['cxx', 'nvcc']:
                    if ext not in extension.extra_compile_args:
                        extension.extra_compile_args[ext] = []
            self._add_compile_flag(extension, '-DTORCH_API_INCLUDE_EXTENSION_H')
            for name in ["COMPILER_TYPE", "STDLIB", "BUILD_ABI"]:
                val = getattr(torch._C, f"_PYBIND11_{name}")
                self._add_compile_flag(extension, f'-DPYBIND11_{name}="{val}"')
            self._define_torch_extension_name(extension)
            self._add_gnu_cpp_abi_flag(extension)
        self.compiler.src_extensions += ['.cu', '.cuh', '.hip']

        def append_std14_if_no_std_present(cflags) -> None:
            cpp_format_prefix = '/{}:' if self.compiler.compiler_type == 'msvc' else '-{}='
            cpp_flag_prefix = cpp_format_prefix.format('std')
            cpp_flag = cpp_flag_prefix + 'c++14'
            if not any(flag.startswith(cpp_flag_prefix) for flag in cflags):
                cflags.append(cpp_flag)

        def convert_to_absolute_paths_inplace(paths):
            if paths is not None:
                for i in range(len(paths)):
                    if not os.path.isabs(paths[i]):
                        paths[i] = os.path.abspath(paths[i])

        def unix_wrap_ninja_compile(sources, output_dir=None, macros=None, include_dirs=None, debug=0,
                                    extra_preargs=None, extra_postargs=None, depends=None):
            output_dir = os.path.abspath(output_dir)
            convert_to_absolute_paths_inplace(self.compiler.include_dirs)
            _, objects, extra_postargs, pp_opts, _ = \
                self.compiler._setup_compile(output_dir, macros, include_dirs, sources, depends, extra_postargs)
            common_cflags = self.compiler._get_cc_args(pp_opts, debug, extra_preargs)
            extra_cc_cflags = self.compiler.compiler_so[1:]
            if (debug):
                print("debug mode")
            else:
                extra_cc_cflags.remove('-g')
                extra_cc_cflags.remove('-Wall')
                print("release mode")
            with_cuda = any(map(_is_cuda_file, sources))
            if isinstance(extra_postargs, dict):
                post_cflags = extra_postargs['cxx']
            else:
                post_cflags = list(extra_postargs)
            if IS_HIP_EXTENSION:
                post_cflags = COMMON_HIP_FLAGS + post_cflags
            append_std14_if_no_std_present(post_cflags)
            cuda_post_cflags = None
            cuda_cflags = None
            if with_cuda:
                cuda_cflags = common_cflags
                if isinstance(extra_postargs, dict):
                    cuda_post_cflags = extra_postargs['nvcc']
                else:
                    cuda_post_cflags = list(extra_postargs)
                if IS_HIP_EXTENSION:
                    cuda_post_cflags = cuda_post_cflags + _get_rocm_arch_flags(cuda_post_cflags)
                    cuda_post_cflags = COMMON_HIP_FLAGS + COMMON_HIPCC_FLAGS + cuda_post_cflags
                append_std14_if_no_std_present(cuda_post_cflags)
                cuda_cflags = [shlex.quote(f) for f in cuda_cflags]
                cuda_post_cflags = [shlex.quote(f) for f in cuda_post_cflags]
            _write_ninja_file_and_compile_objects(sources=sources, objects=objects,
                                                  cflags=[shlex.quote(f) for f in extra_cc_cflags + common_cflags],
                                                  post_cflags=[shlex.quote(f) for f in post_cflags],
                                                  cuda_cflags=cuda_cflags,
                                                  cuda_post_cflags=cuda_post_cflags, build_directory=output_dir,
                                                  verbose=True, with_cuda=with_cuda)
            return objects

        self.compiler.compile = unix_wrap_ninja_compile
        build_ext.build_extensions(self)


def get_version():
    return "0.1"


def get_extensions():
    extensions = []
    include_dirs = []
    define_macros = []
    extra_compile_args = {'cxx': ['-O3']}
    op_files = glob.glob('./lightop/csrc/*.cu') + glob.glob('./lightop/csrc/*.cpp')
    extension = CUDAExtension
    include_dirs.append(os.path.abspath('./lightop/csrc/'))
    ext_ops = extension(
        name="Lightop",
        sources=op_files,
        include_dirs=include_dirs,
        define_macros=define_macros,
        extra_compile_args=extra_compile_args)
    extensions.append(ext_ops)
    return extensions


setup(
    name='Lightop',
    version=get_version(),
    description='Torch Fast Operator Foundation',
    keywords='CV, NLP',
    packages=find_packages(),
    include_package_data=False,
    package_data={
        'lightop': [
            "csrc/*.cuh",
            "csrc/*.cu",
            "csrc/*.hip",
            "csrc/*.cpp"
        ]
    },
    ext_modules=get_extensions(),
    cmdclass={
        'build_ext': BuildReleaseExtension
    },
    zip_safe=False
)

