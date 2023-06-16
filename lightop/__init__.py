import os

try:
    from .version import version, git_hash, git_branch, dtk, abi, torch_version, dcu_version  # noqa: F401
    __version__, __dcu_version__ = version, dcu_version
except ImportError:
    pass
