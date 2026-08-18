"""Compiler selection shared by the LightOp build entry points."""

from dataclasses import dataclass
import os
import shutil
from typing import Optional


@dataclass(frozen=True)
class GPUCompiler:
    name: str
    path: str
    rocm_home: Optional[str]
    requested_mode: str

    @property
    def is_aicc(self) -> bool:
        return self.name == "aicc"

    @property
    def can_fallback(self) -> bool:
        return self.requested_mode == "auto" and self.is_aicc


def _compiler_path(name: str, rocm_home: Optional[str]) -> Optional[str]:
    environment_name = f"LIGHTOP_{name.upper()}_PATH"
    configured_path = os.environ.get(environment_name)
    candidates = [configured_path] if configured_path else []
    if rocm_home:
        candidates.append(os.path.join(rocm_home, "bin", name))
        if os.path.basename(os.path.normpath(rocm_home)) == "hip":
            candidates.append(os.path.join(os.path.dirname(rocm_home), "bin", name))
    candidates.append(shutil.which(name))

    for candidate in candidates:
        if candidate and os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return os.path.realpath(candidate)
    return None


def _infer_rocm_home(compiler_path: str) -> str:
    # Both hipcc and aicc are expected under <ROCM_HOME>/bin/.
    return os.path.dirname(os.path.dirname(compiler_path))


def _normalize_rocm_home(rocm_home: Optional[str]) -> Optional[str]:
    if rocm_home and os.path.basename(os.path.normpath(rocm_home)) == "hip":
        return os.path.dirname(rocm_home)
    return rocm_home


def select_gpu_compiler(torch_rocm_home: Optional[str], mode: Optional[str] = None) -> GPUCompiler:
    requested_mode = (mode or os.environ.get("LIGHTOP_GPU_COMPILER", "auto")).lower()
    if requested_mode not in {"auto", "aicc", "hipcc"}:
        raise RuntimeError(
            "LIGHTOP_GPU_COMPILER must be one of: auto, aicc, hipcc "
            f"(got {requested_mode!r})."
        )

    rocm_home = _normalize_rocm_home(
        os.environ.get("ROCM_HOME") or os.environ.get("ROCM_PATH") or torch_rocm_home
    )
    candidates = ("aicc", "hipcc") if requested_mode == "auto" else (requested_mode,)
    for name in candidates:
        path = _compiler_path(name, rocm_home)
        if path:
            return GPUCompiler(name, path, rocm_home or _infer_rocm_home(path), requested_mode)

    if requested_mode == "auto":
        raise RuntimeError("Neither aicc nor hipcc was found. Set ROCM_HOME or LIGHTOP_AICC_PATH/LIGHTOP_HIPCC_PATH.")
    raise RuntimeError(f"Requested compiler '{requested_mode}' was not found.")
