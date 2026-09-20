"""Runtime environment settings shared by LightOp integrations."""

import os

import torch


SUPPORTED_DEVICES = [
    "gfx936_80cu", "gfx928_128cu", "gfx928_120cu", "gfx938_64cu",
    "gfx938_72cu"
]

device_name = torch.cuda.get_device_properties("cuda").gcnArchName.split(":")[0]
number_cu = torch.cuda.get_device_properties(
    torch.cuda.current_device()
).multi_processor_count

LMSLIM_GPU_NAME = f"{device_name}_{number_cu}cu"

if LMSLIM_GPU_NAME not in SUPPORTED_DEVICES:
    LMSLIM_GPU_NAME = "gfx936_80cu"

if "gfx936" in LMSLIM_GPU_NAME or "gfx938" in LMSLIM_GPU_NAME:
    LMSLIM_USE_LIGHTOP = int(os.getenv("LMSLIM_USE_LIGHTOP", 1))
else:
    LMSLIM_USE_LIGHTOP = 0


def _first_env_int(env_names: tuple[str, ...], default: int) -> int:
    """Read the first configured environment variable as an integer."""
    for name in env_names:
        raw = os.getenv(name)
        if raw is not None and raw != "":
            return int(raw)
    return default


def _first_env_bool(env_names: tuple[str, ...], default: bool) -> bool:
    """Read the first configured environment variable as a boolean."""
    for name in env_names:
        raw = os.getenv(name)
        if raw is not None and raw != "":
            return raw.strip().lower() in ("1", "true", "yes", "on")
    return default


LMSLIM_FUSED_MOE_CHUNK_SIZE = _first_env_int(
    ("LMSLIM_FUSED_MOE_CHUNK_SIZE", "VLLM_FUSED_MOE_CHUNK_SIZE"),
    16 * 1024,
)
LMSLIM_USE_GLOBAL_MOE_CACHE = _first_env_bool(
    ("LMSLIM_USE_GLOBAL_MOE_CACHE", "VLLM_USE_GLOBAL_CACHE13"),
    False,
)
LMSLIM_USE_FUSED_RMS_QUANT = _first_env_bool(
    ("LMSLIM_USE_FUSED_RMS_QUANT", "USE_FUSED_RMS_QUANT"),
    False,
)
LMSLIM_USE_FUSE_SILU_AND_MUL = _first_env_bool(
    ("LMSLIM_USE_FUSE_SILU_AND_MUL", "VLLM_USE_FUSE_SILU_AND_MUL"),
    False,
)


__all__ = [
    "SUPPORTED_DEVICES",
    "device_name",
    "number_cu",
    "LMSLIM_GPU_NAME",
    "LMSLIM_USE_LIGHTOP",
    "LMSLIM_FUSED_MOE_CHUNK_SIZE",
    "LMSLIM_USE_GLOBAL_MOE_CACHE",
    "LMSLIM_USE_FUSED_RMS_QUANT",
    "LMSLIM_USE_FUSE_SILU_AND_MUL",
]
