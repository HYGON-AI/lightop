import hashlib
import os
import sys
import torch

SUPPORTED_DEVICES = ["gfx936_80cu","gfx928_128cu", "gfx928_120cu","gfx938_64cu"]

device_name = torch.cuda.get_device_properties("cuda").gcnArchName.split(':')[0]
number_cu = torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count

LMSLIM_GPU_NAME=device_name+'_'+str(number_cu)+'cu'

if LMSLIM_GPU_NAME not in SUPPORTED_DEVICES:
    LMSLIM_GPU_NAME = "gfx936_80cu"

if "gfx936" in LMSLIM_GPU_NAME or "gfx938" in LMSLIM_GPU_NAME:
    LMSLIM_USE_LIGHTOP=int(os.getenv("LMSLIM_USE_LIGHTOP", 1))
else:
    LMSLIM_USE_LIGHTOP=0


def _first_env_int(env_names: tuple[str, ...], default: int) -> int:
    """从左到右读第一个已设置的环境变量，解析为 int。"""
    for name in env_names:
        raw = os.getenv(name)
        if raw is not None and raw != "":
            return int(raw)
    return default
def _first_env_bool(env_names: tuple[str, ...], default: bool) -> bool:
    """从左到右读第一个已设置的环境变量，解析为 bool（1/true/yes/on 为真）。"""
    for name in env_names:
        raw = os.getenv(name)
        if raw is not None and raw != "":
            return raw.strip().lower() in ("1", "true", "yes", "on")
    return default
# Fused MoE：每批最多处理多少 token（与旧版 vLLM 默认 16*1024 对齐）
LMSLIM_FUSED_MOE_CHUNK_SIZE = _first_env_int(
    ("LMSLIM_FUSED_MOE_CHUNK_SIZE", "VLLM_FUSED_MOE_CHUNK_SIZE"),
    16 * 1024,
)
# 是否复用全局 MoE 中间 buffer（对应原先 VLLM_USE_GLOBAL_CACHE13）
LMSLIM_USE_GLOBAL_MOE_CACHE = _first_env_bool(
    ("LMSLIM_USE_GLOBAL_MOE_CACHE", "VLLM_USE_GLOBAL_CACHE13"),
    False,
)
# 是否在传入 i_q / i_s 时走「已融合 RMS+量化」路径（对应原先 USE_FUSED_RMS_QUANT）
LMSLIM_USE_FUSED_RMS_QUANT = _first_env_bool(
    ("LMSLIM_USE_FUSED_RMS_QUANT", "USE_FUSED_RMS_QUANT"),
    False,
)
LMSLIM_USE_FUSE_SILU_AND_MUL = _first_env_bool(
    ("LMSLIM_USE_FUSE_SILU_AND_MUL", "VLLM_USE_FUSE_SILU_AND_MUL"),
    False,
)
