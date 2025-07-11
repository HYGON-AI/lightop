import functools
from typing import Any, Optional
import os
import json

# Adapted from: https://github.com/sgl-project/sglang/pull/2628
def get_config_file_name(E: int,
                         N: int,
                         dtype: Optional[str],
                         block_shape: Optional[list[int]] = None,) -> str:
    # device_name = current_platform.get_device_name().replace(" ", "_")
    device_name = ""
    dtype_selector = "" if not dtype else f",dtype={dtype}"
    block_shape_selector = ("" if not block_shape or not all(block_shape) else
                            f",block_shape={block_shape}").replace(" ", "")
    return f"E={E},N={N},device_name={device_name}{dtype_selector}{block_shape_selector}.json"  # noqa: E501


# Adapted from: https://github.com/sgl-project/sglang/pull/2628
@functools.lru_cache
def get_moe_cuda_config(
    E: int,
    M: int,
    N: int,
    K: int,
    device_name: str,
    num_cus: str,
    dtype: Optional[str],
    block_n: Optional[int] = None,
    block_k: Optional[int] = None,
) -> Optional[dict[int, Any]]:
    """
    Return optimized configurations for the fused MoE kernel.

    The return value will be a dictionary that maps an irregular grid of
    batch sizes to configurations of the fused_moe kernel. To evaluate the
    kernel on a given batch size bs, the closest batch size in the grid should
    be picked and the associated configuration chosen to invoke the kernel.
    """

    # First look up if an optimized configuration is available in the configs
    # directory
    file_name = f"MOE_BLOCKINT8_CUDA_{N}_{K}_{device_name}_{num_cus}.json"
    config_file_path = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "configs", file_name)
    try:
        with open(config_file_path, 'r') as file:
            config_infos = json.load(file)
            if f"{N}_{K}" in config_infos:
                config_info = config_infos[f"{N}_{K}"]
                int_keys = [int(key) for key in config_info.keys()]
                greater_numbers = [num for num in int_keys if num >= M]
                num_token = min(greater_numbers) if greater_numbers else None
                config = config_info[f"{num_token}"] if f"{num_token}" in config_info else {}
                # 如果config没有，找到最接近的token数
                if not config:
                    min_diff = float('inf')
                    closest_token = None
                    for token in int_keys:
                        diff = abs(token - M)
                        if diff < min_diff:
                            min_diff = diff
                            closest_token = token
                    config = config_info[f"{closest_token}"] if f"{closest_token}" in config_info else {}
            else:
                config={}
    except FileNotFoundError:
        config = {}
        
    if not config: 
        print("Config is not found! Use default config.")
        
    config = config if config else {
        "MODE": 0,
        "BLOCK_SIZE_M": 16,
    }
    return config