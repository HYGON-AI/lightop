import functools
from typing import Any, Optional
import os
import json
import torch

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
    N1: int,
    K1: int,
    N2: int,
    K2: int,
    topk: int,
    device_name: str,
    num_cus: str,
    dtype: Optional[str],
    block_n: Optional[int] = None,
    block_k: Optional[int] = None,
    is_ep: Optional[bool] = False,
    ep_size: Optional[int] = 16
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
    gfx_version = torch.cuda.get_device_properties(0).gcnArchName.split(':')[0]
    
    if not is_ep:
        gemm1_file_name = f"MOE_BLOCKINT8_CUDA_{N1}_{K1}_{gfx_version.upper()}_CU{num_cus}.json"
    else:
        gemm1_file_name = f"MOE_BLOCKINT8_CUDA_EP{ep_size}_{N1}_{K1}_{gfx_version.upper()}_CU{num_cus}.json"
        
    gemm1_config_file_path = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "configs", gemm1_file_name)
    
    try:
        with open(gemm1_config_file_path, 'r') as file:
            config_infos = json.load(file)
            if f"{N1}_{K1}" in config_infos:
                config_info = config_infos[f"{N1}_{K1}"]
                int_keys = [int(key) for key in config_info.keys()]
                greater_numbers = [num for num in int_keys if num >= M]
                num_token = min(greater_numbers) if greater_numbers else None
                gemm1_config = config_info[f"{num_token}"] if f"{num_token}" in config_info else {}
                # 如果config没有，找到最接近的token数
                if not gemm1_config:
                    min_diff = float('inf')
                    closest_token = None
                    for token in int_keys:
                        diff = abs(token - M)
                        if diff < min_diff:
                            min_diff = diff
                            closest_token = token
                    gemm1_config = config_info[f"{closest_token}"] if f"{closest_token}" in config_info else {}
            else:
                gemm1_config={}
    except FileNotFoundError:
        gemm1_config = {}
    
    gemm2_file_name = f"MOE_BLOCKINT8_CUDA_{N2}_{K2}_{gfx_version.upper()}_CU{num_cus}.json"
    gemm2_config_file_path = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "configs", gemm2_file_name)    
    try:
        with open(gemm2_config_file_path, 'r') as file:
            config_infos = json.load(file)
            if f"{N2}_{K2}" in config_infos:
                config_info = config_infos[f"{N2}_{K2}"]
                int_keys = [int(key) for key in config_info.keys()]
                greater_numbers = [num for num in int_keys if num >= M]
                num_token = min(greater_numbers) if greater_numbers else None
                gemm2_config = config_info[f"{num_token}"] if f"{num_token}" in config_info else {}
                # 如果config没有，找到最接近的token数
                if not gemm2_config:
                    min_diff = float('inf')
                    closest_token = None
                    for token in int_keys:
                        diff = abs(token - M)
                        if diff < min_diff:
                            min_diff = diff
                            closest_token = token
                    gemm2_config = config_info[f"{closest_token}"] if f"{closest_token}" in config_info else {}
            else:
                gemm2_config={}
    except FileNotFoundError:
        gemm2_config = {}
        
    status = True    
    if not gemm1_config or not gemm2_config: 
        # print("Config is not found! Use default config.")
        status = False

    # asm files check  
    if gemm1_config and gemm2_config:
        mode1 = gemm1_config['MODE']
        mode2 = gemm2_config['MODE']
        if mode1 >= 1000 or mode2 >= 1000:
            first_stage_co_file = f"moe_w8a8_channel/moe_w8a8_128x256x128_TN_{str(dtype).upper()}_UP.co"
            second_stage_co_file = f"moe_w8a8_channel/moe_w8a8_128x256x128_TN_{str(dtype).upper()}_DOWN.co"
            LIGHTOP_ASM_DIR = os.environ["LIGHTOP_ASM_DIR"]
            path_first_stage_co_file = LIGHTOP_ASM_DIR + first_stage_co_file
            path_second_stage_co_file = LIGHTOP_ASM_DIR + second_stage_co_file
            if not os.path.exists(path_first_stage_co_file) or not os.path.exists(path_second_stage_co_file):
                status=False
            
    return gemm1_config, gemm2_config, status

# Adapted from: https://github.com/sgl-project/sglang/pull/2628
@functools.lru_cache
def get_moe_cuda_marlin_config(
    E: int,
    M: int,
    N1: int,
    K1: int,
    N2: int,
    K2: int,
    topk: int,
    device_name: str,
    num_cus: str,
    dtype: Optional[str],
    block_n: Optional[int] = None,
    block_k: Optional[int] = None,
    is_ep: Optional[bool] = False,
    ep_size: Optional[int] = 16
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
    gfx_version = torch.cuda.get_device_properties(0).gcnArchName.split(':')[0]
    
    if not is_ep:
        gemm1_file_name = f"MOE_BLOCKINT8_CUDA_MARLIN_{N1}_{K1}_{gfx_version.upper()}_CU{num_cus}.json"
    else:
        gemm1_file_name = f"MOE_BLOCKINT8_CUDA_MARLIN_EP{ep_size}_{N1}_{K1}_{gfx_version.upper()}_CU{num_cus}.json"
        
    gemm1_config_file_path = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "configs", gemm1_file_name)
    
    try:
        with open(gemm1_config_file_path, 'r') as file:
            config_infos = json.load(file)
            if f"{N1}_{K1}" in config_infos:
                config_info = config_infos[f"{N1}_{K1}"]
                int_keys = [int(key) for key in config_info.keys()]
                greater_numbers = [num for num in int_keys if num >= M]
                num_token = min(greater_numbers) if greater_numbers else None
                gemm1_config = config_info[f"{num_token}"] if f"{num_token}" in config_info else {}
                # 如果config没有，找到最接近的token数
                if not gemm1_config:
                    min_diff = float('inf')
                    closest_token = None
                    for token in int_keys:
                        diff = abs(token - M)
                        if diff < min_diff:
                            min_diff = diff
                            closest_token = token
                    gemm1_config = config_info[f"{closest_token}"] if f"{closest_token}" in config_info else {}
            else:
                gemm1_config={}
    except FileNotFoundError:
        gemm1_config = {}
    
    gemm2_file_name = f"MOE_BLOCKINT8_CUDA_MARLIN_{N2}_{K2}_{gfx_version.upper()}_CU{num_cus}.json"
    gemm2_config_file_path = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "configs", gemm2_file_name)    
    try:
        with open(gemm2_config_file_path, 'r') as file:
            config_infos = json.load(file)
            if f"{N2}_{K2}" in config_infos:
                config_info = config_infos[f"{N2}_{K2}"]
                int_keys = [int(key) for key in config_info.keys()]
                greater_numbers = [num for num in int_keys if num >= M]
                num_token = min(greater_numbers) if greater_numbers else None
                gemm2_config = config_info[f"{num_token}"] if f"{num_token}" in config_info else {}
                # 如果config没有，找到最接近的token数
                if not gemm2_config:
                    min_diff = float('inf')
                    closest_token = None
                    for token in int_keys:
                        diff = abs(token - M)
                        if diff < min_diff:
                            min_diff = diff
                            closest_token = token
                    gemm2_config = config_info[f"{closest_token}"] if f"{closest_token}" in config_info else {}
            else:
                gemm2_config={}
    except FileNotFoundError:
        gemm2_config = {}

    status = True    
    if not gemm1_config or not gemm2_config: 
        # print("Config is not found! Use default config.")
        status = False

    # asm files check  
    # if gemm1_config and gemm2_config:
    #     mode1 = gemm1_config['MODE']
    #     mode2 = gemm2_config['MODE']
    #     if mode1 >= 1000 or mode2 >= 1000:
    #         first_stage_co_file = f"moe_w8a8_channel/moe_w8a8_marlin_128x256x128_TN_{str(dtype).upper()}_UP.co"
    #         second_stage_co_file = f"moe_w8a8_channel/moe_w8a8_marlin_128x256x128_TN_{str(dtype).upper()}_DOWN.co"
    #         LIGHTOP_ASM_DIR = os.environ["LIGHTOP_ASM_DIR"]
    #         path_first_stage_co_file = LIGHTOP_ASM_DIR + first_stage_co_file
    #         path_second_stage_co_file = LIGHTOP_ASM_DIR + second_stage_co_file
    #         if not os.path.exists(path_first_stage_co_file) or not os.path.exists(path_second_stage_co_file):
    #             status=False
            
    return gemm1_config, gemm2_config, status

# Adapted from: https://github.com/sgl-project/sglang/pull/2628
@functools.lru_cache
def get_moe_cuda_config_w4a8(
    E: int,
    M: int,
    N1: int,
    K1: int,
    N2: int,
    K2: int,
    topk: int,
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
    gemm1_file_name = f"MOE_BLOCKINT4_CUDA_{N1}_{K1}_{device_name}_{num_cus}.json"
    gemm1_config_file_path = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "configs", gemm1_file_name)
    
    try:
        with open(gemm1_config_file_path, 'r') as file:
            config_infos = json.load(file)
            if f"{N1}_{K1}" in config_infos:
                config_info = config_infos[f"{N1}_{K1}"]
                int_keys = [int(key) for key in config_info.keys()]
                greater_numbers = [num for num in int_keys if num >= M]
                num_token = min(greater_numbers) if greater_numbers else None
                gemm1_config = config_info[f"{num_token}"] if f"{num_token}" in config_info else {}
                # 如果config没有，找到最接近的token数
                if not gemm1_config:
                    min_diff = float('inf')
                    closest_token = None
                    for token in int_keys:
                        diff = abs(token - M)
                        if diff < min_diff:
                            min_diff = diff
                            closest_token = token
                    gemm1_config = config_info[f"{closest_token}"] if f"{closest_token}" in config_info else {}
            else:
                gemm1_config={}
    except FileNotFoundError:
        gemm1_config = {}
    
    gemm2_file_name = f"MOE_BLOCKINT4_CUDA_{N2}_{K2}_{device_name}_{num_cus}.json"
    gemm2_config_file_path = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "configs", gemm2_file_name)    
    try:
        with open(gemm2_config_file_path, 'r') as file:
            config_infos = json.load(file)
            if f"{N2}_{K2}" in config_infos:
                config_info = config_infos[f"{N2}_{K2}"]
                int_keys = [int(key) for key in config_info.keys()]
                greater_numbers = [num for num in int_keys if num >= M]
                num_token = min(greater_numbers) if greater_numbers else None
                gemm2_config = config_info[f"{num_token}"] if f"{num_token}" in config_info else {}
                # 如果config没有，找到最接近的token数
                if not gemm2_config:
                    min_diff = float('inf')
                    closest_token = None
                    for token in int_keys:
                        diff = abs(token - M)
                        if diff < min_diff:
                            min_diff = diff
                            closest_token = token
                    gemm2_config = config_info[f"{closest_token}"] if f"{closest_token}" in config_info else {}
            else:
                gemm2_config={}
    except FileNotFoundError:
        gemm2_config = {}
        
    status = True    
    if not gemm1_config or not gemm2_config: 
        # print("Config is not found! Use default config.")
        status = False

    # asm files check  
    # if gemm1_config and gemm2_config:
    #     mode1 = gemm1_config['MODE']
    #     mode2 = gemm2_config['MODE']
    #     if mode1 >= 1000 or mode2 >= 1000:
    #         co_file1 = f"moe_w4a8_channel/moe_w4a8_128x256x128_TN_{str(dtype).upper()}_FirstStage.co"
    #         co_file2 = f"moe_w4a8_channel/moe_w4a8_128x256x128_TN_{str(dtype).upper()}_SecondStage.co"
    #         LIGHTOP_ASM_DIR = os.environ["LIGHTOP_ASM_DIR"]
    #         path_co_file1 = LIGHTOP_ASM_DIR + co_file1
    #         path_co_file2 = LIGHTOP_ASM_DIR + co_file2
    #         if not os.path.exists(path_co_file1) or not os.path.exists(path_co_file2):
    #             status=False
            
        
    gemm2_config = gemm2_config if gemm2_config else {
        "MODE": 0,
        "BLOCK_SIZE_M": 16,
    }
    return gemm1_config, gemm2_config, status

# Adapted from: https://github.com/sgl-project/sglang/pull/2628
@functools.lru_cache
def get_moe_cuda_marlin_config_w4a8(
    E: int,
    M: int,
    N1: int,
    K1: int,
    N2: int,
    K2: int,
    topk: int,
    device_name: str,
    num_cus: str,
    dtype: Optional[str],
    block_n: Optional[int] = None,
    block_k: Optional[int] = None,
    is_ep: Optional[bool] = False,
    ep_size: Optional[int] = 8
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
    device_name = torch.cuda.get_device_properties(0).gcnArchName.split(':')[0]
    gemm1_file_name = f"MOE_BLOCKINT4_CUDA_MARLIN_{N1}_{K1}_{device_name}_{num_cus}.json"
    if is_ep:
        gemm1_file_name = f"MOE_BLOCKINT4_CUDA_MARLIN_EP{ep_size}_{N1}_{K1}_{device_name}_{num_cus}.json"

    gemm1_config_file_path = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "configs", gemm1_file_name)
    
    try:
        with open(gemm1_config_file_path, 'r') as file:
            config_infos = json.load(file)
            if f"{N1}_{K1}" in config_infos:
                config_info = config_infos[f"{N1}_{K1}"]
                int_keys = [int(key) for key in config_info.keys()]
                greater_numbers = [num for num in int_keys if num >= M]
                num_token = min(greater_numbers) if greater_numbers else None
                gemm1_config = config_info[f"{num_token}"] if f"{num_token}" in config_info else {}
                # 如果config没有，找到最接近的token数
                if not gemm1_config:
                    min_diff = float('inf')
                    closest_token = None
                    for token in int_keys:
                        diff = abs(token - M)
                        if diff < min_diff:
                            min_diff = diff
                            closest_token = token
                    gemm1_config = config_info[f"{closest_token}"] if f"{closest_token}" in config_info else {}
            else:
                gemm1_config={}
    except FileNotFoundError:
        gemm1_config = {}
    
    gemm2_file_name = f"MOE_BLOCKINT4_CUDA_MARLIN_{N2}_{K2}_{device_name}_{num_cus}.json"
    if is_ep:
        gemm2_file_name = f"MOE_BLOCKINT4_CUDA_MARLIN_EP{ep_size}_{N2}_{K2}_{device_name}_{num_cus}.json"

    gemm2_config_file_path = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "configs", gemm2_file_name)    
    try:
        with open(gemm2_config_file_path, 'r') as file:
            config_infos = json.load(file)
            if f"{N2}_{K2}" in config_infos:
                config_info = config_infos[f"{N2}_{K2}"]
                int_keys = [int(key) for key in config_info.keys()]
                greater_numbers = [num for num in int_keys if num >= M]
                num_token = min(greater_numbers) if greater_numbers else None
                gemm2_config = config_info[f"{num_token}"] if f"{num_token}" in config_info else {}
                # 如果config没有，找到最接近的token数
                if not gemm2_config:
                    min_diff = float('inf')
                    closest_token = None
                    for token in int_keys:
                        diff = abs(token - M)
                        if diff < min_diff:
                            min_diff = diff
                            closest_token = token
                    gemm2_config = config_info[f"{closest_token}"] if f"{closest_token}" in config_info else {}
            else:
                gemm2_config={}
    except FileNotFoundError:
        gemm2_config = {}
        
    status = True    
    if not gemm1_config or not gemm2_config: 
        # print("Config is not found! Use default config.")
        status = False

    # asm files check  
    # if gemm1_config and gemm2_config:
    #     mode1 = gemm1_config['MODE']
    #     mode2 = gemm2_config['MODE']
    #     if mode1 >= 1000 or mode2 >= 1000:
    #         co_file1 = f"moe_w4a8_channel/moe_w4a8_marlin_128x512x128_TN_{str(dtype).upper()}_FirstStage.co"
    #         co_file2 = f"moe_w4a8_channel/moe_w4a8_marlin_128x512x128_TN_{str(dtype).upper()}_SecondStage.co"
    #         LIGHTOP_ASM_DIR = os.environ["LIGHTOP_ASM_DIR"]
    #         path_co_file1 = LIGHTOP_ASM_DIR + co_file1
    #         path_co_file2 = LIGHTOP_ASM_DIR + co_file2
    #         if not os.path.exists(path_co_file1) or not os.path.exists(path_co_file2):
    #             status=False
            
        
    gemm2_config = gemm2_config if gemm2_config else {
        "MODE": 0,
        "BLOCK_SIZE_M": 16,
    }
    return gemm1_config, gemm2_config, status

# Adapted from: https://github.com/sgl-project/sglang/pull/2628
@functools.lru_cache
def get_moe_cuda_marlin_config_w4a16(
    E: int,
    M: int,
    N1: int,
    K1: int,
    N2: int,
    K2: int,
    topk: int,
    device_name: str,
    num_cus: str,
    dtype: Optional[str],
    block_n: Optional[int] = None,
    block_k: Optional[int] = None,
    is_ep: Optional[bool] = False,
    ep_size: Optional[int] = 8
) -> Optional[dict[int, Any]]:
    """
    Return optimized configurations for the fused MoE kernel.

    The return value will be a dictionary that maps an irregular grid of
    batch sizes to configurations of the fused_moe kernel. To evaluate the
    kernel on a given batch size bs, the closest batch size in the grid should
    be picked and the associated configuration chosen to invoke the kernel.
    """
    assert not is_ep, "lightop.moe.w4a16 not support EP"

    # First look up if an optimized configuration is available in the configs
    # directory
    device_name = torch.cuda.get_device_properties(0).gcnArchName.split(':')[0]  # gfx936
    gemm1_file_name = f"MOE_W4A16_CUDA_MARLIN_{N1}_{K1}_{device_name}_{num_cus}.json"
    if is_ep:
        gemm1_file_name = f"MOE_W4A16_CUDA_MARLIN_EP{ep_size}_{N1}_{K1}_{device_name}_{num_cus}.json"

    gemm1_config_file_path = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "configs", gemm1_file_name)
    
    try:
        with open(gemm1_config_file_path, 'r') as file:
            config_infos = json.load(file)
            if f"{N1}_{K1}" in config_infos:
                config_info = config_infos[f"{N1}_{K1}"]
                int_keys = [int(key) for key in config_info.keys()]
                greater_numbers = [num for num in int_keys if num >= M]
                num_token = min(greater_numbers) if greater_numbers else None
                gemm1_config = config_info[f"{num_token}"] if f"{num_token}" in config_info else {}
                # 如果config没有，找到最接近的token数
                if not gemm1_config:
                    min_diff = float('inf')
                    closest_token = None
                    for token in int_keys:
                        diff = abs(token - M)
                        if diff < min_diff:
                            min_diff = diff
                            closest_token = token
                    gemm1_config = config_info[f"{closest_token}"] if f"{closest_token}" in config_info else {}
            else:
                gemm1_config={}
    except FileNotFoundError:
        gemm1_config = {}
    
    gemm2_file_name = f"MOE_W4A16_CUDA_MARLIN_{N2}_{K2}_{device_name}_{num_cus}.json"
    if is_ep:
        gemm2_file_name = f"MOE_W4A16_CUDA_MARLIN_EP{ep_size}_{N2}_{K2}_{device_name}_{num_cus}.json"

    gemm2_config_file_path = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "configs", gemm2_file_name)    
    try:
        with open(gemm2_config_file_path, 'r') as file:
            config_infos = json.load(file)
            if f"{N2}_{K2}" in config_infos:
                config_info = config_infos[f"{N2}_{K2}"]
                int_keys = [int(key) for key in config_info.keys()]
                greater_numbers = [num for num in int_keys if num >= M]
                num_token = min(greater_numbers) if greater_numbers else None
                gemm2_config = config_info[f"{num_token}"] if f"{num_token}" in config_info else {}
                # 如果config没有，找到最接近的token数
                if not gemm2_config:
                    min_diff = float('inf')
                    closest_token = None
                    for token in int_keys:
                        diff = abs(token - M)
                        if diff < min_diff:
                            min_diff = diff
                            closest_token = token
                    gemm2_config = config_info[f"{closest_token}"] if f"{closest_token}" in config_info else {}
            else:
                gemm2_config={}
    except FileNotFoundError:
        gemm2_config = {}
        
    status = True    
    if not gemm1_config or not gemm2_config: 
        # print("Config is not found! Use default config.")
        status = False

    # # asm files check 
    # if gemm1_config and gemm2_config:
    #     mode1 = gemm1_config['MODE']
    #     mode2 = gemm2_config['MODE']
    #     if mode1 >= 1000 or mode2 >= 1000:
    #         co_file1 = f"moe_w4a8_channel/moe_w4a8_marlin_128x512x128_TN_{str(dtype).upper()}_FirstStage.co"
    #         co_file2 = f"moe_w4a8_channel/moe_w4a8_marlin_128x512x128_TN_{str(dtype).upper()}_SecondStage.co"
    #         LIGHTOP_ASM_DIR = os.environ["LIGHTOP_ASM_DIR"]
    #         path_co_file1 = LIGHTOP_ASM_DIR + co_file1
    #         path_co_file2 = LIGHTOP_ASM_DIR + co_file2
    #         if not os.path.exists(path_co_file1) or not os.path.exists(path_co_file2):
    #             status=False
            
        
    gemm2_config = gemm2_config if gemm2_config else {
        "MODE": 0,
        "BLOCK_SIZE_M": 16,
    }
    return gemm1_config, gemm2_config, status

# Adapted from: https://github.com/sgl-project/sglang/pull/2628
@functools.lru_cache
def get_moe_groupgemm_config_w4a8(
    E: int,
    M: int,
    N1: int,
    K1: int,
    N2: int,
    K2: int,
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
    gemm1_config = {
        "MODE": 1000,
    }
    gemm2_config = {
        "MODE": 1000,
    }
    status = True
    
    return gemm1_config, gemm2_config, status

# Adapted from: https://github.com/sgl-project/sglang/pull/2628
@functools.lru_cache
def get_moe_groupgemm_marlin_config_w4a8(
    E: int,
    M: int,
    N1: int,
    K1: int,
    N2: int,
    K2: int,
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
    gemm1_config = {
        "MODE": 1000,
    }
    gemm2_config = {
        "MODE": 1000,
    }
    status = True
    
    return gemm1_config, gemm2_config, status

@functools.lru_cache
def get_moe_cuda_marlin_config_w16a16(
    E: int,
    M: int,
    N1: int,
    K1: int,
    N2: int,
    K2: int,
    topk: int,
    device_name: str,
    num_cus: str,
    dtype: Optional[str],
    block_n: Optional[int] = None,
    block_k: Optional[int] = None,
    is_ep: Optional[bool] = False,
    ep_size: Optional[int] = 8,
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
    device_name = torch.cuda.get_device_properties(0).gcnArchName.split(':')[0]
    gemm1_file_name = f"MOE_W16A16_CUDA_MARLIN_{N1}_{K1}_{device_name}_{num_cus}.json"
    if is_ep:
        gemm1_file_name = f"MOE_W16A16_CUDA_EP{ep_size}_{N1}_{K1}_{device_name}_{num_cus}.json"
    gemm1_config_file_path = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "configs", gemm1_file_name)
    
    try:
        with open(gemm1_config_file_path, 'r') as file:
            config_infos = json.load(file)
            if f"{N1}_{K1}" in config_infos:
                config_info = config_infos[f"{N1}_{K1}"]
                int_keys = [int(key) for key in config_info.keys()]
                greater_numbers = [num for num in int_keys if num >= M]
                num_token = min(greater_numbers) if greater_numbers else None
                gemm1_config = config_info[f"{num_token}"] if f"{num_token}" in config_info else {}
                # 如果config没有，找到最接近的token数
                if not gemm1_config:
                    min_diff = float('inf')
                    closest_token = None
                    for token in int_keys:
                        diff = abs(token - M)
                        if diff < min_diff:
                            min_diff = diff
                            closest_token = token
                    gemm1_config = config_info[f"{closest_token}"] if f"{closest_token}" in config_info else {}
            else:
                gemm1_config={}
    except FileNotFoundError:
        gemm1_config = {}
    
    gemm2_file_name = f"MOE_W16A16_CUDA_MARLIN_{N2}_{K2}_{device_name}_{num_cus}.json"
    if is_ep:
        gemm2_file_name = f"MOE_W16A16_CUDA_EP{ep_size}_{N2}_{K2}_{device_name}_{num_cus}.json"
    gemm2_config_file_path = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "configs", gemm2_file_name)    
    try:
        with open(gemm2_config_file_path, 'r') as file:
            config_infos = json.load(file)
            if f"{N2}_{K2}" in config_infos:
                config_info = config_infos[f"{N2}_{K2}"]
                int_keys = [int(key) for key in config_info.keys()]
                greater_numbers = [num for num in int_keys if num >= M]
                num_token = min(greater_numbers) if greater_numbers else None
                gemm2_config = config_info[f"{num_token}"] if f"{num_token}" in config_info else {}
                # 如果config没有，找到最接近的token数
                if not gemm2_config:
                    min_diff = float('inf')
                    closest_token = None
                    for token in int_keys:
                        diff = abs(token - M)
                        if diff < min_diff:
                            min_diff = diff
                            closest_token = token
                    gemm2_config = config_info[f"{closest_token}"] if f"{closest_token}" in config_info else {}
            else:
                gemm2_config={}
    except FileNotFoundError:
        gemm2_config = {}
        
    status = True    
    if not gemm1_config or not gemm2_config: 
        # print("Config is not found! Use default config.")
        status = False
            
        gemm1_config = {
            "MODE": 103,
            "BLOCK_SIZE_M": 16,
        }
        gemm2_config = {
            "MODE": 103,
            "BLOCK_SIZE_M": 16,
        }
    return gemm1_config, gemm2_config, status

@functools.lru_cache
def get_moe_cuda_config_w16a16(
    E: int,
    M: int,
    N1: int,
    K1: int,
    N2: int,
    K2: int,
    topk: int,
    device_name: str,
    num_cus: str,
    dtype: Optional[str],
    block_n: Optional[int] = None,
    block_k: Optional[int] = None,
    is_ep: Optional[bool] = False,
    ep_size: Optional[int] = 8,
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
    device_name = torch.cuda.get_device_properties(0).gcnArchName.split(':')[0]
    gemm1_file_name = f"MOE_W16A16_CUDA_{N1}_{K1}_{device_name}_{num_cus}.json"
    if is_ep:
        gemm1_file_name = f"MOE_W16A16_CUDA_EP{ep_size}_{N1}_{K1}_{device_name}_{num_cus}.json"
    gemm1_config_file_path = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "configs", gemm1_file_name)
    
    try:
        with open(gemm1_config_file_path, 'r') as file:
            config_infos = json.load(file)
            if f"{N1}_{K1}" in config_infos:
                config_info = config_infos[f"{N1}_{K1}"]
                int_keys = [int(key) for key in config_info.keys()]
                greater_numbers = [num for num in int_keys if num >= M]
                num_token = min(greater_numbers) if greater_numbers else None
                gemm1_config = config_info[f"{num_token}"] if f"{num_token}" in config_info else {}
                # 如果config没有，找到最接近的token数
                if not gemm1_config:
                    min_diff = float('inf')
                    closest_token = None
                    for token in int_keys:
                        diff = abs(token - M)
                        if diff < min_diff:
                            min_diff = diff
                            closest_token = token
                    gemm1_config = config_info[f"{closest_token}"] if f"{closest_token}" in config_info else {}
            else:
                gemm1_config={}
    except FileNotFoundError:
        gemm1_config = {}
    
    gemm2_file_name = f"MOE_W16A16_CUDA_{N2}_{K2}_{device_name}_{num_cus}.json"
    if is_ep:
        gemm2_file_name = f"MOE_W16A16_CUDA_EP{ep_size}_{N2}_{K2}_{device_name}_{num_cus}.json"
    gemm2_config_file_path = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "configs", gemm2_file_name)    
    try:
        with open(gemm2_config_file_path, 'r') as file:
            config_infos = json.load(file)
            if f"{N2}_{K2}" in config_infos:
                config_info = config_infos[f"{N2}_{K2}"]
                int_keys = [int(key) for key in config_info.keys()]
                greater_numbers = [num for num in int_keys if num >= M]
                num_token = min(greater_numbers) if greater_numbers else None
                gemm2_config = config_info[f"{num_token}"] if f"{num_token}" in config_info else {}
                # 如果config没有，找到最接近的token数
                if not gemm2_config:
                    min_diff = float('inf')
                    closest_token = None
                    for token in int_keys:
                        diff = abs(token - M)
                        if diff < min_diff:
                            min_diff = diff
                            closest_token = token
                    gemm2_config = config_info[f"{closest_token}"] if f"{closest_token}" in config_info else {}
            else:
                gemm2_config={}
    except FileNotFoundError:
        gemm2_config = {}
        
    status = True    
    if not gemm1_config or not gemm2_config: 
        # print("Config is not found! Use default config.")
        status = False
        gemm1_config = {
            "MODE": 500,
            "BLOCK_SIZE_M": 16,
        }
        gemm2_config = {
            "MODE": 500,
            "BLOCK_SIZE_M": 16,
        }
    return gemm1_config, gemm2_config, status

@functools.lru_cache(maxsize=1024)
def get_batched_gemm_w8a8_config(
    M: int,
    N: int,
    K: int,
) -> Optional[dict[int, Any]]:
    """
    Load a GEMM configuration using the standardized M_LEQ_x/M_GEQ_y/any format.

    Args:
        M: M dimension of the GEMM
        N: N dimension of the GEMM
        K: K dimension of the GEMM

    Returns:
        Dictionary with the config params,
        bool indicating if the config is tuned.(True if tuned, False otherwise)
    """
    # Input validation
    assert M >= 0, "M must be positive."
    assert N > 0, "N must be positive when provided."
    assert K > 0, "K must be positive when provided."

    def _load_config_file(
        cache_dict: dict,
        cache_key: str,
        fpath: str,
        config_key: str,
        fpath_should_exist: bool = False,
    ) -> bool:
        """
        Helper function to load a config file and cache it.
        """
        if os.path.exists(fpath):
            with open(fpath, "r") as file:
                config = json.load(file)
            cache_dict[cache_key][config_key] = config
            return True
        elif fpath_should_exist:
            # raise AssertionError(f"Required config file doesn't exist: {fpath}")
            return False
        return False
    
    if not hasattr(get_batched_gemm_w8a8_config, "_config_cache"):
        get_batched_gemm_w8a8_config._config_cache = {}

    gfx_version = torch.cuda.get_device_properties(0).gcnArchName.split(':')[0]
    cu_number = torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count
    cache_key = f"BATCHED_GEMM_W8A8_{N}_{K}"
    nk_key = f"{N}_{K}_{gfx_version.upper()}_CU{cu_number}"
    filename = f"BATCHED_GEMM_W8A8_{N}_{K}_{gfx_version.upper()}_CU{cu_number}.json"
    config_file_path = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "configs", filename)

    if cache_key not in get_batched_gemm_w8a8_config._config_cache:
        get_batched_gemm_w8a8_config._config_cache[cache_key] = {}

        # Load default config (must exist)
        fpath = config_file_path
        _load_config_file(
            get_batched_gemm_w8a8_config._config_cache,
            cache_key,
            fpath,
            nk_key,
            fpath_should_exist=True,
        )

    config_dict_key = None

    if N is not None and K is not None:
        if nk_key not in get_batched_gemm_w8a8_config._config_cache[cache_key]:
            # load specialized config
            config_dict_key = None
        else:
            config_dict_key = nk_key

    best_batch_gemm_config = {}
    status = True

    if config_dict_key is not None:
        config_dict = get_batched_gemm_w8a8_config._config_cache[cache_key][config_dict_key]
        int_keys = [int(key) for key in config_dict.keys()]
        greater_numbers = [num for num in int_keys if num >= M]
        batch_size = min(greater_numbers) if greater_numbers else None
        best_batch_gemm_config = config_dict[f"{batch_size}"] if f"{batch_size}" in config_dict else {}
        # 如果config没有，找到最接近的token数
        if not best_batch_gemm_config:
            min_diff = float('inf')
            closest_batch_size = None
            for bs in int_keys:
                diff = abs(bs - M)
                if diff < min_diff:
                    min_diff = diff
                    closest_batch_size = bs
            best_batch_gemm_config = config_dict[f"{closest_batch_size}"] if f"{closest_batch_size}" in config_dict else {}
        status = True
    else:
        best_batch_gemm_config = {'MODE': 0}
        status = False
        
    return best_batch_gemm_config, status
        
