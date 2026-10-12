# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

import argparse
import json
import time
from contextlib import nullcontext
from datetime import datetime
from itertools import product
from typing import Any, Dict, List, Tuple, Callable, TypedDict, Optional

import ray
import torch
import triton
from ray.experimental.tqdm_ray import tqdm
from transformers import AutoConfig

# from vllm.model_executor.layers.fused_moe.fused_moe import *
from vllm.platforms import current_platform
from vllm.utils import FlexibleArgumentParser

# 用于tuning时正确性验证和 性能测试的文件
from fuse_moegemm import grouped_topk, blockM_gemm2_prefill_configs, gemm2_kernel_valid, gemm2_kernel_prof
# 用于模型侧性能测试的文件
from fused_moe_cuda import fused_moe_impl
from fused_moe_cuda_valid import fused_moe_impl_valid

from lightop.quant import (
    per_token_group_quant_int8,
    per_token_quant_int8,
)
import math
import torch.nn.functional as F
import pandas as pd
from typing import Optional, Type,Any, Dict, List, Tuple

import os
os.environ["TRITON_CACHE_DIR"] = f".triton/cache"

triton_json_dir="./cache/moe"
device_name=torch.cuda.get_device_properties('cuda').name.replace(" ","_")

M_THRESHOLD = 128

# 生成 [m, topk] 的整数张量，每行元素不重复
def generate_unique_int_tensor(m, topk, low=0, high=256):
    # 检查 topk 是否超过范围
    if topk > (high - low):
        raise ValueError(f"topk ({topk}) 不能超过范围 ({high - low})")
    
    # 生成张量
    tensor = torch.zeros((m, topk), dtype=torch.int)
    
    for i in range(m):
        # 对每一行生成不重复的随机整数
        row = torch.randperm(high - low)[:topk] + low
        tensor[i] = row
    
    return tensor

def _fused_moe_helper(m: int,
                             n: int,
                             k: int,
                             n2:int,
                             k2:int,
                             topk:int,
                             E:int,
                             out_dtype: Type[torch.dtype] = torch.bfloat16,
                             device: str = "cuda",
                             block_size: List[int]=[128,128],
                             best_config:Optional[list] = None):

    input = (torch.randn((m, n2), device=device) * 5).to(dtype=out_dtype)
    weight13 = to_int8(torch.randn((E,n ,k), device=device) * 5)
    weight2 = to_int8(torch.randn((E,n2 ,k2), device=device) * 5)

    weight13_scale = (torch.randn((E, n ,1), device=device,
                           dtype=torch.float32))
    weight2_scale = (torch.randn((E, n2, 1), device=device,
                           dtype=torch.float32))   

    tensor = torch.rand((m, topk),device=device)
    topk_weights = F.softmax(tensor, dim=1)

    topk_ids=generate_unique_int_tensor(m=m, topk=topk, low=0, high=E).cuda()

    
    return input,weight13,weight2,weight13_scale,weight2_scale,topk_weights,topk_ids

def to_int8(tensor: torch.Tensor):
    return torch.round(tensor.clamp(min=-128, max=127)).to(dtype=torch.int8)

class BenchmarkConfig(TypedDict):
    BLOCK_SIZE_M: int
    BLOCK_SIZE_N: int
    BLOCK_SIZE_K: int
    GROUP_SIZE_M: int
    num_warps: int
    num_stages: int
    enable_mmacfuse: int

def benchmark_config_gemm2(
                            num_tokens: int,
                            num_experts: int,
                            shard_intermediate_size: int,
                            hidden_size: int,
                            topk: int,
                            dtype: torch.dtype,
                            use_fp8_w8a8: bool,
                            use_int8_w8a16: bool,
                            use_int8_w4a8: bool,
                            num_iters: int = 100,
                            block_quant_shape: List[int] = None,
                            delta: int=-1,
                            mode_idx: int=-1,
                            block_m: int=-1
                            ) -> float:
    init_dtype = torch.float16 if use_fp8_w8a8 else dtype
    x = torch.randn(num_tokens, hidden_size, dtype=dtype)
    w1_scale = None
    w2_scale = None
    a1_scale = None
    a2_scale = None
    w1 = torch.randint(-127,
                        127, (
                            num_experts,
                            shard_intermediate_size, # n
                            hidden_size//2, # k
                        ),
                        dtype=torch.int8)
    w2 = torch.randint(-127,
                        127, (
                            num_experts,
                            hidden_size, # n2
                            shard_intermediate_size//4, # k2
                        ),
                        dtype=torch.int8)

    w1_scale = (torch.randn((num_experts, shard_intermediate_size ,1),
                           dtype=torch.float32))
    w2_scale = (torch.randn((num_experts, hidden_size, 1),
                           dtype=torch.float32))    
    
    gating_output = torch.randn(num_iters,
                                num_tokens,
                                num_experts,
                                dtype=torch.float32)
    
    def prepare(i: int):
        input_gating.copy_(gating_output[i])

    input_gating = torch.randn(num_tokens, num_experts, dtype=torch.float32)
    
    latencies: list[float] = []
    for i in range(num_iters):
        prepare(i)
        torch.cuda.synchronize()

        topk_weights, topk_ids = grouped_topk(x, input_gating, topk, True, 8, 4)
        
        kernel_time = gemm2_kernel_prof(
                                        hidden_states=x,
                                        w1=w1,
                                        w2=w2,
                                        topk_weights=topk_weights,
                                        topk_ids=topk_ids,
                                        inplace=False,
                                        use_int8_w4a8=True,
                                        w1_scale=w1_scale,
                                        w2_scale=w2_scale,
                                        block_shape=None,
                                        per_channel_quant=True,
                                        mode_idx=mode_idx, # 表示当前准确性验证的 mode 索引
                                        block_m=block_m,
                                        delta=delta
                                        )

    
        latencies.append(kernel_time)

    avg = sum(latencies) / (num_iters)
    return avg

def moe_cuda_graph(
        num_tokens: int,
        num_experts: int,
        shard_intermediate_size: int,
        hidden_size: int,
        topk: int,
        dtype: torch.dtype,
        use_fp8_w8a8: bool,
        use_int8_w8a16: bool,
        use_int8_w4a8: bool,
        num_iters: int=100,
        block_quant_shape: List[int] = None,
        seed: int=0,
        device_id: int=0,
    ) -> tuple[dict[str, int], float]:
        from vllm.platforms import current_platform
        current_platform.seed_everything(seed)
        print("====================== num_tokens: ", num_tokens, "  ====================")
        device="cuda:"+str(device_id)   
        x = torch.randn(num_tokens, hidden_size, device=device, dtype=dtype)
        w1_scale = None
        w2_scale = None
        a1_scale = None
        a2_scale = None

        w1 = torch.randint(-127,
                            127, (
                                num_experts,#256
                                shard_intermediate_size, # n:512
                                hidden_size// 2, # k:3584
                            ),
                            device=device,
                            dtype=torch.int8)
        w2 = torch.randint(-127,
                            127, (
                                num_experts,
                                hidden_size, # n2
                                shard_intermediate_size // 4, # 128
                            ),
                            device=device,
                            dtype=torch.int8)
        
        w1_scale = (torch.randn((num_experts, shard_intermediate_size ,1),
                           dtype=torch.float32, device=device))
        w2_scale = (torch.randn((num_experts, hidden_size, 1),
                           dtype=torch.float32, device=device))    
           
        gating_output = torch.randn(num_iters,
                                num_tokens,
                                num_experts,
                                device=device,
                                dtype=torch.float32) 
        def prepare(i: int):
            input_gating.copy_(gating_output[i])

        input_gating = torch.randn(num_tokens, num_experts, device=device, dtype=torch.float32)

        # 正确性验证
        is_true = fused_moe_impl_valid(x,
                                        w1,
                                        w2,
                                        input_gating,
                                        topk,
                                        renormalize=True,
                                        inplace=False,
                                        use_fp8_w8a8=use_fp8_w8a8,
                                        use_int8_w8a16=use_int8_w8a16,
                                        use_int8_w4a8=use_int8_w4a8,
                                        w1_scale=w1_scale,
                                        w2_scale=w2_scale,
                                        a1_scale=a1_scale,
                                        a2_scale=a2_scale,
                                        block_shape=block_quant_shape,
                                        config=None)
        if not is_true:
            print("正确性验证存在错误！！")
            assert False
        else:
            print("通过正确性验证！！！")

        def run():
            fused_moe_impl(x,
                            w1,
                            w2,
                            input_gating,
                            topk,
                            renormalize=True,
                            inplace=False,
                            use_fp8_w8a8=use_fp8_w8a8,
                            use_int8_w8a16=use_int8_w8a16,
                            use_int8_w4a8=use_int8_w4a8,
                            w1_scale=w1_scale,
                            w2_scale=w2_scale,
                            a1_scale=a1_scale,
                            a2_scale=a2_scale,
                            block_shape=block_quant_shape,
                            config=None)
        
        run()
        torch.cuda.synchronize()

        # Capture 10 invocations with CUDA graph
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for _ in range(10):
                run()
        torch.cuda.synchronize()

        # Warmup
        for _ in range(5):
            graph.replay()
        torch.cuda.synchronize()

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        latencies: list[float] = []
        for i in range(num_iters):
            prepare(i)
            torch.cuda.synchronize()
            
            start_event.record()
            graph.replay()
            end_event.record()
            end_event.synchronize()
            latencies.append(start_event.elapsed_time(end_event))
        avg = sum(latencies) / (num_iters * 10) * 1000  # us
        graph.reset()
        return avg
        
@ray.remote(num_gpus=1)
class BenchmarkWorker:

    def __init__(self, seed: int, device_id: int) -> None:
        from vllm.platforms import current_platform
        torch.set_default_device("cuda:"+ str(device_id))
        current_platform.seed_everything(seed)
        self.seed = seed

        self.device_id = device_id
    def benchmark(
        self,
        num_tokens: int,
        num_experts: int,
        shard_intermediate_size: int,
        hidden_size: int,
        topk: int,
        dtype: torch.dtype,
        use_fp8_w8a8: bool,
        use_int8_w8a16: bool,
        use_int8_w4a8: bool,
        block_quant_shape: List[int] = None,
        use_deep_gemm: bool = False,
    ) -> tuple[dict[str, int], float]:
        
        kernel_time = moe_cuda_graph(num_tokens, num_experts,
                                       shard_intermediate_size, hidden_size,
                                       topk, dtype, use_fp8_w8a8,
                                       use_int8_w8a16, use_int8_w4a8, block_quant_shape=block_quant_shape,seed=self.seed, device_id=self.device_id)

        return kernel_time

    def tune(
        self,
        num_tokens: int,
        num_experts: int,
        shard_intermediate_size: int,
        hidden_size: int,
        topk: int,
        dtype: torch.dtype,
        use_fp8_w8a8: bool,
        use_int8_w8a16: bool,
        use_int8_w4a8: bool,
        block_quant_shape: list[int],
        use_deep_gemm: bool,
    ) -> dict[str, int]:
        from vllm.platforms import current_platform
        current_platform.seed_everything(self.seed)
        device="cuda:"+str(self.device_id)

        print("======================================= num_tokens: ", num_tokens, " =============================")
        best_config_kernel0 = None
        best_config_kernel1 = None
        best_time = float("inf")
        with torch.cuda.device(self.device_id):
            # 遍历当前 num_tokens 下 各个block_m的性能

            # 准备 gemm1 和 gemm2 都需要的原始数据
            input,weight13,weight2,weight13_scale,weight2_scale,topk_weights,topk_ids=_fused_moe_helper(m=num_tokens,n=shard_intermediate_size,k=(hidden_size//2),n2=hidden_size,k2=(shard_intermediate_size // 4),topk=topk,E=num_experts,device=device,block_size=None,out_dtype=dtype)
            
            # for block_m in [16, 32, 64, 128]:
            # 当前汇编版本仅支持 128
            for block_m in [128]:

                # ==============================  gemm1 调优逻辑 =========================== 
                time_kernel0 = float('inf')
                config_kernel0 = None
        
                time_kernel0 = 0.0
                best_config_0 = {}
                best_config_0['BLOCK_SIZE_M'] = 128
                best_config_0['MODE'] = 1000
                best_config_0['DELTA'] = 1
                config_kernel0 = best_config_0
                
                # ================================== gemm2 调优逻辑 ================================
                
                # gemm2 使用 mode 索引
                time_kernel1 = float('inf')
                config_kernel1 = None
                # 遍历所有已支持的 mode
                for mode_idx in range(len(blockM_gemm2_prefill_configs)):
                    # 获取当前mode的 block_m
                    mode_blcok_m = blockM_gemm2_prefill_configs[mode_idx]

                    delta = 128 // mode_blcok_m

                    # mode 的 blokc_m 不符合当前block_m的直接跳过
                    if mode_blcok_m * delta  != block_m:
                        continue

                    # 先进行 正确性测试
                    is_true = gemm2_kernel_valid(
                                                    hidden_states=input,
                                                    w1=weight13,
                                                    w2=weight2,
                                                    topk_weights=topk_weights,
                                                    topk_ids=topk_ids,
                                                    inplace=False,
                                                    use_int8_w4a8=True,
                                                    w1_scale=weight13_scale,
                                                    w2_scale=weight2_scale,
                                                    block_shape=None,
                                                    per_channel_quant=True,
                                                    mode_idx=mode_idx, # 表示当前准确性验证的 mode 索引
                                                    block_m=block_m,
                                                    delta=delta
                                                    )

                    # 正确性测试不通过的直接跳过
                    if not is_true:
                        continue

                    # 随机 100次专家选择 做 性能测试
                    kernel_time = benchmark_config_gemm2(
                                                        num_tokens,
                                                        num_experts,
                                                        shard_intermediate_size,
                                                        hidden_size,
                                                        topk,
                                                        dtype,
                                                        use_fp8_w8a8,
                                                        use_int8_w8a16,
                                                        use_int8_w4a8,
                                                        num_iters=10,
                                                        block_quant_shape=block_quant_shape,
                                                        mode_idx=mode_idx,
                                                        block_m=block_m,
                                                        delta=delta
                                                        )

                    if kernel_time < time_kernel1:
                        time_kernel1 = kernel_time
                        best_config_1 = {}
                        best_config_1['BLOCK_SIZE_M'] = block_m
                        best_config_1['MODE'] = mode_idx
                        best_config_1['DELTA'] = delta

                        config_kernel1 = best_config_1

                print("Completed tuning for batch_size: ", num_tokens, "block_m: ", block_m)
                if (time_kernel0 + time_kernel1) < best_time:
                    best_time = time_kernel0 + time_kernel1
                    best_config_kernel0 = config_kernel0
                    best_config_kernel1 = config_kernel1

        return best_config_kernel0, best_config_kernel1


def sort_config(config: BenchmarkConfig) -> BenchmarkConfig:
    
    config_first_stage, config_second_stage = config
    config = config_first_stage
    first_stage_dict =  {
        "BLOCK_SIZE_M": config["BLOCK_SIZE_M"],
        "BLOCK_SIZE_N": config["BLOCK_SIZE_N"],
        "BLOCK_SIZE_K": config["BLOCK_SIZE_K"],
        "GROUP_SIZE_M": config["GROUP_SIZE_M"],
        "num_warps": config["num_warps"],
        "num_stages": config["num_stages"],
        "enable_mmacfuse": config["enable_mmacfuse"],
        **({"waves_per_eu": config["waves_per_eu"]} if "waves_per_eu" in config else {}),
        **({"matrix_instr_nonkdim": config["matrix_instr_nonkdim"]} if "matrix_instr_nonkdim" in config else {}),
        **({"kpack": config["kpack"]} if "kpack" in config else {}),
    }
    
    config = config_second_stage
    second_stage_dict =  {
        "BLOCK_SIZE_M": config["BLOCK_SIZE_M"],
        "BLOCK_SIZE_N": config["BLOCK_SIZE_N"],
        "BLOCK_SIZE_K": config["BLOCK_SIZE_K"],
        "GROUP_SIZE_M": config["GROUP_SIZE_M"],
        "num_warps": config["num_warps"],
        "num_stages": config["num_stages"],
        "enable_mmacfuse": config["enable_mmacfuse"],
        **({"waves_per_eu": config["waves_per_eu"]} if "waves_per_eu" in config else {}),
        **({"matrix_instr_nonkdim": config["matrix_instr_nonkdim"]} if "matrix_instr_nonkdim" in config else {}),
        **({"kpack": config["kpack"]} if "kpack" in config else {}),
    }
    
    configs_tuple = (first_stage_dict, second_stage_dict)

    return configs_tuple

def save_configs(configs_first_stage: dict[int, BenchmarkConfig], configs_second_stage: dict[int, BenchmarkConfig], num_experts: int,
                 shard_intermediate_size: int, hidden_size: int, topk: int,
                 dtype: torch.dtype, use_fp8_w8a8: bool, use_int8_w8a16: bool, use_int8_w4a8: bool,
                 block_quant_shape: List[int]) -> None:

    N=shard_intermediate_size
    K=hidden_size//2
    filename = f"MOE_BLOCKINT4_CUDA_{N}_{K}_{device_name}_{torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count}.json"

    print(f"Writing best config to {filename}...")
    try:
        with open(filename, 'r') as file:
                config_infos = json.load(file)
                if f"{N}_{K}" in config_infos:
                    config_info = config_infos[f"{N}_{K}"]
                else:
                    config_info = {}
                
    except FileNotFoundError:
        config_infos = {}
        config_info = {}  
    
    for key, value in configs_first_stage.items():
        config_info[f"{key}"] = value  
        
    config_infos[f"{N}_{K}"] = config_info
    # save best config
    with open(filename, 'w') as file:
        json.dump(config_infos, file, indent=4)  
     
    N=hidden_size
    K=shard_intermediate_size // 4
    filename = f"MOE_BLOCKINT4_CUDA_{N}_{K}_{device_name}_{torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count}.json"

    print(f"Writing best config to {filename}...")
    try:
        with open(filename, 'r') as file:
                config_infos = json.load(file)
                if f"{N}_{K}" in config_infos:
                    config_info = config_infos[f"{N}_{K}"]
                else:
                    config_info = {}
                
    except FileNotFoundError:
        config_infos = {}
        config_info = {}  
    
    for key, value in configs_second_stage.items():
        config_info[f"{key}"] = value  
        
    config_infos[f"{N}_{K}"] = config_info
    # save best config
    with open(filename, 'w') as file:
        json.dump(config_infos, file, indent=4) 

def get_weight_block_size_safety(config, default_value=None):

    quantization_config = getattr(config, 'quantization_config', {})
    if isinstance(quantization_config, dict):
        return quantization_config.get('weight_block_size', default_value)
    return default_value


def main(args: argparse.Namespace):
    print(args)
    
    moe_ep_size = args.moe_ep_size
    tp_size = args.tp_size
    if moe_ep_size > 1:
        tp_size = tp_size // moe_ep_size
    
    block_quant_shape = None
    topk = 8
    dtype = torch.bfloat16 if current_platform.is_rocm() else config.torch_dtype
    use_fp8_w8a8 = args.dtype == "fp8_w8a8"
    use_int8_w8a16 = args.dtype == "int8_w8a16"
    use_int8_w4a8 = args.dtype == "int8_w4a8"

    E = 256
    shard_intermediate_size=512
    hidden_size=7168


    if args.batch_size is None:
        # batch_sizes = [
        #     # 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 6144, 8192, 16384
        #     1,# 2, 4, 8, 16, 32, 64, 128,
        # ]
        # batch_sizes = [256]
        batch_sizes = [256, 512, 1024]

    else:
        batch_sizes = [args.batch_size]

    use_deep_gemm = bool(args.use_deep_gemm)
    ray.init(address=None,
                 ignore_reinit_error=True,
                 num_gpus=args.num_gpus)
    num_gpus = int(ray.available_resources()["GPU"])
    workers = [BenchmarkWorker.remote(args.seed, i) for i in range(num_gpus)]
    def _distribute(method: str, inputs: list[Any]) -> list[Any]:
        outputs = []
        worker_idx = 0
        for input_args in inputs:
            worker = workers[worker_idx]
            worker_method = getattr(worker, method)
            output = worker_method.remote(*input_args)
            outputs.append(output)
            worker_idx = (worker_idx + 1) % num_gpus
        return ray.get(outputs)

    if args.tune:

        print(f"Start tuning ...")
        start = time.time()
        configs = _distribute(
            "tune", [(batch_size, E, shard_intermediate_size, hidden_size,
                      topk, dtype, use_fp8_w8a8, use_int8_w8a16, use_int8_w4a8,
                      block_quant_shape, use_deep_gemm)
                     for batch_size in batch_sizes])
        
        best_configs_first_stage = {}
        best_configs_second_stage = {}

        for M, config in zip(batch_sizes, configs):
            first_stage_config, second_stage_config = config
            best_configs_first_stage[M] = first_stage_config
            best_configs_second_stage[M] = second_stage_config          
            
        save_configs(best_configs_first_stage, best_configs_second_stage, E, shard_intermediate_size, hidden_size,
                     topk, dtype, use_fp8_w8a8, use_int8_w8a16, use_int8_w4a8,
                     block_quant_shape)
        end = time.time()
        print(f"Tuning took {end - start:.2f} seconds")
    else:
        outputs = _distribute(
            "benchmark",
            [(batch_size, E, shard_intermediate_size, hidden_size, topk, dtype,
              use_fp8_w8a8, use_int8_w8a16, use_int8_w4a8, block_quant_shape, use_deep_gemm)
             for batch_size in batch_sizes])

        _w13n=[]
        _w13k=[]
        
        _w2n=[]
        _w2k=[]
        _m=[]
        _total_time=[]
        
        for batch_size, kernel_time in zip(batch_sizes, outputs):
            _w13n.append(shard_intermediate_size)
            _w13k.append(hidden_size//2)
            _w2n.append(hidden_size)
            _w2k.append(shard_intermediate_size//4)
            _m.append(batch_size)
            _total_time.append(kernel_time)
            print(f"Batch size: {batch_size}, Kernel time: {kernel_time:.2f} us")
        
        df = pd.DataFrame({'m':_m,'w13n':_w13n,'w13k':_w13k,'w2n':_w2n,'w2k':_w2k,'moe 量化算子总耗时(us)':_total_time})
        # 将 DataFrame 写入 Excel 文件
        df.to_excel('moegemmoutput.xlsx', index=False)


if __name__ == "__main__":
    parser = FlexibleArgumentParser()
    parser.add_argument("--model",
                        type=str,
                        default="/public/opendas/DL_DATA/llm-models/deepseek-r1/DeepSeek-R1-INT8")
    parser.add_argument("--tp-size",
                        "-tp",
                        "--tensor-parallel-size",
                        type=int,
                        default=32)
    parser.add_argument("--dtype",
                        type=str,
                        choices=["auto", "fp8_w8a8", "int8_w8a16", "int8_w8a8", "int8_w4a8"],
                        default="int8_w4a8")
    parser.add_argument("--use-deep-gemm", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, required=False)
    parser.add_argument("--tune", action="store_true")
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--correctness", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true",default=True)
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--moe-ep-size", "-ep", type=int, default=1)
    args = parser.parse_args()

    main(args)

    # 单独执行计算进行排错
    # benchmark_config_profile(49152, 256, 256, 7168, 8, torch.bfloat16, use_fp8_w8a8=False, use_int8_w8a16=False, use_int8_w4a8=True, num_iters=10, block_quant_shape=[128,128])

    # num_tokens = 256
    # num_experts = 256
    # shard_intermediate_size = 512
    # hidden_size = 7168
    # topk = 8
    # dtype = torch.bfloat16
    # use_fp8_w8a8 = False
    # use_int8_w8a16 = False
    # use_int8_w4a8 = True
    # block_quant_shape = None

    # kernel_time = moe_cuda_graph(num_tokens, num_experts,
    #                             shard_intermediate_size, hidden_size,
    #                             topk, dtype, use_fp8_w8a8,
    #                             use_int8_w8a16, use_int8_w4a8, block_quant_shape=block_quant_shape, seed=0, device_id=0)