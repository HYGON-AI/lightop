# SPDX-License-Identifier: Apache-2.0

import argparse
import time
from datetime import datetime
from itertools import product
from typing import Any, Dict, List, Tuple, TypedDict
import numpy as np
import ray
import torch
torch.set_printoptions(precision=4, profile="default", sci_mode=False)
import triton
from ray.experimental.tqdm_ray import tqdm
from transformers import AutoConfig
from fused_time_moe_triton import do_bench_fused_time_moe_triton
# from vllm.model_executor.layers.fused_moe.fused_moe import *
# from fused_moe import *
# from fused_moe_3 import *
from typing import Optional
import os
import pandas as pd
import triton
import triton.language as tl

from marlin_quant import w16a16_marlin_weight
# 引入适配后的 marlin fused_moe实现
# from fused_marlin_moe_chunk_size import fused_marlin_moe
from fused_marlin_moe import fused_marlin_moe
from fused_marlin_moe_valid import fused_marlin_moe_valid

from vllm.model_executor.layers.fused_moe.fused_moe import fused_topk

from vllm.platforms import current_platform
from vllm.utils import FlexibleArgumentParser
from vllm.scalar_type import scalar_types

FP8_DTYPE = torch.float8_e4m3fnuz if current_platform.is_rocm(
) else torch.float8_e4m3fn


BLOCK_M_TOTAL = 80

def grouped_topk(hidden_states: torch.Tensor,
                 gating_output: torch.Tensor,
                 topk: int,
                 renormalize: bool,
                 num_expert_group: int = 0,
                 topk_group: int = 0,
                 scoring_func: str = "softmax",
                 e_score_correction_bias: Optional[torch.Tensor] = None):

    assert hidden_states.shape[0] == gating_output.shape[0], (
        "Number of tokens mismatch")

    if scoring_func == "softmax":
        scores = torch.softmax(gating_output, dim=-1)
    elif scoring_func == "sigmoid":
        scores = gating_output.sigmoid()
    else:
        raise ValueError(f"Unsupported scoring function: {scoring_func}")

    if e_score_correction_bias is not None:
        # Store original scores before applying correction bias. We use biased
        # scores for expert selection but original scores for routing weights
        original_scores = scores
        scores = scores + e_score_correction_bias.unsqueeze(0)

    num_token = scores.shape[0]
    group_scores = scores.view(num_token, num_expert_group,
                               -1).max(dim=-1).values  # [n, n_group]
    group_idx = torch.topk(group_scores, k=topk_group, dim=-1,
                           sorted=False)[1]  # [n, top_k_group]
    group_mask = torch.zeros_like(group_scores)  # [n, n_group]
    group_mask.scatter_(1, group_idx, 1)  # [n, n_group]
    score_mask = group_mask.unsqueeze(-1).expand(
        num_token, num_expert_group,
        scores.shape[-1] // num_expert_group).reshape(num_token, -1)  # [n, e]
    tmp_scores = scores.masked_fill(~score_mask.bool(), 0.0)  # [n, e]

    if e_score_correction_bias is not None:
        topk_ids = torch.topk(tmp_scores, k=topk, dim=-1, sorted=False)[1]
        # Use original unbiased scores for the routing weights
        topk_weights = original_scores.gather(1, topk_ids)
    else:
        topk_weights, topk_ids = torch.topk(tmp_scores,
                                            k=topk,
                                            dim=-1,
                                            sorted=False)

    if renormalize:
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

    return topk_weights.to(torch.float32), topk_ids.to(torch.int32)

def benchmark_config(
    num_tokens: int,
    num_experts: int,
    shard_intermediate_size: int,
    hidden_size: int,
    topk: int
) -> float:

    print("m = ", num_tokens, " 性能测试......")
    # print("num_experts: ", num_experts)
    # print("shard_intermediate_size: ", shard_intermediate_size)
    # print("hidden_size: ", hidden_size)
    # print("topk: ", topk)

    torch.manual_seed(0)

    # 随机100次 专家选择
    num_iters = 100


    #  =========================================== 准备 gemm1 相关数据 =============================================
    x = torch.randn((num_tokens, hidden_size),device='cuda',dtype=torch.bfloat16) / 10 # [256*8, 7168] (bf16)
    # x = torch.ones_like(x)
    w1 = torch.randn((num_experts, shard_intermediate_size, hidden_size), device=x.device, dtype=torch.bfloat16)
    # w1 = torch.ones_like(w1)
    w1_scale = (torch.randn((num_experts, shard_intermediate_size ,1), dtype=torch.float32, device=x.device))
    # print("x",x)
    # print("w1",w1)
    # # torch.set_printoptions(profile="full")           
    # print("w1",w1[37][0])    
    torch.set_printoptions(profile="default")
    #  =========================================== 准备 gemm2 相关数据 =============================================
    w2 = torch.randn((num_experts, hidden_size, shard_intermediate_size // 2), device=x.device, dtype=torch.bfloat16)
    w2_scale = (torch.randn((num_experts, hidden_size, 1), dtype=torch.float32, device=x.device))


    # ============================================ 准备 gating_output 和 score ======================================
    gating_output = torch.randn(num_iters, num_tokens, num_experts, dtype=torch.float32)
    score = torch.randn((num_tokens, num_experts), device="cuda", dtype=torch.float32)

    def prepare(i: int):
        score.copy_(gating_output[i])

    topk_weights, topk_ids = grouped_topk(x, score, topk, True, 8, 4)
    # 先进行准确度测试
    # GEMM1_Peak,GEMM2_Peak,bandwith1,bandwith2 = fused_marlin_moe_valid(   
    fused_marlin_moe_valid(                      
                            x,
                            w1,
                            w2,
                            w1_scale,
                            w2_scale,
                            score,
                            topk_weights,
                            topk_ids
                            )
    # return
    # 预热
    for i in range(10):
        # 使用新的专家得分
        prepare(i)
        # 获取新专家得分的 算子时间
        topk_weights, topk_ids = grouped_topk(x, score, topk, True, 8, 4)
        for j in range(10): # 一次随机专家运行10次
            fused_marlin_moe(
                            x,
                            w1,
                            w2,
                            w1_scale,
                            w2_scale,
                            score,
                            topk_weights,
                            topk_ids
                            )
            

    gemm1_time_us_total = 0 
    gemm2_time_us_total = 0
    for i in range(num_iters):
        # 使用新的专家得分
        prepare(i)
        # 获取新专家得分的 算子时间
        topk_weights, topk_ids = grouped_topk(x, score, topk, True, 8, 4)
        for j in range(10): # 一次随机专家运行10次
            gemm1_time_us, gemm2_time_us =  fused_marlin_moe(
                                                            x,
                                                            w1,
                                                            w2,
                                                            w1_scale,
                                                            w2_scale,
                                                            score,
                                                            topk_weights,
                                                            topk_ids
                                                            )
            
            gemm1_time_us_total += gemm1_time_us
            gemm2_time_us_total += gemm2_time_us

    print("      GEMM1_Time_us: ", gemm1_time_us_total/num_iters/10, "GEMM2_Time_us: ", gemm2_time_us_total/num_iters/10)
    # for i in range(10):
    #     # 使用新的专家得分
    #     prepare(i)
    #     # 获取新专家得分的 算子时间
    #     topk_weights, topk_ids = grouped_topk(x, score, topk, True, 8, 4)
    #     for j in range(10): # 一次随机专家运行10次
    #         fused_time_moe_triton(
    #                         x,
    #                         w1,
    #                         w2,
    #                         w1_scale,
    #                         w2_scale,
    #                         score,
    #                         topk_weights,
    #                         topk_ids
    #                         )
            

    # gemm1_time_us_total = 0 
    # gemm2_time_us_total = 0
    # for i in range(num_iters):
    #     # 使用新的专家得分
    #     prepare(i)
    #     # 获取新专家得分的 算子时间
    #     topk_weights, topk_ids = grouped_topk(x, score, topk, True, 8, 4)
    #     for j in range(10): # 一次随机专家运行10次
    #         gemm1_time_us, gemm2_time_us =  fused_time_moe_triton(
    #                                                         x,
    #                                                         w1,
    #                                                         w2,
    #                                                         w1_scale,
    #                                                         w2_scale,
    #                                                         score,
    #                                                         topk_weights,
    #                                                         topk_ids
    #                                                         )
            
    #         gemm1_time_us_total += gemm1_time_us
    #         gemm2_time_us_total += gemm2_time_us

    # print("      GEMM3_Time_us: ", gemm1_time_us_total/num_iters/10, "GEMM4_Time_us: ", gemm2_time_us_total/num_iters/10)
    


    # return {
    #     'M': num_tokens,
    #     'N1': shard_intermediate_size,
    #     'K1': hidden_size//2,
    #     'N2': hidden_size,
    #     'K2': shard_intermediate_size // 4,
    #     'E': num_experts,
    #     'TopK': topk,
    #     'GEMM1_Time_us': gemm1_time_us_total/num_iters/10,
    #     'GEMM2_Time_us': gemm2_time_us_total/num_iters/10,
    # }


    # add triton tunning 
    # for i in range(10):
    #     # 使用新的专家得分
    #     prepare(i)
    #     # 获取新专家得分的 算子时间
    #     topk_weights, topk_ids = grouped_topk(x, score, topk, True, 8, 4)
    #     for j in range(10): # 一次随机专家运行10次
    #         do_bench_fused_time_moe_triton(
    #                         x,
    #                         w1,
    #                         w2,
    #                         w1_scale,
    #                         w2_scale,
    #                         score,
    #                         topk_weights,
    #                         topk_ids
    #                         )
            

    # gemm1_time_us_total = 0 
    # gemm2_time_us_total = 0
    # for i in range(num_iters):
    #     # 使用新的专家得分
    #     prepare(i)
    #     # 获取新专家得分的 算子时间
    #     topk_weights, topk_ids = grouped_topk(x, score, topk, True, 8, 4)
    #     for j in range(10): # 一次随机专家运行10次
    #         gemm1_time_us, gemm2_time_us =  do_bench_fused_time_moe_triton(
    #                                                         x,
    #                                                         w1,
    #                                                         w2,
    #                                                         w1_scale,
    #                                                         w2_scale,
    #                                                         score,
    #                                                         topk_weights,
    #                                                         topk_ids
    #                                                         )
            
    #         gemm1_time_us_total += gemm1_time_us
    #         gemm2_time_us_total += gemm2_time_us

    # print("      GEMM3_Time_us: ", gemm1_time_us_total/num_iters/10, "GEMM4_Time_us: ", gemm2_time_us_total/num_iters/10)
    


    # return {
    #     'M': num_tokens,
    #     'N1': shard_intermediate_size,
    #     'K1': hidden_size//2,
    #     'N2': hidden_size,
    #     'K2': shard_intermediate_size // 4,
    #     'E': num_experts,
    #     'TopK': topk,
    #     'GEMM1_Time_us': gemm1_time_us_total/num_iters/10,
    #     'GEMM2_Time_us': gemm2_time_us_total/num_iters/10,
    # }


@ray.remote(num_gpus=1)
class BenchmarkWorker:

    def __init__(self, seed: int) -> None:
        torch.set_default_device("cuda")
        self.seed = seed
        # Get the device ID to allocate tensors and kernels
        # on the respective GPU. This is required for Ray to work
        # correctly with multi-GPU tuning on the ROCm platform.
        # self.device_id = int(ray.get_gpu_ids()[0])
        gpu_ids = ray.get_gpu_ids()
        if len(gpu_ids) == 0:
            self.device_id = 0  # 或者你想用的 GPU id
        else:
            self.device_id = int(gpu_ids[0])        

    def benchmark(
        self,
        num_tokens: int,
        num_experts: int,
        shard_intermediate_size: int,
        hidden_size: int,
        topk: int
    ) -> Tuple[Dict[str, int], float]:
        # 测试                                         
        result = benchmark_config(
                                num_tokens, 
                                num_experts,
                                shard_intermediate_size, 
                                hidden_size,
                                topk)
        return result

def main(args: argparse.Namespace):
    
    num_experts = 128
    shard_intermediate_size=768
    hidden_size=2048
    renormalize=True
    topk = 8

    if len(args.batch_size) == 0:
        print("进行全size性能测试: ")
        # batch_sizes = [2048]
        # batch_sizes = [512, 1024, 2048, 4096, 6144, 8192, 16384]
        # batch_sizes = [2048]
        # batch_sizes = [512, 1024, 2048, 4096, 6144, 8192, 16384]
        batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128]
        # batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 6144, 8192, 16384]
    else:
        batch_sizes = args.batch_size

    ray.init(address=None,
                 ignore_reinit_error=True,
                 num_gpus=1, local_mode = True) # local_mode = True用于debug
    num_gpus = int(ray.available_resources()["GPU"])
    workers = [BenchmarkWorker.remote(args.seed) for _ in range(num_gpus)]

    def _distribute(method: str, inputs: List[Any]) -> List[Any]:
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
        assert False
    else:
       _distribute(
            "benchmark", [(batch_size, num_experts, shard_intermediate_size, hidden_size,
                           topk) for batch_size in batch_sizes])
        # print(outputs)

        # df = pd.DataFrame(outputs)

        # # 将数据写入 Excel 文件
        # try:
        #     df.to_excel('./marlin_prof.xlsx', index=False)
        #     print("数据已成功写入到 marlin_prof.xlsx 文件")
        # except Exception as e:
        #     print(f"写入文件时出错: {e}")    



if __name__ == "__main__":
    parser = FlexibleArgumentParser()
    parser.add_argument("--model",
                        type=str,
                        default="./DeepSeek-R1-AWQ")
    parser.add_argument("--tp-size",
                        "-tp",
                        "--tensor-parallel-size",
                        type=int,
                        default=8)
    parser.add_argument("--dtype",
                        type=str,
                        choices=["auto", "fp8_w8a8", "int8_w8a16", "int4_w4a16"],
                        default="int4_w4a16")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=list, default=[])
    parser.add_argument("--tune", action="store_true",default=False)
    parser.add_argument("--nn_moe", type=bool, default=False)
    parser.add_argument("--trust-remote-code", action="store_true", default=True)
    args = parser.parse_args()

    main(args)

    # benchmark_config(4096, 256, 512, 7168, 8)
