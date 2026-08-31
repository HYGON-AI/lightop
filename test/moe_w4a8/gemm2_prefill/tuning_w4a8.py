import torch
import time
# from lightop._lmslim_native.quantize import quant_ops
from typing import Optional, Type, Any, Callable, Dict, List, Tuple
import pandas as pd
import os
import json
import triton
import triton.language as tl
import pandas as pd
import logging
import math
import torch.nn.functional as F
# from per_token_group_test import per_token_group_quant_int8
import functools
import logging
from vllm.platforms import current_platform
import vllm.envs as envs
from vllm.logger import init_logger
from vllm import _custom_ops as ops
# import _custom_ops as ops
from lightop.moe import moe_gemm_w8a8, get_moe_cuda_config
from vllm.model_executor.layers.fused_moe.fused_moe import moe_kernel_prepare_input
from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
    moe_align_block_size)

from triton_w4a8 import invoke_fused_moe_kernel_w4a8
import pandas as pd

from lightop.moe import moe_gemm_w4a8

blockM_configs = {
    0 : 16,
    1 : 32,
    2 : 64,

    3 : 16,
    4 : 32,
    5 : 16,

    6 : 32,
    7 : 32,
    8 : 64,
    9 : 64,

    10: 64,

    11: 64,
    12: 32,

    13: 128,
}


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

def tuning_gemm2():
    torch.manual_seed(0)
    num_experts = 256

    
    shard_intermediate_size=512
    hidden_size=7168
    renormalize=True
    topk = 8     

    for num_tokens in [256, 512, 1024]:
        print("=========== num_tokens ", num_tokens, " ==================")
        x = torch.randn((num_tokens, hidden_size),device='cuda',dtype=torch.bfloat16) / 10 # [256*8, 7168] (bf16)
        x_2 = torch.randn((num_tokens * topk, shard_intermediate_size//2),device='cuda',dtype=torch.bfloat16)  # [256*8, 7168] (bf16)

        
        # w2: [256, 7168, 512//2//2] = [256, 7168, 128] int8
        w2 = torch.randint(-127, 127, (num_experts, hidden_size, shard_intermediate_size // 4), device=x_2.device, dtype=torch.int8)
        # w1_scale: [256, 7168, 1]
        w2_scale = (torch.randn((num_experts, hidden_size, 1), dtype=torch.float32, device=x_2.device))   

        gating_output = torch.randn((num_tokens, num_experts),device=x_2.device,dtype=torch.float32)

        topk_weights, topk_ids = grouped_topk(
            hidden_states=x,
            gating_output=gating_output,
            topk=topk,
            renormalize=renormalize,
            num_expert_group=8,
            topk_group=4,
        )

    
        qcurr_hidden_states, qa2_scale = moe_kernel_prepare_input(
            A=x_2,
            B=w2,
            A_scale=None,
            B_scale=w2_scale,
            use_fp8_w8a8=None,
            use_int8_w8a8=True,
            use_int8_w8a16=None,
            use_int4_w4a16=None,
            per_channel_quant=True,
            block_shape=None)

        # 传入一个用于debug的Tensor
        debug_output = torch.zeros((32, 32),device='cuda',dtype=torch.int8)
        res_output = torch.zeros((256, 16),device='cuda',dtype=torch.int32)
        # 传入int32的Tensor，用来存储 mmac 的计算结果，判断
        bf16_output = torch.zeros((32, 128),device='cuda',dtype=torch.bfloat16)
        # 传入int32的Tensor，用来存储 mmac 的计算结果，判断

        for block_m in [16, 32, 64, 128]:
            print("=========== block_m ", block_m, " ==================")
            sorted_token_ids, expert_ids, num_tokens_post_padded = (moe_align_block_size(topk_ids, block_m, num_experts))

            best_time = 100000
            best_mode = 0

            for mode_idx in range(len(blockM_configs)):
                
                mode_blcok_m = blockM_configs[mode_idx]
                if mode_blcok_m != block_m:
                    continue
                # =====================================  cuda 版本实现  ======================================================
                gemm2_output = torch.zeros((num_tokens, topk, hidden_size), device=x_2.device, dtype=x_2.dtype)
                config_lightop = {                
                    'DELTA': 1
                }
                config_lightop['MODE'] = mode_idx
                moe_gemm_w4a8(
                    qcurr_hidden_states,
                    w2,
                    gemm2_output,
                    qa2_scale,
                    w2_scale,
                    topk_weights,
                    sorted_token_ids,
                    expert_ids,
                    num_tokens_post_padded,
                    1, # gemm2的topk应该置为 1
                    config_lightop
                )

                triton_output_bf16 = torch.zeros((num_tokens, topk, hidden_size), device=x_2.device, dtype=torch.bfloat16)

                config = {"BLOCK_SIZE_M": 32,
                        "BLOCK_SIZE_N": 128,
                        "BLOCK_SIZE_K": 32,
                        "GROUP_SIZE_M": 1
                        }
                config["BLOCK_SIZE_M"] = mode_blcok_m

                compute_type = tl.bfloat16 if x.dtype == torch.bfloat16 else tl.float16
                invoke_fused_moe_kernel_w4a8(
                                            qcurr_hidden_states,
                                            w2,
                                            triton_output_bf16,
                                            qa2_scale,
                                            w2_scale,
                                            None, # w2_zero
                                            topk_weights,
                                            topk_ids,
                                            sorted_token_ids,
                                            expert_ids,
                                            num_tokens_post_padded,
                                            True,  # mul_routed_weight
                                            1, # top_k
                                            config,
                                            compute_type=compute_type,
                                            use_fp8_w8a8=False,
                                            use_int8_w8a16=False,
                                            use_int8_w8a8 = False,
                                            use_int4_w4a8=True,
                                            block_shape=None,
                                )


                if not (torch.allclose(gemm2_output, triton_output_bf16, rtol=1e-2, atol=1e-2)):
                    print("num_tokens: ", num_tokens, "block_m: ", block_m, "mode_idx: ", mode_idx,  "未通过！！！")
                else:
                    # print("num_tokens: ", num_tokens, "block_m: ", block_m, "mode_idx: ", mode_idx,  "通过！！！")
                    # =========================== 统计耗时 ================================================
                    ms_cuda_time = triton.testing.do_bench(lambda: moe_gemm_w4a8(
                                                                                qcurr_hidden_states,
                                                                                w2,
                                                                                gemm2_output,
                                                                                qa2_scale,
                                                                                w2_scale,
                                                                                topk_weights,
                                                                                sorted_token_ids,
                                                                                expert_ids,
                                                                                num_tokens_post_padded,
                                                                                1, # gemm2的topk应该置为 1
                                                                                config_lightop
                                                                            ), warmup=10, rep=100)

                    if(ms_cuda_time < best_time):
                        best_time = ms_cuda_time
                        best_mode = mode_idx

            # 输出各 block_m 的最优 mode     
            print("num_tokens: ", num_tokens, "block_m: ", block_m, "best_mode: ", best_mode)
                    
                
if __name__ == "__main__":
    tuning_gemm2()
