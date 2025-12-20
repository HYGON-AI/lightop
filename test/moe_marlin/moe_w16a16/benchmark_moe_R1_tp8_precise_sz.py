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
from fused_marlin_moe_valid import fused_marlin_moe_valid, fused_triton_moe

from vllm.model_executor.layers.fused_moe.fused_moe import fused_topk

from vllm.platforms import current_platform
from vllm.utils import FlexibleArgumentParser
from vllm.scalar_type import scalar_types

FP8_DTYPE = torch.float8_e4m3fnuz if current_platform.is_rocm(
) else torch.float8_e4m3fn


BLOCK_M_TOTAL = 80

def _try_parse_layer_index(layer_name: str) -> Optional[int]:
    # Typical: "model.layers.37.mlp.experts"
    try:
        prefix = "model.layers."
        if not layer_name.startswith(prefix):
            return None
        rest = layer_name[len(prefix):]
        return int(rest.split(".", 1)[0])
    except Exception:
        return None


def _resolve_router_logits_pt(
    source: Optional[str],
    num_tokens: int,
    *,
    strict: bool = False,
) -> Optional[str]:
    if not source:
        return None
    source = os.path.expanduser(source)
    if os.path.isdir(source):
        # Convention: bs256.pt / bs390.pt
        cand = os.path.join(source, f"bs{num_tokens}.pt")
        if os.path.exists(cand):
            return cand
        if strict:
            raise FileNotFoundError(
                f"router_logits source is a dir but {cand} not found.")
        return None
    if os.path.exists(source):
        return source
    if strict:
        raise FileNotFoundError(f"router_logits pt not found: {source}")
    return None


def _load_router_logits_samples(
    router_logits_source: Optional[str],
    num_tokens: int,
    num_experts: int,
    device: torch.device,
    *,
    layer: Optional[str] = None,
    dtype: torch.dtype = torch.float32,
    strict: bool = False,
) -> Optional[List[torch.Tensor]]:
    pt_path = _resolve_router_logits_pt(router_logits_source,
                                        num_tokens,
                                        strict=strict)
    if pt_path is None:
        return None

    payload = torch.load(pt_path, map_location="cpu")
    if not isinstance(payload, dict) or "layers" not in payload:
        raise ValueError(
            f"Unexpected router_logits .pt format: {pt_path} (missing 'layers')"
        )
    layers = payload["layers"]
    if not isinstance(layers, dict) or not layers:
        raise ValueError(
            f"Unexpected router_logits .pt format: {pt_path} ('layers' empty)")

    layer_names = list(layers.keys())
    layer_names.sort(
        key=lambda n: (_try_parse_layer_index(n) is None, _try_parse_layer_index(n) or 0, n)
    )

    selected_layer_names: List[str]
    if layer is None or str(layer).strip() == "":
        selected_layer_names = layer_names
    else:
        layer_s = str(layer).strip()
        if layer_s.isdigit():
            want_idx = int(layer_s)
            selected_layer_names = [
                n for n in layer_names if _try_parse_layer_index(n) == want_idx
            ]
            if not selected_layer_names:
                raise ValueError(
                    f"Layer index {want_idx} not found in {pt_path}")
        else:
            if layer_s not in layers:
                raise ValueError(
                    f"Layer name {layer_s} not found in {pt_path}. "
                    f"Available example: {layer_names[0]}")
            selected_layer_names = [layer_s]

    samples: List[torch.Tensor] = []
    for layer_name in selected_layer_names:
        state = layers[layer_name]
        if not isinstance(state, dict):
            continue
        chunks = state.get("router_logits_chunks", None)
        if chunks is None:
            continue
        if not isinstance(chunks, list) or len(chunks) == 0:
            continue

        logits = torch.cat(chunks, dim=0)
        if logits.dim() != 2:
            continue
        if logits.shape[0] != num_tokens or logits.shape[1] != num_experts:
            continue

        samples.append(logits.to(device=device, dtype=dtype, non_blocking=True))

    if not samples:
        raise ValueError(
            f"No router_logits samples matched shape [{num_tokens}, {num_experts}] in {pt_path}"
        )
    print(
        f"[router_logits] loaded {len(samples)} sample(s) from {pt_path} (layer={layer or 'ALL'})"
    )
    return samples


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
    topk: int,
    dtype: Optional[str],
    router_logits_source: Optional[str] = None,
    router_logits_layer: Optional[str] = None,
    router_topk_impl: str = "auto",
    router_logits_strict: bool = False,
    use_nn_moe: bool = False,
) -> float:

    print("m = ", num_tokens, " 性能测试......",dtype)
    # print("num_experts: ", num_experts)
    # print("shard_intermediate_size: ", shard_intermediate_size)
    # print("hidden_size: ", hidden_size)
    # print("topk: ", topk)

    torch.manual_seed(0)

    # 随机100次 专家选择
    num_iters = 100


    #  =========================================== 准备 gemm1 相关数据 =============================================
    x = torch.randn((num_tokens, hidden_size),device='cuda',dtype=dtype) / 10 # [256*8, 7168] (bf16)
    # torch.set_printoptions(profile="full")       
    # x = torch.ones_like(x)
    w1 = torch.randn((num_experts, shard_intermediate_size, hidden_size), device=x.device, dtype=dtype)
    # w1 = torch.ones_like(w1)            
    torch.set_printoptions(profile="default")
    # -------- 处理 gemm1 权重矩阵 ------------------
    w1_marlin_list = []
    for i in range(w1.shape[0]):
        w1_marlin_in = w16a16_marlin_weight(w1[i])
        w1_marlin_list.append(w1_marlin_in)
    w1_marlin = torch.stack(w1_marlin_list, dim=0)
    # print("gemm1 权重重排完成 重排前size为：重排后size为：",w1.size(),w1_marlin.size())

    #  =========================================== 准备 gemm2 相关数据 =============================================
    w2 = torch.randn((num_experts, hidden_size, shard_intermediate_size // 2), device=x.device, dtype=dtype)
    w2_marlin_list = []
    for i in range(w2.shape[0]):
        w2_marlin_in = w16a16_marlin_weight(w2[i])
        w2_marlin_list.append(w2_marlin_in)
    w2_marlin = torch.stack(w2_marlin_list, dim=0)
    # print("gemm2 权重重排完成 重排前size为：重排后size为：",w2.size(),w2_marlin.size())
    # ============================================ 准备 gating_output 和 score ======================================
    score = torch.randn((num_tokens, num_experts), device="cuda", dtype=torch.float32)
    # w2 = torch.ones_like(w2)

    router_logits_samples = _load_router_logits_samples(
        router_logits_source,
        num_tokens,
        num_experts,
        score.device,
        layer=router_logits_layer,
        dtype=torch.float32,
        strict=router_logits_strict,
    )
    if router_logits_samples is None:
        gating_output = torch.randn(num_iters, num_tokens, num_experts, dtype=torch.float32)

        def prepare(i: int):
            score.copy_(gating_output[i])
    else:

        def prepare(i: int):
            score.copy_(router_logits_samples[i % len(router_logits_samples)])

    def select_topk() -> Tuple[torch.Tensor, torch.Tensor]:
        if router_topk_impl == "auto":
            # Keep routing/topk implementation consistent across
            # random logits and dataset-captured logits.
            impl = "fused"
        else:
            impl = router_topk_impl

        if impl == "fused":
            topk_weights, topk_ids, _ = fused_topk(x, score, topk, True)
            return topk_weights, topk_ids
        if impl == "grouped":
            return grouped_topk(x, score, topk, True, 8, 4)
        raise ValueError(
            f"Unknown router_topk_impl={router_topk_impl} (expected: auto|fused|grouped)"
        )

    topk_weights, topk_ids = select_topk()

    _, _, _, _, triton_gemm1_input, triton_gemm2_input, triton_gemm1_output, triton_gemm2_output = fused_marlin_moe_valid(                         
                                                                                                                            x,
                                                                                                                            w1,
                                                                                                                            w2,
                                                                                                                            w1_marlin,
                                                                                                                            w2_marlin,
                                                                                                                            score,
                                                                                                                            topk_weights,
                                                                                                                            topk_ids,
                                                                                                                            debug_status=True
                                                                                                                            )
    # return
    # 预热
    for i in range(10):
        # 使用新的专家得分
        prepare(i)
        # 获取新专家得分的 算子时间
        topk_weights, topk_ids = select_topk()
        for j in range(10): # 一次随机专家运行10次
            fused_marlin_moe(
                            x,
                            w1,
                            w2,
                            w1_marlin,
                            w2_marlin,
                            score,
                            topk_weights,
                            topk_ids
                            )
            fused_triton_moe(
                            triton_gemm1_input,
                            triton_gemm2_input,
                            w1,
                            w2,
                            w1_marlin,
                            w2_marlin,
                            score,
                            topk_weights,
                            topk_ids,
                            use_nn_moe=use_nn_moe)
            

    gemm1_time_us_total = 0 
    gemm2_time_us_total = 0
    gemm1_time_us_total_triton = 0 
    gemm2_time_us_total_triton = 0
    kernel_num_iters = 1
    for i in range(num_iters):
        # 使用新的专家得分
        prepare(i)
        # 获取新专家得分的 算子时间
        topk_weights, topk_ids = select_topk()
        GEMM1_Peak, GEMM2_Peak, bandwith1, bandwith2, triton_gemm1_input, triton_gemm2_input, triton_gemm1_output, triton_gemm2_output = fused_marlin_moe_valid(                         
                                                                                                                                                                x,
                                                                                                                                                                w1,
                                                                                                                                                                w2,
                                                                                                                                                                w1_marlin,
                                                                                                                                                                w2_marlin,
                                                                                                                                                                score,
                                                                                                                                                                topk_weights,
                                                                                                                                                                topk_ids
                                                                                                                                                                )
        triton_opt_gemm1_output = None
        triton_opt_gemm2_output = None  
        for j in range(kernel_num_iters): # 一次随机专家运行kernel_num_iters次
            gemm1_time_us, gemm2_time_us =  fused_marlin_moe(
                                                            x,
                                                            w1,
                                                            w2,
                                                            w1_marlin,
                                                            w2_marlin,
                                                            score,
                                                            topk_weights,
                                                            topk_ids
                                                            )
            gemm1_time_us_triton, gemm2_time_us_triton, triton_opt_gemm1_output, triton_opt_gemm2_output =  fused_triton_moe(
                                                                                                                                triton_gemm1_input,
                                                                                                                                triton_gemm2_input,
                                                                                                                                w1,
                                                                                                                                w2,
                                                                                                                                w1_marlin,
                                                                                                                                w2_marlin,
                                                                                                                                score,
                                                                                                                                topk_weights,
                                                                                                                                topk_ids,
                                                                                                                                use_nn_moe=use_nn_moe
                                                                                                                                )
            
            gemm1_time_us_total += gemm1_time_us
            gemm2_time_us_total += gemm2_time_us
            gemm1_time_us_total_triton += gemm1_time_us_triton
            gemm2_time_us_total_triton += gemm2_time_us_triton
        # import pdb
        # pdb.set_trace()
        torch.testing.assert_close(triton_opt_gemm1_output, triton_gemm1_output, atol=1e-2, rtol=1e-2)
        # print("do bench kernel triton_opt对比triton gemm1 success!!!!")
        torch.testing.assert_close(triton_opt_gemm2_output, triton_gemm2_output, atol=1e-2, rtol=1e-2)
        # print("do bench kernel triton_opt对比triton gemm2 success!!!!")

    gemm1_time_us_total_avg = gemm1_time_us_total/num_iters/kernel_num_iters
    gemm2_time_us_total_avg = gemm2_time_us_total/num_iters/kernel_num_iters
    gemm1_time_us_total_avg_triton = gemm1_time_us_total_triton/num_iters/kernel_num_iters
    gemm2_time_us_total_avg_triton = gemm2_time_us_total_triton/num_iters/kernel_num_iters
    
    print("      GEMM1_Time_us: ", gemm1_time_us_total_avg, "GEMM2_Time_us: ", gemm2_time_us_total_avg)
    print("      GEMM1_Time_us_triton: ", gemm1_time_us_total_avg_triton, "GEMM2_Time_us_triton: ", gemm2_time_us_total_avg_triton)
    print("GEMM1峰值:",GEMM1_Peak/(gemm1_time_us_total_avg /1000/1000)/10**12)
    print("GEMM2峰值:",GEMM2_Peak/(gemm2_time_us_total_avg /1000/1000)/10**12)
    print("GEMM1+GEMM2总峰值:",GEMM1_Peak/(gemm1_time_us_total_avg /1000/1000)/10**12 + GEMM2_Peak/(gemm2_time_us_total_avg /1000/1000)/10**12)
    print("GEMM1带宽GB/s:",bandwith1/(gemm1_time_us_total_avg/1000/1000))
    print("GEMM2带宽GB/s:",bandwith2/(gemm2_time_us_total_avg/1000/1000))

    gemm1_gemm2_time_us_total_avg = gemm1_time_us_total_avg + gemm2_time_us_total_avg
    gemm1_gemm2_time_us_total_avg_triton = gemm1_time_us_total_avg_triton + gemm2_time_us_total_avg_triton
    gemm1_peak = GEMM1_Peak/(gemm1_time_us_total_avg /1000/1000)/10**12
    gemm2_peak = GEMM2_Peak/(gemm2_time_us_total_avg /1000/1000)/10**12
    gemm1_bandwidth = bandwith1/(gemm1_time_us_total_avg/1000/1000)
    gemm2_bandwidth = bandwith2/(gemm2_time_us_total_avg/1000/1000)
    
    return {
        'M': num_tokens,
        'N1': shard_intermediate_size,
        'K1': hidden_size,
        'N2': hidden_size,
        'K2': shard_intermediate_size // 2,
        'E': num_experts,
        'TopK': topk,
        'GEMM1_Time_us_triton': gemm1_time_us_total_avg_triton,
        'GEMM2_Time_us_triton': gemm2_time_us_total_avg_triton,
        'GEMM1+GEMM2_us_triton': gemm1_gemm2_time_us_total_avg_triton,
        'GEMM1_Time_us': gemm1_time_us_total_avg,
        'GEMM2_Time_us': gemm2_time_us_total_avg,
        'GEMM1+GEMM2_us': gemm1_gemm2_time_us_total_avg,
        'GEMM1_TFlops': gemm1_peak,
        'GEMM2_TFlops': gemm2_peak,
        'GEMM1_GB/s': gemm1_bandwidth,
        'GEMM2_GB/s': gemm2_bandwidth,
    }



@ray.remote(num_gpus=1)
class BenchmarkWorker:

    def __init__(
        self,
        seed: int,
        router_logits_source: Optional[str] = None,
        router_logits_layer: Optional[str] = None,
        router_topk_impl: str = "auto",
        router_logits_strict: bool = False,
        use_nn_moe: bool = False,
    ) -> None:
        torch.set_default_device("cuda")
        self.seed = seed
        self.router_logits_source = router_logits_source
        self.router_logits_layer = router_logits_layer
        self.router_topk_impl = router_topk_impl
        self.router_logits_strict = router_logits_strict
        self.use_nn_moe = use_nn_moe
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
        topk: int,
        dtype: Optional[str],
    ) -> Tuple[Dict[str, int], float]:
        # 测试                                         
        result = benchmark_config(
                                num_tokens, 
                                num_experts,
                                shard_intermediate_size, 
                                hidden_size,
                                topk,
                                dtype,
                                router_logits_source=self.router_logits_source,
                                router_logits_layer=self.router_logits_layer,
                                router_topk_impl=self.router_topk_impl,
                                router_logits_strict=self.router_logits_strict,
                                use_nn_moe=self.use_nn_moe)
        return result

def main(args: argparse.Namespace):
    
    #qwen3-30b-A3B-tp2
    num_experts = 128
    shard_intermediate_size=768
    hidden_size=2048
    
    # # #qwen3-235b-A22B-tp8
    # num_experts = 128
    # shard_intermediate_size=384
    # hidden_size=4096
    
    # # #qwen3-235b-A22B-tp16
    # num_experts = 128
    # shard_intermediate_size=192
    # hidden_size=4096
    
    # # #qwen3-480b-A35B-tp8
    # num_experts = 160
    # shard_intermediate_size=640
    # hidden_size=6144
    
    # # #qwen3-480b-A35B-tp16
    # num_experts = 160
    # shard_intermediate_size=320
    # hidden_size=6144
    
    topk = 8
    renormalize=True
    dtype = torch.float16
    # dtype = torch.bfloat16
    if len(args.batch_size) == 0:
        print("进行全size性能测试: ")
        # batch_sizes = [256]
        # batch_sizes = [512, 1024, 2048, 4096, 6144, 8192, 16384]
        batch_sizes = [1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128, 256, 512, 1024, 1536, 2048, 3072, 4096] 
        # batch_sizes = [ 256, 390] 
        # batch_sizes = [ 390] 
        # batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128] 
        # batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128]
        # batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 6144, 8192, 16384]
    else:
        batch_sizes = args.batch_size

    ray.init(address=None,
                 ignore_reinit_error=True,
                 num_gpus=1, local_mode = True) # local_mode = True用于debug
    num_gpus = int(ray.available_resources()["GPU"])
    workers = [
        BenchmarkWorker.remote(
            args.seed,
            args.router_logits_source,
            args.router_logits_layer,
            args.router_topk_impl,
            args.router_logits_strict,
            args.nn_moe,
        ) for _ in range(num_gpus)
    ]

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
      results = _distribute(
                "benchmark", [(batch_size, num_experts, shard_intermediate_size, hidden_size,
                            topk,dtype) for batch_size in batch_sizes])
      output_file = "moe_w16a16_gemm_channel_marlin_wise.xlsx"
      df = pd.DataFrame(results)
      df.to_excel(output_file, index=False) 

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
    parser.add_argument("--batch-size", type=int, nargs="*", default=[])
    parser.add_argument("--tune", action="store_true",default=False)
    parser.add_argument("--nn-moe", action="store_true", default=False,
                        dest="nn_moe",
                        help="Use nn-moe layout (default: False, uses tn-moe)")
    parser.add_argument(
        "--router-logits-source",
        type=str,
        default="",
        help=
        "Path to router_logits .pt file, or a dir containing bs{num_tokens}.pt (from router_capture.py)."
    )
    parser.add_argument(
        "--router-logits-layer",
        type=str,
        default="",
        help=
        "Layer selector for router_logits: empty=all layers (cycled), digit=layer index, otherwise full layer name."
    )
    parser.add_argument(
        "--router-topk-impl",
        type=str,
        choices=["auto", "fused", "grouped"],
        default="fused",
        help=
        "Top-k selection implementation: fused (vllm fused_topk), grouped (this script grouped_topk), auto (same as fused)."
    )
    parser.add_argument(
        "--router-logits-strict",
        action="store_true",
        help=
        "If set, missing bs{num_tokens}.pt (when source is a dir) raises instead of falling back to random."
    )
    parser.add_argument("--trust-remote-code", action="store_true", default=True)
    args = parser.parse_args()

    main(args)

    # benchmark_config(4096, 256, 512, 7168, 8)
    # 默认随机数测试
    # HIP_VISIBLE_DEVICES=1 python3 benchmark_moe_R1_tp8_precise_sz.py --nn-moe
    # 如果有真实数据集，可以用真实数据集测试
    # HIP_VISIBLE_DEVICES=1 python3 benchmark_moe_R1_tp8_precise_sz.py -tp 2 --router-logits-source /home/data/qwen3-moe/logist/ --nn-moe
