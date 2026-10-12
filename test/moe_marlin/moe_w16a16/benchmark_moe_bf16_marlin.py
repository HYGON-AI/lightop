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
from fuse_moegemm_marlin import fused_moe_impl, fused_moe_first_stage_tune, fused_moe_second_stage_tune, \
                            grouped_topk, blockM_all_configs

from vllm.platforms import current_platform
from vllm.model_executor.layers.fused_moe.router.fused_topk_router import fused_topk

from lightop.quant import (
    per_token_group_quant_int8, per_token_quant_int8)
import math
import torch.nn.functional as F
import pandas as pd
from fused_moe_valid_w16a16_marlin import fused_moe_valid_w16a16_marlin
from marlin_quant import w16a16_marlin_weight

import os
os.environ["TRITON_CACHE_DIR"] = f".triton/cache"

triton_json_dir="./cache/moe"
device_name=torch.cuda.get_device_properties('cuda').name.replace(" ","_")


batch_size_router_logits_list = [256, 390, 512] # 仅支持bs=256;390;512的router_logits用于tuning

def is_supoort_router_logits(batch_size: int, hidden_size: int, experts_num: int) -> bool:

    if batch_size in batch_size_router_logits_list and hidden_size == 2048 and experts_num == 128:
        return True
    return False

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

# For test
def native_per_token_group_quant_int8(x, group_size, eps=1e-10, dtype=torch.int8):
    """Function to perform per-token-group quantization on an input tensor `x` using native torch.

    It converts the tensor values into float8 values and returns the
    quantized tensor along with the scaling factor used for quantization.
    Note that only `torch.float8_e4m3fn` is supported for now.
    """
    assert (
        x.shape[-1] % group_size == 0
    ), "the last dimension of `x` cannot be divisible by `group_size`"
    assert x.is_contiguous(), "`x` is not contiguous"

    iinfo = torch.iinfo(dtype)
    int8_min = iinfo.min
    int8_max = iinfo.max

    x_ = x.reshape(x.numel() // group_size, group_size)
    amax = x_.abs().max(dim=-1, keepdim=True)[0].clamp(min=eps).to(torch.float32)
    x_s = amax / int8_max
    x_q = (x_ / x_s).clamp(min=int8_min, max=int8_max).to(dtype)
    x_q = x_q.reshape(x.shape)
    x_s = x_s.reshape(x.shape[:-1] + (x.shape[-1] // group_size,))

    return x_q, x_s


# For test
def native_w8a8_block_int8_matmul(A, B, As, Bs, block_size, output_dtype=torch.float16):
    """This function performs matrix multiplication with block-wise quantization using native torch.

    It takes two input tensors `A` and `B` with scales `As` and `Bs`.
    The output is returned in the specified `output_dtype`.
    """

    A = A.to(torch.float32)
    B = B.to(torch.float32)
    assert A.shape[-1] == B.shape[-1]
    assert B.ndim == 2 and B.is_contiguous() and Bs.ndim == 2
    assert len(block_size) == 2
    block_n, block_k = block_size[0], block_size[1]
    assert (A.shape[-1] + block_k - 1) // block_k == As.shape[-1]
    assert A.shape[:-1] == As.shape[:-1]

    M = A.numel() // A.shape[-1]
    N, K = B.shape
    origin_C_shape = A.shape[:-1] + (N,)
    A = A.reshape(M, A.shape[-1])
    As = As.reshape(M, As.shape[-1])
    n_tiles = (N + block_n - 1) // block_n
    k_tiles = (K + block_k - 1) // block_k
    assert n_tiles == Bs.shape[0]
    assert k_tiles == Bs.shape[1]

    C_shape = (M, N)
    C = torch.zeros(C_shape, dtype=torch.float32, device=A.device)

    A_tiles = [A[:, i * block_k : min((i + 1) * block_k, K)] for i in range(k_tiles)]
    B_tiles = [
        [
            B[
                j * block_n : min((j + 1) * block_n, N),
                i * block_k : min((i + 1) * block_k, K),
            ]
            for i in range(k_tiles)
        ]
        for j in range(n_tiles)
    ]
    C_tiles = [C[:, j * block_n : min((j + 1) * block_n, N)] for j in range(n_tiles)]
    As_tiles = [As[:, i : i + 1] for i in range(k_tiles)]

    for i in range(k_tiles):
        for j in range(n_tiles):
            a = A_tiles[i]
            b = B_tiles[j][i]
            c = C_tiles[j]
            s = As_tiles[i] * Bs[j][i]
            c[:, :] += torch.matmul(a, b.t()) * s

    C = C.reshape(origin_C_shape).to(output_dtype)
    return C


def native_w8a8_per_token_matmul(A, B, As, Bs, output_dtype=torch.float16):
    """Matrix multiplication function that supports per-token input quantization and per-column weight quantization"""
    A = A.to(torch.float32)
    B = B.to(torch.float32)

    assert A.shape[-1] == B.shape[-1], "Dimension mismatch"
    assert B.ndim == 2 and B.is_contiguous(), "B must be a 2D contiguous tensor"

    # Reshape input
    M = A.numel() // A.shape[-1]
    B = B.t()  # Transpose weight matrix
    N, K = B.shape
    origin_C_shape = A.shape[:-1] + (K,)
    A = A.reshape(M, N)

    # As is per-token [M, 1], Bs is per-column [1, K]
    C = torch.matmul(A, B)  # [M, K]
    C = As * C * Bs.view(1, -1)  # Broadcast per-column scale

    return C.reshape(origin_C_shape).to(output_dtype)

def torch_w8a8_per_column_moe(a, w1, w2, w1_s, w2_s, topk_weight, topk_ids,topk):
    """This function performs fused moe with per-column int8 quantization using native torch."""

    B, D = a.shape
    # Perform per-token quantization
    a_q, a_s = per_token_quant_int8(a)
    # Repeat tokens to match topk
    a_q = a_q.view(B, -1, D).repeat(1, topk, 1).reshape(-1, D)
    # Also repeat the scale
    a_s = a_s.view(B, -1, 1).repeat(1, topk, 1).reshape(-1, 1)  # [B*topk, 1]

    out = torch.zeros(B * topk, w2.shape[1], dtype=a.dtype, device=a.device)

    # Calculate routing
    # score = torch.softmax(score, dim=-1, dtype=torch.float32)
    # topk_weight, topk_ids = torch.topk(score, topk)
    topk_weight = topk_weight.view(-1)
    topk_ids = topk_ids.view(-1)
    # Process each expert
    for i in range(w1.shape[0]):
        mask = topk_ids == i
        if mask.sum():
            # First MLP layer: note that a_s is now per-token
            inter_out = native_w8a8_per_token_matmul(
                a_q[mask], w1[i], a_s[mask], w1_s[i], output_dtype=a.dtype
            )
            d = inter_out.shape[-1] // 2
            act_out = F.silu(inter_out[..., :d]) * inter_out[..., d:]
            # Quantize activation output with per-token
            act_out_q, act_out_s = per_token_quant_int8(act_out)

            # Second MLP layer
            out[mask] = native_w8a8_per_token_matmul(
                act_out_q, w2[i], act_out_s, w2_s[i], output_dtype=a.dtype
            )
    # Apply routing weights and sum
    return (
        out.view(B, -1, w2.shape[1]) * topk_weight.view(B, -1, 1).to(out.dtype)
    ).sum(dim=1)

def _torch_fused_moegemm_baseline(hidden_states: torch.Tensor,
                       w1: torch.Tensor,
                       w2: torch.Tensor,
                       topk_weights: torch.Tensor,
                       topk_ids: torch.Tensor,
                       inplace: bool = False,
                       use_fp8_w8a8: bool = False,
                       use_int8_w8a8: bool = False,
                       use_int8_w8a16: bool = False,
                       use_int4_w4a16: bool = False,
                       w1_scale: Optional[torch.Tensor] = None,
                       w2_scale: Optional[torch.Tensor] = None,
                       w1_zp: Optional[torch.Tensor] = None,
                       w2_zp: Optional[torch.Tensor] = None,
                       a1_scale: Optional[torch.Tensor] = None,
                       a2_scale: Optional[torch.Tensor] = None,
                       block_shape: Optional[List[int]] = None,
                       use_nn_moe: Optional[bool] = False,
                       moe_ep_size: Optional[int] = 1,
                       start_expert: Optional[int] = -1,
                       end_expert: Optional[int] = -1):

    ## ref https://github.com/sgl-project/sglang/blob/main/test/srt/test_block_int8.py
    # out_hidden_states = torch_w8a8_block_int8_moe(hidden_states,
    #                                               w1,
    #                                               w2,
    #                                               w1_s = w1_scale,
    #                                               w2_s = w2_scale,
    #                                               topk = topk_ids.shape[1],
    #                                               block_shape=block_shape,
    #                                               topk_weight=topk_weights,
    #                                               topk_ids=topk_ids)
    out_hidden_states = torch_w8a8_per_column_moe(hidden_states,
                              w1,
                              w2,
                              w1_s = w1_scale,
                              w2_s = w2_scale,
                              topk_weight=topk_weights,
                              topk_ids=topk_ids,
                              topk = topk_ids.shape[1])

    return out_hidden_states

def torch_moe_impl( hidden_states: torch.Tensor,
                    w1: torch.Tensor,
                    w2: torch.Tensor,
                    gating_output: torch.Tensor,
                    topk: int,
                    renormalize: bool,
                    inplace: bool = False,
                    use_grouped_topk: bool = False,
                    num_expert_group: Optional[int] = None,
                    topk_group: Optional[int] = None,
                    custom_routing_function: Optional[Callable] = None,
                    use_fp8_w8a8: bool = False,
                    use_int8_w8a8: bool = False,
                    use_int8_w8a16: bool = False,
                    use_int4_w4a16: bool = False,
                    w1_scale: Optional[torch.Tensor] = None,
                    w2_scale: Optional[torch.Tensor] = None,
                    w1_zp: Optional[torch.Tensor] = None,
                    w2_zp: Optional[torch.Tensor] = None,
                    a1_scale: Optional[torch.Tensor] = None,
                    a2_scale: Optional[torch.Tensor] = None,
                    block_shape: Optional[List[int]] = None,
                    use_nn_moe: Optional[bool] = False,
                    moe_ep_size: Optional[int] = 1,
                    start_expert: Optional[int] = None,
                    end_expert: Optional[int] = None,
                    config:  Dict[str, Any] = None,
) -> torch.Tensor:
    """
    This function computes a Mixture of Experts (MoE) layer using two sets of
    weights, w1 and w2, and top-k gating mechanism.

    Parameters:
    - hidden_states (torch.Tensor): The input tensor to the MoE layer.
    - w1 (torch.Tensor): The first set of expert weights.
    - w2 (torch.Tensor): The second set of expert weights.
    - gating_output (torch.Tensor): The output of the gating operation
        (before softmax).
    - topk (int): The number of top-k experts to select.
    - renormalize (bool): If True, renormalize the top-k weights to sum to 1.
    - inplace (bool): If True, perform the operation in-place.
        Defaults to False.
    - num_expert_group: Optional[int]: additional parameter for grouped_topk
    - topk_group: Optional[int]: additional parameter for grouped_topk
    - use_grouped_topk: If True, use grouped_topk instead of fused_topk
        note: Deepseekv2 model uses grouped_topk
    - use_fp8_w8a8 (bool): If True, use fp8 arithmetic to compute the inner
        products for w1 and w2. Defaults to False.
    - use_int8_w8a16 (bool): If True, use matmul of int8 weight and bf16/fp16
        activation to compute the inner products for w1 and w2.
        Defaults to False.
    - use_int4_w4a16 (bool): If True, use matmul of int4 weight and bf16/fp16
        activation to compute the inner products for w1 and w2.
        Defaults to False.
    - w1_scale (Optional[torch.Tensor]): Optional scale to be used for
        w1.
    - w2_scale (Optional[torch.Tensor]): Optional scale to be used for
        w2.
    - a1_scale (Optional[torch.Tensor]): Optional scale to be used for
        a1.
    - a2_scale (Optional[torch.Tensor]): Optional scale to be used for
        a2.
    - block_shape: (Optional[List[int]]): Optional block size for block-wise
        quantization.

    Returns:
    - torch.Tensor: The output tensor after applying the MoE layer.
    """
    # Check constraints.
    assert gating_output.shape[1] == w1.shape[0], "Number of experts mismatch"

    if use_grouped_topk:
        assert num_expert_group is not None and topk_group is not None
        topk_weights, topk_ids = grouped_topk(hidden_states, gating_output,
                                              topk, renormalize,
                                              num_expert_group, topk_group)
    elif custom_routing_function is None:
        topk_weights, topk_ids = fused_topk(hidden_states, gating_output, topk,
                                            renormalize)
    else:
        topk_weights, topk_ids = custom_routing_function(
            hidden_states, gating_output, topk, renormalize)

    return _torch_fused_moegemm_baseline(hidden_states=hidden_states,
                                            w1=w1,
                                            w2=w2,
                                            topk_weights=topk_weights,
                                            topk_ids=topk_ids,
                                            inplace=False,
                                            use_int8_w8a8=True,
                                            w1_scale=w1_scale,
                                            w2_scale=w2_scale,
                                            block_shape=block_shape)

class BenchmarkConfig(TypedDict):
    BLOCK_SIZE_M: int
    BLOCK_SIZE_N: int
    BLOCK_SIZE_K: int
    GROUP_SIZE_M: int
    num_warps: int
    num_stages: int
    enable_mmacfuse: int


def benchmark_config(config: BenchmarkConfig,
                     num_tokens: int,
                     num_experts: int,
                     shard_intermediate_size: int,
                     hidden_size: int,
                     topk: int,
                     dtype: torch.dtype,
                     use_fp8_w8a8: bool,
                     use_int8_w8a16: bool,
                     use_int8_w8a8: bool,
                     num_iters: int = 100,
                     block_quant_shape: List[int] = None,
                     stages: int=1,
                     router_logits_source: Optional[str] = None,
                     router_logits_layer: Optional[str] = None,
                     router_topk_impl: str = "auto",
                     router_logits_strict: bool = False,) -> float:

    w1_scale = None
    w2_scale = None
    a1_scale = None
    a2_scale = None

    #  =========================================== 准备 gemm1 相关数据 =============================================
    x = torch.randn((num_tokens, hidden_size),device='cuda',dtype=torch.bfloat16) / 10 # [256*8, 7168] (bf16)
    w1 = torch.randn((num_experts, shard_intermediate_size, hidden_size), device=x.device, dtype=torch.bfloat16)
    # -------- 处理 gemm1 权重矩阵 ------------------
    w1_marlin_list = []
    for i in range(w1.shape[0]):
        w1_marlin_in = w16a16_marlin_weight(w1[i])
        w1_marlin_list.append(w1_marlin_in)
    w1_marlin = torch.stack(w1_marlin_list, dim=0)
    w1_scale = (torch.randn((num_experts, shard_intermediate_size ,1), dtype=torch.float32, device=x.device))

    #  =========================================== 准备 gemm2 相关数据 =============================================
    w2 = torch.randn((num_experts, hidden_size, shard_intermediate_size // 2), device=x.device, dtype=torch.bfloat16)
    # -------- 处理 gemm2 权重矩阵 ------------------
    w2_marlin_list = []
    for i in range(w2.shape[0]):
        w2_marlin_in = w16a16_marlin_weight(w2[i])
        w2_marlin_list.append(w2_marlin_in)
    w2_marlin = torch.stack(w2_marlin_list, dim=0)
    w2_scale = (torch.randn((num_experts, hidden_size, 1), dtype=torch.float32, device=x.device))


    # ============================================ 准备 gating_output 和 score ======================================
    score = torch.randn(num_tokens, num_experts, dtype=torch.float32)
    # gating_output = torch.randn(num_iters, num_tokens, num_experts, dtype=torch.float32)

    router_logits_samples = _load_router_logits_samples(
        router_logits_source,
        num_tokens,
        num_experts,
        score.device,
        layer=router_logits_layer,
        dtype=torch.float32,
        strict=router_logits_strict,
    )

    if router_logits_samples is None or not is_supoort_router_logits(num_tokens, hidden_size, num_experts): # 不支持router_logits的bs,hidden_size,experts_num
        gating_output = torch.randn(num_iters, num_tokens, num_experts, dtype=torch.float32)

        def prepare(i: int):
            score.copy_(gating_output[i])
    else:

        def prepare(i: int):
            score.copy_(router_logits_samples[i % len(router_logits_samples)])

    print("Tuing config: ", config)
    latencies: list[float] = []
    for i in range(num_iters):
        prepare(i)
        torch.cuda.synchronize()
        kernel_time = 0.0
        if stages == 1:
            kernel_time = fused_moe_first_stage_tune(x,
                                                    w1,
                                                    w2,
                                                    w1_marlin,
                                                    w2_marlin,
                                                    score,
                                                    topk,
                                                    renormalize=True,
                                                    inplace=True,
                                                    use_fp8_w8a8=use_fp8_w8a8,
                                                    use_int8_w8a16=use_int8_w8a16,
                                                    w1_scale=w1_scale,
                                                    w2_scale=w2_scale,
                                                    a1_scale=a1_scale,
                                                    a2_scale=a2_scale,
                                                    block_shape=block_quant_shape,
                                                    config=config)

        elif stages == 2:
            kernel_time = fused_moe_second_stage_tune(x,
                                                    w1,
                                                    w2,
                                                    w1_marlin,
                                                    w2_marlin,
                                                    score,
                                                    topk,
                                                    renormalize=True,
                                                    inplace=True,
                                                    use_fp8_w8a8=use_fp8_w8a8,
                                                    use_int8_w8a16=use_int8_w8a16,
                                                    w1_scale=w1_scale,
                                                    w2_scale=w2_scale,
                                                    a1_scale=a1_scale,
                                                    a2_scale=a2_scale,
                                                    block_shape=block_quant_shape,
                                                    config=config)
        else:
            assert False

        latencies.append(kernel_time)

    avg = sum(latencies) / (num_iters) * 1000  # us
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
        use_int8_w8a8: bool,
        num_iters: int=100,
        block_quant_shape: List[int] = None,
        seed: int=0,
        device_id: int=0,
    ) -> tuple[dict[str, int], float]:
        from vllm.platforms import current_platform
        current_platform.seed_everything(seed)

        device="cuda:"+str(device_id)
        x = torch.randn(num_tokens, hidden_size, device=device, dtype=dtype)
        w1_scale = None
        w2_scale = None
        a1_scale = None
        a2_scale = None
        # num_iters = 10
        w1 = torch.randint(-127,
                            127, (
                                num_experts,
                                shard_intermediate_size, # n
                                hidden_size, # k
                            ),
                            device=device,
                            dtype=torch.int8)
        w2 = torch.randint(-127,
                            127, (
                                num_experts,
                                hidden_size, # n2
                                shard_intermediate_size // 2, # k2
                            ),
                            device=device,
                            dtype=torch.int8)
        # w1_scale = (torch.randn((256,math.ceil(shard_intermediate_size/block_quant_shape[0]), math.ceil(hidden_size/block_quant_shape[1])),device=device,
        #                     dtype=torch.float32))
        # w2_scale = (torch.randn((256,math.ceil(hidden_size/block_quant_shape[0]), math.ceil((shard_intermediate_size // 2)/block_quant_shape[1])),device=device,
        #                     dtype=torch.float32))
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
                            use_int8_w8a8=use_int8_w8a8,
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

def fuse_moe_correctness(
        num_tokens: int,
        num_experts: int,
        shard_intermediate_size: int,
        hidden_size: int,
        topk: int,
        dtype: torch.dtype,
        use_fp8_w8a8: bool,
        use_int8_w8a16: bool,
        use_int8_w8a8: bool,
        num_iters: int=100,
        block_quant_shape: List[int] = None,
        seed: int=0,
        device_id: int=0,
    ) -> tuple[dict[str, int], float]:
        from vllm.platforms import current_platform
        current_platform.seed_everything(seed)

        # device="cuda:"+str(device_id)
        # x = torch.randn(num_tokens, hidden_size, device=device, dtype=dtype)
        # w1_scale = None
        # w2_scale = None
        # a1_scale = None
        # a2_scale = None

        # w1 = torch.randint(-127,
        #                     127, (
        #                         num_experts,
        #                         shard_intermediate_size, # n
        #                         hidden_size, # k
        #                     ),
        #                     device=device,
        #                     dtype=torch.int8)
        # w2 = torch.randint(-127,
        #                     127, (
        #                         num_experts,
        #                         hidden_size, # n2
        #                         shard_intermediate_size // 2, # k2
        #                     ),
        #                     device=device,
        #                     dtype=torch.int8)
        # # w1_scale = (torch.randn((256,math.ceil(shard_intermediate_size/block_quant_shape[0]), math.ceil(hidden_size/block_quant_shape[1])),device=device,
        # #                     dtype=torch.float32))
        # # w2_scale = (torch.randn((256,math.ceil(hidden_size/block_quant_shape[0]), math.ceil((shard_intermediate_size // 2)/block_quant_shape[1])),device=device,
        # #                     dtype=torch.float32))
        # w1_scale = (torch.randn((num_experts, shard_intermediate_size ,1),
        #                    dtype=torch.float32))
        # w2_scale = (torch.randn((num_experts, hidden_size, 1),
        #                    dtype=torch.float32))

        # input_gating = torch.randn(num_tokens, num_experts, device=device, dtype=torch.float32)

        # # triton_output = fused_moe_impl(x,
        # #                     w1,
        # #                     w2,
        # #                     input_gating,
        # #                     topk,
        # #                     renormalize=True,
        # #                     inplace=False,
        # #                     use_fp8_w8a8=use_fp8_w8a8,
        # #                     use_int8_w8a16=use_int8_w8a16,
        # #                     w1_scale=w1_scale,
        # #                     w2_scale=w2_scale,
        # #                     a1_scale=a1_scale,
        # #                     a2_scale=a2_scale,
        # #                     block_shape=block_quant_shape,
        # #                     config=None)

        # triton_output = fused_moe_impl_test(x,
        #                     w1,
        #                     w2,
        #                     input_gating,
        #                     topk,
        #                     renormalize=True,
        #                     inplace=False,
        #                     use_fp8_w8a8=use_fp8_w8a8,
        #                     use_int8_w8a16=use_int8_w8a16,
        #                     w1_scale=w1_scale,
        #                     w2_scale=w2_scale,
        #                     a1_scale=a1_scale,
        #                     a2_scale=a2_scale,
        #                     block_shape=block_quant_shape,
        #                     config=None)

        # torch_output = torch_moe_impl(x,
        #                     w1,
        #                     w2,
        #                     input_gating,
        #                     topk,
        #                     renormalize=True,
        #                     inplace=False,
        #                     use_fp8_w8a8=use_fp8_w8a8,
        #                     use_int8_w8a16=use_int8_w8a16,
        #                     w1_scale=w1_scale,
        #                     w2_scale=w2_scale,
        #                     a1_scale=a1_scale,
        #                     a2_scale=a2_scale,
        #                     block_shape=block_quant_shape,
        #                     config=None)
        # torch.cuda.synchronize()
        # print(f"Processing batch_size={num_tokens}")
        # if (torch.mean(torch.abs(triton_output.to(torch.float32) - torch_output.to(torch.float32))) / torch.mean(torch.abs(torch_output.to(torch.float32)))) < 0.02:
        #     print("triton all精度检查合格")
        # else:
        #     print("triton 精度检查不合格！！！")

def fuse_moe_correctness_preprocess(
        num_tokens: int,
        num_experts: int,
        shard_intermediate_size: int,
        hidden_size: int,
        topk: int,
        dtype: torch.dtype,
        use_fp8_w8a8: bool,
        use_int8_w8a16: bool,
        use_int8_w8a8: bool,
        num_iters: int=100,
        block_quant_shape: List[int] = None,
        block_m_configs: dict[str, int]=None,
        seed: int=0,
        device_id: int=0,
    ) -> tuple[dict[str, int], float]:

    mode_ranges = [mode for mode in block_m_configs.keys()]
    gemm1_compare_failed_list = [] # [111,169]: #qwen3-480b gemm1
    gemm2_compare_failed_list = [] # [111,169]: #qwen3-480b gemm2
    device="cuda:"+str(device_id)

    for mode in mode_ranges:
        mode0 = mode
        mode1 = mode

        # if mode0 in [111,169]: #qwen3-480b gemm1
        #     mode0 = 300
        # if mode1 in [111,169]: #qwen3-480b gemm2
        #     mode1 = 300

        mode_list=[mode0,mode1]

        #  =========================================== 准备 gemm1 相关数据 =============================================
        x = torch.randn((num_tokens, hidden_size),device=device,dtype=torch.bfloat16) / 10 # [256*8, 7168] (bf16)
        w1 = torch.randn((num_experts, shard_intermediate_size, hidden_size), device=x.device, dtype=torch.bfloat16)
        # -------- 处理 gemm1 权重矩阵 ------------------
        w1_marlin_list = []
        for i in range(w1.shape[0]):
            w1_marlin_in = w16a16_marlin_weight(w1[i])
            w1_marlin_list.append(w1_marlin_in)
        w1_marlin = torch.stack(w1_marlin_list, dim=0)
        w1_scale = (torch.randn((num_experts, shard_intermediate_size ,1), dtype=torch.float32, device=x.device))

        #  =========================================== 准备 gemm2 相关数据 =============================================
        w2 = torch.randn((num_experts, hidden_size, shard_intermediate_size // 2), device=x.device, dtype=torch.bfloat16)
        w2_scale = (torch.randn((num_experts, hidden_size, 1), dtype=torch.float32, device=x.device))
        # -------- 处理 gemm2 权重矩阵 ------------------
        w2_marlin_list = []
        for i in range(w2.shape[0]):
            w2_marlin_in = w16a16_marlin_weight(w2[i])
            w2_marlin_list.append(w2_marlin_in)
        w2_marlin = torch.stack(w2_marlin_list, dim=0)
        w2_scale = (torch.randn((num_experts, hidden_size, 1), dtype=torch.float32, device=x.device))


        # ============================================ 准备 gating_output 和 score ======================================
        gating_output = torch.randn(num_iters, num_tokens, num_experts, dtype=torch.float32)
        score = torch.randn((num_tokens, num_experts), device=device, dtype=torch.float32)

        topk_weights, topk_ids = grouped_topk(x, score, topk, True, 8, 4)

        # input,weight13,weight2,weight13_scale,weight2_scale,topk_weights,topk_ids=_fused_moe_helper(m=num_tokens,n=shard_intermediate_size,k=hidden_size,n2=hidden_size,k2=(shard_intermediate_size // 2),topk=topk,E=num_experts,device=device,block_size=None,out_dtype=dtype)
        # # triton moe: output
        # output=fused_experts_impl_int8_preprocess(hidden_states=input,
        #                                             w1=weight13,
        #                                             w2=weight2,
        #                                             topk_weights=topk_weights,
        #                                             topk_ids=topk_ids,
        #                                             inplace=False,
        #                                             use_int8_w8a8=True,
        #                                             w1_scale=weight13_scale,
        #                                             w2_scale=weight2_scale,
        #                                             block_shape=None,
        #                                             config={"BLOCK_SIZE_M": 32},
        #                                             per_channel_quant=True,
        #                                             mode=mode_list,
        #                                             gemm1_compare_list=gemm1_compare_failed_list,
        #                                             gemm2_compare_list=gemm2_compare_failed_list)
        output = fused_moe_valid_w16a16_marlin(
                                                x,
                                                w1,
                                                w2,
                                                w1_marlin,
                                                w2_marlin,
                                                score,
                                                topk_weights,
                                                topk_ids,
                                                modes=mode_list,
                                                gemm1_compare_failed_list=gemm1_compare_failed_list,
                                                gemm2_compare_failed_list=gemm2_compare_failed_list,
                                                impl_status_tune = False)

    print("gemm1_compare_failed_list: ", gemm1_compare_failed_list)
    print("gemm2_compare_failed_list: ", gemm2_compare_failed_list)
    return gemm1_compare_failed_list, gemm2_compare_failed_list

## benchmark_config_profile(8, 256, 256, 7168, 8, torch.bfloat16, use_fp8_w8a8=False, use_int8_w8a16=False, use_int8_w8a8=True, num_iters=10, block_quant_shape=[128,128])
def benchmark_config_profile(
        num_tokens: int,
        num_experts: int,
        shard_intermediate_size: int,
        hidden_size: int,
        topk: int,
        dtype: torch.dtype,
        use_fp8_w8a8: bool,
        use_int8_w8a16: bool,
        use_int8_w8a8: bool,
        num_iters: int=100,
        block_quant_shape: List[int] = None,
) -> float:

    device="cuda"
    x = torch.randn(num_tokens, hidden_size, device=device, dtype=dtype)
    w1_scale = None
    w2_scale = None
    a1_scale = None
    a2_scale = None

    w1 = torch.randint(-127,
                        127, (
                            num_experts,
                            shard_intermediate_size, # n
                            hidden_size, # k
                        ),
                        device=device,
                        dtype=torch.int8)
    w2 = torch.randint(-127,
                        127, (
                            num_experts,
                            hidden_size, # n2
                            shard_intermediate_size // 2, # k2
                        ),
                        device=device,
                        dtype=torch.int8)
    w1_scale = (torch.randn((256,math.ceil(num_experts/block_quant_shape[0]), math.ceil(hidden_size/block_quant_shape[1])), device=device,
                        dtype=torch.float32))
    w2_scale = (torch.randn((256,math.ceil(hidden_size/block_quant_shape[0]), math.ceil((shard_intermediate_size // 2)/block_quant_shape[1])), device=device,
                        dtype=torch.float32))

    gating_output = torch.randn(num_iters,
                            num_tokens,
                            num_experts,
                            device=device,
                            dtype=torch.float32)
    def prepare(i: int):
        input_gating.copy_(gating_output[i])

    input_gating = torch.randn(num_tokens, num_experts, device=device, dtype=torch.float32)

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
                        w1_scale=w1_scale,
                        w2_scale=w2_scale,
                        a1_scale=a1_scale,
                        a2_scale=a2_scale,
                        block_shape=block_quant_shape,
                        config=None)

    # JIT compilation & warmup
    run()
    torch.cuda.synchronize()

    # Capture 10 invocations with CUDA graph
    # graph = torch.cuda.CUDAGraph()
    # with torch.cuda.graph(graph):
    #     for _ in range(10):
    #         run()
    # torch.cuda.synchronize()

    # Warmup
    for _ in range(5):
        #graph.replay()
        run()
    torch.cuda.synchronize()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    latencies: List[float] = []
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA,],record_shapes=True,with_stack=True) as prof:

        for i in range(num_iters):
            prepare(i)
            torch.cuda.synchronize()

            start_event.record()
            #graph.replay()from
            run()
            end_event.record()
            end_event.synchronize()
            latencies.append(start_event.elapsed_time(end_event))

    print('Prepare time report')
    print(prof.key_averages(group_by_input_shape=True).table(sort_by="self_cuda_time_total", row_limit=-1))
    prof.export_chrome_trace('./dpsk_r1_bw_triton-tp16.json')
    avg = sum(latencies) / (num_iters) * 1000  # us
    #graph.reset()
    return avg

def get_rocm_tuning_space(use_fp16):

    mode_range = [mode for mode in blockM_all_configs.keys()]
    param_ranges = {
        "MODE": mode_range,
    }
    return param_ranges


def get_configs_compute_bound(use_fp16,
                              block_quant_shape) -> list[dict[str, int]]:
    configs = []

    if current_platform.is_rocm():
        param_ranges = get_rocm_tuning_space(use_fp16)
    else:
        # Reduced search space for faster tuning.
        # TODO(woosuk): Increase the search space and use a performance model to
        # prune the search space.
        block_m_range = [16, 32, 64, 128, 256]
        block_n_range = [64, 128, 256]
        block_k_range = [64, 128, 256]
        num_warps_range = [2, 4, 8]
        group_m_range = [1, 2, 4, 8, 16]
        num_stage_range = [0, 1, 2, 3]

        param_ranges = {
            "BLOCK_SIZE_M": block_m_range,
            "BLOCK_SIZE_N": block_n_range,
            "BLOCK_SIZE_K": block_k_range,
            "GROUP_SIZE_M": group_m_range,
            "num_warps": num_warps_range,
            "num_stages": num_stage_range,
        }

    keys, values = zip(*param_ranges.items())
    for config_values in product(*values):
        config = dict(zip(keys, config_values))
        configs.append(config)

    return configs


def prune_rocm_search_space_kernel0(num_tokens, shard_intermediate_size, hidden_size,
                            search_space, is_fp16, topk, gemm_compare_failed_list):
    N1, K1 = shard_intermediate_size, hidden_size
    pruned_space_0 = prune_rocm_configs(num_tokens, N1, K1,
                                        search_space, is_fp16, gemm_compare_failed_list)
    return pruned_space_0

def prune_rocm_search_space_kernel1(num_tokens, shard_intermediate_size, hidden_size,
                            search_space, is_fp16, topk, gemm_compare_failed_list):
    N2, K2 = hidden_size, shard_intermediate_size // 2
    pruned_space_1 = prune_rocm_configs(num_tokens, N2, K2,
                                        search_space, is_fp16, gemm_compare_failed_list)
    return pruned_space_1


# The following code is inspired by ROCm/Triton GEMM tuning script:
# https://github.com/ROCm/triton/blob/triton-mlir/scripts/amd/gemm/tune_gemm.py#L89
def prune_rocm_configs(M, N, K, configs, is_fp16=True, gemm_compare_failed_list=[]):
    pruned_configs = []
    for config in configs:
        mode = config.get("MODE")
        exclude_modes = gemm_compare_failed_list
        if mode in exclude_modes:
            continue
        pruned_configs.append(config)

    return pruned_configs


def need_split_k(SIZE_M, SIZE_N, SIZE_K):
    return (SIZE_M < 64 or SIZE_N < 64) and SIZE_K > 1024


def merge_unique_dicts(list1, list2):
    result = []
    combined_list = list1.copy()
    combined_list.extend(list2)
    for dictionary in combined_list:
        if dictionary not in result:
            result.append(dictionary)
    return result


@ray.remote(num_gpus=1)
class BenchmarkWorker:

    def __init__(self,
                 seed: int,
                 device_id: int,
                 router_logits_source: Optional[str] = None,
                 router_logits_layer: Optional[str] = None,
                 router_topk_impl: str = "auto",
                 router_logits_strict: bool = False) -> None:
        from vllm.platforms import current_platform
        torch.set_default_device("cuda:"+ str(device_id))
        current_platform.seed_everything(seed)
        self.seed = seed
        # Get the device ID to allocate tensors and kernels
        # on the respective GPU. This is required for Ray to work
        # correctly with multi-GPU tuning on the ROCm platform.
        self.device_id = device_id
        self.router_logits_source = router_logits_source
        self.router_logits_layer = router_logits_layer
        self.router_topk_impl = router_topk_impl
        self.router_logits_strict = router_logits_strict

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
        use_int8_w8a8: bool,
        block_quant_shape: List[int] = None,
        use_deep_gemm: bool = False,
    ) -> tuple[dict[str, int], float]:

        kernel_time = moe_cuda_graph(num_tokens, num_experts,
                                       shard_intermediate_size, hidden_size,
                                       topk, dtype, use_fp8_w8a8,
                                       use_int8_w8a16, use_int8_w8a8, block_quant_shape=block_quant_shape,seed=self.seed, device_id=self.device_id)

        return kernel_time

    def correctness(
        self,
        num_tokens: int,
        num_experts: int,
        shard_intermediate_size: int,
        hidden_size: int,
        topk: int,
        dtype: torch.dtype,
        use_fp8_w8a8: bool,
        use_int8_w8a16: bool,
        use_int8_w8a8: bool,
        block_quant_shape: List[int] = None,
        use_deep_gemm: bool = False,
    ) -> tuple[dict[str, int], float]:

        fuse_moe_correctness(num_tokens, num_experts,
                                       shard_intermediate_size, hidden_size,
                                       topk, dtype, use_fp8_w8a8,
                                       use_int8_w8a16, use_int8_w8a8, block_quant_shape=block_quant_shape,seed=self.seed, device_id=self.device_id)

        return

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
        use_int8_w8a8: bool,
        search_space,
        block_quant_shape: list[int],
        use_deep_gemm: bool,
        ep_size: int = 1,
    ) -> dict[str, int]:
        best_config_kernel0 = None
        best_config_kernel1 = None
        best_time = float("inf")
        from vllm.platforms import current_platform
        current_platform.seed_everything(self.seed)

        mode_pattern = "MODE"
        block_configs = blockM_all_configs

        gemm1_compare_failed_list, gemm2_compare_failed_list = fuse_moe_correctness_preprocess(num_tokens, num_experts,
                                                                        shard_intermediate_size, hidden_size,
                                                                        topk, dtype, use_fp8_w8a8,
                                                                        use_int8_w8a16, use_int8_w8a8, block_quant_shape=block_quant_shape,
                                                                        block_m_configs = block_configs,seed=self.seed, device_id=self.device_id)


        if current_platform.is_rocm():
            is_fp16 = is_fp16 = not (use_fp8_w8a8 or use_int8_w8a16 or use_int8_w8a8)
            search_space_kernel0 = prune_rocm_search_space_kernel0(num_tokens,
                                                   shard_intermediate_size,
                                                   hidden_size, search_space,
                                                   is_fp16, topk, gemm1_compare_failed_list)
            search_space_kernel1 = prune_rocm_search_space_kernel1(num_tokens,
                                                   shard_intermediate_size,
                                                   hidden_size, search_space,
                                                   is_fp16, topk, gemm2_compare_failed_list)

        # import pdb
        # pdb.set_trace()

        block_m_range = [16,32,64,128]
        if num_tokens <= 32:
            block_m_range = [16,32]
        elif num_tokens <= 64:
            block_m_range = [16,32,64]
        elif num_tokens <= 128:
            block_m_range = [16,32,64,128]

        with torch.cuda.device(self.device_id):

            for block_m in tqdm(block_m_range, desc="Processing block_m......"):
                time_kernel0 = float('inf')
                config_kernel0 = None

                time_kernel1 = float('inf')
                config_kernel1 = None
                for config0 in search_space_kernel0:
                    if block_configs[config0[mode_pattern]] != block_m:
                        continue
                    try:
                        kernel_time = benchmark_config(
                                            config0,
                                            num_tokens,
                                            num_experts,
                                            shard_intermediate_size,
                                            hidden_size,
                                            topk,
                                            dtype,
                                            use_fp8_w8a8,
                                            use_int8_w8a16,
                                            use_int8_w8a8,
                                            num_iters=10,
                                            block_quant_shape=block_quant_shape,
                                            stages=1,
                                            router_logits_source=self.router_logits_source,
                                            router_logits_layer=self.router_logits_layer,
                                            router_topk_impl=self.router_topk_impl,
                                            router_logits_strict=self.router_logits_strict)
                    except triton.runtime.autotuner.OutOfResources:
                        # Some configurations may be invalid and fail to compile.
                        # 捕获lds资源溢出错误
                        continue
                    if kernel_time < time_kernel0:
                        time_kernel0 = kernel_time
                        config_kernel0 = config0
                for config1 in search_space_kernel1:
                    if block_configs[config1[mode_pattern]] > block_m:
                        continue
                    try:
                        kernel_time = benchmark_config(
                                            config1,
                                            num_tokens,
                                            num_experts,
                                            shard_intermediate_size,
                                            hidden_size,
                                            topk,
                                            dtype,
                                            use_fp8_w8a8,
                                            use_int8_w8a16,
                                            use_int8_w8a8,
                                            num_iters=10,
                                            block_quant_shape=block_quant_shape,
                                            stages=2,
                                            router_logits_source=self.router_logits_source,
                                            router_logits_layer=self.router_logits_layer,
                                            router_topk_impl=self.router_topk_impl,
                                            router_logits_strict=self.router_logits_strict)
                    except triton.runtime.autotuner.OutOfResources:
                        # Some configurations may be invalid and fail to compile.
                        # 捕获lds资源溢出错误
                        continue
                    if kernel_time < time_kernel1:
                        time_kernel1 = kernel_time
                        config_kernel1 = config1
                now = datetime.now()
                print(f"{now.ctime()}] Completed tuning for batch_size={num_tokens}")
                if (time_kernel0 + time_kernel1) < best_time:
                    best_time = time_kernel0 + time_kernel1
                    best_config_kernel0 = config_kernel0
                    best_config_kernel1 = config_kernel1

        if best_config_kernel0 and best_config_kernel1:
            best_config_kernel0['MODE'] = best_config_kernel0.pop(mode_pattern)
            best_config_kernel1['MODE'] = best_config_kernel1.pop(mode_pattern)

        best_config_kernel0['DELTA'] = 1
        best_config_kernel1['DELTA'] = int(best_config_kernel0['BLOCK_SIZE_M']/best_config_kernel1['BLOCK_SIZE_M'])
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

def save_single_config(batch_size: int, first_stage_config: BenchmarkConfig, second_stage_config: BenchmarkConfig,
                       num_experts: int, shard_intermediate_size: int, hidden_size: int, topk: int,
                       dtype: torch.dtype, use_fp8_w8a8: bool, use_int8_w8a16: bool, use_int8_w8a8: bool,
                       moe_ep_size: int = 1, block_quant_shape: List[int] = None) -> None:
    """保存单个batch_size的配置结果，支持增量保存"""
    # 保存第一阶段配置
    N = shard_intermediate_size
    K = hidden_size
    gfx_version = torch.cuda.get_device_properties(0).gcnArchName.split(':')[0]
    if moe_ep_size > 1:
        filename = f"MOE_W16A16_CUDA_MARLIN_EP{moe_ep_size}_{N}_{K}_{gfx_version.lower()}_{torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count}.json"
    elif moe_ep_size == 1:
        filename = f"MOE_W16A16_CUDA_MARLIN_{N}_{K}_{gfx_version.lower()}_{torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count}.json"
    else:
        raise ValueError(f"Invalid moe_ep_size: {moe_ep_size}")

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

    # 更新当前batch_size的配置
    config_info[f"{batch_size}"] = first_stage_config
    config_infos[f"{N}_{K}"] = config_info

    # 保存文件
    with open(filename, 'w') as file:
        json.dump(config_infos, file, indent=4)
    print(f"Saved first stage config for batch_size={batch_size} to {filename}")

    # 保存第二阶段配置
    N = hidden_size
    K = shard_intermediate_size // 2
    if moe_ep_size > 1:
        filename = f"MOE_W16A16_CUDA_MARLIN_EP{moe_ep_size}_{N}_{K}_{gfx_version.lower()}_{torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count}.json"
    elif moe_ep_size == 1:
        filename = f"MOE_W16A16_CUDA_MARLIN_{N}_{K}_{gfx_version.lower()}_{torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count}.json"
    else:
        raise ValueError(f"Invalid moe_ep_size: {moe_ep_size}")

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

    # 更新当前batch_size的配置
    config_info[f"{batch_size}"] = second_stage_config
    config_infos[f"{N}_{K}"] = config_info

    # 保存文件
    with open(filename, 'w') as file:
        json.dump(config_infos, file, indent=4)
    print(f"Saved second stage config for batch_size={batch_size} to {filename}")

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

    # config = AutoConfig.from_pretrained(
    #     args.model, trust_remote_code=args.trust_remote_code)
    # if config.architectures[0] == "DbrxForCausalLM":
    #     E = config.ffn_config.moe_num_experts
    #     E = E // moe_ep_size
    #     topk = config.ffn_config.moe_top_k
    #     intermediate_size = config.ffn_config.ffn_hidden_size
    #     shard_intermediate_size = 2 * intermediate_size // tp_size
    # elif config.architectures[0] == "JambaForCausalLM":
    #     E = config.num_experts
    #     E = E // moe_ep_size
    #     topk = config.num_experts_per_tok
    #     intermediate_size = config.intermediate_size
    #     shard_intermediate_size = 2 * intermediate_size // tp_size
    # elif (config.architectures[0] == "DeepseekV3ForCausalLM"
    #       or config.architectures[0] == "DeepseekV2ForCausalLM"):
    #     E = config.n_routed_experts
    #     E = E // moe_ep_size
    #     topk = config.num_experts_per_tok
    #     intermediate_size = config.moe_intermediate_size
    #     shard_intermediate_size = 2 * intermediate_size // tp_size
    #     block_quant_shape = get_weight_block_size_safety(config)
    # elif config.architectures[0] == "Qwen2MoeForCausalLM":
    #     E = config.num_experts
    #     E = E // moe_ep_size
    #     topk = config.num_experts_per_tok
    #     intermediate_size = config.moe_intermediate_size
    #     shard_intermediate_size = 2 * intermediate_size // tp_size
    # else:
    #     if not hasattr(config, "hidden_size"):
    #         # Support for llama4
    #         config = config.text_config
    #     # Default: Mixtral.
    #     E = config.num_local_experts
    #     E = E // moe_ep_size
    #     topk = config.num_experts_per_tok
    #     intermediate_size = config.intermediate_size
    #     shard_intermediate_size = 2 * intermediate_size // tp_size

    # # int8 fuse-moe
    # shard_intermediate_size = shard_intermediate_size * 2
    # # block_quant_shape = [128, 128]
    # block_quant_shape = None
    # hidden_size = config.hidden_size
    dtype = torch.bfloat16
    use_fp8_w8a8 = False
    use_int8_w8a16 = False
    use_int8_w8a8 = False
    topk = 8
    ep_size = 1



    # # 字节算子测试
    # E = 64
    # shard_intermediate_size=2048
    # hidden_size=4096
    # topk = 4

    ## deepseek-r1-tp16
    # E = 256
    # shard_intermediate_size=256
    # hidden_size=7168

    # ## deepseek-r1-tp8
    # E = 256
    # shard_intermediate_size=512
    # hidden_size=7168

    # # ## deepseek-671b-tp4
    # E = 256
    # shard_intermediate_size=1024
    # hidden_size=7168

    # ## deepseek-671b-tp2
    # E = 256
    # shard_intermediate_size=2048
    # hidden_size=7168

    # ## deepseek-671b-tp1
    # E = 256
    # shard_intermediate_size=4096
    # hidden_size=7168

    # E = 128
    # # #qwen3-30b-A3B-tp1
    # shard_intermediate_size=1536
    # hidden_size=2048
    # #qwen3-30b-A3B-tp2
    # E = 128
    # shard_intermediate_size=768
    # hidden_size=2048

    #qwen3-30b-A3B-tp4
    E = 128
    shard_intermediate_size=384
    hidden_size=2048

    # #qwen3-480b-tp8
    # E = 160
    # shard_intermediate_size=640
    # hidden_size=6144

    #qwen3-480b-tp16
    # E = 160
    # shard_intermediate_size=320
    # hidden_size=6144

    # #qwen3-235b-A22B-tp2
    # E = 128
    # shard_intermediate_size=1536
    # hidden_size=4096

    # # #qwen3-235b-A22B-tp8
    # E = 128
    # shard_intermediate_size=384
    # hidden_size=4096

    # # #qwen3-235b-A22B-tp16
    # E = 128
    # shard_intermediate_size=192
    # hidden_size=4096

    # deepseek-r1-EP16
    # E = 16
    # shard_intermediate_size=4096
    # hidden_size=7168

    # # glm4.5v-tp2
    # E = 128
    # shard_intermediate_size=1408
    # hidden_size=4096

    # # glm4.5v-tp4
    # E = 128
    # shard_intermediate_size=704
    # hidden_size=4096

    # # glm5-tp16
    # E = 256
    # shard_intermediate_size=256
    # hidden_size=6144

    # # qwen3.5-35b-a3b-tp1
    # E = 256
    # shard_intermediate_size=1024
    # hidden_size=2048

    # # qwen3.5-35b-a3b-tp2
    # E = 256
    # shard_intermediate_size=512
    # hidden_size=2048

    # # qwen3.5-35b-a3b-tp4
    # E = 256
    # shard_intermediate_size=256
    # hidden_size=2048

    # # qwen3.5-122b-a10b-tp2
    # E = 256
    # shard_intermediate_size=1024
    # hidden_size=3072

    # # qwen3.5-122b-a10b-tp8
    # E = 256
    # shard_intermediate_size=256
    # hidden_size=3072

    # # qwen3.5-397b-a17b-tp8
    # E = 512
    # shard_intermediate_size=256
    # hidden_size=4096

    # # qwen3.5-397b-a17b-tp16
    # E = 512
    # shard_intermediate_size=128
    # hidden_size=4096

    # step3.5-flash-tp8-ep1
    #E = 288
    #shard_intermediate_size=320
    #hidden_size=4096

    if args.batch_size is None:
        batch_sizes = [
            # 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 6144, 8192, 16384
            1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128,
            256,390,512,1024,1536,
            2048, 4096, 6144, 8192, 16384,

            # 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 128,
            # 144, 160, 176, 192, 208, 224, 240, 256,
            # 288, 320, 352, 384, 416, 448, 480, 512,

            # 640, 768, 896, 1024,
            # 1152, 1280, 1408, 1536, 1664, 1792, 1920, 2048,
            # 2176, 2304, 2432, 2560, 2688, 2816, 2944, 3072,
            # 3200, 3328, 3456, 3584, 3712, 3840, 3968, 4096,
            # 6144, 8192, 10240, 12288, 14336, 16384

            # 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048,
            # 1, 2, 4, 8, 16, 32, 64, 8192, 16384, 32768
            # 4096, 6144, 8192, 16384, 32768
            # 16384
            # 65536,32768,
            # 32768, 49152, 65536, 131072
            # 6144, 8192, 16384, 32768, 40960
            # 512,
            # 1024,2048,4096
        ]
    else:
        batch_sizes = [args.batch_size]

    use_deep_gemm = bool(args.use_deep_gemm)
    ray.init(address=None,
                 ignore_reinit_error=True,
                 num_gpus=args.num_gpus)
    num_gpus = int(ray.available_resources()["GPU"])
    workers = [BenchmarkWorker.remote(args.seed, i, router_logits_source=args.router_logits_source, router_logits_layer=args.router_logits_layer,
                                      router_topk_impl=args.router_topk_impl, router_logits_strict=args.router_logits_strict) for i in range(num_gpus)]
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
        is_fp16 = not (use_fp8_w8a8 or use_int8_w8a16 or use_int8_w8a8)
        search_space = get_configs_compute_bound(is_fp16, block_quant_shape)
        print(f"Start tuning over {len(search_space)} configurations...")
        start = time.time()
        # 提交所有任务
        worker_idx = 0
        task_refs = []
        batch_size_to_ref = {}
        for batch_size in batch_sizes:
            worker = workers[worker_idx]
            task_ref = worker.tune.remote(
                batch_size, E, shard_intermediate_size, hidden_size,
                topk, dtype, use_fp8_w8a8, use_int8_w8a16, use_int8_w8a8, search_space,
                block_quant_shape, use_deep_gemm, ep_size
            )
            task_refs.append(task_ref)
            batch_size_to_ref[task_ref] = batch_size
            worker_idx = (worker_idx + 1) % num_gpus

        # 逐个获取结果并立即保存（增量保存）
        best_configs_first_stage = {}
        best_configs_second_stage = {}
        completed_count = 0

        while task_refs:
            # 等待至少一个任务完成
            ready_refs, remaining_refs = ray.wait(task_refs, num_returns=1, timeout=None)
            task_refs = remaining_refs

            for ref in ready_refs:
                try:
                    config = ray.get(ref)
                    batch_size = batch_size_to_ref[ref]
                    first_stage_config, second_stage_config = config
                    best_configs_first_stage[batch_size] = first_stage_config
                    best_configs_second_stage[batch_size] = second_stage_config

                    # 立即保存当前结果
                    save_single_config(
                        batch_size, first_stage_config, second_stage_config,
                        E, shard_intermediate_size, hidden_size, topk,
                        dtype, use_fp8_w8a8, use_int8_w8a16, use_int8_w8a8,
                        ep_size, block_quant_shape
                    )
                    completed_count += 1
                    print(f"Progress: {completed_count}/{len(batch_sizes)} batch sizes completed")
                except Exception as e:
                    batch_size = batch_size_to_ref[ref]
                    print(f"Error processing batch_size={batch_size}: {e}")
                    # 继续处理其他任务

        end = time.time()
        print(f"Tuning took {end - start:.2f} seconds")
        print(f"All {len(batch_sizes)} batch sizes have been tuned and saved")
    elif args.correctness:
        outputs = _distribute(
            "correctness",
            [(batch_size, E, shard_intermediate_size, hidden_size, topk, dtype,
              use_fp8_w8a8, use_int8_w8a16, use_int8_w8a8, block_quant_shape, use_deep_gemm)
             for batch_size in batch_sizes])

        print(f"Correctness Done!")
    else:
        outputs = _distribute(
            "benchmark",
            [(batch_size, E, shard_intermediate_size, hidden_size, topk, dtype,
              use_fp8_w8a8, use_int8_w8a16, use_int8_w8a8, block_quant_shape, use_deep_gemm)
             for batch_size in batch_sizes])

        _w13n=[]
        _w13k=[]

        _w2n=[]
        _w2k=[]
        _m=[]
        _total_time=[]

        for batch_size, kernel_time in zip(batch_sizes, outputs):
            _w13n.append(shard_intermediate_size)
            _w13k.append(hidden_size)
            _w2n.append(hidden_size)
            _w2k.append(shard_intermediate_size//2)
            _m.append(batch_size)
            _total_time.append(kernel_time)
            print(f"Batch size: {batch_size}, Kernel time: {kernel_time:.2f} us")

        df = pd.DataFrame({'m':_m,'w13n':_w13n,'w13k':_w13k,'w2n':_w2n,'w2k':_w2k,'moe 量化算子总耗时(us)':_total_time})
        # 将 DataFrame 写入 Excel 文件
        df.to_excel('moegemmoutput.xlsx', index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
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
                        choices=["auto", "fp8_w8a8", "int8_w8a16", "int8_w8a8"],
                        default="bf16_w16a16")
    parser.add_argument("--use-deep-gemm", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, required=False)
    parser.add_argument("--tune", action="store_true")
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--correctness", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true",default=True)
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--moe-ep-size", "-ep", type=int, default=1)
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
    args = parser.parse_args()

    main(args)
    # benchmark_config_profile(49152, 256, 256, 7168, 8, torch.bfloat16, use_fp8_w8a8=False, use_int8_w8a16=False, use_int8_w8a8=True, num_iters=10, block_quant_shape=[128,128])
    # HIP_VISIBLE_DEVICES=3 python benchmark_moe_bf16_marlin.py --num-gpus 1 --tune --router-logits-source /home/data/qwen3-moe/logist/
