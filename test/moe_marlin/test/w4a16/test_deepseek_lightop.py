import pandas as pd
import torch
import triton
import triton.language as tl
from tqdm import trange
from typing import Any, Callable, Dict, List, Optional, Tuple

import vllm._custom_ops as vllm_ops
from vllm.model_executor.layers.fused_moe.fused_moe import (
    fused_moe_kernel_gptq_awq,
    moe_align_block_size,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    quantize_weights,
)
from vllm.scalar_type import scalar_types
# import marlin  # 确保导入了 marlin 包
import lightop  # 确保导入了 marlin 包
from marlin_quant import awq_marlin_quantize

global_group_size = 64

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

def run_moe_marlin(inputs, weight, topk_ids, bit):
    torch.cuda.manual_seed_all(0)
    num_experts, n, k = weight.shape
    num_tokens, top_k = topk_ids.shape
    assert inputs.size(0) == num_tokens and inputs.size(1) == k

    for block_size in [16, 32, 48, 64, 80]:
        if num_tokens * top_k / num_experts / block_size < 0.9:
            break
    # block_size = 16

    # print("block_m: ", block_size)

    sorted_token_ids, expert_ids, num_tokens_post_padded = \
        moe_align_block_size(topk_ids, block_size, num_experts)
    group_size = global_group_size
    weight_ref_list = []
    qweight_list = []
    # scales_list = []
    # qzeros_list = []
    scale_zeros_list = []

    quant_type = scalar_types.uint4 if bit == 4 else scalar_types.uint8
    for i in range(weight.shape[0]):
        weight_ref, qweight, scale_zeros = awq_marlin_quantize(
            weight[i].T, quant_type, group_size)

        weight_ref_list.append(weight_ref)
        qweight_list.append(qweight)
        # scales_list.append(scales)
        # qzeros_list.append(qzeros)
        scale_zeros_list.append(scale_zeros)

    device = torch.device("cuda:0")
    weight_ref = torch.stack(weight_ref_list, dim=0)
    qweight = torch.stack(qweight_list, dim=0)
    # scales = torch.stack(scales_list, dim=0)
    # qzeros = torch.stack(qzeros_list, dim=0)
    scale_zeros = torch.stack(scale_zeros_list, dim=0)

    g_idx = torch.empty((0, 0), dtype=torch.int32, device=device)
    sort_indices = torch.empty((0, 0), dtype=torch.int32, device=device)
    topk_weights = torch.empty((0, ), dtype=torch.float, device=device)
    sms = torch.cuda.get_device_properties(device).multi_processor_count
    workspace = torch.zeros((sms * 3, ), dtype=torch.int, device=device)

    def run():
        # output = torch.empty((num_tokens * top_k, n), device=inputs.device, dtype=inputs.dtype)
        # output.zero_()
        return lightop.moe.moe_marlin_w4a16(
                                            inputs, 
                                            None, 
                                            qweight, 
                                            scale_zeros,
                                            # scales, 
                                            # qzeros, 
                                            g_idx, 
                                            sort_indices,
                                            workspace,
                                            sorted_token_ids,
                                            expert_ids,
                                            num_tokens_post_padded,
                                            topk_weights,
                                            block_size,
                                            top_k,
                                            False,
                                            False,
                                            num_tokens,
                                            n,
                                            k,
                                            True,
                                            True,
                                            True,
                                            False
                                        )

    torch.cuda.synchronize()
    is_full_zero = torch.all(workspace == 0)
    # print("工作空间是不是全为零: ", is_full_zero)
    # ms_per_iter = triton.testing.do_bench(lambda: run(), warmup=10, rep=100)
    is_full_zero = torch.all(workspace == 0)
    # print("工作空间是不是全为零: ", is_full_zero)
    output = run()
    ms_per_iter = 0.0
    return output, ms_per_iter

def run_moe_triton_gemm(inputs, weight, topk_ids, bit):
    torch.cuda.manual_seed_all(0)

    num_experts, n, k = weight.shape
    num_tokens, top_k = topk_ids.shape
    assert inputs.size(0) == num_tokens and inputs.size(1) == k

    if num_tokens <= num_experts:
        config = {
            "BLOCK_SIZE_M": 16,
            "BLOCK_SIZE_N": 32,
            "BLOCK_SIZE_K": 64,
            "GROUP_SIZE_M": 1,
            "num_warps": 4,
            "num_stages": 1,
        }
    else:
        config = {
            "BLOCK_SIZE_M": 64,
            "BLOCK_SIZE_N": 64,
            "BLOCK_SIZE_K": 32,
            "GROUP_SIZE_M": 8,
        }

    sorted_token_ids, expert_ids, num_tokens_post_padded = \
        moe_align_block_size(topk_ids, config["BLOCK_SIZE_M"], num_experts)

    group_size = global_group_size
    weight_ref_list = []
    qweight_list = []
    scales_list = []
    qzeros_list = []

    quant_type = scalar_types.uint4 if bit == 4 else scalar_types.uint8
    for i in range(weight.shape[0]):
        weight_ref, qweight, scales, qzeros = quantize_weights(
            weight[i].T, quant_type, group_size, True, False)

        weight_ref = weight_ref.T
        qweight = qweight.T.contiguous().to(torch.uint8)
        scales = scales.T
        qzeros = qzeros.T.contiguous().to(torch.uint8)

        if bit == 4:
            qweight = qweight[:, 1::2] * 16 + qweight[:, ::2]
            qzeros = qzeros[1::2, :] * 16 + qzeros[::2, :]

        weight_ref_list.append(weight_ref)
        qweight_list.append(qweight)
        scales_list.append(scales)
        qzeros_list.append(qzeros)

    weight_ref = torch.stack(weight_ref_list, dim=0)
    qweight = torch.stack(qweight_list, dim=0)
    scales = torch.stack(scales_list, dim=0)
    qzeros = torch.stack(qzeros_list, dim=0)

    outputs = torch.empty(
        (num_tokens, top_k, n),
        dtype=torch.half, device="cuda:0"
    )

    A, B, C, B_scale, B_zp = inputs, qweight, outputs, scales, qzeros

    EM = sorted_token_ids.shape[0]
    if A.shape[0] < config["BLOCK_SIZE_M"]:
        EM = min(sorted_token_ids.shape[0],
                 A.shape[0] * top_k * config['BLOCK_SIZE_M'])

    blocks = triton.cdiv(EM, config['BLOCK_SIZE_M']) * \
        triton.cdiv(B.shape[1], config['BLOCK_SIZE_N'])
    grid = (blocks, )

    def run():
        outputs_run = torch.empty((num_tokens, top_k, n), dtype=torch.half, device="cuda:0")
        fused_moe_kernel_gptq_awq[grid](
            A,
            B,
            outputs_run,
            B_scale,
            B_zp,
            None,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            B.shape[1],
            A.shape[1],
            EM,
            topk_ids.numel(),
            A.stride(0),
            A.stride(1),
            B.stride(0),
            B.stride(2),
            B.stride(1),
            C.stride(1),
            C.stride(2),
            B_scale.stride(0),
            B_scale.stride(2),
            B_scale.stride(1),
            B_zp.stride(0) if B_zp is not None else 0,
            B_zp.stride(2) if B_zp is not None else 0,
            B_zp.stride(1) if B_zp is not None else 0,
            block_k_diviable=A.shape[1] % config["BLOCK_SIZE_K"] == 0,
            group_size=group_size,
            MUL_ROUTED_WEIGHT=False,
            top_k=top_k,
            compute_type=tl.float16,
            has_zp=B_zp is not None,
            use_int4_w4a16=bit == 4,
            use_int8_w8a16=bit == 8,
            **config,
        )
        return outputs_run

    # ms_per_iter = triton.testing.do_bench(lambda: run(), warmup=10, rep=100)

    output = run()
    # return output, ms_per_iter
    return output, 0.0

def perf_deepseek_v3(use_bf16):
    weight_shapes = [
        (256, 8, 256 * 2, 7168),
        (256, 8, 7168, 256),
    ]
    res = []
    for num_experts, top_k, n, k in weight_shapes:
        # print("num_experts: ", num_experts, "top_k", top_k, "n", n, "k", k)
        bit = 4
        # for i in trange(12):
        # for i in [1, 2, 4]:
        # for i in [1, 2, 4]:
        for i in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]:
        # for i in [512, 1024, 2048, 4096, 8192, 16384]:
        # for i in [32]:
        # for i in [32]:
        # for i in [4096]:
        # for i in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]:
        # for i in [1024]:
        # for i in [32]:
            # print("     执行 num_tokens: ", i, " 性能测试......")         
            num_tokens = i 
            print("num_tokens: ", num_tokens, "use_bf16: ", use_bf16, "test......")
            torch.cuda.manual_seed_all(0)
            inputs = torch.randn(
                (num_tokens, k),
                dtype=torch.half, device="cuda:0"
            ) / 10
            weight = torch.randn(
                (num_experts, n, k),
                dtype=torch.half, device="cuda:0"
            ) / 10

            scores = torch.randn(
                (num_tokens, num_experts),
                dtype=torch.float, device="cuda:0"
            )
            # _, topk_ids = scores.topk(top_k, 1)

            _, topk_ids = grouped_topk(inputs, scores, top_k, True, num_expert_group=8, topk_group = 4)

            # 准确性对比
            triton_output, triton_time = run_moe_triton_gemm(inputs, weight, topk_ids, bit)
            triton_output = triton_output.reshape(-1, n)
            if use_bf16:
                inputs = inputs.to(torch.bfloat16)
            marlin_output, marlin_time = run_moe_marlin(inputs, weight, topk_ids, bit)

            # print("triton_output: ", triton_output)
            # print("marlin_output: ", marlin_output)
            if use_bf16:
                is_true = torch.allclose(marlin_output.to(torch.float16), triton_output, rtol=1e-1, atol=0)
            else:
                is_true = torch.allclose(marlin_output, triton_output, atol=1e-2, rtol=0)

            res.append({
                        "num_tokens": num_tokens,
                        "use_bf16": use_bf16,
                        "is_true": is_true
            })
    return res
            

if __name__ == "__main__":
    result_acc_total = []
    result_acc_fp16 = perf_deepseek_v3(use_bf16=False)
    result_acc_total.extend(result_acc_fp16)

    # result_acc_bf16 = perf_deepseek_v3(use_bf16=True)
    # result_acc_total.extend(result_acc_bf16)

    file_name = "test_w4a16_fused_moe_tp.xlsx"
    # 列表转换为 pandas DataFrame（表格数据结构）
    df = pd.DataFrame(result_acc_total)

    # 写入 Excel 文件（index=False 表示不写入行索引）
    df.to_excel(file_name, index=False, engine='openpyxl')

    print("数据已成功写入 ", file_name)

