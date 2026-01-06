import random
import torch
import pytest
from typing import Dict, List, Any

import copy
from datetime import datetime
import time

from vllm import _custom_ops as vllm_ops


from lightop import fuse_groupgemm_permute_quant

_cuda_available = True
# seed = 1234
# torch.manual_seed(seed)
# torch.cuda.manual_seed(seed)

def pytorch_permute_mask_map(tokens, routing_map):
    # 获取输入token张量的维度信息
    num_tokens, _ = tokens.shape
    num_experts = routing_map.shape[1]

    # 将routing_map转换为布尔类型，转置并确保内存连续
    routing_map = routing_map.bool().T.contiguous()

    # 创建token索引矩阵
    token_indices = (
        torch.arange(num_tokens, device=routing_map.device).unsqueeze(0).expand(num_experts, -1)
    )

    # 使用mask提取被选中的token索引
    sorted_indices = token_indices.masked_select(routing_map)

    # 根据排序后的索引重新排列token
    permuted_input = tokens.index_select(0, sorted_indices)

    # 对排列后的 token 进行量化
    scaled_permuted_input, scales, _ = vllm_ops.scaled_int8_quant(permuted_input)

    # 返回重排序并量化后的token、量化尺度和排序索引
    return scaled_permuted_input, scales, sorted_indices

# ------------------------------------------------------------------------------------
# 测试函数
# ------------------------------------------------------------------------------------

def _test_permutation_mask_map(
    num_tokens,
    num_expert,
    hidden_size,
    topK,
    dtype,
) -> Dict[str, Any]:
    num_out_tokens = num_tokens * topK

    # 创建测试数据
    pytorch_permute_fwd_input = torch.rand((num_tokens, hidden_size), dtype=dtype).cuda()

    # 生成routing_map
    _tmp_tensor = torch.zeros((num_tokens * num_expert,))
    _tmp_tensor[: int(num_out_tokens)] = 1.0
    _tmp_idx = torch.randperm(num_tokens * num_expert)
    routing_map = torch.reshape(_tmp_tensor[_tmp_idx], (num_tokens, num_expert)).bool().cuda()

    # --- 1. Correctness Check ---
    pytorch_permute_output_quant, pytorch_scale, sorted_indices = pytorch_permute_mask_map(
        pytorch_permute_fwd_input, routing_map
    )
    
    cuda_permute_output_quant, cuda_scale, cuda_sorted_indices, expert_m_count = fuse_groupgemm_permute_quant(
        pytorch_permute_fwd_input, routing_map
    )

    permute_correct = torch.allclose(pytorch_permute_output_quant, cuda_permute_output_quant, rtol=1e-5, atol=1e-5)
    scale_correct = torch.allclose(pytorch_scale, cuda_scale, rtol=1e-5, atol=1e-5)
    indices_correct = torch.allclose(sorted_indices.float(), cuda_sorted_indices.float())
    # print("permute_correct", permute_correct)
    # print("scale_correct", scale_correct)
    # print("indices_correct", indices_correct)
    # print("pytorch_permute_output_quant",pytorch_permute_output_quant)
    # print("cuda_permute_output_quant",cuda_permute_output_quant)
    
    # --- 2. Performance Benchmark ---
    num_trials = 100
    num_warmup = 10

    # Benchmark PyTorch (permute + quant)
    for _ in range(num_warmup):
        _ = pytorch_permute_mask_map(pytorch_permute_fwd_input, routing_map)
    torch.cuda.synchronize()
    start_time = time.time()
    for _ in range(num_trials):
        _ = pytorch_permute_mask_map(pytorch_permute_fwd_input, routing_map)
    torch.cuda.synchronize()
    pytorch_permute_time_ms = ((time.time() - start_time) / num_trials) * 1000

    # Benchmark CUDA (fused permute + quant)
    for _ in range(num_warmup):
        _ = fuse_groupgemm_permute_quant(pytorch_permute_fwd_input, routing_map)
    torch.cuda.synchronize()
    start_time = time.time()
    for _ in range(num_trials):
        _ = fuse_groupgemm_permute_quant(pytorch_permute_fwd_input, routing_map)
    torch.cuda.synchronize()
    cuda_permute_time_ms = ((time.time() - start_time) / num_trials) * 1000

    return {
        "permute": {
            "correct": "Pass" if  permute_correct and scale_correct and indices_correct else "FAIL",
            "pytorch_ms": pytorch_permute_time_ms,
            "cuda_ms": cuda_permute_time_ms,
            "speedup": pytorch_permute_time_ms / cuda_permute_time_ms if cuda_permute_time_ms > 0 else float('inf'),
        }
    }


def run_benchmark_suite(test_params: Dict[str, Any]):
    print("\n" + "="*80)
    print(f"  Running Benchmark Suite")
    print("="*80)

    permute_results_list = []

    for num_tokens in test_params['num_tokens_list']:
        # print(f"Testing num_tokens: {num_tokens}...")
        results = _test_permutation_mask_map(
            num_tokens=num_tokens,
            num_expert=test_params["num_expert"],
            hidden_size=test_params["hidden_size"],
            topK=test_params["topK"],
            dtype=test_params["dtype"],
        )
        permute_results_list.append({"num_tokens": num_tokens, **results['permute']})

    print("\n--- Permute Performance ---")
    print(f"{'Tokens':<8} | {'Correct':<10} | {'PyTorch (ms)':<15} | {'CUDA (ms)':<12} | {'Speedup':<10}")
    print("-" * 70)
    for res in permute_results_list:
        print(f"{res['num_tokens']:<8} | {res['correct']:<10} | {res['pytorch_ms']:<15.3f} | {res['cuda_ms']:<12.3f} | {res['speedup']:.2f}x")
    
    print("\n" + "="*80)
    print("  Benchmark Suite Finished")
    print("="*80)


def test_permutation_single_case():
    print("GPU:", torch.cuda.get_device_name(0))

    test_params = {
        "num_expert": 256,
        "hidden_size": 7168,
        # "hidden_size": 3072,
        "topK": 8,
        "dtype": torch.bfloat16,
        "num_tokens_list": [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192],
        # "num_tokens_list": [1],
    }

    # 运行一次测试套件
    run_benchmark_suite(test_params)


if __name__ == "__main__":
    test_permutation_single_case()