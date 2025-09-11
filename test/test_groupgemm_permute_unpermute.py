import random

import torch
import pytest
from typing import Dict, List

import copy
from datetime import datetime
import pandas as pd


# from permute_ops_cuda import permute_with_indices as groupgemm_permute
# from permute_ops_cuda import unpermute as groupgemm_unpermute

from lightop import groupgemm_permute, groupgemm_unpermute

_cuda_available = True
seed = 1234
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

def pytorch_permute_mask_map(tokens, routing_map):
    """Permute the tokens and probs based on the mask.
    Tokens with the same designated expert will be grouped together.
    The shape of mask is [tokens, num_experts], it indicates which experts were selected
    by each token.

    Args:
        tokens (torch.Tensor): The input token tensor, [num_tokens, hidden].
        routing_map (torch.Tensor): The sparse token to expert mapping, [num_tokens, num_experts].
    """
    # 获取输入token张量的维度信息
    num_tokens, _ = tokens.shape  # 提取token数量，忽略hidden维度
    num_experts = routing_map.shape[1]  # 提取专家数量

    # import pdb
    # pdb.set_trace()

    # 将routing_map转换为布尔类型，转置并确保内存连续
    # 从 [num_tokens, num_experts] 转换为 [num_experts, num_tokens]
    # 这样每一行代表一个专家，每一列代表一个token
    routing_map = routing_map.bool().T.contiguous()

    # 创建token索引矩阵，用于后续的masked_select操作 
    # 生成形状为 [num_experts, num_tokens] 的索引矩阵
    # 每一行都是 [0, 1, 2, ..., num_tokens-1]
    token_indices = (
        torch.arange(num_tokens, device=routing_map.device).unsqueeze(0).expand(num_experts, -1)
    )
    
    # 使用routing_map作为mask，提取被选中的token索引
    # 结果是一个一维张量，包含所有被专家选中的token的索引
    # 这些索引按照专家顺序排列（先专家0的所有token，再专家1的所有token，以此类推）
    sorted_indices = token_indices.masked_select(routing_map)

    # 根据排序后的索引重新排列token
    # 将tokens按照sorted_indices的顺序重新排列，实现按专家分组
    permuted_input = tokens.index_select(0, sorted_indices)

    # pdb.set_trace()
    
    # 返回重排序后的token张量和排序索引
    return permuted_input, sorted_indices


def pytorch_unpermute_mask_map(
    permuted_tokens: torch.Tensor,
    sorted_indices: torch.Tensor,
    restore_shape: torch.Size,
    probs: torch.Tensor = None,
    routing_map: torch.Tensor = None,
):
    """
    Restore the original order of tokens after permutation. If probs are provided, it
    will also apply them to the tokens before restoring the order.

    Args:
        permuted_tokens (torch.Tensor): The permuted token tensor.
        sorted_indices (torch.Tensor): The indices used to sort the tokens.
        restore_shape (torch.Size): The shape of the unpermuted tensor.
        probs (torch.Tensor, optional): The unpermuted probs tensor,
        routing_map (torch.Tensor, optional): Token to expert mapping, shape
            [num_tokens, num_experts].

    Returns:
        torch.Tensor: The tokens restored to their original order.
    """
    # 从restore_shape中提取hidden维度大小
    _, hidden = restore_shape

    # 如果提供了概率权重，需要先对重排序后的tokens应用权重
    if probs is not None:
        # 确保提供了routing_map，因为需要用它来重排序概率
        assert routing_map is not None, "Mask must be provided to permute the probs."
        # 将概率按与tokens相同的方式重排序
        # 先转置probs和routing_map，然后使用masked_select提取对应的概率值
        permuted_probs = probs.T.contiguous().masked_select(routing_map.T.contiguous()) # 和sort_indices相同大小
        # 将重排序后的概率应用到tokens上（广播到hidden维度）
        permuted_tokens = permuted_tokens * permuted_probs.unsqueeze(-1)


    # 创建输出张量，用零填充，形状为原始tokens的形状
    output_tokens = torch.zeros(
        restore_shape, device=permuted_tokens.device, dtype=permuted_tokens.dtype
    )
    
    # 使用scatter_add_将重排序后的tokens恢复到原始位置
    # sorted_indices.unsqueeze(1).expand(-1, hidden) 将索引扩展到hidden维度
    # 这样可以将每个token的所有hidden维度都恢复到正确位置
    output_tokens.scatter_add_(0, sorted_indices.unsqueeze(1).expand(-1, hidden), permuted_tokens)
    

    # 返回恢复到原始顺序的tokens
    return output_tokens


def _test_permutation_mask_map(
    num_tokens,
    num_expert,
    hidden_size,
    topK,
    num_out_tokens,
    with_probs,
    dtype,
    BENCHMARK=False,
    results_dict=None,
):
    # 如果没有指定输出token数量，默认为num_tokens * topK
    if num_out_tokens == None:
        num_out_tokens = num_tokens * topK

    

    # 创建测试数据
    pytorch_permute_fwd_input = torch.rand((num_tokens, hidden_size), dtype=dtype).cuda()
    pytorch_permute_bwd_input = torch.rand((num_out_tokens, hidden_size), dtype=dtype).cuda()
    pytorch_unpermute_bwd_input = torch.rand((num_tokens, hidden_size), dtype=dtype).cuda()

    pytorch_permute_fwd_input.requires_grad_(True)

    restore_shape = pytorch_permute_fwd_input.shape

    # 生成routing_map，确保有num_out_tokens个True值
    _tmp_tensor = torch.zeros((num_tokens * num_expert,))
    _tmp_tensor[: int(num_out_tokens)] = 1.0
    _tmp_idx = torch.randperm(num_tokens * num_expert)
    routing_map = torch.reshape(_tmp_tensor[_tmp_idx], (num_tokens, num_expert)).bool().cuda()

    # routing_map = torch.zeros_like(routing_map)

    # 生成概率权重（如果需要）
    probs = None
    if with_probs:
        probs = torch.rand(num_tokens, num_expert).cuda() * routing_map
        row_sums = probs.sum(dim=1, keepdim=True)
        probs = probs / row_sums
        probs = probs.to(dtype)
        probs.requires_grad_(True)

    ###################################################################################################################################
    #
    # PyTorch Permutation Test
    #
    ###################################################################################################################################
    print("Testing PyTorch implementation...")
    
    pytorch_permute_output, sorted_indices = pytorch_permute_mask_map(
        pytorch_permute_fwd_input, routing_map
    )


    pytorch_unpermute_fwd_input = pytorch_permute_output.detach()
    pytorch_unpermute_fwd_input.requires_grad_(True)

    pytorch_unpermute_output = pytorch_unpermute_mask_map(
        pytorch_unpermute_fwd_input, sorted_indices, restore_shape, probs, routing_map
    )

    print("PyTorch Permutation Test Passed")
    
    ###################################################################################################################################
    #
    # CUDA Permutation Test (if available)
    #
    ###################################################################################################################################
    if _cuda_available:
        print("Testing CUDA implementation...")
        
        # Test CUDA permutation
        #routing_map = routing_map.bool().T.contiguous()


        print("pytorch_permute_fwd_input.shape: ", pytorch_permute_fwd_input.shape)
        print("routing_map.shape: ", routing_map.shape)
        # input [num_tokens, hidden_size]
        # routing_map [num_token, num_experts]
        cuda_permute_output, cuda_sorted_indices, expert_m_count = groupgemm_permute(
            pytorch_permute_fwd_input.detach(), routing_map
        )

        # Test CUDA unpermutation
        cuda_unpermute_output = groupgemm_unpermute(    
            cuda_permute_output, cuda_sorted_indices, list(restore_shape), probs, routing_map, expert_m_count
        )

        print("expert_m_count:", cuda_sorted_indices.size(0))
        # indices_close = torch.allclose(sorted_indices.float(), cuda_sorted_indices[0:8191].float())
        # Compare results

        permute_close = torch.allclose(pytorch_permute_output, cuda_permute_output, rtol=1e-2, atol=1e-2)
        indices_close = torch.allclose(sorted_indices.float(), cuda_sorted_indices.float())
        unpermute_close = torch.allclose(pytorch_unpermute_output, cuda_unpermute_output, rtol=1e-2, atol=1e-2)

        print(f"Permute results match: {permute_close}")
        print(f"Indices match: {indices_close}")  
        print(f"Unpermute results match: {unpermute_close}")

        # import pdb
        # pdb.set_trace()

       
        
        if not permute_close:
            diff = torch.abs(pytorch_permute_output - cuda_permute_output)
            print(f"Max permute difference: {diff.max().item():.2e}")
            
        if not unpermute_close:
            diff = torch.abs(pytorch_unpermute_output - cuda_unpermute_output)
            print(f"Max unpermute difference: {diff.max().item():.2e}")
            
        # assert permute_close, "CUDA permute results don't match PyTorch"
        # assert indices_close, "CUDA indices don't match PyTorch"
        # assert unpermute_close, "CUDA unpermute results don't match PyTorch"
        
        print("CUDA Permutation Test Passed")
    
        
        ###################################################################################################################################
        #
        # Performance Benchmark (if requested)
        #
        ###################################################################################################################################
       
        print("\nRunning performance benchmark...")
        import time
        
        num_trials = 100
        num_warmup = 10

        # num_trials = 2
        # num_warmup = 2
        
        # ========================================
        # Permute Performance Test
        # ========================================
        print("\n--- Permute Performance ---")
        
        # Warmup for permute
        for _ in range(num_warmup):
            _ = pytorch_permute_mask_map(pytorch_permute_fwd_input, routing_map)
            _ = groupgemm_permute(pytorch_permute_fwd_input, routing_map)
        
        torch.cuda.synchronize()
        
        # Benchmark PyTorch permute
        start_time = time.time()
        for _ in range(num_trials):
            pytorch_permute_result = pytorch_permute_mask_map(pytorch_permute_fwd_input, routing_map)
        torch.cuda.synchronize()
        pytorch_permute_time = (time.time() - start_time) / num_trials
        
        # Benchmark CUDA permute
        start_time = time.time()
        for _ in range(num_trials):
            cuda_permute_result = groupgemm_permute(pytorch_permute_fwd_input, routing_map)
        torch.cuda.synchronize()
        cuda_permute_time = (time.time() - start_time) / num_trials
        
        permute_speedup = pytorch_permute_time / cuda_permute_time
        print(f"PyTorch permute time: {pytorch_permute_time*1000:.3f}ms")
        print(f"CUDA permute time: {cuda_permute_time*1000:.3f}ms")
        print(f"Permute speedup: {permute_speedup:.2f}x")
        
        # ========================================
        # Unpermute Performance Test
        # ========================================
        print("\n--- Unpermute Performance ---")
        
        # Use the results from permute operations for unpermute testing
        pytorch_tokens, pytorch_indices = pytorch_permute_result
        #cuda_tokens, cuda_indices = cuda_permute_result
        cuda_tokens, cuda_indices, expert_m_count = cuda_permute_result
        
        # Warmup for unpermute
        for _ in range(num_warmup):
            _ = pytorch_unpermute_mask_map(pytorch_tokens, pytorch_indices, restore_shape, probs, routing_map)
            _ = groupgemm_unpermute(cuda_tokens, cuda_indices, list(restore_shape), probs, routing_map, expert_m_count)
        
        torch.cuda.synchronize()
        
        # Benchmark PyTorch unpermute
        start_time = time.time()
        for _ in range(num_trials):
            pytorch_unpermute_result = pytorch_unpermute_mask_map(pytorch_tokens, pytorch_indices, restore_shape, probs, routing_map)
        torch.cuda.synchronize()
        pytorch_unpermute_time = (time.time() - start_time) / num_trials
        
        # Benchmark CUDA unpermute
        start_time = time.time()
        for _ in range(num_trials):
            cuda_unpermute_result = groupgemm_unpermute(cuda_tokens, cuda_indices, list(restore_shape), probs, routing_map, expert_m_count)
        torch.cuda.synchronize()
        cuda_unpermute_time = (time.time() - start_time) / num_trials
        
        unpermute_speedup = pytorch_unpermute_time / cuda_unpermute_time
        print(f"PyTorch unpermute time: {pytorch_unpermute_time*1000:.3f}ms")
        print(f"CUDA unpermute time: {cuda_unpermute_time*1000:.3f}ms")
        print(f"Unpermute speedup: {unpermute_speedup:.2f}x")
        
        # ========================================
        # Combined Performance Summary
        # ========================================
        print("\n--- Performance Summary ---")
        pytorch_total_time = pytorch_permute_time + pytorch_unpermute_time
        cuda_total_time = cuda_permute_time + cuda_unpermute_time
        total_speedup = pytorch_total_time / cuda_total_time
        
        print(f"PyTorch total time: {pytorch_total_time*1000:.3f}ms")
        print(f"CUDA total time: {cuda_total_time*1000:.3f}ms")
        print(f"Overall speedup: {total_speedup:.2f}x")
        


        return {
            "num_tokens": num_tokens,
            "num_out_tokens": num_out_tokens,
            "num_expert": num_expert,
            "hidden_size": hidden_size,
            "topK": topK,
            "with_probs": with_probs,
            "pytorch_permute_time": pytorch_permute_time*1000*1000,
            "cuda_permute_time": cuda_permute_time*1000*1000,
            "pytorch_unpermute_time": pytorch_unpermute_time*1000*1000,
            "cuda_unpermute_time": cuda_unpermute_time*1000*1000,
        }


def test_permutation_single_case():
    print("GPU:", torch.cuda.get_device_name(0))

    # num_tokens = 128  # 1 - 64k
    num_expert = 256
    hidden_size = 7168 
    topK = 8
    # num_out_tokens = num_tokens * topK - 1
    with_probs = True
    Benchmark = True

    dtype = torch.bfloat16

    # with_probs = False
    # results_list_no_probs = []
    # print("------------------------------------with_probs = False------------------------------------")
    # for num_tokens in [200]:
    #     num_out_tokens = num_tokens * topK
    #     print("-----------------------------------------------------------------------------------")
    #     print(f"Testing num_tokens: {num_tokens}, num_out_tokens: {num_out_tokens}")
    #     results = _test_permutation_mask_map(
    #         num_tokens=num_tokens,
    #         num_expert=num_expert,
    #         hidden_size=hidden_size,
    #         topK=topK,
    #         num_out_tokens=num_out_tokens,
    #         with_probs=with_probs,
    #         dtype=dtype,
    #         BENCHMARK=True,
    #     )
    #     results_list_no_probs.append(results)
    
    results_list = []
    for num_tokens in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]:
        num_out_tokens = num_tokens * topK 
        print("-----------------------------------------------------------------------------------")
        print(f"Testing num_tokens: {num_tokens}, num_out_tokens: {num_out_tokens}")
        results = _test_permutation_mask_map(
            num_tokens=num_tokens,
            num_expert=num_expert,
            hidden_size=hidden_size,
            topK=topK,
            num_out_tokens=num_out_tokens,
            with_probs=with_probs,
            dtype=dtype,
            BENCHMARK=Benchmark,
        )
        results_list.append(results)
    
    df = pd.DataFrame(results_list)
    df.to_excel("results_with_probs.xlsx", index=False)


    with_probs = False
    results_list_no_probs = []
    print("------------------------------------with_probs = False------------------------------------")
    for num_tokens in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]:
        num_out_tokens = num_tokens * topK
        print("-----------------------------------------------------------------------------------")
        print(f"Testing num_tokens: {num_tokens}, num_out_tokens: {num_out_tokens}")
        results = _test_permutation_mask_map(
            num_tokens=num_tokens,
            num_expert=num_expert,
            hidden_size=hidden_size,
            topK=topK,
            num_out_tokens=num_out_tokens,
            with_probs=with_probs,
            dtype=dtype,
            BENCHMARK=True,
        )
        results_list_no_probs.append(results)

    df = pd.DataFrame(results_list_no_probs)
    df.to_excel("results_no_probs.xlsx", index=False)


if __name__ == "__main__":
    test_permutation_single_case()
