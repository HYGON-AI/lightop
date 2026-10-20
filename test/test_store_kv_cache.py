# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

import torch
import time
import traceback

def store_kvcache_contiguous_reference(
    packed_qkv: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    q_lens: torch.Tensor,
    accum_q_lens: torch.Tensor,
    cache_lens: torch.Tensor,
    cache_slot_ids: torch.Tensor,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    kv_cache_dtype_str: str,
    q_head_num: int,
    kv_head_num: int
):
    batch_size = q_lens.shape[0]
    torch_dst_dtype = k_cache.dtype

    for i in range(batch_size):
        q_len = q_lens[i].item()
        if q_len == 0:
            continue

        q_offset = accum_q_lens[i].item()
        cur_cache_len = cache_lens[i].item()
        cur_slot_id = cache_slot_ids[i].item()

        token_start = q_offset
        token_end = q_offset + q_len

        k_head_start = q_head_num
        k_head_end = q_head_num + kv_head_num
        v_head_start = q_head_num + kv_head_num
        v_head_end = q_head_num + kv_head_num * 2

        cache_start = cur_cache_len
        cache_end = cur_cache_len + q_len

        cur_k = packed_qkv[token_start:token_end, k_head_start:k_head_end, :].transpose(0, 1)
        cur_v = packed_qkv[token_start:token_end, v_head_start:v_head_end, :].transpose(0, 1)

        if kv_cache_dtype_str == "auto":
            k_cache[cur_slot_id, :, cache_start:cache_end, :] = cur_k.to(torch_dst_dtype)
            v_cache[cur_slot_id, :, cache_start:cache_end, :] = cur_v.to(torch_dst_dtype)
        elif kv_cache_dtype_str == "int8":
            k_cache[cur_slot_id, :, cache_start:cache_end, :] = torch.round(
                torch.mul(cur_k, k_scale.unsqueeze(1))
            ).to(torch_dst_dtype)
            
            v_cache[cur_slot_id, :, cache_start:cache_end, :] = torch.round(
                torch.mul(cur_v, v_scale.unsqueeze(1))
            ).to(torch_dst_dtype)
        else:
            raise ValueError(f"Unsupported kv_cache_dtype_str: {kv_cache_dtype_str}")


#cuda kernal
def store_kvcache_contiguous_custom(
    packed_qkv: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    q_lens: torch.Tensor,
    accum_q_lens: torch.Tensor,
    cache_lens: torch.Tensor,
    cache_slot_ids: torch.Tensor,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    kv_cache_dtype_str: str,
    q_head_num: int,
    kv_head_num: int
):
    
    #自定义 CUDA/HIP kernel 的包装器。
    try:
        from lightop import op
        op.store_kv_cache(
            packed_qkv, k_cache, v_cache,
            q_lens, accum_q_lens, cache_lens, cache_slot_ids,
            k_scale, v_scale,
            kv_cache_dtype_str, q_head_num, kv_head_num
        )
    except (ImportError, AttributeError) as e:
        print("\n\n" + "=" * 80)
        print("错误: 无法导入或调用 'lightop.op.store_kv_cache_contiguous'。")
        print(traceback.format_exc())
        print("=" * 80 + "\n\n")
        return None
    return True



def benchmark(func, args_dict, warmup_runs=20, timed_runs=100):
    for _ in range(warmup_runs):
        func(**args_dict)
    torch.cuda.synchronize()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    start_event.record()
    for _ in range(timed_runs):
        func(**args_dict)
    end_event.record()
    
    torch.cuda.synchronize()
    
    elapsed_time_ms = start_event.elapsed_time(end_event)
    avg_time_ms = elapsed_time_ms / timed_runs
    return avg_time_ms


def run_test_case(
    case_name: str,
    batch_size: int,
    q_len_config, # 可以是整数或元组 (min, max)
    kv_cache_dtype_str: str,
    q_head_num: int,
    kv_head_num: int,
    head_dim: int,
    max_kv_len: int,
    input_dtype: torch.dtype,
    device: str
):
    """
    一个通用的测试函数，用于执行单个测试场景。
    """
    print(f"\n{'='*25} {case_name}: DTYPE={kv_cache_dtype_str.upper()} {'='*25}")

    # --- 根据模式动态设置 Cache 数据类型 ---
    if kv_cache_dtype_str == "auto":
        cache_dtype = input_dtype
    else: # int8
        cache_dtype = torch.int8

    print("===== Test Configuration =====")
    print(f"Batch Size: {batch_size}, Q Heads: {q_head_num}, KV Heads: {kv_head_num}, Head Dim: {head_dim}")
    print(f"Max KV Len: {max_kv_len}, Input DType: {input_dtype}, Cache DType: {cache_dtype}")
    
    # --- 数据准备 ---
    if isinstance(q_len_config, int):
        q_lens = torch.full((batch_size,), q_len_config, device=device, dtype=torch.int32)
    else:
        min_q, max_q = q_len_config
        q_lens = torch.randint(min_q, max_q, (batch_size,), device=device, dtype=torch.int32)
    
    # 确保 cache_lens 不会超出范围
    max_q_len_val = q_lens.max().item()
    
    cache_lens = torch.randint(0, max_kv_len - max_q_len_val, (batch_size,), device=device, dtype=torch.int32)
    
    total_tokens = torch.sum(q_lens).item()
    accum_q_lens = torch.cumsum(q_lens, dim=0, dtype=torch.int32) - q_lens
    cache_slot_ids = torch.arange(batch_size, device=device, dtype=torch.int32)

    total_head_num = q_head_num + 2 * kv_head_num
    packed_qkv = torch.randn((total_tokens, total_head_num, head_dim), dtype=input_dtype, device=device)
    
    cache_shape = (batch_size, kv_head_num, max_kv_len, head_dim)
    k_cache_initial = torch.zeros(cache_shape, dtype=cache_dtype, device=device)
    v_cache_initial = torch.zeros(cache_shape, dtype=cache_dtype, device=device)
    
    print(f"Query Lengths (q_lens): {q_lens.tolist()}")
    print(f"Total Tokens in Batch: {total_tokens}, Packed QKV Shape: {packed_qkv.shape}")
    print("========================================================\n")

    # --- 生成缩放因子 ---
    if kv_cache_dtype_str == "int8":
        qmax = 127.0
        k_abs_max = torch.randn(kv_head_num, head_dim, device=device).abs() + 1.0 
        v_abs_max = torch.randn(kv_head_num, head_dim, device=device).abs() + 1.0
        k_scale = (qmax / k_abs_max).to(torch.float32)
        v_scale = (qmax / v_abs_max).to(torch.float32)
    else:
        k_scale = torch.ones(kv_head_num, head_dim, dtype=torch.float32, device=device)
        v_scale = torch.ones(kv_head_num, head_dim, dtype=torch.float32, device=device)

    args = {
        "packed_qkv": packed_qkv, "q_lens": q_lens, "accum_q_lens": accum_q_lens,
        "cache_lens": cache_lens, "cache_slot_ids": cache_slot_ids,
        "k_scale": k_scale, "v_scale": v_scale, "kv_cache_dtype_str": kv_cache_dtype_str,
        "q_head_num": q_head_num, "kv_head_num": kv_head_num
    }

    # --- 1. 正确性验证 ---
    print("--- Running Correctness Verification ---")
    k_cache_ref, v_cache_ref = k_cache_initial.clone(), v_cache_initial.clone()
    k_cache_custom, v_cache_custom = k_cache_initial.clone(), v_cache_initial.clone()
    
    store_kvcache_contiguous_reference(**{**args, "k_cache": k_cache_ref, "v_cache": v_cache_ref})
    custom_kernel_available = store_kvcache_contiguous_custom(**{**args, "k_cache": k_cache_custom, "v_cache": v_cache_custom})
    
    if custom_kernel_available:
        # 修复点: 将 allclose 检查移到 CPU 以避免 OOM
        print("Moving tensors to CPU for correctness check...")
        k_correct = torch.allclose(k_cache_ref.cpu(), k_cache_custom.cpu(), atol=1)
        v_correct = torch.allclose(v_cache_ref.cpu(), v_cache_custom.cpu(), atol=1)
        if k_correct and v_correct:
            print("✅ Correctness Test Passed: Custom kernel matches the torch reference.")
        else:
            print("❌ Correctness Test Failed: Outputs do NOT match.")
    else:
        print("Skipping correctness test for custom kernel as it could not be imported/run.")

    # # --- 2. 性能比较 ---
    # print("\n--- Running Performance Comparison ---")
    # args_ref_bench = {**args, "k_cache": k_cache_initial.clone(), "v_cache": v_cache_initial.clone()}
    # time_ref = benchmark(store_kvcache_contiguous_reference, args_ref_bench)
    # print(f"Reference PyTorch : {time_ref:9.6f} ms (Baseline)")

    # if custom_kernel_available:
    #     args_custom_bench = {**args, "k_cache": k_cache_initial.clone(), "v_cache": v_cache_initial.clone()}
    #     time_custom = benchmark(store_kvcache_contiguous_custom, args_custom_bench)
    #     speedup = time_ref / time_custom if time_custom > 0 else float('inf')
    #     print(f"Custom Kernel     : {time_custom:9.6f} ms ({speedup:.2f}x speedup vs Reference)")


if __name__ == "__main__":
    # --- 全局测试配置 ---
    common_params = {
        "q_head_num": 32,
        "kv_head_num": 16,
        "head_dim": 256,
        "max_kv_len": 4096,
        "input_dtype": torch.bfloat16,
        "device": 'cuda'
    }

    # --- 定义所有测试场景 ---
    test_scenarios = {
        "CASE 1: BATCH PREFILL": {
            "batch_size": 16,
            "q_len_config": (1, 1024) # q_len 是 1 到 1024 之间的随机数
        },
        "CASE 2: SINGLE-REQUEST PREFILL": {
            "batch_size": 1,
            "q_len_config": (2, 1024) # q_len 是 2 到 1024 之间的随机数
        },
        "CASE 3: SINGLE-REQUEST DECODING": {
            "batch_size": 1,
            "q_len_config": 1 # q_len 固定为 1
        }
    }

    # --- 循环执行所有测试 ---
    for case_name, params in test_scenarios.items():
        for kv_cache_dtype in ["int8", "auto"]:      
            run_test_case(
                case_name=case_name,
                kv_cache_dtype_str=kv_cache_dtype,
                **common_params,
                **params
            )
