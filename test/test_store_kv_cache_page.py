import torch
import time
import traceback
import random

def store_kvcache_paged_reference(
    key: torch.Tensor,
    value: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    q_lens: torch.Tensor,
    accum_q_lens: torch.Tensor,
    cache_lens: torch.Tensor,
    block_table: torch.Tensor,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    kv_cache_dtype_str: str,
    block_size: int
):
    """
    PyTorch reference implementation for storing key-value cache in a paged format.
    This function simulates the behavior of the custom CUDA kernel for verification.
    """
    batch_size = q_lens.shape[0]
    torch_dst_dtype = k_cache.dtype

    for i in range(batch_size):
        q_len = q_lens[i].item()
        if q_len == 0:
            continue

        q_offset = accum_q_lens[i].item()
        cur_cache_len = cache_lens[i].item()
        cur_block_table = block_table[i]

        # Extract the key and value tensors for the current sequence
        cur_k = key[q_offset : q_offset + q_len]
        cur_v = value[q_offset : q_offset + q_len]

        # Iterate through each new token to be cached
        for token_idx_in_q in range(q_len):
            src_k_token = cur_k[token_idx_in_q] # Shape: [kv_head_num, head_dim]
            src_v_token = cur_v[token_idx_in_q]

            # Calculate the logical destination position in the cache
            dst_token_idx = cur_cache_len + token_idx_in_q

            # Determine the physical block and the offset within that block
            logical_block_idx = dst_token_idx // block_size
            offset_in_block = dst_token_idx % block_size
            physical_block_id = cur_block_table[logical_block_idx].item()

            # Quantize and store the data
            if kv_cache_dtype_str == "auto":
                k_cache[physical_block_id, :, offset_in_block, :] = src_k_token.to(torch_dst_dtype)
                v_cache[physical_block_id, :, offset_in_block, :] = src_v_token.to(torch_dst_dtype)
            elif kv_cache_dtype_str == "int8":
                k_quant = torch.round(torch.mul(src_k_token, k_scale)).to(torch_dst_dtype)
                v_quant = torch.round(torch.mul(src_v_token, v_scale)).to(torch_dst_dtype)
                k_cache[physical_block_id, :, offset_in_block, :] = k_quant
                v_cache[physical_block_id, :, offset_in_block, :] = v_quant
            else:
                raise ValueError(f"Unsupported kv_cache_dtype_str: {kv_cache_dtype_str}")

def store_kvcache_paged_custom(
    key: torch.Tensor,
    value: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    q_lens: torch.Tensor,
    accum_q_lens: torch.Tensor,
    cache_lens: torch.Tensor,
    block_table: torch.Tensor,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    kv_cache_dtype_str: str
):
    """
    Wrapper for the custom CUDA/HIP kernel for paged KV cache storage.
    """
    try:
        from lightop import op
        # Note: The C++ function infers dimensions from tensors.
        op.store_kv_cache_paged(
            key, value, k_cache, v_cache,
            q_lens, accum_q_lens, cache_lens, block_table,
            k_scale, v_scale, kv_cache_dtype_str
        )
    except (ImportError, AttributeError) as e:
        print("\n\n" + "=" * 80)
        print("Error: Could not import or call 'lightop.op.store_kv_cache_paged'.")
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
    q_len_config,
    kv_cache_dtype_str: str,
    kv_head_num: int,
    head_dim: int,
    max_seq_len: int,
    block_size: int,
    num_blocks: int,
    input_dtype: torch.dtype,
    device: str
):
    """
    A general test function to execute a single test scenario for paged KV cache.
    """
    print(f"\n{'='*25} {case_name}: DTYPE={kv_cache_dtype_str.upper()} {'='*25}")

    cache_dtype = input_dtype if kv_cache_dtype_str == "auto" else torch.int8

    print("===== Test Configuration =====")
    print(f"Batch Size: {batch_size}, KV Heads: {kv_head_num}, Head Dim: {head_dim}")
    print(f"Block Size: {block_size}, Total Blocks: {num_blocks}, Max Seq Len: {max_seq_len}")
    print(f"Input DType: {input_dtype}, Cache DType: {cache_dtype}")

    # --- Data Preparation ---
    if isinstance(q_len_config, int):
        q_lens = torch.full((batch_size,), q_len_config, device=device, dtype=torch.int32)
    else:
        min_q, max_q = q_len_config
        q_lens = torch.randint(min_q, max_q, (batch_size,), device=device, dtype=torch.int32)

    max_q_len_val = q_lens.max().item()
    cache_lens = torch.randint(0, max_seq_len - max_q_len_val, (batch_size,), device=device, dtype=torch.int32)

    total_tokens = torch.sum(q_lens).item()
    accum_q_lens = torch.cumsum(q_lens, dim=0, dtype=torch.int32) - q_lens

    key = torch.randn((total_tokens, kv_head_num, head_dim), dtype=input_dtype, device=device)
    value = torch.randn((total_tokens, kv_head_num, head_dim), dtype=input_dtype, device=device)

    cache_shape = (num_blocks, kv_head_num, block_size, head_dim)
    k_cache_initial = torch.zeros(cache_shape, dtype=cache_dtype, device=device)
    v_cache_initial = torch.zeros(cache_shape, dtype=cache_dtype, device=device)

    # --- Block Table Generation ---
    max_blocks_per_seq = (max_seq_len + block_size - 1) // block_size
    assert num_blocks >= batch_size * max_blocks_per_seq, "Not enough physical blocks to run test."
    
    # Generate a pool of unique physical block IDs and assign them
    block_pool = random.sample(range(num_blocks), batch_size * max_blocks_per_seq)
    block_table = torch.tensor(block_pool, device=device, dtype=torch.int32).view(batch_size, max_blocks_per_seq)

    print(f"Query Lengths (q_lens): {q_lens.tolist()}")
    print(f"Total Tokens in Batch: {total_tokens}, Key/Value Shape: {key.shape}")
    print(f"Block Table Shape: {block_table.shape}")
    print("========================================================\n")

    # --- Scale Factor Generation ---
    if kv_cache_dtype_str == "int8":
        qmax = 127.0
        k_abs_max = torch.randn(kv_head_num, head_dim, device=device).abs() + 1.0
        v_abs_max = torch.randn(kv_head_num, head_dim, device=device).abs() + 1.0
        k_scale = (qmax / k_abs_max).to(torch.float32)
        v_scale = (qmax / v_abs_max).to(torch.float32)
    else:
        k_scale = torch.ones(kv_head_num, head_dim, dtype=torch.float32, device=device)
        v_scale = torch.ones(kv_head_num, head_dim, dtype=torch.float32, device=device)

    # --- Correctness Verification ---
    print("--- Running Correctness Verification ---")
    args_ref = {
        "key": key, "value": value, "k_cache": k_cache_initial.clone(), "v_cache": v_cache_initial.clone(),
        "q_lens": q_lens, "accum_q_lens": accum_q_lens, "cache_lens": cache_lens, "block_table": block_table,
        "k_scale": k_scale, "v_scale": v_scale, "kv_cache_dtype_str": kv_cache_dtype_str, "block_size": block_size
    }
    store_kvcache_paged_reference(**args_ref)

    args_custom = {
        "key": key, "value": value, "k_cache": k_cache_initial.clone(), "v_cache": v_cache_initial.clone(),
        "q_lens": q_lens, "accum_q_lens": accum_q_lens, "cache_lens": cache_lens, "block_table": block_table,
        "k_scale": k_scale, "v_scale": v_scale, "kv_cache_dtype_str": kv_cache_dtype_str
    }
    custom_kernel_available = store_kvcache_paged_custom(**args_custom)

    if custom_kernel_available:
        k_correct = torch.allclose(args_ref["k_cache"], args_custom["k_cache"], atol=1)
        v_correct = torch.allclose(args_ref["v_cache"], args_custom["v_cache"], atol=1)
        if k_correct and v_correct:
            print("✅ Correctness Test Passed: Custom kernel matches the torch reference.")
        else:
            print("❌ Correctness Test Failed: Outputs do NOT match.")
    else:
        print("Skipping correctness test for custom kernel as it could not be imported/run.")

    # # --- Performance Comparison ---
    # print("\n--- Running Performance Comparison ---")
    # time_ref = benchmark(store_kvcache_paged_reference, args_ref)
    # print(f"Reference PyTorch : {time_ref:9.6f} ms (Baseline)")

    # if custom_kernel_available:
    #     time_custom = benchmark(store_kvcache_paged_custom, args_custom)
    #     speedup = time_ref / time_custom if time_custom > 0 else float('inf')
    #     print(f"Custom Kernel     : {time_custom:9.6f} ms ({speedup:.2f}x speedup vs Reference)")


if __name__ == "__main__":
    # --- Global Test Configuration ---
    common_params = {
        "kv_head_num": 16,
        "head_dim": 256,
        "max_seq_len": 4096,
        "block_size": 16,
        "num_blocks": 16000,
        "input_dtype": torch.bfloat16,
        "device": 'cuda'
    }

    # --- Test Scenarios ---
    test_scenarios = {
        "CASE 1: BATCH PREFILL": {
            "batch_size": 16,
            "q_len_config": (1, 1024)
        },
        "CASE 2: SINGLE-REQUEST PREFILL": {
            "batch_size": 1,
            "q_len_config": (2, 1024)
        },
        "CASE 3: SINGLE-REQUEST DECODING": {
            "batch_size": 1,
            "q_len_config": 1
        }
    }

    # --- Run All Tests ---
    for case_name, params in test_scenarios.items():
        for kv_cache_dtype in ["int8", "auto"]:
            run_test_case(
                case_name=case_name,
                kv_cache_dtype_str=kv_cache_dtype,
                **common_params,
                **params
            )
