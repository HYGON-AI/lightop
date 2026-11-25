# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

import sys
import random

import pytest
import torch
import time
import traceback

FP8_DTYPE_ALIASES = ("fp8", "fp8_e4m3", "float8", "float8_e4m3fn")
# OCP E4M3 finite max; used for static scale generation (matches custom_ops style).
FP8_E4M3_MAX = 448.0


def _is_fp8_dtype(kv_cache_dtype_str: str) -> bool:
    return kv_cache_dtype_str in FP8_DTYPE_ALIASES


def _cache_dtype_for(kv_cache_dtype_str: str, input_dtype: torch.dtype) -> torch.dtype:
    if kv_cache_dtype_str == "auto":
        return input_dtype
    if kv_cache_dtype_str == "int8":
        return torch.int8
    if _is_fp8_dtype(kv_cache_dtype_str):
        # Kernel writes 1-byte E4M3 bit patterns via uint8_t*.
        return torch.uint8
    raise ValueError(f"Unsupported kv_cache_dtype_str: {kv_cache_dtype_str}")


def _quantize_fp8_e4m3(x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Scale then cast to float8_e4m3fn; return uint8 bit storage."""
    # scale broadcast: x is [kv_head, tokens, dim] or [kv_head, dim]
    q = torch.mul(x.float(), scale).to(torch.float8_e4m3fn)
    return q.view(torch.uint8)


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
        elif _is_fp8_dtype(kv_cache_dtype_str):
            k_cache[cur_slot_id, :, cache_start:cache_end, :] = _quantize_fp8_e4m3(
                cur_k, k_scale.unsqueeze(1)
            )
            v_cache[cur_slot_id, :, cache_start:cache_end, :] = _quantize_fp8_e4m3(
                cur_v, v_scale.unsqueeze(1)
            )
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

    cache_dtype = _cache_dtype_for(kv_cache_dtype_str, input_dtype)

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
    elif _is_fp8_dtype(kv_cache_dtype_str):
        k_abs_max = torch.randn(kv_head_num, head_dim, device=device).abs() + 1.0
        v_abs_max = torch.randn(kv_head_num, head_dim, device=device).abs() + 1.0
        k_scale = (FP8_E4M3_MAX / k_abs_max).to(torch.float32)
        v_scale = (FP8_E4M3_MAX / v_abs_max).to(torch.float32)
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

    passed = False
    if custom_kernel_available:
        # 修复点: 将 allclose 检查移到 CPU 以避免 OOM
        print("Moving tensors to CPU for correctness check...")
        # int8/fp8: allow ±1 code mismatch from RNE/HW vs torch cast edge cases
        atol = 1 if kv_cache_dtype_str != "auto" else 0
        if kv_cache_dtype_str == "auto":
            k_correct = torch.allclose(k_cache_ref.cpu().float(), k_cache_custom.cpu().float(), atol=1e-3, rtol=1e-3)
            v_correct = torch.allclose(v_cache_ref.cpu().float(), v_cache_custom.cpu().float(), atol=1e-3, rtol=1e-3)
        else:
            k_correct = torch.allclose(k_cache_ref.cpu(), k_cache_custom.cpu(), atol=atol)
            v_correct = torch.allclose(v_cache_ref.cpu(), v_cache_custom.cpu(), atol=atol)
        if k_correct and v_correct:
            print("✅ Correctness Test Passed: Custom kernel matches the torch reference.")
            passed = True
        else:
            k_diff = (k_cache_ref.cpu().to(torch.int16) - k_cache_custom.cpu().to(torch.int16)).abs().max().item() if k_cache_ref.dtype in (torch.uint8, torch.int8) else float('nan')
            v_diff = (v_cache_ref.cpu().to(torch.int16) - v_cache_custom.cpu().to(torch.int16)).abs().max().item() if v_cache_ref.dtype in (torch.uint8, torch.int8) else float('nan')
            print(f"❌ Correctness Test Failed: Outputs do NOT match. max|k_diff|={k_diff}, max|v_diff|={v_diff}")
    else:
        print("Skipping correctness test for custom kernel as it could not be imported/run.")

    return passed


# --- module-level configuration (shared by the pytest cases and the sweep) ---
COMMON_PARAMS = {
    "q_head_num": 32,
    "kv_head_num": 16,
    "head_dim": 256,
    "max_kv_len": 4096,
    "input_dtype": torch.bfloat16,
    "device": "cuda",
}

TEST_SCENARIOS = {
    "CASE 1: BATCH PREFILL": {
        "batch_size": 16,
        "q_len_config": (1, 1024),  # q_len 是 1 到 1024 之间的随机数
    },
    "CASE 2: SINGLE-REQUEST PREFILL": {
        "batch_size": 1,
        "q_len_config": (2, 1024),  # q_len 是 2 到 1024 之间的随机数
    },
    "CASE 3: SINGLE-REQUEST DECODING": {
        "batch_size": 1,
        "q_len_config": 1,  # q_len 固定为 1
    },
    # batch=1 + small kv_head triggers flat_prefill path (threads_needed < 64)
    "CASE 4: FLAT PREFILL (kv_head=1)": {
        "batch_size": 1,
        "q_len_config": 128,
        "kv_head_num": 1,
        "q_head_num": 8,
    },
}

# CI keeps the runtime bounded: two representative cases x two quantized dtypes.
# int8/fp8 are the paths this file is meant to cover; `auto` is exercised by the
# manual sweep below.
CI_CASES = (
    "CASE 3: SINGLE-REQUEST DECODING",
    "CASE 4: FLAT PREFILL (kv_head=1)",
)
CI_DTYPES = ("int8", "fp8")


@pytest.mark.parametrize("case_name", CI_CASES)
@pytest.mark.parametrize("kv_cache_dtype", CI_DTYPES)
def test_store_kv_cache(case_name: str, kv_cache_dtype: str) -> None:
    if not torch.cuda.is_available():
        pytest.skip("requires HCU")
    torch.manual_seed(0)
    random.seed(0)
    case_params = {**COMMON_PARAMS, **TEST_SCENARIOS[case_name]}
    ok = run_test_case(
        case_name=case_name,
        kv_cache_dtype_str=kv_cache_dtype,
        **case_params,
    )
    assert ok, (
        f"{case_name} / {kv_cache_dtype}: custom kernel unavailable or output "
        "mismatch against the torch reference (see stdout above)"
    )


def run_full_sweep() -> int:
    """Manual sweep: every scenario x every dtype (not used by CI)."""
    torch.manual_seed(0)
    random.seed(0)
    dtypes = ["int8", "auto", "fp8"]
    results = []
    for case_name, params in TEST_SCENARIOS.items():
        for kv_cache_dtype in dtypes:
            case_params = {**COMMON_PARAMS, **params}
            ok = run_test_case(
                case_name=case_name,
                kv_cache_dtype_str=kv_cache_dtype,
                **case_params,
            )
            results.append((case_name, kv_cache_dtype, ok))

    failed = [r for r in results if not r[2]]
    print("\n===== SUMMARY =====")
    for name, dtype, ok in results:
        print(f"{'PASS' if ok else 'FAIL'}: {name} / {dtype}")
    return 1 if failed else 0


if __name__ == "__main__":
    if "--full" in sys.argv:
        raise SystemExit(run_full_sweep())
    raise SystemExit(pytest.main([__file__, "-q"]))
