# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

import csv
import random

import pytest
import torch
from lightop.attention import mqa_logits, paged_mqa_logits


DECODE_MODES = [(False, False, False), (True, False, False), (False, True, False), (False, False, True)]
CSV_FIELDS = [
    "Mode", "Impl", "Dtype", "Chunk", "Q Shape", "KV Shape", "Weights Shape", "Context Lens Shape",
    "Page Table Shape", "Max Context Len", "Correctness", "Latency (us)", "TFLOPS",
    "Logical BW (GB/s)", "Raw Samples (us)",
]


def generate_cp_test_data(seq_len, seq_len_kv):
    assert seq_len_kv % seq_len == 0 and seq_len % 2 == 0
    chunk_size = seq_len // 2
    cp_size = seq_len_kv // seq_len
    # Select an arbitrary CP rank
    cp_id = cp_size // 3
    ks = torch.zeros(seq_len, dtype=torch.int, device='cuda')
    ke = torch.zeros(seq_len, dtype=torch.int,  device='cuda')
    for i in range(chunk_size):
        ke[i] = cp_id * chunk_size + i
        ke[i + chunk_size] = (cp_size * 2 - 1 - cp_id) * chunk_size + i
    return ks, ke

def ceil_div(x: int, y: int) -> int:
    return (x + y - 1) // y

def kv_cache_cast_to_fp8(x: torch.Tensor) -> torch.Tensor:
    num_blocks, block_size, num_heads, head_dim = x.shape
    assert num_heads == 1
    x_amax = x.abs().float().amax(dim=3, keepdim=True).clamp(1e-4)
    sf = x_amax / 448.0
    x_scaled = (x * (1.0 / sf)).to(torch.float8_e4m3fn)
    x_fp8 = torch.empty((num_blocks, block_size * (head_dim + 4)), device=x.device, dtype=torch.uint8)
    x_fp8[ :, : block_size * head_dim] = x_scaled.view(num_blocks, block_size * head_dim).view(dtype=torch.uint8)
    x_fp8[ :, block_size * head_dim :] = sf.view(num_blocks, block_size).view(dtype=torch.uint8)
    return x_fp8.view(num_blocks, block_size, num_heads, head_dim + 4)

def calc_diff(x: torch.Tensor, y: torch.Tensor):
    x, y = x.double(), y.double()
    denominator = (x * x + y * y).sum()
    sim = 2 * (x * y).sum() / denominator
    return 1 - sim

def count_bytes(*tensors):
    total = 0
    for t in tensors:
        if isinstance(t, (tuple, list)):
            total += count_bytes(*t)
        elif t is not None:
            total += t.numel() * t.element_size()
    return total

def bench_kineto(func, warmups=5, reps=20):
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    for _ in range(warmups):
        func()

    torch.cuda.synchronize()

    total_time_ms = 0.0
    for _ in range(reps):
        start_event.record()
        func()
        end_event.record()
        torch.cuda.synchronize()
        total_time_ms += start_event.elapsed_time(end_event)

    avg_time_s = (total_time_ms / reps) / 1000.0
    return avg_time_s

def ref_paged_mqa_logits(q: torch.Tensor, kv_cache: torch.Tensor, weights: torch.Tensor,
                         context_lens: torch.Tensor, block_tables: torch.Tensor, max_model_len: int,
                         kv_scales: torch.Tensor = None):
    batch_size, next_n = q.size(0), q.size(1)
    block_size = kv_cache.size(1)
    logits = torch.full([batch_size * next_n, max_model_len], float('-inf'), device=q.device, dtype=torch.float32)
    context_lens = context_lens.tolist()
    for i in range(batch_size):
        context_len = context_lens[i]
        q_offsets = torch.arange(context_len - next_n, context_len, device='cuda')
        weight_slice = weights[i * next_n:(i + 1) * next_n, :].transpose(0, 1).contiguous()
        for block_rk in range(ceil_div(context_len, block_size)):
            block_idx = block_tables[i][block_rk]
            # qx, kx = q[i], kv_cache[block_idx]
            qx, kx = q[i].to(torch.float32), kv_cache[block_idx].to(torch.float32)
            k_offsets = torch.arange(block_rk * block_size, (block_rk + 1) * block_size, device='cuda')
            mask = (k_offsets[None, :] < context_len) & (k_offsets[None, :] <= q_offsets[:, None])
            s = torch.where(mask[None, :, :], (qx.transpose(0, 1) @ kx.transpose(0, 1).transpose(1, 2)).to(logits.dtype), float('-inf'))
            s = torch.relu(s) * weight_slice[..., None]
            s = s.sum(dim=0)
            if kv_scales is not None:
                s *= kv_scales[block_idx]
            logits[i * next_n:(i + 1) * next_n, block_rk * block_size: (block_rk + 1) * block_size] = torch.where(k_offsets[None, :] <= q_offsets[:, None], s, float('-inf'))
    return logits

@pytest.mark.parametrize("use_fp8,use_mixed,use_int8", DECODE_MODES, ids=["bf16", "fp8", "mixed", "int8"])
def test_paged_mqa_logits(use_fp8, use_mixed, use_int8, csv_writer=None):
    assert sum((use_fp8, use_mixed, use_int8)) <= 1

    if not torch.cuda.is_available():
        pytest.skip("A CUDA/ROCm device is required")
    random.seed(20260914)
    torch.manual_seed(20260914)
    device_props = torch.cuda.get_device_properties(0)
    device_name = (
        device_props.gcnArchName.split(':')[0] if hasattr(device_props, 'gcnArchName') else device_props.name
    )
    if use_fp8 and 'gfx938' not in device_name:
        pytest.skip(f'FP8 Paged MQA Logits requires gfx938, got {device_name}')

    max_model_len = 320 if use_int8 else 111 * 1000
    batch_next_cases = [(2, 1), (2, 2)] if use_int8 else [(1, 1), (64, 1), (64, 2), (128, 1)]
    avg_kv_values = (287,) if use_int8 else (4096, 72000)
    block_sizes = (4, 8, 16, 32, 64) if use_int8 else (64,)
    for batch_size, next_n in batch_next_cases:
        for heads, index_dim in [(8, 128), (16, 128), (32, 128), (64, 128)]:
            for avg_kv, blocksize in (
                (avg_kv, blocksize)
                for avg_kv in avg_kv_values
                for blocksize in block_sizes
            ):
                if use_int8:
                    context_lens = torch.tensor(
                        [min(max_model_len, blocksize * 3 + 3), 287], device='cuda', dtype=torch.int32
                    )
                    page_counts = [ceil_div(int(context_len), blocksize) for context_len in context_lens]
                    num_blocks = sum(page_counts) + 11
                else:
                    num_blocks = max_model_len * 3

                if use_int8:
                    q = torch.randint(
                        -127, 128, (batch_size, next_n, heads, index_dim), device='cuda', dtype=torch.int8
                    )
                    kv_cache = torch.randint(
                        -127, 128, (num_blocks, blocksize, 1, index_dim), device='cuda', dtype=torch.int8
                    )
                    kv_scales = torch.rand(
                        (num_blocks, blocksize), device='cuda', dtype=torch.float32
                    ) * 0.09 + 0.01
                else:
                    q = torch.randn(
                        (batch_size, next_n, heads, index_dim), device='cuda', dtype=torch.bfloat16
                    )
                    kv_cache = torch.randn(
                        (num_blocks, blocksize, 1, index_dim), device='cuda', dtype=torch.bfloat16
                    )
                    kv_scales = None
                weights = torch.randn((batch_size * next_n, heads), device='cuda', dtype=torch.float32)

                if not use_int8:
                    context_lens = torch.randint(
                        int(0.9 * avg_kv), int(1.1 * avg_kv), (batch_size,), device='cuda', dtype=torch.int32
                    )
                    page_counts = [ceil_div(int(context_len), blocksize) for context_len in context_lens]
                max_block_len = (
                    max(page_counts)
                    if use_int8
                    else (context_lens.max().item() + blocksize - 1) // blocksize * blocksize
                )
                block_tables = torch.zeros((batch_size, max_block_len), device='cuda', dtype=torch.int32)

                counter = 0
                block_idx_pool = list(range(num_blocks))
                random.shuffle(block_idx_pool)
                for i in range(batch_size):
                    for j in range(page_counts[i]):
                        block_tables[i][j] = block_idx_pool[counter]
                        counter += 1

                schedule_metadata = None
                torch.cuda.synchronize()

                if use_int8:
                    test_name = 'INT8'
                    q_input = q
                    kv_cache_input = pack_int8_fused_cache(kv_cache, kv_scales)
                elif use_fp8:
                    test_name = 'FP8'
                    q_input = q.to(torch.float8_e4m3fn)
                    kv_cache_input = kv_cache_cast_to_fp8(kv_cache)
                elif use_mixed:
                    test_name = 'BF16-Q/FP8-KV'
                    q_input = q
                    kv_cache_input = kv_cache_cast_to_fp8(kv_cache)
                else:
                    test_name = 'BF16'
                    q_input = q
                    kv_cache_input = kv_cache

                logits = paged_mqa_logits(
                    q_input, kv_cache_input, weights, context_lens, block_tables, schedule_metadata, max_model_len,
                    clean_logits=True
                )
                torch.cuda.synchronize()

                ref_logits = ref_paged_mqa_logits(
                    q, kv_cache, weights, context_lens, block_tables, max_model_len, kv_scales=kv_scales
                )
                positions = torch.arange(max_model_len, device='cuda').unsqueeze(0).expand(batch_size * next_n, -1)
                row_indices = torch.arange(batch_size * next_n, device='cuda') // next_n
                next_n_offset = torch.arange(batch_size * next_n, device='cuda') % next_n
                ref_neginf_mask = ~(
                    positions <= (context_lens[row_indices] - next_n + next_n_offset).unsqueeze(1)
                )
                neginf_mask = logits == float('-inf')
                assert torch.equal(neginf_mask, ref_neginf_mask)

                logits = logits.masked_fill(neginf_mask, 0)
                ref_logits = ref_logits.masked_fill(ref_neginf_mask, 0)
                if use_int8:
                    assert torch.isfinite(logits).all()
                    max_abs = (logits - ref_logits).abs().max()
                    max_ref = ref_logits.abs().max()
                    normalized_error = max_abs / max_ref.clamp_min(1.0)
                    assert normalized_error < 2e-5, f'{max_abs=}, {normalized_error=}, not pass'
                elif use_fp8 or use_mixed:
                    diff = calc_diff(logits, ref_logits)
                    tolerance = 1.1e-3 if batch_size == 1 else 1e-3
                    assert diff < tolerance, f'{diff=}, not pass'
                else:
                    assert torch.allclose(logits, ref_logits, rtol=1e-3, atol=1e-3)

                sum_lens = sum(context_lens.to(torch.int64))
                tflops = 2 * sum_lens * next_n * heads * index_dim / 1e12
                quantized_kv = use_fp8 or use_mixed or use_int8
                input_bytes = (
                    count_bytes(q_input, weights, context_lens)
                    + sum_lens * (index_dim * (1 if quantized_kv else 2) + (4 if quantized_kv else 0))
                    + (sum_lens / blocksize) * 4
                )
                output_bytes = batch_size * max_model_len * next_n * 4
                t = bench_kineto(
                    lambda: paged_mqa_logits(
                        q_input, kv_cache_input, weights, context_lens, block_tables, schedule_metadata, max_model_len,
                        clean_logits=False
                    )
                )
                print(f'decode {test_name} Q={tuple(q_input.shape)} KV={tuple(kv_cache_input.shape)} | '
                      f'PASS | {t * 1e6:.2f} us | {tflops / t:.2f} TFLOPS | '
                      f'{(input_bytes + output_bytes) / t / 1e9:.2f} GB/s')
                if csv_writer is not None:
                    csv_writer.writerow({
                        "Mode": "decode", "Impl": "paged", "Dtype": test_name,
                        "Q Shape": tuple(q_input.shape), "KV Shape": tuple(kv_cache_input.shape),
                        "Weights Shape": tuple(weights.shape), "Context Lens Shape": tuple(context_lens.shape),
                        "Page Table Shape": tuple(block_tables.shape), "Max Context Len": max_model_len,
                        "Correctness": "PASS", "Latency (us)": f"{t * 1e6:.2f}",
                        "TFLOPS": f"{tflops / t:.2f}",
                        "Logical BW (GB/s)": f"{(input_bytes + output_bytes) / t / 1e9:.2f}",
                    })


def pack_int8_fused_cache(kv: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    num_blocks, block_kv, num_kv_heads, head_dim = kv.shape
    assert num_kv_heads == 1 and head_dim == 128

    packed = torch.empty((num_blocks, block_kv * (head_dim + 4)), device=kv.device, dtype=torch.int8)
    packed[:, :block_kv * head_dim] = kv.reshape(num_blocks, block_kv * head_dim)
    packed[:, block_kv * head_dim:] = scales.contiguous().view(dtype=torch.int8)
    return packed.view(num_blocks, block_kv, 1, head_dim + 4)


def make_prefix_prefill_case(chunk, device='cuda', seed=20260914):
    """The four CP8 chunks of one 128K request, with one shared KV sequence.

    Packed pages contain all 64 INT8 keys followed by all 64 FP32 scales;
    the apparent last dimension of 132 is not a per-token interleaved layout.
    The cache and physical page permutation are identical across chunks.
    """
    if chunk not in (1, 2, 3, 4):
        raise ValueError('chunk must be in [1, 4]')
    tokens, heads, head_dim, block_size, physical_pages = 4096, 64, 128, 64, 1161
    max_context_len = chunk * 8192
    generator = torch.Generator(device=device).manual_seed(seed)
    kv = torch.randint(-128, 128, (physical_pages, block_size, 1, head_dim),
                       dtype=torch.int8, device=device, generator=generator)
    scales = torch.rand((physical_pages, block_size), dtype=torch.float32,
                        device=device, generator=generator) * 0.09 + 0.01
    shared_pages = torch.randperm(physical_pages, device=device, generator=generator)[:512].int()
    # Materialize the supplied [4096, pages] shape. Every row has the same prefix.
    page_table = shared_pages[:max_context_len // block_size].repeat(tokens, 1)
    generator.manual_seed(seed + chunk)
    q = torch.randint(-128, 128, (tokens, 1, heads, head_dim), dtype=torch.int8,
                      device=device, generator=generator)
    weights = torch.randn((tokens, heads), dtype=torch.float32,
                          device=device, generator=generator)
    q_scales = torch.rand((tokens, heads), dtype=torch.float32,
                         device=device, generator=generator) * 0.09 + 0.01
    weights *= q_scales
    context_lens = torch.arange(tokens, dtype=torch.int32, device=device) * 2 + (chunk - 1) * 8192
    return dict(q=q, fused_kv_cache=pack_int8_fused_cache(kv, scales), weights=weights,
                context_lens=context_lens, block_table=page_table, schedule_meta=None,
                max_context_len=max_context_len)


@pytest.mark.parametrize("chunk", [1, 2, 3, 4], ids=["chunk1", "chunk2", "chunk3", "chunk4"])
def test_paged_mqa_logits_prefill(chunk, csv_writer=None):
    if not torch.cuda.is_available():
        pytest.skip("A CUDA/ROCm device is required")
    props = torch.cuda.get_device_properties(0)
    arch = getattr(props, 'gcnArchName', props.name).split(':')[0]
    if arch != 'gfx936':
        pytest.skip(f'INT8 prefix prefill requires gfx936, got {arch}')

    previous_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        case = make_prefix_prefill_case(chunk)
        q, weights, lengths = case['q'], case['weights'], case['context_lens']
        tokens, _, heads, dim = q.shape
        max_len = case['max_context_len']
        num_blocks, block_size = case['fused_kv_cache'].shape[:2]
        raw = case['fused_kv_cache'].view(num_blocks, -1)
        pages = case['block_table'][0].long()
        k = raw[:, :block_size * dim].reshape(num_blocks, block_size, dim)[pages].reshape(max_len, dim)
        scales = raw[:, block_size * dim:].contiguous().view(torch.float32)[pages].reshape(max_len)
        ks = torch.zeros_like(lengths)
        calls = (
            ('paged', lambda: paged_mqa_logits(**case, clean_logits=True, is_prefill=True)),
            ('native', lambda: mqa_logits(q[:, 0], k, weights, ks, lengths, scales, clean_logit=True)),
        )
        outputs = [call() for _, call in calls]
        # Full independent reference: exact FP32 INT8 dots, FP64 weighted reduction.
        k_float, positions = k.float(), torch.arange(max_len, device=q.device)
        max_error = torch.zeros(len(outputs), device=q.device, dtype=torch.float64)
        max_reference = torch.zeros((), device=q.device, dtype=torch.float64)
        for row in range(0, tokens, 32):
            dots = q[row:row + 32, 0].reshape(-1, dim).float() @ k_float.T
            reference = (dots.reshape(-1, heads, max_len).double().relu()
                         * weights[row:row + 32, :, None].double()).sum(1) * scales.double()
            valid = positions[None, :] < lengths[row:row + 32, None]
            max_reference = torch.maximum(max_reference, reference.masked_fill(~valid, 0).abs().max())
            for index, output in enumerate(outputs):
                assert output.shape == (tokens, max_len) and output.dtype == torch.float32
                actual = output[row:row + 32]
                assert torch.equal(torch.isneginf(actual), ~valid)
                assert torch.isfinite(actual[valid]).all()
                error = (actual.double() - reference).masked_fill(~valid, 0).abs().max()
                max_error[index] = torch.maximum(max_error[index], error)
        normalized_errors = (max_error / max_reference.clamp_min(1)).tolist()
        assert max(normalized_errors) < 2e-5, f'{chunk=}, {normalized_errors=}'

        del outputs, output, actual, dots, reference, valid, k_float, error
        sum_lens = lengths.long().sum().item()
        page_bytes = ((lengths.long() + block_size - 1) // block_size).sum().item() * 4
        for name, call in calls:
            for _ in range(10):
                call()
            torch.cuda.synchronize()
            start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
            samples_us = []
            for _ in range(5):
                start.record()
                for _ in range(20):
                    call()
                end.record()
                torch.cuda.synchronize()
                samples_us.append(start.elapsed_time(end) * 1000 / 20)
            median_us = sorted(samples_us)[2]
            a_bytes = count_bytes(q, weights, lengths) + (count_bytes(ks) if name == 'native' else 0)
            b_bytes = sum_lens * (dim + 4) + (page_bytes if name == 'paged' else 0)
            c_bytes = tokens * max_len * 4
            q_shape = tuple(q.shape if name == 'paged' else q[:, 0].shape)
            kv_shape = tuple(case['fused_kv_cache'].shape if name == 'paged' else k.shape)
            print(f'prefill chunk{chunk} {name} Q={q_shape} KV={kv_shape} | '
                  f'PASS | {median_us:.2f} us | '
                  f'{2 * sum_lens * heads * dim / median_us / 1e6:.2f} TFLOPS | '
                  f'{(a_bytes + b_bytes + c_bytes) / median_us / 1000:.2f} GB/s')
            if csv_writer is not None:
                csv_writer.writerow({
                    "Mode": "prefill", "Impl": name, "Dtype": "INT8", "Chunk": chunk,
                    "Q Shape": q_shape, "KV Shape": kv_shape,
                    "Weights Shape": tuple(weights.shape), "Context Lens Shape": tuple(lengths.shape),
                    "Page Table Shape": tuple(case['block_table'].shape) if name == 'paged' else "",
                    "Max Context Len": max_len, "Correctness": "PASS",
                    "Latency (us)": f"{median_us:.2f}",
                    "TFLOPS": f"{2 * sum_lens * heads * dim / median_us / 1e6:.2f}",
                    "Logical BW (GB/s)": f"{(a_bytes + b_bytes + c_bytes) / median_us / 1000:.2f}",
                    "Raw Samples (us)": "; ".join(f"{sample:.3f}" for sample in samples_us),
                })
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous_tf32


if __name__ == '__main__':
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA/ROCm device is required")
    csv_filename = "paged_mqa_logits_results.csv"
    # Match ci_tests: pytest validates cases; the standalone entry writes CSV.
    # Logical BW counts algorithmic traffic, not measured physical DRAM traffic.
    with open(csv_filename, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for mode in DECODE_MODES:
            try:
                test_paged_mqa_logits(*mode, csv_writer=writer)
            except pytest.skip.Exception as error:
                print(f"SKIP: {error}")
            csv_file.flush()
        for chunk in (1, 2, 3, 4):
            try:
                test_paged_mqa_logits_prefill(chunk, csv_writer=writer)
            except pytest.skip.Exception as error:
                print(f"SKIP: {error}")
            csv_file.flush()
    print(f"CSV saved => {csv_filename}")
