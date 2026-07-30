import random

import torch
from lightop.attention import paged_mqa_logits


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

def test_paged_mqa_logits(use_fp8=False, use_mixed=False, use_int8=False):
    assert sum((use_fp8, use_mixed, use_int8)) <= 1

    device_props = torch.cuda.get_device_properties(0)
    device_name = (
        device_props.gcnArchName.split(':')[0] if hasattr(device_props, 'gcnArchName') else device_props.name
    )
    if use_fp8 and 'gfx938' not in device_name:
        print(f'Skipping FP8 Paged MQA Logits on {device_name}')
        return

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

                print(f'Testing {test_name} Paged MQA Logits:')
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
                print(f' > BSZ={batch_size:3}, NextN={next_n:1}, H={heads:2}, D={index_dim:2}, '
                      f'B={blocksize:2}, L={avg_kv:6}: '
                      f'{tflops / t:4.0f} TFLOPS, {t * 1e6:3.0f} us, '
                      f'{(input_bytes + output_bytes) / t / 1e9:4.0f} GB/s | ')
                print()
    print()


def pack_int8_fused_cache(kv: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    num_blocks, block_kv, num_kv_heads, head_dim = kv.shape
    assert num_kv_heads == 1 and head_dim == 128

    packed = torch.empty((num_blocks, block_kv * (head_dim + 4)), device=kv.device, dtype=torch.int8)
    packed[:, :block_kv * head_dim] = kv.reshape(num_blocks, block_kv * head_dim)
    packed[:, block_kv * head_dim:] = scales.contiguous().view(dtype=torch.int8)
    return packed.view(num_blocks, block_kv, 1, head_dim + 4)


if __name__ == '__main__':
    test_paged_mqa_logits()
    test_paged_mqa_logits(use_fp8=True)
    test_paged_mqa_logits(use_mixed=True)
    test_paged_mqa_logits(use_int8=True)
