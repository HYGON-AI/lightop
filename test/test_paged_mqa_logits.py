import torch
import random
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

def ref_fp8_paged_mqa_logits(q: torch.Tensor, kv_cache: torch.Tensor,
                             weights: torch.Tensor, context_lens: torch.Tensor, block_tables: torch.Tensor,
                             max_model_len: int):
    batch_size, next_n, heads, dim = q.size()
    num_block, block_size, _, dim = kv_cache.size()
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
            logits[i * next_n:(i + 1) * next_n, block_rk * block_size: (block_rk + 1) * block_size] = torch.where(k_offsets[None, :] <= q_offsets[:, None], s, float('-inf'))
    return logits

def test_paged_mqa_logits(use_fp8 = True):
    max_model_len = 111 * 1000
    for batch_size, next_n in [(1, 1), (64, 1), (64, 2), (128, 1)]:
        for heads, index_dim in [(16, 128), (32, 128), (64, 128)]:
            for avg_kv in (4096, 72000,):
                num_blocks, blocksize = max_model_len * 3, 64

                q = torch.randn((batch_size, next_n, heads, index_dim), device='cuda', dtype=torch.bfloat16)
                kv_cache = torch.randn((num_blocks, blocksize, 1, index_dim), device='cuda', dtype=torch.bfloat16)
                weights = torch.randn((batch_size * next_n, heads), device='cuda', dtype=torch.float32)

                context_lens = torch.randint(int(0.9 * avg_kv), int(1.1 * avg_kv), (batch_size, )).cuda().to(torch.int32)
                max_block_len = (context_lens.max().item() + blocksize - 1) // blocksize * blocksize
                block_tables = torch.zeros((batch_size, max_block_len), device='cuda', dtype=torch.int32)

                counter = 0
                block_idx_pool = list(range(num_blocks))
                random.shuffle(block_idx_pool)
                for i in range(batch_size):
                    ctx_len = context_lens[i].item()
                    for j in range(ceil_div(ctx_len, blocksize)):
                        block_tables[i][j] = block_idx_pool[counter]
                        counter += 1

                device_props = torch.cuda.get_device_properties(0)
                device_name = device_props.gcnArchName.split(':')[0] if hasattr(device_props, 'gcnArchName') else device_props.name
                num_cus = device_props.multi_processor_count

                schedule_metadata = None
                torch.cuda.synchronize()

                if 'gfx938' in device_name and use_fp8:
                    print('Testing FP8 Paged MQA Logits:')
                    q_fp8 = q.to(torch.float8_e4m3fn)
                    kv_cache_fp8 = kv_cache_cast_to_fp8(kv_cache)

                    logits = paged_mqa_logits(
                        q_fp8, kv_cache_fp8, weights, context_lens, block_tables, schedule_metadata, max_model_len, clean_logits=True)
                    torch.cuda.synchronize()

                    ref_logits = ref_fp8_paged_mqa_logits(q, kv_cache, weights, context_lens, block_tables, max_model_len)

                    positions = torch.arange(max_model_len, device='cuda').unsqueeze(0).expand(batch_size * next_n, -1)
                    row_indices = torch.arange(batch_size * next_n, device='cuda') // next_n
                    next_n_offset = torch.arange(batch_size * next_n, device='cuda') % next_n
                    ref_neginf_mask = ~(positions <= (context_lens[row_indices] - next_n + next_n_offset).unsqueeze(1))

                    neginf_mask = (logits == float('-inf'))
                    assert torch.equal(neginf_mask, ref_neginf_mask)

                    logits = logits.masked_fill(neginf_mask, 0)
                    ref_logits = ref_logits.masked_fill(ref_neginf_mask, 0)

                    diff = calc_diff(logits, ref_logits)
                    if batch_size == 1:
                        assert diff < 1.1e-3, f'{diff=}, not pass'
                    else:
                        assert diff < 1e-3, f'{diff=}, not pass'

                else:
                    print('Testing BF16 Paged MQA Logits:')
                    q_bf16 = q
                    kv_cache_bf16 = kv_cache

                    logits = paged_mqa_logits(
                        q_bf16, kv_cache_bf16, weights, context_lens, block_tables, schedule_metadata, max_model_len, clean_logits=True)
                    torch.cuda.synchronize()

                    ref_logits = ref_fp8_paged_mqa_logits(q_bf16, kv_cache_bf16, weights, context_lens, block_tables, max_model_len)

                    positions = torch.arange(max_model_len, device='cuda').unsqueeze(0).expand(batch_size * next_n, -1)
                    row_indices = torch.arange(batch_size * next_n, device='cuda') // next_n
                    next_n_offset = torch.arange(batch_size * next_n, device='cuda') % next_n
                    ref_neginf_mask = ~(positions <= (context_lens[row_indices] - next_n + next_n_offset).unsqueeze(1))

                    neginf_mask = (logits == float('-inf'))
                    assert torch.equal(neginf_mask, ref_neginf_mask)

                    logits = logits.masked_fill(neginf_mask, 0)
                    ref_logits = ref_logits.masked_fill(ref_neginf_mask, 0)

                    assert torch.allclose(logits, ref_logits, rtol=1e-3, atol=1e-3)

                sum_lens = sum(context_lens.to(torch.int64))
                tflops = 2 * sum_lens * next_n * heads * index_dim / 1e12
                input_bytes = count_bytes(q) / (2 if 'gfx938' in device_name and use_fp8 else 1) + count_bytes(weights, context_lens) + sum_lens * (index_dim * (1 if 'gfx938' in device_name and use_fp8 else 2) + (4 if 'gfx938' in device_name and use_fp8 else 0)) + (sum_lens / blocksize) * 4
                output_bytes = batch_size * max_model_len * next_n * 4
                t = bench_kineto(lambda: paged_mqa_logits(
                    q_fp8 if 'gfx938' in device_name and use_fp8 else q_bf16,
                    kv_cache_fp8 if 'gfx938' in device_name and use_fp8 else kv_cache_bf16,
                    weights, context_lens, block_tables, schedule_metadata, max_model_len, clean_logits=False
                ))
                clean_bytes = (batch_size * next_n * max_model_len - neginf_mask.sum().item()) * 4 + count_bytes(context_lens)
                print(f' > BSZ={batch_size:3}, NextN={next_n:1}, H={heads:2}, D={index_dim:2}, L={avg_kv:6}: '
                      f'{tflops / t:4.0f} TFLOPS, {t * 1e6:3.0f} us, '
                      f'{(input_bytes + output_bytes) / t / 1e9:4.0f} GB/s | ')
                      # f'clean: {clean_t * 1e6:3.0f} us, {clean_bytes / clean_t / 1e9:4.0f} GB/s')
                print()
    print()

if __name__ == '__main__':
    test_paged_mqa_logits(use_fp8=True)
    test_paged_mqa_logits(use_fp8=False)