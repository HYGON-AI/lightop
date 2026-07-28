import torch
import triton
import triton.language as tl
import pandas as pd
import matplotlib.pyplot as plt
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.utils.torch_utils import set_random_seed
from vllm.model_executor.layers.rotary_embedding import DeepseekScalingRotaryEmbedding
from lightop.attention import rotary_embedding_deepseek_fuse
import time 

@triton.jit
def deepseek_scaling_rotary_emb_kernel_gptj(cos_sin, q, stride1: int,
                                         stride2: int, stride_cs: int,
                                         dim1: int, dim2: int, dim3: int,
                                         BLOCK_SIZE: tl.constexpr):
    pid0 = tl.program_id(0)
    pid1 = tl.program_id(1)
    pid2 = tl.program_id(2)
    offsets_cs = tl.arange(0, BLOCK_SIZE) + pid2 * BLOCK_SIZE
    offsets_q = tl.arange(0, BLOCK_SIZE * 2) + pid2 * BLOCK_SIZE * 2

    offsets = pid0 * stride1 + pid1 * stride2 + offsets_q
    mask = offsets_cs < dim3
    mask2 = offsets_q < dim3 * 2

    v_cos = tl.load(cos_sin + pid0 * stride_cs + offsets_cs, mask=mask)
    v_cos2 = tl.interleave(v_cos, v_cos)
    v_sin = tl.load(cos_sin + pid0 * stride_cs + dim3 + offsets_cs, mask=mask)
    v_sin2 = tl.interleave(v_sin, v_sin)
    x12 = tl.load(q + offsets, mask=mask2)
    x1, x2 = tl.split(x12.reshape([BLOCK_SIZE, 2]))
    
    tl.debug_barrier()
    x12_ = tl.ravel(tl.join(-x2, x1))
    x12 = x12 * v_cos2 + x12_ * v_sin2
    tl.store(q + offsets, x12, mask=mask2)

def triton_call(q, rotary_dim, cos_sin):
    BLOCK_SIZE = 64
    grid = (
        q.shape[-3],
        q.shape[-2],
        triton.cdiv(rotary_dim // 2, BLOCK_SIZE),
    )
    deepseek_scaling_rotary_emb_kernel_gptj[grid](
        cos_sin, q,
        stride1=q.stride()[-3], stride2=q.stride()[-2],
        stride_cs=cos_sin.stride()[-2],
        dim1=q.shape[0], dim2=q.shape[1],
        dim3=rotary_dim // 2,
        BLOCK_SIZE=BLOCK_SIZE, num_warps=1)

# ------------------------------------------------------------------
# Main Benchmark Logic
# ------------------------------------------------------------------
if __name__ == '__main__':
    device = torch.device("cuda:0")
    set_random_seed(0)
    torch.set_default_device("cuda:0")

    base = 10000
    max_position = 4096
    is_neox_style = False
    rotary_dim = 32
    head_size = 64
    scaling_factor = 40.0
    DTYPE = torch.bfloat16
    # DTYPE = torch.float16
    
    # batch_sizes = [2, 4, 6, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8000]
    batch_sizes = [512,504,496,488,480,472,464,456,448,440,432,424,416,408,400,392,384,376,368,360,352,344,336,328,320,312,304,296,288,280,272,264,256,248,240,232,224,216,208,200,192,184,176,168,160,152,144,136,128,120,112,104,96,88,80,72,64,56,48,40,32,24,16,8,4,2,1]
    num_heads_list = [128, 64, 32, 16, 8]

    with set_current_vllm_config(VllmConfig()):
        rot = DeepseekScalingRotaryEmbedding(
            head_size, rotary_dim, max_position, base, is_neox_style,
            scaling_factor, DTYPE,
        ).to(device)
    cos_sin_cache = rot._compute_cos_sin_cache() #[max_position*scaling_factor,rotary_dim]
    if cos_sin_cache.dtype != DTYPE:
        cos_sin_cache = cos_sin_cache.to(DTYPE)
    results = []
    ATOL = 1e-2 if DTYPE in [torch.float16, torch.bfloat16] else 1e-7
    
    for num_heads in num_heads_list:
        print(f"\n\n### Benchmarking for Query Heads = {num_heads}\n")
        print(f"{'Batch Size':<12} | {'Correctness':<12} | {'Triton (us)':<15} | {'LightOp (us)':<15} | {'Speedup':<10}")
        print("-" * 80)

        for bs in batch_sizes:
            positions = torch.randint(0, max_position, (bs,), device=device)
            
            # offsets = torch.ones_like(positions)
            offsets = torch.randint_like(positions, low=0, high=10)
            
            effective_positions_triton = torch.add(positions, offsets)
            cos_sin = cos_sin_cache[effective_positions_triton] 
            
            q_orig = torch.randn(bs, num_heads, head_size, dtype=DTYPE, device=device)
            k_orig = torch.randn(bs, 1, head_size, dtype=DTYPE, device=device)

            # --- 1. Correctness Check ---
            q_triton = q_orig.clone()
            k_triton = k_orig.clone()
            triton_call(q_triton, rotary_dim, cos_sin)
            triton_call(k_triton, rotary_dim, cos_sin)

            q_lightop = q_orig.clone()
            k_lightop = k_orig.clone()
            
            rotary_embedding_deepseek_fuse(positions, q_lightop, k_lightop, head_size, cos_sin_cache, is_neox_style, offsets)
            
            is_correct = torch.allclose(q_triton, q_lightop, atol=ATOL) and torch.allclose(k_triton, k_lightop, atol=ATOL)
            correctness_status = " Pass" if is_correct else " Fail"
            assert is_correct, f"Correctness check failed for bs={bs}, num_heads={num_heads}"

            # --- 2. Performance Test using manual timing ---
            warmup_runs = 20
            test_runs = 100

            # --- Test Triton Kernel ---
            q_triton_bench = q_orig.clone()
            k_triton_bench = k_orig.clone()
            for _ in range(warmup_runs):
                triton_call(q_triton_bench, rotary_dim, cos_sin)
                triton_call(k_triton_bench, rotary_dim, cos_sin)
            torch.cuda.synchronize()

            triton_avg_time = triton.testing.do_bench(lambda: [triton_call(q_triton_bench, rotary_dim, cos_sin), triton_call(q_triton_bench, rotary_dim, cos_sin)])
            triton_us = triton_avg_time * 1_000

            # --- Test LightOp Kernel ---
            q_lightop_bench = q_orig.clone()
            k_lightop_bench = k_orig.clone()
            for _ in range(warmup_runs):
                rotary_embedding_deepseek_fuse(positions, q_lightop_bench, k_lightop_bench, head_size, cos_sin_cache, is_neox_style, offsets)
            torch.cuda.synchronize()

            start_time = time.time()
            for _ in range(test_runs):
                rotary_embedding_deepseek_fuse(positions, q_lightop_bench, k_lightop_bench, head_size, cos_sin_cache, is_neox_style, offsets)
            torch.cuda.synchronize()
            lightop_avg_time = (time.time() - start_time) / test_runs
            lightop_us = lightop_avg_time * 1_000_000

            speedup = triton_us / lightop_us

            print(f"{bs:<12} | {correctness_status:<12} | {triton_us:<15.2f} | {lightop_us:<15.2f} | {speedup:<9.2f}x")

            results.extend([
                {'bs': bs, 'num_heads': num_heads, 'impl': 'Triton (Unfused)', 'us': triton_us},
                {'bs': bs, 'num_heads': num_heads, 'impl': 'LightOp (Fused)', 'us': lightop_us}
            ])