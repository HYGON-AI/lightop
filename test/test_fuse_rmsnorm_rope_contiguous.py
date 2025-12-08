import torch
from vllm import _custom_ops as vllm_ops
from triton.testing import do_bench 
from lightop import fused_rms_norm_rope_contiguous
from lightop import fused_rms_norm_contiguous
import pdb
from lightop import op
from vllm import _custom_ops as ops


def _compute_inv_freq(base,rotary_dim, device) -> torch.Tensor:
    """Compute the inverse frequency."""
    # NOTE(woosuk): To exactly match the HF implementation, we need to
    # use CPU to compute the cache and then move it to GPU. However, we
    # create the cache on GPU for faster initialization. This may cause
    # a slight numerical difference between the HF implementation and ours.
    inv_freq = 1.0 / (base**(torch.arange(
        0, rotary_dim, 2, dtype=torch.float, device=device) / rotary_dim))
    return inv_freq

def _compute_cos_sin_cache(max_position_embeddings,base,rotary_dim, device) -> torch.Tensor:
    """Compute the cos and sin cache."""
    inv_freq = _compute_inv_freq(base,rotary_dim, device=device)
    t = torch.arange(max_position_embeddings, dtype=torch.float, device=device)

    freqs = torch.einsum("i,j -> ij", t, inv_freq)
    cos = freqs.cos()
    sin = freqs.sin()
    cache = torch.cat((cos, sin), dim=-1)
    return cache

def _create_mla_cache(
    num_blocks: int,
    block_size: int,
    entry_size: int,
    dtype: torch.dtype,
    kv_cache_dtype: str,
    device: str,
) -> torch.Tensor:
    cache_dtype = torch.uint8
    return torch.zeros(num_blocks,
                       block_size,
                       entry_size,
                       dtype=cache_dtype,
                       device=device)


def rms_rope_single_test(kv_cache_dtype_str,batch_size,num_head,q_c_dim,kv_c_dim,rot_dim,num_kv_cache_blocks,kv_cache_blocks_size,test_speed = False):
    DEVICE = "cuda"
    variance_epsilon = 1e-6
    DTYPE = torch.bfloat16    
    is_neox_style = False
    max_position = 163840
    base = 1e4
    if kv_cache_dtype_str == "fp8_e4m3":
        KV_CACHE_DTYPE = torch.float8_e4m3fn
    elif kv_cache_dtype_str == "fp8_e5m2":
        KV_CACHE_DTYPE = torch.float8_e5m2
    elif kv_cache_dtype_str == "fp16":
        KV_CACHE_DTYPE = torch.float16
    elif kv_cache_dtype_str == "bf16":
        KV_CACHE_DTYPE = torch.bfloat16
    else:
        raise ValueError(f"Unsupported kv_cache_dtype_str: {kv_cache_dtype_str}")
    kv_cache_scale = 1.0
    # rope(Q)的相关tensor创建
    q_tensor = torch.randn(batch_size, num_head,q_c_dim + rot_dim, dtype=DTYPE, device=DEVICE)
    cos_sin_cache = _compute_cos_sin_cache(max_position,base,rot_dim, device=DEVICE)
    if cos_sin_cache.dtype != DTYPE:
        cos_sin_cache = cos_sin_cache.to(DTYPE)
    positions = torch.randint(0,max_position, (batch_size,), device=DEVICE)
    q_pe = q_tensor[...,q_c_dim:]
    q_pe_no_fuse = q_pe.clone()
    q_pe_ori = q_pe.clone()
    # rope(K)的相关tensor创建
    k_tensor = torch.randn(batch_size, kv_c_dim + rot_dim, dtype=DTYPE, device=DEVICE)
    k_pe = k_tensor[...,kv_c_dim:]
    k_pe_no_fuse = k_pe.clone()
    kv_c = k_tensor[...,:kv_c_dim]
    kv_c_normed = torch.empty_like(kv_c).contiguous()
    weight = torch.randn(kv_c_dim, dtype=DTYPE, device=DEVICE)
    kv_c_no_fuse = kv_c.clone()
    kv_c_no_fuse_out = torch.empty_like(kv_c_no_fuse).contiguous()
    # kv_cache相关
    kv_cache = _create_mla_cache(num_kv_cache_blocks, kv_cache_blocks_size, kv_c_dim+rot_dim, DTYPE,
                                  kv_cache_dtype_str, DEVICE)
    kv_cache_no_fuse = kv_cache.clone()
    kv_scale_tensor = torch.tensor(1., dtype=torch.float32, device=DEVICE)
    slot_mapping = torch.randperm(num_kv_cache_blocks * kv_cache_blocks_size, device=DEVICE)[:batch_size]

    if KV_CACHE_DTYPE == torch.float16:
        kv_cache = kv_cache.half()
        kv_cache_no_fuse = kv_cache_no_fuse.half()
    elif KV_CACHE_DTYPE == torch.bfloat16:
        kv_cache = kv_cache.bfloat16()
        kv_cache_no_fuse = kv_cache_no_fuse.bfloat16()

    def run_fuse_opt():
        fused_rms_norm_rope_contiguous(positions,q_pe,k_pe,kv_c,kv_c_normed,weight,cos_sin_cache,slot_mapping,kv_cache,kv_cache_dtype_str,kv_cache_scale,is_neox_style,variance_epsilon)
    def run_no_fuse_opt():
        op.rotary_embedding_deepseek_fuse(positions, q_pe_no_fuse, k_pe_no_fuse, rot_dim, cos_sin_cache, is_neox_style, None)
        fused_rms_norm_contiguous(kv_c_no_fuse_out, kv_c_no_fuse, weight, variance_epsilon)
        cache_dtype = kv_cache_dtype_str
        if cache_dtype == "fp16" or cache_dtype == "bf16":
            cache_dtype = "auto"
        ops.concat_and_cache_mla(kv_c_no_fuse_out, k_pe_no_fuse, kv_cache_no_fuse, slot_mapping, cache_dtype, kv_scale_tensor)
    if test_speed:
        warmup_runs = 25
        test_runs = 100
        avg_time_no_fuse_opt = do_bench(lambda: run_no_fuse_opt(), 
                                 warmup=warmup_runs, rep=test_runs) 
        avg_time_fuse_opt = do_bench(lambda: run_fuse_opt(), 
                                 warmup=warmup_runs, rep=test_runs)
        print(f"bs: {batch_size}, num_head: {num_head}, opt non-fused time: {avg_time_no_fuse_opt * 1000:.2f} us, fuse opt time: {avg_time_fuse_opt * 1000:.2f} us")
    else:
        run_no_fuse_opt()
        run_fuse_opt()
        diff_max_q = torch.max(torch.abs(q_pe - q_pe_no_fuse))
        diff_max_k_pe = torch.max(torch.abs(k_pe - k_pe_no_fuse))
        diff_max_kv_c = torch.max(torch.abs(kv_c_no_fuse_out - kv_c_normed))
        diff_max_kv_cache = torch.max(torch.abs(kv_cache.float() - kv_cache_no_fuse.float()))
        print(f"bs: {batch_size}, num_head: {num_head}, diff_max_q: {diff_max_q}, diff_max_k_pe: {diff_max_k_pe}, diff_max_kv_c: {diff_max_kv_c}, diff_max_kv_cache: {diff_max_kv_cache}")

if __name__ == "__main__":
    # rms_rope_single_test(kv_cache_dtype_str = "bf16", batch_size=4096,num_head=128,q_c_dim=128, kv_c_dim=512, rot_dim=64, num_kv_cache_blocks=1000, kv_cache_blocks_size=16, test_speed=False)
    # rms_rope_single_test(kv_cache_dtype_str = "fp8_e5m2",batch_size=32,num_head=128,q_c_dim=128, kv_c_dim=512, rot_dim=64, num_kv_cache_blocks=1000, kv_cache_blocks_size=16, test_speed=True)
    for bs in [2048,4096,8192,16384]:
       for num_head in [128,]:
        rms_rope_single_test(kv_cache_dtype_str = "bf16",batch_size=bs,num_head=num_head,q_c_dim=128, kv_c_dim=512, rot_dim=64, num_kv_cache_blocks=10000, kv_cache_blocks_size=64, test_speed=True)



