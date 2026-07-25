import torch
# from vllm import _custom_ops as vllm_ops
from triton.testing import do_bench
from lightop.attention import fuse_rmsnorm_rope_quant_qkv
from lightop.attention import rotary_embedding_deepseek_fuse
from lightop.norm import fused_rms_norm_contiguous
import pdb
import torch
import triton
import triton.language as tl
# 设置随机种子以确保可复现性
torch.manual_seed(40)


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


@triton.jit
def _per_token_quant_fp8(
    x_ptr,
    xq_ptr,
    scale_ptr,
    stride_x,
    stride_xq,
    N,
    BLOCK: tl.constexpr,
    fp8_min,
    fp8_max
):
    row_id = tl.program_id(0)

    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(x_ptr + row_id * stride_x + cols, mask=mask,
                other=0.0).to(tl.float32)
    absmax = tl.maximum(tl.max(tl.abs(x)), 1e-10)
    scale_x = absmax / fp8_max
    x_q = tl.clamp(x / scale_x, min=fp8_min, max=fp8_max)
    tl.store(xq_ptr + row_id * stride_xq + cols, x_q, mask=mask)
    tl.store(scale_ptr + row_id, scale_x)

def per_token_quant_fp8_triton(x,dtype):
    """XSL验证代码版本 - Triton实现"""
    M = x.numel() // x.shape[-1]
    N = x.shape[-1]
    # x_q = torch.empty_like(x, device=x.device, dtype=torch.float8_e4m3fn)
    x_q = torch.empty_like(x, device=x.device, dtype=torch.float)
    scales = torch.empty(x.shape[:-1] + (1, ),
                         device=x.device,
                         dtype=torch.float32)
    BLOCK = triton.next_power_of_2(N)
    # heuristics for number of warps
    num_warps = min(max(BLOCK // 256, 1), 8)
    finfo = torch.finfo(dtype)
    # pdb.set_trace()
    fp8_min = finfo.min
    fp8_max = finfo.max
    #assert x.is_contiguous()
    _per_token_quant_fp8[(M, )](
        x,
        x_q,
        scales,
        stride_x=x.stride(-2),
        stride_xq=x_q.stride(-2),
        N=N,
        BLOCK=BLOCK,
        num_warps=num_warps,
        num_stages=1,
        fp8_min=fp8_min,
        fp8_max=fp8_max
    )
    return x_q.to(dtype), scales





def rms_rope_single_test(kv_cache_dtype_str,k_head_stride,batch_size,num_head,q_c_dim,kv_c_dim,rot_dim,num_kv_cache_blocks,kv_cache_blocks_size,test_speed = False):
    DEVICE = "cuda"
    variance_epsilon = 1e-6
    DTYPE = torch.bfloat16    
    is_neox_style = False
    max_position = 163840
    base = 1e4
    dtype_mapping = {
        "fp8_e4m3": torch.float8_e4m3fn,
        "fp8_e5m2": torch.float8_e5m2,
        "fp16": torch.float16,
        "bf16": torch.bfloat16
    }
    KV_CACHE_DTYPE = dtype_mapping.get(kv_cache_dtype_str)
    if KV_CACHE_DTYPE is None:
        raise ValueError(f"Unsupported kv_cache_dtype_str: {kv_cache_dtype_str}")
    kv_cache_scale = 1.0
    # rope(Q)的相关tensor创建
    q_tensor = torch.randn(batch_size, num_head,q_c_dim + rot_dim, dtype=DTYPE, device=DEVICE)
    q_quant_gt = torch.empty_like(q_tensor[:,:,:q_c_dim], dtype=KV_CACHE_DTYPE, device=DEVICE)
    q_quant = torch.empty_like(q_tensor, dtype=KV_CACHE_DTYPE, device=DEVICE)
    cos_sin_cache = _compute_cos_sin_cache(max_position,base,rot_dim, device=DEVICE)
    if cos_sin_cache.dtype != DTYPE:
        cos_sin_cache = cos_sin_cache.to(DTYPE)
    positions = torch.randperm(max_position, device=DEVICE)[:batch_size]
    q_c = q_tensor[...,:q_c_dim]
    q_pe = q_tensor[...,q_c_dim:]
    q_pe_no_fuse = q_pe.clone()
    q_pe_ori = q_pe.clone()
    q_scale = torch.empty((batch_size, num_head, 1), dtype=torch.float32, device=DEVICE)
    q_scale_gt = torch.empty((batch_size, num_head, 1), dtype=torch.float32, device=DEVICE)
    # rope(K)的相关tensor创建
    k_tensor = torch.randn(batch_size, k_head_stride, dtype=DTYPE, device=DEVICE)
    k_pe = k_tensor[..., k_head_stride -rot_dim:]
    k_pe_no_fuse = k_pe.clone()
    kv_c = k_tensor[..., k_head_stride - (kv_c_dim+rot_dim):k_head_stride - rot_dim]
    kv_c_normed = torch.empty(kv_c.shape, dtype=DTYPE, device=DEVICE)
    weight = torch.randn(kv_c_dim, dtype=DTYPE, device=DEVICE)
    kv_c_no_fuse = kv_c.clone()
    kv_c_no_fuse_out = torch.empty_like(kv_c_no_fuse).contiguous()
    # kv_cache相关
    kv_cache = _create_mla_cache(num_kv_cache_blocks, kv_cache_blocks_size, kv_c_dim+rot_dim, DTYPE,
                                  kv_cache_dtype_str, DEVICE)
    kv_cache_no_fuse = kv_cache.clone()
    kv_scale_tensor = torch.tensor(1., dtype=torch.float32, device=DEVICE)
    total_slots = num_kv_cache_blocks * kv_cache_blocks_size
    slot_mapping = torch.randperm(total_slots, device=DEVICE)[:batch_size]

    if KV_CACHE_DTYPE == torch.float16:
        kv_cache = kv_cache.half()
        kv_cache_no_fuse = kv_cache_no_fuse.half()
    elif KV_CACHE_DTYPE == torch.bfloat16:
        kv_cache = kv_cache.bfloat16()
        kv_cache_no_fuse = kv_cache_no_fuse.bfloat16()

    def run_fuse_opt():
        fuse_rmsnorm_rope_quant_qkv(positions,q_c,q_pe,q_quant,q_scale,k_pe,kv_c,kv_c_normed,weight,cos_sin_cache,slot_mapping,kv_cache,kv_cache_dtype_str,kv_cache_scale,is_neox_style,variance_epsilon)
    def run_no_fuse_opt():
        rotary_embedding_deepseek_fuse(positions, q_pe_no_fuse, k_pe_no_fuse, rot_dim, cos_sin_cache, is_neox_style, None)
        fused_rms_norm_contiguous(kv_c_no_fuse_out, kv_c_no_fuse, weight, variance_epsilon)
        cache_dtype = kv_cache_dtype_str
        if cache_dtype == "fp16" or cache_dtype == "bf16":
            cache_dtype = "auto"
        # ops.concat_and_cache_mla(kv_c_no_fuse_out, k_pe_no_fuse, kv_cache_no_fuse, slot_mapping, cache_dtype, kv_scale_tensor)
        # q_quant_gt[:] = q_tensor[:,:,:q_c_dim].to(KV_CACHE_DTYPE)
        q_quant_gt[:], q_scale_gt[:] = per_token_quant_fp8_triton(q_tensor[:,:,:q_c_dim],KV_CACHE_DTYPE)
        kv_cache_no_fuse.view(-1, 576)[slot_mapping, :kv_c_dim] = kv_c_no_fuse_out.to(KV_CACHE_DTYPE).view(torch.uint8)
        kv_cache_no_fuse.view(-1, 576)[slot_mapping, kv_c_dim:] = k_pe_no_fuse.to(KV_CACHE_DTYPE).view(torch.uint8)
        # pdb.set_trace()
    if test_speed:
        warmup_runs = 25
        test_runs = 100
        avg_time_no_fuse_opt = do_bench(lambda: run_no_fuse_opt(), 
                                 warmup=warmup_runs, rep=test_runs) 
        avg_time_fuse_opt = do_bench(lambda: run_fuse_opt(), 
                                 warmup=warmup_runs, rep=test_runs)
        print(f"bs: {batch_size}, num_head: {num_head}, opt non-fused time: {avg_time_no_fuse_opt * 1000:.2f} us, fuse opt time: {avg_time_fuse_opt * 1000:.2f} us")
    else:
        run_fuse_opt()
        run_no_fuse_opt()
        diff_max_k_pe = torch.max(torch.abs(k_pe - k_pe_no_fuse))
        diff_max_kv_c = torch.max(torch.abs(kv_c_normed - kv_c_no_fuse_out))
        kv_cache = kv_cache.view(-1, 576).view(KV_CACHE_DTYPE).float()[slot_mapping[:],:]
        kv_cache_no_fuse = kv_cache_no_fuse.view(-1, 576).view(KV_CACHE_DTYPE).float()[slot_mapping[:],:]
        diff_max_kv_cache = torch.abs(kv_cache - kv_cache_no_fuse).max()
        # 这里view(torch.uint8)是因为q_quant和q_quant_gt是为了方便比较最后一个bit。 ±1的误差是允许的
        diff_max_q_c_quant = torch.max(torch.abs(q_quant[:,:,:q_c_dim].float() - q_quant_gt.float()))
        # pdb.set_trace()
        diff_max_q_pe_quant = torch.max(torch.abs(q_quant.float()[:,:,q_c_dim:] - q_pe_no_fuse.to(KV_CACHE_DTYPE).float()))
        diff_max_q_scale = torch.max(torch.abs(q_scale - q_scale_gt))
        assert diff_max_k_pe < 1e-6
        assert diff_max_kv_c < 1e-6
        assert diff_max_kv_cache <= 1
        assert diff_max_q_pe_quant < 1e-6
        assert diff_max_q_c_quant <= 32
        assert diff_max_q_scale < 1e-6
        print(f"bs: {batch_size}, num_head: {num_head},  diff_max_k_pe: {diff_max_k_pe}, diff_max_kv_c: {diff_max_kv_c}, diff_max_kv_cache: {diff_max_kv_cache}, diff_max_q_pe_quant: {diff_max_q_pe_quant}, diff_max_q_c_quant: {diff_max_q_c_quant}, diff_max_q_scale: {diff_max_q_scale}")
        # print(f"fuse_rmsnorm_rope_quant_qkv bs: {batch_size}, num_head: {num_head} passed")

        # pdb.set_trace()

if __name__ == "__main__":
    # rms_rope_single_test(kv_cache_dtype_str = "fp8_e4m3", batch_size=32,num_head=128,q_c_dim=128, kv_c_dim=512, rot_dim=64, num_kv_cache_blocks=1000, kv_cache_blocks_size=16, test_speed=False)
    # rms_rope_single_test(kv_cache_dtype_str = "fp8_e5m2",batch_size=32,num_head=128,q_c_dim=128, kv_c_dim=512, rot_dim=64, num_kv_cache_blocks=1000, kv_cache_blocks_size=16, test_speed=True)
    for bs in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]:
       for num_head in [16, 32, 64, 128]:
        rms_rope_single_test(kv_cache_dtype_str = "fp8_e4m3", k_head_stride=576,batch_size=bs,num_head=128,q_c_dim=128, kv_c_dim=512, rot_dim=64, num_kv_cache_blocks=10000, kv_cache_blocks_size=16, test_speed=False)


