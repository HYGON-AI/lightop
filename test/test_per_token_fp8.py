import torch
import triton
import triton.language as tl
import time

try:
    from lightop import op
except ImportError:
    print("警告: 无法导入 lightop.op，请确保您的 C++ 扩展已正确编译并安装。")


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
    x_q = tl.clamp(x / scale_x, min=fp8_min, max=fp8_max).to(xq_ptr.dtype.element_ty)
    tl.store(xq_ptr + row_id * stride_xq + cols, x_q, mask=mask)
    tl.store(scale_ptr + row_id, scale_x)


def per_token_quant_fp8_triton(x):
    M = x.numel() // x.shape[-1]
    N = x.shape[-1]
    x_q = torch.empty_like(x, device=x.device, dtype=torch.float8_e4m3fn)
    scales = torch.empty(x.shape[:-1] + (1, ),
                         device=x.device,
                         dtype=torch.float32)
    BLOCK = triton.next_power_of_2(N)
    # heuristics for number of warps
    num_warps = min(max(BLOCK // 256, 1), 8)
    finfo = torch.finfo(x_q.dtype)
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
    return x_q, scales

def run_benchmark():
    num_tokens_list = [128,256,512,1024,2048,4096,8192,16384]
    hidden_size = 4096
    
    warmup_iters = 10
    test_iters = 100

    print("======================================== FP8 Quantization Benchmark ========================================")
    print(f"{'Tokens':<8} | {'Hidden':<8} | {'Scale MaxDiff':<15} | {'FP8 Match Rate':<15} | {'Triton (us)':<12} | {'Custom (us)':<12} | {'Speedup':<8}")
    print("-" * 105)

    for num_tokens in num_tokens_list:

        x = torch.randn((num_tokens, hidden_size), dtype=torch.float16, device='cuda')
        
        # ----------------------------------------------------------------------
        # 1. 精度验证
        # ----------------------------------------------------------------------
        # Triton 运行
        triton_q, triton_s = per_token_quant_fp8_triton(x)
        
        # Custom Op 运行
        custom_q = torch.empty_like(x, device=x.device, dtype=torch.float8_e4m3fn)
        custom_s = torch.empty((num_tokens, 1), device=x.device, dtype=torch.float32)
        #out input scale
        op.per_token_quant_fp8(custom_q, x, custom_s)
        
        # 验证 scales 差异
        scale_max_diff = torch.abs(triton_s - custom_s).max().item()
        
        # 验证 FP8 量化结果一致性 (转为 fp32 对比)
        # 注意：由于舍入模式(Round-to-Nearest-Even)与精度实现细微不同，这里统计完全匹配的比例
        t_q_f = triton_q.to(torch.float32)
        c_q_f = custom_q.to(torch.float32)
        match_rate = (t_q_f == c_q_f).float().mean().item()
        # import pdb
        # pdb.set_trace()
        # ----------------------------------------------------------------------
        # 2. 性能测试 (Triton)
        # ----------------------------------------------------------------------
        # 预热
        for _ in range(warmup_iters):
            per_token_quant_fp8_triton(x)
        torch.cuda.synchronize()
        
        # 计时
        start_time = time.perf_counter()
        for _ in range(test_iters):
            per_token_quant_fp8_triton(x)
        torch.cuda.synchronize()
        triton_time_us = (time.perf_counter() - start_time) * 1e6 / test_iters

        # ----------------------------------------------------------------------
        # 3. 性能测试 (Custom)
        # ----------------------------------------------------------------------
        # 预热
        for _ in range(warmup_iters):
            op.per_token_quant_fp8(custom_q, x, custom_s)
        torch.cuda.synchronize()
        
        # 计时
        start_time = time.perf_counter()
        for _ in range(test_iters):
            op.per_token_quant_fp8(custom_q, x, custom_s)
        torch.cuda.synchronize()
        custom_time_us = (time.perf_counter() - start_time) * 1e6 / test_iters

        speedup = triton_time_us / custom_time_us

        # 打印结果行
        print(f"{num_tokens:<8} | {hidden_size:<8} | {scale_max_diff:<15.6e} | {match_rate:<15.2%} | {triton_time_us:<12.2f} | {custom_time_us:<12.2f} | {speedup:<8.2f}x")

if __name__ == "__main__":
    run_benchmark()