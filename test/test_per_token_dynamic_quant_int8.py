import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from lightop.quant import per_token_dynamic_quant_int8

@triton.jit
def _per_token_dynamic_quant_int8_kernel(
    x_ptr,
    xq_ptr,
    smoothscale_ptr,
    scale_ptr,
    stride_x,
    stride_xq,
    stride_smoothscale,
    N,
    BLOCK: tl.constexpr,
):
    # Adapted from lmdeploy/lmdeploy/pytorch/kernels/cuda/w8a8_triton_kernels.py at 086481ed84b59bee3b8e4274e5fc6962004
    row_id = tl.program_id(0)

    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(x_ptr + row_id * stride_x + cols, mask=mask,
                other=0.0).to(tl.float32)
    s = tl.load(smoothscale_ptr + cols * stride_smoothscale, mask=mask, other=1.0) 

    x_scaled = x * s
    absmax = tl.maximum(tl.max(tl.abs(x_scaled)), 1e-10)
    scale_x = absmax / 127
    # x_q = x_scaled * (127 / absmax)
    x_q = x_scaled / scale_x
    x_q = libdevice.nearbyint(x_q).to(tl.int8)

    tl.store(xq_ptr + row_id * stride_xq + cols, x_q, mask=mask)
    tl.store(scale_ptr + row_id, scale_x)

def per_token_dynamic_quant_int8_triton(x, smooth_scale):
    M = x.numel() // x.shape[-1]
    N = x.shape[-1]
    x_q = torch.empty_like(x, device=x.device, dtype=torch.int8)
    scales = torch.empty(x.shape[:-1],
                         device=x.device,
                         dtype=torch.float32)
    BLOCK = triton.next_power_of_2(N)
    # heuristics for number of warps
    num_warps = min(max(BLOCK // 256, 1), 8)

    assert smooth_scale.shape[-1] == x.shape[-1]
    assert x.is_contiguous()

    _per_token_dynamic_quant_int8_kernel[(M, )](
        x,
        x_q,
        smooth_scale,
        scales,
        stride_x=x.stride(-2),
        stride_xq=x_q.stride(-2),
        stride_smoothscale = smooth_scale.stride(-1),
        N=N,
        BLOCK=BLOCK,
        num_warps=num_warps,
        num_stages=1,
    )

    return x_q, scales

# ------------------ 接口 ------------------
def int8_quant_init(output: torch.Tensor, input: torch.Tensor,
                    smoothscale: torch.Tensor):
    """
    Replaced implementation with the provided Triton kernel.
    The 'output' argument is no longer used but kept for API compatibility.
    """
    ref_out, ref_scales = per_token_dynamic_quant_int8_triton(input, smoothscale)
    return ref_out, ref_scales

def int8_quant_opt(input: torch.Tensor, smoothscale: torch.Tensor):
    return per_token_dynamic_quant_int8(input, smoothscale)



def test_per_token_dynamic_quant_int8_category_matches_triton() -> None:
    torch.manual_seed(7)
    input_tensor = torch.randn(
        (4, 128), device="cuda", dtype=torch.bfloat16
    )
    smooth_scale = torch.rand(
        (128,), device="cuda", dtype=torch.float32
    ).contiguous()

    expected_q, expected_scale = per_token_dynamic_quant_int8_triton(
        input_tensor, smooth_scale
    )
    actual_q, actual_scale = per_token_dynamic_quant_int8(
        input_tensor, smooth_scale
    )
    torch.cuda.synchronize()

    torch.testing.assert_close(actual_q, expected_q, rtol=0, atol=0)
# ------------------ 正确性 ------------------
def check_correctness_once(num_tokens: int, hidden_sizes: int, rtol=1e-2, atol=1e-2):
    device = torch.device("cuda")
    # 数据类型
    dtype = torch.bfloat16
    # dtype = torch.float
    # dtype = torch.float16
    m=num_tokens
    num_local_tokens = [m]
    num_local_tokens_tensor = torch.tensor(num_local_tokens,device=device,dtype=torch.int32)
    
    input = torch.randn((num_tokens, hidden_sizes), device=device, dtype=dtype)
    out_init_placeholder = torch.empty((num_tokens, hidden_sizes), device=device, dtype=torch.int8)

    smoothscale = torch.ones(
        hidden_sizes,
        dtype=torch.float32,
        device = device
    ).contiguous()
    
    out_init, scales_init = int8_quant_init(out_init_placeholder, input, smoothscale)
    out_init = out_init[:m]
    scales_init = scales_init[:m]
    
    out_opt , scales_opt = int8_quant_opt(input, smoothscale)
    out_opt = out_opt[:m]
    scales_opt = scales_opt[:m]

    if not torch.allclose(scales_init, scales_opt, rtol=1e-2, atol=1e-2):#lj
        diff = (scales_init - scales_opt).abs()
        max_abs = diff.max().item()
        # print("out_shape:",out_init.shape)
        raise AssertionError(f"scales_max_abs={max_abs:.4e}")

    if not torch.allclose(out_init, out_opt, rtol=0.0, atol=atol): # Increased atol for int8 comparison
        diff = (out_init - out_opt).abs()
        max_abs = diff.max().item()
        print("out_init_shape:",out_init.shape)
        print("out_opt_shape:",out_opt.shape)
        raise AssertionError(f"out_max_abs={max_abs:.4e}")

def run_correctness_suite():
    # torch.manual_seed(23)
    numtokens_list = [2**item for item in range(1, 18)] 
    # numtokens_list = [8192,32768]
    hiddensizes_list = [1024, 2048, 4096, 7168, 8192] #2048, 7168
    # hiddensizes_list = [7168] #2048, 7168

    for num_tokens in numtokens_list:
        for hidden_size in hiddensizes_list:
            check_correctness_once(num_tokens, hidden_size)
    print("[Correctness] All tests passed ")

# ------------------ 指标计算函数 ------------------
def calculate_io_metrics(num_tokens, hidden_sizes, input_dtype=torch.bfloat16):
    input_elem_bytes = torch.finfo(input_dtype).bits // 8
    read_byte = num_tokens * hidden_sizes * input_elem_bytes
    
    write_xq_byte = num_tokens * hidden_sizes * 1 
    write_scale_byte = num_tokens * 1 * 4
    write_byte = write_xq_byte + write_scale_byte
    
    io_bytes = read_byte + write_byte
    
    return read_byte, write_byte, io_bytes

# ------------------ 性能 ------------------
# numtokens_list = [8192, 65536]
numtokens_list = [2**item for item in range(1, 18)] 
hiddensizes_list = [4096] #2048, 7168, 4096
# numtokens_list = list(range(1, 33))
# hiddensizes_list = [2048] #2048, 7168, 4096
configs = [(num_tokens, hidden_sizes) for num_tokens in numtokens_list for hidden_sizes in hiddensizes_list]

def bench_quant_op(provider, input, output, smoothscale, rep=1000):
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    times = []
    for _ in range(rep):
        if provider == "init": 
            start.record()
            int8_quant_init(output, input, smoothscale)
            end.record()
        elif provider == "opt":
            start.record()
            int8_quant_opt(input, smoothscale)
            end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    
    ms = torch.tensor(times).mean().item()
    min_ms = torch.tensor(times).min().item()
    max_ms = torch.tensor(times).max().item()
    
    return ms, min_ms, max_ms

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["shape"], 
        x_vals=[f"[{num_tokens},{hidden_sizes}]" for num_tokens, hidden_sizes in configs], 
        line_arg="provider",
        line_vals=["init", "opt"],
        line_names=["Init(Triton)", "Opt (ops)"], # Updated line name
        styles=[("blue", "-"), ("red", "-")],
        ylabel="Latency (us)",
        plot_name="per-token-quant-int8-performance",
        args={},
    )
)

def benchmark(shape, provider, warmup=10):
    num_tokens, hidden_sizes = map(int, shape.strip("[]").split(","))
    dtype = torch.bfloat16
    # dtype = torch.float
    device = torch.device("cuda")
    input = torch.randn((num_tokens, hidden_sizes), device=device, dtype=dtype)
    output = torch.empty((num_tokens, hidden_sizes), device=device, dtype=torch.int8)

    smoothscale = torch.ones(
        hidden_sizes,
        dtype=torch.float32,
        device = device
    ).contiguous()
    
    m = num_tokens 
    num_local_tokens = [m]
    # num_local_tokens_tensor = torch.tensor(num_local_tokens,device=device,dtype=torch.int32)

    smoothscale = torch.ones(
        hidden_sizes,
        dtype=torch.float32,
        device = device
    ).contiguous()

    quantiles = [0.5, 0.2, 0.8]

    if provider == "init":
        # Warmup for the Triton implementation
        for _ in range(warmup):
            out_init, scales_init = int8_quant_init(output, input, smoothscale)
        torch.cuda.synchronize()
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: int8_quant_init(output, input, smoothscale),
            quantiles=quantiles,
            warmup=warmup, # 使用传入的 1000 次预热
            rep=100        # 设置重复运行 100 次用于计算平均值
        )
        # ms, min_ms, max_ms = bench_quant_op(
        #     provider="init",
        #     input=input,
        #     output=output,
        #     smoothscale=smoothscale,
        #     rep=100,   # 原 100 次重复
        # )
    elif provider == "opt":
        for _ in range(warmup):
            out_opt, scales_opt = int8_quant_opt(input, smoothscale)
        torch.cuda.synchronize()
        ms, min_ms, max_ms = bench_quant_op(
            provider="opt",
            input=input,
            output=output,
            smoothscale=smoothscale,
            rep=100,   # 原 100 次重复
        )
    else:
        raise ValueError(f"Unknown provider: {provider}")
    
    # ------------------ 计算全量指标 ------------------
    # 1. 延迟：ms -> us（微秒）
    latency_us = 1000 * ms
    latency_us_min = 1000 * min_ms
    latency_us_max = 1000 * max_ms
    
    # 2. IO字节数
    read_byte, write_byte, io_bytes = calculate_io_metrics(num_tokens, hidden_sizes, dtype)
    
    # 3. 内存带宽（GB/s）：IO字节数(GB) / 延迟(秒)
    # 字节转GB：1GB = 1024^3 字节；微秒转秒：1秒=1e6微秒
    io_gb = io_bytes / (1024 ** 3)
    time_s = latency_us / 1e6
    mem_bw = io_gb / time_s
    
    # 最大/最小带宽（对应最小/最大延迟）
    time_s_min = latency_us_min / 1e6
    time_s_max = latency_us_max / 1e6
    mem_bw_max = io_gb / time_s_min
    mem_bw_min = io_gb / time_s_max
    
    # if provider == "opt":

    # ------------------ 打印全量指标------------------
    print(f"\n=== Shape: [{num_tokens},{hidden_sizes}] ===")
    print(f"name:             : {provider}")
    print(f"Latency (us)      : avg={latency_us:.2f} | min={latency_us_min:.2f} | max={latency_us_max:.2f}")
    print(f"Read Byte         : {read_byte:,} B ({read_byte/(1024**2):.2f} MB)")
    print(f"Write Byte        : {write_byte:,} B ({write_byte/(1024**2):.2f} MB)")
    print(f"IO Byte           : {io_bytes:,} B ({io_bytes/(1024**2):.2f} MB)")
    print(f"Memory BW (GB/s)  : avg={mem_bw:.2f} | max={mem_bw_max:.2f} | min={mem_bw_min:.2f}")

    # 返回延迟
    return 1000 * ms, 1000 * max_ms, 1000 * min_ms

# ------------------ 入口 ------------------
if __name__ == "__main__":
    # run_correctness_suite()
    benchmark.run(print_data=True, warmup=100)

