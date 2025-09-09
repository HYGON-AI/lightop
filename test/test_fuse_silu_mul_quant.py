import torch
import triton
from vllm import _custom_ops as vllm_ops
from lightop import fuse_silu_mul_quant

# ------------------ 接口 ------------------
def fuse_silu_mul_quant_init(output: torch.Tensor, input: torch.Tensor):
    torch.ops._C.silu_and_mul_opt(output, input)
    ref_out, ref_scales, _= vllm_ops.scaled_int8_quant(output)
    return ref_out, ref_scales

def fuse_silu_mul_quant_opt(input: torch.Tensor):
    opt_out, opt_scales=fuse_silu_mul_quant(input)
    return opt_out, opt_scales

# ------------------ 正确性 ------------------
def check_correctness_once(num_tokens: int, hidden_sizes: int, rtol=1e-7, atol=1e-7):
    device = torch.device("cuda")
    dtype = torch.bfloat16
    # dtype = torch.float
    # dtype = torch.float16
    input = torch.randn((num_tokens, hidden_sizes), device=device, dtype=dtype)
    out_init = torch.empty((num_tokens, hidden_sizes//2), device=device, dtype=dtype)

    out_init, scales_init = fuse_silu_mul_quant_init(out_init, input)
    out_opt , scales_opt = fuse_silu_mul_quant_opt(input)

    if not torch.allclose(scales_init, scales_opt, rtol=rtol, atol=atol):
        diff = (scales_init - scales_opt).abs()
        max_abs = diff.max().item()
        print("out_shape:",out_init.shape)
        raise AssertionError(f"scales_max_abs={max_abs:.4e}")

    
    if not torch.allclose(out_init, out_opt, rtol=rtol, atol=atol):
        diff = (out_init - out_opt).abs()
        max_abs = diff.max().item()
        print("out_shape:",out_init.shape)
        raise AssertionError(f"scales_max_abs={max_abs:.4e}")
    

def run_correctness_suite():
    # torch.manual_seed(23)
    numtokens_list = [2**item for item in range(0, 18)]  # [1, 2, ... , 65536, 131072]
    hiddensizes_list = [1024, 512, 256] #m=256，k=256

    for num_tokens in numtokens_list:
        for hidden_size in hiddensizes_list:
            check_correctness_once(num_tokens, hidden_size)
    print("[Correctness] All tests passed ")

# ------------------ 性能 ------------------
numtokens_list = [2**item for item in range(0, 18)]  
hiddensizes_list = [1024, 512, 256]
configs = [(num_tokens, hidden_sizes) for num_tokens in numtokens_list for hidden_sizes in hiddensizes_list]

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["shape"],                                  
        x_vals=[f"[{num_tokens},{hidden_sizes}]" for num_tokens, hidden_sizes in configs],        
        line_arg="provider",
        line_vals=["init", "opt"],
        line_names=["Init(vllm)", "Opt (ops)"],
        styles=[("blue", "-"), ("red", "-")],
        ylabel="us",
        plot_name="fuse-silu-mul-quant-performance",
        args={},
    )
)
def benchmark(shape, provider, warmup=10):
    num_tokens, hidden_sizes = map(int, shape.strip("[]").split(","))
    dtype = torch.bfloat16
    # dtype = torch.float
    device = torch.device("cuda")
    input = torch.randn((num_tokens, hidden_sizes), device=device, dtype=dtype)
    output = torch.empty((num_tokens, hidden_sizes//2), device=device, dtype=dtype)

    quantiles = [0.5, 0.2, 0.8]

    if provider == "init":
        for _ in range(warmup):
            out_init, scales_init = fuse_silu_mul_quant_init(output, input)
        torch.cuda.synchronize()
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: fuse_silu_mul_quant_init(output, input),
            quantiles=quantiles,
        )
    elif provider == "opt":
        for _ in range(warmup):
            out_opt, scales_opt = fuse_silu_mul_quant_opt(input)
        torch.cuda.synchronize()
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: fuse_silu_mul_quant_opt(input),
            quantiles=quantiles,
        )
    else:
        raise ValueError(f"Unknown provider: {provider}")

    return 1000 * ms, 1000 * max_ms, 1000 * min_ms

# ------------------ 入口 ------------------
if __name__ == "__main__":
    run_correctness_suite()
    benchmark.run(print_data=True, warmup=10)