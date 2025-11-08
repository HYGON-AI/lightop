import torch
import triton
from vllm import _custom_ops as vllm_ops
from lightop import fuse_silu_mul_quant_ep
from typing import Optional

# ------------------ 接口 ------------------
def fuse_silu_mul_quant_init(output: torch.Tensor, input: torch.Tensor):
    torch.ops._C.silu_and_mul_opt(output, input)
    ref_out, ref_scales, _= vllm_ops.scaled_int8_quant(output)
    return ref_out, ref_scales

def fuse_silu_mul_quant_opt(input: torch.Tensor,
                            tokens_per_expert: Optional[torch.Tensor] = None): 
    opt_out, opt_scales = fuse_silu_mul_quant_ep(input, tokens_per_expert) 
    return opt_out, opt_scales

# ------------------ 正确性 ------------------
def check_correctness_once(T_dim: int, H_dim: int, E_dim: int, rtol=1e-7, atol=1e-7):
    device = torch.device("cuda")
    dtype = torch.bfloat16
    
    tokens_per_expert = torch.randint(0, T_dim + 1, (E_dim,), device=device, dtype=torch.int32)
    print("tokens_per_expert", tokens_per_expert)
    d_dim = H_dim // 2

    input_3d = torch.randn((E_dim, T_dim, H_dim), device=device, dtype=dtype)
    out_init_3d = torch.empty((E_dim, T_dim, d_dim), device=device, dtype=dtype)
    out_opt_3d = torch.empty((E_dim, T_dim, d_dim), dtype=torch.int8, device=device) 
    scales_opt_3d = torch.empty((E_dim, T_dim, 1), dtype=torch.float32, device=device)

    input_2d_ref = input_3d.view(-1, H_dim)
    out_init_2d_ref = out_init_3d.view(-1, d_dim)
    
    out_init, scales_init = fuse_silu_mul_quant_init(out_init_2d_ref, input_2d_ref)
    
    out_init = out_init.view(E_dim, T_dim, d_dim)
    scales_init = scales_init.view(E_dim, T_dim, 1)

    out_opt, scales_opt = fuse_silu_mul_quant_opt(input_3d, tokens_per_expert)
    
    t_indices = torch.arange(T_dim, device=device, dtype=torch.int64).unsqueeze(0)
    mask = (t_indices < tokens_per_expert.unsqueeze(1).to(torch.int64))

    mask_for_out = mask.unsqueeze(-1)
    mask_for_scales = mask.unsqueeze(-1)

    scales_init_valid = scales_init.masked_select(mask_for_scales)
    scales_opt_valid = scales_opt.masked_select(mask_for_scales) #lj
    # scales_opt_valid = scales_opt

    if not torch.allclose(scales_init_valid, scales_opt_valid, rtol=rtol, atol=atol):
        diff = (scales_init_valid - scales_opt_valid).abs()
        max_abs = diff.max().item() if diff.numel() > 0 else 0.0
        print(f"Shapes (E,T,d): ({E_dim}, {T_dim}, {d_dim})")
        print(f"Tokens per expert: {tokens_per_expert.tolist()}")
        print("scales_init (valid):", scales_init_valid.flatten()[:10])
        print("scales_opt (valid):", scales_opt_valid.flatten()[:10])
        raise AssertionError(f"scales_max_abs={max_abs:.4e}")

    out_init_valid = out_init.masked_select(mask_for_out)
    out_opt_valid = out_opt.masked_select(mask_for_out) #lj
    # out_opt_valid = out_opt

    if not torch.allclose(out_init_valid, out_opt_valid, rtol=rtol, atol=atol):
        diff = (out_init_valid - out_opt_valid).abs()
        max_abs = diff.max().item() if diff.numel() > 0 else 0.0
        print(f"Shapes (E,T,d): ({E_dim}, {T_dim}, {d_dim})")
        print(f"Tokens per expert: {tokens_per_expert.tolist()}")
        raise AssertionError(f"output_max_abs={max_abs:.4e}")
    

def run_correctness_suite():
    T_dim_list = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192] 
    E_dim = 8
    H_dim_list = [4096]
    
    for T_dim in T_dim_list:
        for H_dim in H_dim_list:
            check_correctness_once(T_dim, H_dim, E_dim)
    print("[Correctness] All tests passed ")

# ------------------ 性能 ------------------
numtokens_list = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192] 
hiddensizes_list = [4096]

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
        plot_name="fuse-silu-mul-quant-performance (E=8)",
        args={},
    )
)
def benchmark(shape, provider, warmup=10):
    num_tokens, hidden_sizes = map(int, shape.strip("[]").split(","))
    dtype = torch.bfloat16
    device = torch.device("cuda")
    E_dim = 8 

    input_3d = torch.randn((E_dim, num_tokens, hidden_sizes), device=device, dtype=dtype)
    output_3d = torch.empty((E_dim, num_tokens, hidden_sizes//2), device=device, dtype=dtype)
    
    quantiles = [0.5, 0.2, 0.8]

    if provider == "init":
        input_2d_ref = input_3d.view(-1, hidden_sizes)
        output_2d_ref = output_3d.view(-1, hidden_sizes//2)
        
        f = lambda: fuse_silu_mul_quant_init(output_2d_ref, input_2d_ref)
        for _ in range(warmup):
            f()
        torch.cuda.synchronize()
        ms, min_ms, max_ms = triton.testing.do_bench(f, quantiles=quantiles)
        
    elif provider == "opt":
        f = lambda: fuse_silu_mul_quant_opt(input_3d) 
        
        for _ in range(warmup):
            f()
        torch.cuda.synchronize()
        ms, min_ms, max_ms = triton.testing.do_bench(f, quantiles=quantiles)
    else:
        raise ValueError(f"Unknown provider: {provider}")

    return 1000 * ms, 1000 * max_ms, 1000 * min_ms

# ------------------ 入口 ------------------
if __name__ == "__main__":
    run_correctness_suite()
    benchmark.run(print_data=True, warmup=10)