import torch
import triton
from vllm import _custom_ops as vllm_ops
from lightop import op as ops
import pdb
# -------------------------------
# 包一层，便于替换/调试
# -------------------------------
def biased_grouped_topk_org_init(scores, bias, num_expert_group, topk_group, topk, n_share_experts_fusion, routed_scaling_factor):
    return vllm_ops.moe_fused_gate(scores, bias, num_expert_group, topk_group, topk, n_share_experts_fusion, routed_scaling_factor)

def biased_grouped_topk_org_opt(scores, bias, num_expert_group, topk_group, topk, n_share_experts_fusion, routed_scaling_factor):
    return ops.moe_fused_gate(scores, bias, num_expert_group, topk_group, topk, n_share_experts_fusion, routed_scaling_factor)

# -------------------------------
# 正确性验证 (仅 bf16)
# -------------------------------
def check_correctness_once(
    token_num: int,
    num_experts: int,
    num_expert_group: int,
    topk_group: int,
    topk: int,
    n_share_experts_fusion: int, 
    routed_scaling_factor: float,
    rtol=1e-6,
    atol=1e-6
):
    device = torch.device("cuda")
    dtype = torch.bfloat16

    scores = torch.randn((token_num, num_experts), device=device, dtype=dtype)
    bias = torch.randn(num_experts, device=device, dtype=dtype)
    #vllm 使用的还是混合精度，lightop全部强转为了float
    w_init, i_init = biased_grouped_topk_org_init(
        scores.float(), bias.float(), num_expert_group, topk_group, topk, n_share_experts_fusion, routed_scaling_factor
    )
    w_opt, i_opt = biased_grouped_topk_org_opt(
        scores.clone(), bias.clone(), num_expert_group, topk_group, topk, n_share_experts_fusion, routed_scaling_factor
    )
    # pdb.set_trace()
    assert w_init.shape == w_opt.shape, f"weights shape mismatch: {w_init.shape} vs {w_opt.shape}"
    assert i_init.shape == i_opt.shape, f"indices shape mismatch: {i_init.shape} vs {i_opt.shape}"

    if not torch.equal(i_init, i_opt):
        diff = (i_init != i_opt)
        n_diff = diff.sum().item()
        bad_pos = torch.nonzero(diff)[:10].tolist()
        raise AssertionError(
            f"indices mismatch: {n_diff} elements differ. First few positions: {bad_pos}"
        )

    if not torch.allclose(w_init, w_opt, rtol=rtol, atol=atol):
        max_abs = (w_init - w_opt).abs().max().item()
        max_rel = ((w_init - w_opt).abs() / (w_opt.abs() + 1e-12)).max().item()
        raise AssertionError(
            f"weights not close: max_abs={max_abs:.4e}, max_rel={max_rel:.4e}, "
            f"rtol={rtol}, atol={atol}"
        )

def run_correctness_suite():
    torch.manual_seed(0)
    assert torch.cuda.is_available(), "CUDA is required"
    n_share_experts_fusion = 0
    routed_scaling_factor = 2.5

    extra = [600, 700, 800, 900, 1000, 2000, 3000, 4000,
            5000, 10000, 15000, 20000, 25000, 30000, 35000, 40000,
            45000, 50000, 55000, 60000]

    cases = [
        (n, 256, 8, 4, 8, n_share_experts_fusion, routed_scaling_factor)
        for n in list(range(12, 13)) 
    ]

    for (token_num, ne, ng, tg, tk, nshar, rout) in cases:
        check_correctness_once(
            token_num=token_num,
            num_experts=ne,
            num_expert_group=ng,
            topk_group=tg,
            topk=tk,
            n_share_experts_fusion=nshar, 
            routed_scaling_factor=rout
        )
    print("[Correctness] All bf16 tests passed ✅")

# -------------------------------
# 性能基准 (仅 bf16，支持 warmup)
# -------------------------------
token_num_range = list(range(12, 13))
# more_tokens = [
    
#     #, 65000, 70000, 75000, 80000, 85000, 90000, 95000, 100000,
# ]
# token_num_range.extend(more_tokens)

configs = [(sq,) for sq in token_num_range]

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["token_num"],
        x_vals=[list(_) for _ in configs],
        line_arg="provider",
        line_vals=["init", "opt"],
        line_names=["Init", "Opt"],
        styles=[("blue", "-"), ("red", "-")],
        ylabel="us",
        plot_name="moe-fused-gate-performance",
        args={},
    )
)
def benchmark(token_num, provider, warmup=5):
    dtype = torch.bfloat16
    device = torch.device("cuda")
    num_experts, num_expert_group, topk_group, topk = 256, 8, 4, 8
    n_share_experts_fusion = 0
    routed_scaling_factor = 2.5

    scores = torch.randn((token_num, num_experts), device=device, dtype=dtype)
    bias = torch.rand(num_experts, device=device, dtype=dtype)

    quantiles = [0.5, 0.2, 0.8]

    if provider == "init":
        for _ in range(warmup):
            biased_grouped_topk_org_init(scores.clone(), bias.clone(), num_expert_group, topk_group, topk, n_share_experts_fusion, routed_scaling_factor)
        torch.cuda.synchronize()
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: biased_grouped_topk_org_init(scores.clone(), bias.clone(), num_expert_group, topk_group, topk, n_share_experts_fusion, routed_scaling_factor),
            quantiles=quantiles,
        )
    elif provider == "opt":
        for _ in range(warmup):
            biased_grouped_topk_org_opt(scores.clone(), bias.clone(), num_expert_group, topk_group, topk, n_share_experts_fusion, routed_scaling_factor)
        torch.cuda.synchronize()
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: biased_grouped_topk_org_opt(scores.clone(), bias.clone(), num_expert_group, topk_group, topk, n_share_experts_fusion, routed_scaling_factor),
            quantiles=quantiles,
        )
    else:
        raise ValueError(f"Unknown provider: {provider}")

    return 1000 * ms, 1000 * max_ms, 1000 * min_ms

# -------------------------------
# 入口
# -------------------------------
if __name__ == "__main__":
    run_correctness_suite()
    # warmup 可定义，比如 10
    benchmark.run(print_data=True, warmup=10)
