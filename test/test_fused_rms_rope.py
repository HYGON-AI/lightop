import torch
import triton
import pandas as pd
from vllm import _custom_ops as ops
from lightop import rms_rotary_embedding_fuse
# from triton.testing import do_bench # 不再需要
import itertools
import os

# =====================================
# 1. 全局配置
# =====================================
torch.manual_seed(0)
DEVICE = "cuda"
DTYPE = torch.float16  # 修改此处切换精度: float16 | float32 | bfloat16
RMS_EPS = 1e-6
ROPE_THETA = 10000.0
OUTPUT_FILE = "benchmark_final_results_manual.xlsx"

# 计时配置
WARMUP_ITERS = 10
BENCH_ITERS = 100

# 定义测试形状列表: (BATCH_Q, SEQ_Q, NUM_HEADS_Q, NUM_HEADS_K, HEAD_SIZE)
TEST_SHAPES = (
    [(1, d1, 3, 1, 128) for d1 in (1 << i for i in range(17))]
)

TEST_NEOX_STYLES = [True]
TEST_RESIDUALS = [False]

# =====================================
# 2. 辅助工具 (Cos/Sin & Data Gen)
# =====================================
def build_cos_sin_cache(max_position, dim, base=10000.0, dtype=torch.float32, device="cuda"):
    half_dim = dim // 2
    freq_seq = torch.arange(0, half_dim, dtype=dtype, device=device)
    inv_freq = 1.0 / (base ** (freq_seq / half_dim))
    t = torch.arange(0, max_position, dtype=dtype, device=device)
    freqs = torch.einsum("i,j->ij", t, inv_freq)
    cos = torch.cos(freqs)
    sin = torch.sin(freqs)
    return torch.cat([cos, sin], dim=-1).contiguous()

def get_tensors(b_q, s_q, h_q, h_k, dim, device, dtype, generate_residual):
    q = torch.randn(b_q, s_q, h_q, dim, device=device, dtype=dtype)
    k = torch.randn(b_q, s_q, h_k, dim, device=device, dtype=dtype)
    positions = torch.arange(s_q, device=device).unsqueeze(0).expand(b_q, s_q).contiguous()
    weight_q = torch.ones(dim, device=device, dtype=dtype)
    weight_k = torch.ones(dim, device=device, dtype=dtype)
    res_q = torch.randn_like(q) if generate_residual else None
    res_k = torch.randn_like(k) if generate_residual else None
    return q, k, positions, weight_q, weight_k, res_q, res_k

# =====================================
# 3. 核心 Kernel 封装
# =====================================
def baseline_kernel_adapter(q, k, positions, w_q, w_k, res_q, res_k, cache, is_neox, use_res):
    if use_res:
        ops.fused_add_rms_norm_opt(q, res_q, w_q, RMS_EPS)
        ops.fused_add_rms_norm_opt(k, res_k, w_k, RMS_EPS)
    else:
        ops.rms_norm_opt(q, q, w_q, RMS_EPS)
        ops.rms_norm_opt(k, k, w_k, RMS_EPS)
    ops.rotary_embedding(positions, q, k, w_q.shape[-1], cache, is_neox)

def fused_kernel_adapter(q, k, positions, w_q, w_k, res_q, res_k, cache, is_neox, use_res):
    rms_rotary_embedding_fuse(
        positions, q, k, w_q.shape[-1], cache, is_neox,
        w_q, w_k, 
        residual_q=res_q, residual_k=res_k, 
        epsilon=RMS_EPS
    )

# =====================================
# 4. 手动 Benchmark 工具 (GPU Event 计时)
# =====================================
def benchmark_func(func, warmup=10, rep=100):
    """
    手动执行 Warmup 和 Timing Loop，返回平均执行时间 (us)
    """
    # 1. 预热 (Warmup)
    for _ in range(warmup):
        func()
    
    # 确保预热完成
    torch.cuda.synchronize()

    # 2. 正式计时 (Measurement)
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    for _ in range(rep):
        func()
    end_event.record()

    # 等待执行结束
    torch.cuda.synchronize()

    # 计算总时间并取平均 (elapsed_time 返回毫秒)
    total_ms = start_event.elapsed_time(end_event)
    avg_us = (total_ms * 1000) / rep # 转换为微秒 (us)
    return avg_us

# =====================================
# 5. 精度测试逻辑
# =====================================
def get_precision_tolerance(dtype):
    if dtype == torch.float32:
        return 1e-6, 1e-6
    else:
        return 1e-2, 1e-2

def compare(a, b, name, dtype):
    atol, rtol = get_precision_tolerance(dtype)
    max_diff = (a - b).abs().max().item()
    mean_diff = (a - b).abs().mean().item()
    print(f"   > {name}: max_diff={max_diff:.6e}, mean_diff={mean_diff:.6e}")
    assert torch.allclose(a, b, atol=atol, rtol=rtol), f"{name} mismatch!"
    return max_diff

# =====================================
# 6. 单次测试流程 (Runner)
# =====================================
def run_single_case(shape_config, is_neox, use_residual):
    batch, seq, h_q, h_k, dim = shape_config
    
    # ---------------------------
    # 准备数据
    # ---------------------------
    q_org, k_org, positions, w_q, w_k, res_q_org, res_k_org = get_tensors(
        batch, seq, h_q, h_k, dim, DEVICE, DTYPE, use_residual
    )
    
    # 准备 Cache
    cos_sin_cache = build_cos_sin_cache(seq, dim, ROPE_THETA, torch.float32, DEVICE)
    if DTYPE != torch.float32:
        cos_sin_cache = cos_sin_cache.to(DTYPE)

    # ---------------------------
    # Accuracy Test
    # ---------------------------
    print(f"🧪 Checking: Shape={shape_config}, Neox={is_neox}, Res={use_residual}")
    
    q_ref, k_ref = q_org.clone(), k_org.clone()
    r_q_ref = res_q_org.clone() if use_residual else None
    r_k_ref = res_k_org.clone() if use_residual else None

    q_tgt, k_tgt = q_org.clone(), k_org.clone()
    r_q_tgt = res_q_org.clone() if use_residual else None
    r_k_tgt = res_k_org.clone() if use_residual else None

    # Run Baseline
    baseline_kernel_adapter(q_ref, k_ref, positions, w_q, w_k, r_q_ref, r_k_ref, cos_sin_cache, is_neox, use_residual)
    # Run Fused
    fused_kernel_adapter(q_tgt, k_tgt, positions, w_q, w_k, r_q_tgt, r_k_tgt, cos_sin_cache, is_neox, use_residual)

    status = "Pass"
    max_diff_record = 0.0

    try:
        diff_k = compare(k_tgt, k_ref, "Key", DTYPE)
        diff_q = compare(q_tgt, q_ref, "Query", DTYPE)
        max_diff_record = max(diff_q, diff_k)
        print("   ✅ Accuracy Verified")
    except AssertionError as e:
        status = "Fail"
        print(f"   ❌ Accuracy Failed: {e}")

    # ---------------------------
    # Speed Test (Manual Timing)
    # ---------------------------
    
    def baseline_fn():
        baseline_kernel_adapter(q_org, k_org, positions, w_q, w_k, res_q_org, res_k_org, cos_sin_cache, is_neox, use_residual)

    # 为了公平，给 Fused Kernel 一个独立的数据副本进行 Bench
    q_bench, k_bench = q_org.clone(), k_org.clone()
    r_q_bench = res_q_org.clone() if use_residual else None
    r_k_bench = res_k_org.clone() if use_residual else None

    def fused_fn():
        fused_kernel_adapter(q_bench, k_bench, positions, w_q, w_k, r_q_bench, r_k_bench, cos_sin_cache, is_neox, use_residual)

    try:
        base_us = benchmark_func(baseline_fn, warmup=WARMUP_ITERS, rep=BENCH_ITERS)
        fuse_us = benchmark_func(fused_fn, warmup=WARMUP_ITERS, rep=BENCH_ITERS)
    except Exception as e:
        print(f"   ❌ Benchmark Error: {e}")
        return None

    # 不需要再 * 1000，benchmark_func 已经返回 us
    speedup = base_us / fuse_us

    return {
        "Batch": batch, "Seq": seq, "H_Q": h_q, "H_K": h_k, "Dim": dim,
        "Neox": is_neox, "Resid": use_residual,
        "Status": status,
        "Max_Diff": max_diff_record,
        "Base(us)": round(base_us, 2),
        "Fuse(us)": round(fuse_us, 2),
        "Speedup": round(speedup, 3)
    }

# =====================================
# 7. 主程序
# =====================================
def main():
    results = []
    combinations = list(itertools.product(TEST_SHAPES, TEST_NEOX_STYLES, TEST_RESIDUALS))
    
    print(f"🚀 开始测试 | DTYPE: {DTYPE} | Warmup: {WARMUP_ITERS} | Reps: {BENCH_ITERS}")
    print("=" * 100)

    for shape, neox, resid in combinations:
        res = run_single_case(shape, neox, resid)
        if res:
            results.append(res)
            print(f"   ⏱️  Result: Base={res['Base(us)']}us, Fuse={res['Fuse(us)']}us, Speedup={res['Speedup']}x")
            print("-" * 50)

    # 写入 Excel
    if results:
        df = pd.DataFrame(results)
        df.to_excel(OUTPUT_FILE, index=False)
        
        print("\n" + "="*100)
        print(f"✅ 测试完成。结果已保存至: {os.path.abspath(OUTPUT_FILE)}")
        print(f"📊 平均加速比: {df['Speedup'].mean():.2f}x")
        
        fail_df = df[df['Status'] == 'Fail']
        if not fail_df.empty:
            print(f"⚠️  警告: 共有 {len(fail_df)} 个 Case 精度未通过!")
        else:
            print("✨ 所有 Case 精度验证通过。")

if __name__ == "__main__":
    main()