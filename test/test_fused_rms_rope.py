import torch
import triton
import pandas as pd
from vllm import _custom_ops as ops
from lightop import rms_rotary_embedding_fuse
from triton.testing import do_bench
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
OUTPUT_FILE = "benchmark_final_results.xlsx"

# 定义测试形状列表: (BATCH_Q, SEQ_Q, NUM_HEADS_Q, NUM_HEADS_K, HEAD_SIZE)
TEST_SHAPES = [
    (1, 128, 16, 4, 128),    # User Provided Case
    (1, 128, 32, 8, 128),   # Batched
]

TEST_NEOX_STYLES = [True, False]
TEST_RESIDUALS = [True, False]

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
# 4. 你的精度测试逻辑 (保持原样)
# =====================================
def get_precision_tolerance(dtype):
    if dtype == torch.float32:
        return 1e-6, 1e-6
    else:
        return 5e-3, 5e-3

def compare(a, b, name, dtype):
    atol, rtol = get_precision_tolerance(dtype)
    # print(a) # 为了日志整洁，批量测试时建议注释掉打印 Tensor
    # print(b)
    max_diff = (a - b).abs().max().item()
    mean_diff = (a - b).abs().mean().item()
    print(f"   > {name}: max_diff={max_diff:.6e}, mean_diff={mean_diff:.6e}")
    assert torch.allclose(a, b, atol=atol, rtol=rtol), f"{name} mismatch!"
    return max_diff

# =====================================
# 5. 单次测试流程 (Runner)
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
        diff_q = compare(q_tgt, q_ref, "Query", DTYPE)
        diff_k = compare(k_tgt, k_ref, "Key", DTYPE)
        max_diff_record = max(diff_q, diff_k)
        print("   ✅ Accuracy Verified")
    except AssertionError as e:
        status = "Fail"
        print(f"   ❌ Accuracy Failed: {e}")

    # ---------------------------
    # Speed Test
    # ---------------------------
    # 定义闭包用于 Benchmark
    # 注意：vLLM ops 是 In-place 修改的，do_bench 循环无法重置数据。
    # 我们这里主要测量计算吞吐，忽略因数据数值变化带来的极微小延迟差异。
    
    def baseline_fn():
        baseline_kernel_adapter(q_org, k_org, positions, w_q, w_k, res_q_org, res_k_org, cos_sin_cache, is_neox, use_residual)

    # 为了公平，给 Fused Kernel 一个独立的数据副本进行 Bench
    q_bench, k_bench = q_org.clone(), k_org.clone()
    r_q_bench = res_q_org.clone() if use_residual else None
    r_k_bench = res_k_org.clone() if use_residual else None

    def fused_fn():
        fused_kernel_adapter(q_bench, k_bench, positions, w_q, w_k, r_q_bench, r_k_bench, cos_sin_cache, is_neox, use_residual)

    try:
        # do_bench 返回的是毫秒(ms)
        ms_base = do_bench(baseline_fn, rep=1000, grad_to_none=[q_org, k_org])
        ms_fuse = do_bench(fused_fn, rep=1000, grad_to_none=[q_bench, k_bench])
    except Exception as e:
        print(f"   ❌ Benchmark Error: {e}")
        return None

    # 转换为微秒 (us)
    base_us = ms_base * 1000
    fuse_us = ms_fuse * 1000
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
# 6. 主程序
# =====================================
def main():
    results = []
    combinations = list(itertools.product(TEST_SHAPES, TEST_NEOX_STYLES, TEST_RESIDUALS))
    
    print(f"🚀 开始测试 | DTYPE: {DTYPE} | Total Cases: {len(combinations)}")
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