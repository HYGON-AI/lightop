import torch
import time

try:
    from lightop.tensor import ds_cat
    # 如果您打包后的模块名叫 lightop，这里导入您的实际算子
    from lightop import op
    HAS_LIGHTOP = True
except ImportError:
    HAS_LIGHTOP = False
    print("未检测到 lightop 模块，将使用 PyTorch 原生算子作为 Mock 运行逻辑。")
    # 构建 Mock 函数以便在无 C++ 环境下测试逻辑
    def ds_cat(A, B, C, mode):
        C.copy_(torch.cat([A, B], dim=2))
    
    class MockOp:
        # 匹配 Canvas 中最新的 C++ 代码 QKV 融合接口 (带 zero-cost split)
        def ds_fused_qkv_cast_fp8(self, q, kv_nope, k_pe, qk_nope_head_dim, v_head_dim):
            # Mock 中使用 split 模拟 C++ 的指针偏移逻辑
            k_nope, v = kv_nope.split([qk_nope_head_dim, v_head_dim], dim=-1)
            
            out_q = q.to(torch.float8_e4m3fn)
            out_k = torch.cat([k_nope, k_pe], dim=2).to(torch.float8_e4m3fn)
            out_v = v.to(torch.float8_e4m3fn)
            return [out_q, out_k, out_v]
            
    op = MockOp()

def lightop_concat_prefill_helper(A: torch.Tensor, B: torch.Tensor, dim: int, use_stride_zero : bool):
    if use_stride_zero:
        output_shape = list(A.shape)
        output_shape[dim] = A.shape[dim] + B.shape[dim]  
        C = torch.empty(output_shape, device=A.device, dtype=A.dtype)
        mode = 6
        ds_cat(A, B, C, mode)
        return C
    else:
        return torch.cat([A, B], dim=dim)

def run_fused_qkv_benchmark(M, N_q, N_kv, q_dim, a_section, b_section, v_dim, use_stride_zero=False):
    """
    运行单次 QKV 融合算子评测，返回 (最小位匹配率, 最大绝对误差, 非融合基准耗时_us, 融合算子耗时_us)
    这里的 a_section 即 qk_nope_head_dim, v_dim 即 v_head_dim
    """
    device = "cuda"
    dtype = torch.bfloat16
    
    # --- 准备输入张量 ---
    q = torch.randn((M, N_q, q_dim), dtype=dtype, device=device)
    # 按照实际前向传播生成完整的 kv_nope (k_nope 和 v 在内存中连续)
    kv_nope = torch.randn((M, N_kv, a_section + v_dim), dtype=dtype, device=device)
    
    # 模拟 k_pe (带广播的情况)
    if use_stride_zero:
        B_origin = torch.randn((M, 1, b_section), dtype=dtype, device=device)
        B = B_origin.expand(M, N_kv, b_section)
    else:
        B = torch.randn((M, N_kv, b_section), dtype=dtype, device=device)
    
    # --- 阶段 1: 精度校验 ---
    # 基准 GT: PyTorch 走 Split -> Concat -> 3 次 FP8 Cast
    q_gt = q.to(torch.float8_e4m3fn)
    k_nope, v_unfused = kv_nope.split([a_section, v_dim], dim=-1)
    C_unfused = lightop_concat_prefill_helper(k_nope, B, dim=2, use_stride_zero=use_stride_zero)
    k_gt = C_unfused.to(torch.float8_e4m3fn)
    v_gt = v_unfused.to(torch.float8_e4m3fn)
    
    # 融合算子测试 (一次调用完成全部: split, concat, cast)
    q_custom, k_custom, v_custom = op.ds_fused_qkv_cast_fp8(q, kv_nope, B, a_section, v_dim)
    
    # 验证辅助函数
    def check_diff(gt, custom):
        gt_bytes = gt.view(torch.uint8)
        custom_bytes = custom.view(torch.uint8)
        match_rate = (gt_bytes == custom_bytes).float().mean().item()
        max_diff = (gt.to(torch.float32) - custom.to(torch.float32)).abs().max().item()
        return match_rate, max_diff

    # 计算 Q, K, V 三者的误差
    q_match, q_diff = check_diff(q_gt, q_custom)
    k_match, k_diff = check_diff(k_gt, k_custom)
    v_match, v_diff = check_diff(v_gt, v_custom)
    
    # 取最差的结果作为代表
    min_match_rate = min(q_match, k_match, v_match)
    max_diff_all = max(q_diff, k_diff, v_diff)
    
    # 如果位匹配率低于 90% 且数值误差过大，判定为失败
    if min_match_rate < 0.90 and max_diff_all > 0.1:
        return min_match_rate, max_diff_all, 0.0, 0.0

    # --- 阶段 2: 性能评测 ---
    test_loops = 100

    # 基准预热
    for _ in range(10):
        _q = q.to(torch.float8_e4m3fn)
        _k_nope, _v = kv_nope.split([a_section, v_dim], dim=-1)
        _k = lightop_concat_prefill_helper(_k_nope, B, dim=2, use_stride_zero=use_stride_zero).to(torch.float8_e4m3fn)
        _v = _v.to(torch.float8_e4m3fn)
    torch.cuda.synchronize()
    
    # 基准计时
    start_unfused = time.perf_counter()
    for _ in range(test_loops):
        _q = q.to(torch.float8_e4m3fn)
        _k_nope, _v = kv_nope.split([a_section, v_dim], dim=-1)
        _k = lightop_concat_prefill_helper(_k_nope, B, dim=2, use_stride_zero=use_stride_zero).to(torch.float8_e4m3fn)
        _v = _v.to(torch.float8_e4m3fn)
    torch.cuda.synchronize()
    avg_unfused_us = (time.perf_counter() - start_unfused) * 1e6 / test_loops

    # 融合算子预热
    for _ in range(10):
        _ = op.ds_fused_qkv_cast_fp8(q, kv_nope, B, a_section, v_dim)
    torch.cuda.synchronize() 
    
    # 融合算子计时
    start_fused = time.perf_counter()
    for _ in range(test_loops):
        _ = op.ds_fused_qkv_cast_fp8(q, kv_nope, B, a_section, v_dim)
    torch.cuda.synchronize() 
    avg_fused_us = (time.perf_counter() - start_fused) * 1e6 / test_loops

    return min_match_rate, max_diff_all, avg_unfused_us, avg_fused_us

if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("当前环境无可用 GPU，请在带有 CUDA 的环境执行！")
        exit(0)
    use_stride_zero = True
    print(f"k_pe broadcast mode (use_stride_zero): {use_stride_zero}")
    
    # (M, N_q, N_kv, q_dim, qk_nope_head_dim, k_pe_dim, v_head_dim, use_s0, desc)
    test_cases = [
        (1024,  32, 8, 192, 128, 64, 128, use_stride_zero, "Decode"),
        (4096,  32, 8, 192, 128, 64, 128, use_stride_zero, "Prefill 4K"),
        (8192,  16, 4, 192, 128, 64, 128, use_stride_zero, "Prefill 8K"),
        (12649, 64, 64, 192, 128, 64, 128, use_stride_zero, "Prefill 16K"),
    ]

    mode_info = "真实环境" if HAS_LIGHTOP else "Mock环境 (仅测试逻辑)"
    print(f"正在运行 Fused QKV FP8 Cast 基准测试 [{mode_info}]...")
    print("-" * 125)
    print(f"{'M (Seq)':<8} | {'N_q/kv':<8} | {'q/a/b/v_dim':<15} | {'Bit_Match':<9} | {'Max_Diff':<8} | {'Unfused(us)':<11} | {'Fused(us)':<10} | {'Speedup':<8}")
    print("-" * 125)

    for M, N_q, N_kv, q_dim, a_sec, b_sec, v_dim, use_s0, desc in test_cases:
        try:
            match_rate, max_diff, unfused_us, fused_us = run_fused_qkv_benchmark(
                M, N_q, N_kv, q_dim, a_sec, b_sec, v_dim, use_s0)
            
            match_str = f"{match_rate*100:.1f}%"
            diff_str = f"{max_diff:.4f}"
            
            # 用于排版显示的字符串
            head_str = f"{N_q}/{N_kv}"
            dim_str = f"{q_dim}/{a_sec}/{b_sec}/{v_dim}"
            
            # 放宽严格等于判定，转为依赖 max_diff
            if max_diff > 0.1:
                match_str = "FAIL"
                speedup_str = "N/A"
            else:
                speedup = (unfused_us / fused_us) if fused_us > 1e-3 else 0.0
                speedup_str = f"{speedup:.2f}x"
                
            print(f"{M:<8} | {head_str:<8} | {dim_str:<15} | {match_str:<9} | {diff_str:<8} | {unfused_us:<11.2f} | {fused_us:<10.2f} | {speedup_str:<8}")
            
        except torch.cuda.OutOfMemoryError:
            print(f"{M:<8} | {head_str:<8} | {dim_str:<15} | {'OOM':<9} | {'-':<8} | {'-':<11} | {'-':<10} | {'-':<8}")
            torch.cuda.empty_cache()

    print("-" * 125)
    print("测试完成。Unfused 代表分步计算 (Split + Q/K/V独立FP8 Cast + 拼接), Fused 代表单 Kernel 零开销完成！")