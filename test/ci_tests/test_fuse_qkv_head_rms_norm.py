import torch
import time
import csv
import pytest
from lightop import op

# =====================================================================
# 1. PyTorch 基准实现 (修复了原地修改 bug)
# =====================================================================
def baseline_qk_rms_norm(token_data, q_norm_weight, k_norm_weight, q_head_num, kv_head_num, qk_head_dim, eps=1e-5):
    out = token_data.clone()

    q_start = 0
    q_end = q_head_num * qk_head_dim
    k_start = q_head_num * qk_head_dim
    k_end = k_start + kv_head_num * qk_head_dim

    # 提取 Q 和 K，转为 float32 计算 norm 防止溢出
    q_head_data = out[:, q_start:q_end].view(-1, q_head_num, qk_head_dim).float()
    k_head_data = out[:, k_start:k_end].view(-1, kv_head_num, qk_head_dim).float()

    q_normed = torch.nn.functional.rms_norm(
        q_head_data,
        normalized_shape=(qk_head_dim,),
        weight=q_norm_weight.float(),
        eps=eps
    )
    k_normed = torch.nn.functional.rms_norm(
        k_head_data,
        normalized_shape=(qk_head_dim,),
        weight=k_norm_weight.float(),
        eps=eps
    )

    out[:, q_start:q_end] = q_normed.to(out.dtype).view(-1, q_head_num * qk_head_dim)
    out[:, k_start:k_end] = k_normed.to(out.dtype).view(-1, kv_head_num * qk_head_dim)

    return out

# =====================================================================
# 2. 测试与验证主逻辑
# =====================================================================
NUM_TOKENS_LIST = [
    32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 128,
    144, 160, 176, 192, 208, 224, 240, 256,
    288, 320, 352, 384, 416, 448, 480, 512,
    640, 768, 896, 1024,
    1152, 1280, 1408, 1536, 1664, 1792, 1920, 2048,
    2176, 2304, 2432, 2560, 2688, 2816, 2944, 3072,
    3200, 3328, 3456, 3584, 3712, 3840, 3968, 4096,
    6144, 8192, 10240, 12288, 14336, 16384, 18432, 20480,
    22528, 24576, 26624, 28672, 32768
]


def run_case(num_tokens, q_head_num=8, kv_head_num=8, qk_head_dim=128, v_head_dim=128):
    norm_dim = (q_head_num + kv_head_num) * qk_head_dim
    not_norm_dim = kv_head_num * v_head_dim
    total_dim = norm_dim + not_norm_dim

    dtype = torch.bfloat16
    device = "cuda"
    eps = 1e-5
    warmup_iters = 10
    test_iters = 100

    # 随机初始化数据
    token_data = torch.randn((num_tokens, total_dim), dtype=dtype, device=device)
    q_weight = torch.ones(qk_head_dim, dtype=torch.float32, device=device)
    k_weight = torch.ones(qk_head_dim, dtype=torch.float32, device=device)

    # 1. 精度验证 (Correctness)
    base_out = baseline_qk_rms_norm(token_data, q_weight, k_weight, q_head_num, kv_head_num, qk_head_dim, eps)

    cuda_out = token_data.clone()
    op.fuse_qkv_head_rms_norm(cuda_out, q_weight, k_weight, q_head_num, kv_head_num, qk_head_dim, eps)

    # 验证整体的精度差异 (使用 allclose，atol 和 rtol 设为 1e-2 适应 bf16 精度)
    is_close = torch.allclose(base_out, cuda_out, atol=1e-2, rtol=1e-2)
    assert is_close, (
        f"Q/KV norm 精度不达标 (num_tokens={num_tokens}): "
        f"max_abs_diff="
        f"{(base_out.float() - cuda_out.float()).abs().max().item():.4e}"
    )

    # 验证 V 区域是否严格保持不变 (可以直接用 torch.equal 严格判断)
    v_start = norm_dim
    is_v_intact = torch.equal(token_data[:, v_start:], cuda_out[:, v_start:])
    assert is_v_intact, "V_head data was incorrectly modified!"

    # 2. PyTorch 性能测试
    for _ in range(warmup_iters):
        _ = baseline_qk_rms_norm(token_data, q_weight, k_weight, q_head_num, kv_head_num, qk_head_dim, eps)
    torch.cuda.synchronize()
    start_time = time.perf_counter()
    for _ in range(test_iters):
        _ = baseline_qk_rms_norm(token_data, q_weight, k_weight, q_head_num, kv_head_num, qk_head_dim, eps)
    torch.cuda.synchronize()
    base_time_us = (time.perf_counter() - start_time) * 1e6 / test_iters

    # 3. CUDA Kernel 性能测试
    cuda_out_perf = token_data.clone() # 保持数据结构
    for _ in range(warmup_iters):
        op.fuse_qkv_head_rms_norm(cuda_out_perf, q_weight, k_weight, q_head_num, kv_head_num, qk_head_dim, eps)
    torch.cuda.synchronize()
    start_time = time.perf_counter()
    for _ in range(test_iters):
        op.fuse_qkv_head_rms_norm(cuda_out_perf, q_weight, k_weight, q_head_num, kv_head_num, qk_head_dim, eps)
    torch.cuda.synchronize()
    cuda_time_us = (time.perf_counter() - start_time) * 1e6 / test_iters

    # 4. 计算带宽 (Bandwidth)
    # 读：Q 和 K 的 Token 数据，外加 weight。写：修改后的 Q 和 K Token 数据。 (V 不参与访存读写计算)
    bytes_per_elem = token_data.element_size()
    io_bytes = num_tokens * norm_dim * bytes_per_elem * 2 # 读 + 写
    io_bytes += (qk_head_dim * 2 * 4) # Q/K Weight (FP32 -> 4 bytes)

    bandwidth_gb_s = (io_bytes / 1e9) / (cuda_time_us / 1e6)
    speedup = base_time_us / cuda_time_us if cuda_time_us > 0 else float('inf')

    return {
        "num_tokens": num_tokens,
        "qkv_heads": f"{q_head_num}_{kv_head_num}",
        "qk_head_dim": qk_head_dim,
        "base_time_us": base_time_us,
        "cuda_time_us": cuda_time_us,
        "speedup": speedup,
        "bandwidth_gb_s": bandwidth_gb_s,
    }


def run_benchmark(save_to_csv=False, csv_filename="benchmark_results.csv"):
    print(f"\n{'='*115}")
    print(f"{'Tokens':<8} | {'Q/KV Heads':<10} | {'Dim':<5} | {'Result':<10} | {'PyTorch(us)':<12} | {'CUDA(us)':<10} | {'Speedup':<8} | {'Bandwidth(GB/s)':<15}")
    print(f"{'-'*115}")

    # CSV 初始化
    csv_file = None
    csv_writer = None
    if save_to_csv:
        csv_file = open(csv_filename, mode='w', newline='', encoding='utf-8')
        csv_writer = csv.writer(csv_file)
        # 写入 CSV 表头
        csv_writer.writerow(['Tokens', 'Q/KV Heads', 'Dim', 'Result', 'PyTorch(us)', 'CUDA(us)', 'Speedup', 'Bandwidth(GB/s)'])

    for num_tokens in NUM_TOKENS_LIST:
        result = run_case(num_tokens)

        # 将分隔符 "/" 改为 "_"，避免 Excel 自动将 "8/8" 解析为日期 "8月8日"
        print(f"{result['num_tokens']:<8} | {result['qkv_heads']:<10} | {result['qk_head_dim']:<5} | {'Pass ✅':<10} | {result['base_time_us']:<12.2f} | {result['cuda_time_us']:<10.2f} | {result['speedup']:<8.2f} | {result['bandwidth_gb_s']:<15.2f}")

        # 写入单行数据
        if save_to_csv and csv_writer is not None:
            csv_writer.writerow([
                result["num_tokens"],
                result["qkv_heads"],
                result["qk_head_dim"],
                "Pass",  # CSV中去掉表情符号以便后续处理更干净
                f"{result['base_time_us']:.2f}",
                f"{result['cuda_time_us']:.2f}",
                f"{result['speedup']:.2f}",
                f"{result['bandwidth_gb_s']:.2f}"
            ])

    # 关闭文件句柄
    if save_to_csv and csv_file is not None:
        csv_file.close()
        print(f"\n[INFO] Benchmark results have been successfully saved to: {csv_filename}")

    print(f"{'='*115}\n")


@pytest.mark.parametrize("num_tokens", NUM_TOKENS_LIST)
def test_fuse_qkv_head_rms_norm(num_tokens):
    if not torch.cuda.is_available():
        pytest.skip("需要 CUDA/ROCm 设备")
    run_case(num_tokens)


if __name__ == "__main__":
    run_benchmark(save_to_csv=True, csv_filename="qkv_norm_benchmark_results.csv")
