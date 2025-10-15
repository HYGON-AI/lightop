import torch
from vllm import _custom_ops as vllm_ops
from triton.testing import do_bench 
from lightop import fused_rms_norm_contiguous

def init(out: torch.Tensor, x: torch.Tensor, weight: torch.Tensor, epsilon: float):
    """基准函数：直接在（连续的）输入上运行 op。"""
    x_con = x.contiguous()
    vllm_ops.rms_norm_opt(
        out,
        x_con,
        weight,
        epsilon,
    )

def opt(out: torch.Tensor, x: torch.Tensor, weight: torch.Tensor, epsilon: float):
    """待测函数：在运行 op 之前，对输入进行 .contiguous() 操作。"""
    fused_rms_norm_contiguous(
        out,
        x,
        weight,
        epsilon
    )

# --- 测试参数定义 ---
HIDDEN_SIZE = 512
# DTYPE = torch.float16
DTYPE = torch.bfloat16
DEVICE = "cuda"
variance_epsilon = 1e-6

# 定义要测试的一系列批处理大小 (bs)
batch_sizes_to_test = [1, 4, 8, 9 , 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8000]

# do_bench 的运行次数
warmup_runs = 25
test_runs = 100

# 存储所有测试结果
results = []

print(f"--- 开始基准测试 ---")
print(f"对比 `init` (处理连续张量) vs `opt` (处理非连续张量)")
print(f"使用 triton.testing.do_bench 进行性能测量。")
print(f"将要测试的批处理大小: {batch_sizes_to_test}")
print("-" * 65)

# --- 循环测试不同的批处理大小 ---
for bs in batch_sizes_to_test:
    print(f"\n>>> 正在测试批处理大小 (bs): {bs} <<<")

    total_elements = bs * 576
    x_data = torch.randn(total_elements, dtype=DTYPE, device=DEVICE)
    x_str = torch.as_strided(x_data, size=(bs, HIDDEN_SIZE), stride=(576, 1))
    x = x_str.contiguous()
    
    weight = torch.ones(HIDDEN_SIZE, dtype=DTYPE, device=DEVICE)
    
    out_init = torch.empty_like(x)
    out_opt = torch.empty_like(x)

    # -- 正确性验证 --
    # 这里的输入应该都是连续版本，以确保 op 本身计算结果一致
    init_correctness_out = torch.empty_like(x)
    opt_correctness_out = torch.empty_like(x)
    init(init_correctness_out, x_str, weight, variance_epsilon)
    opt(opt_correctness_out, x_str, weight, variance_epsilon) 
    are_close = torch.allclose(init_correctness_out, opt_correctness_out, atol=1e-3)
    print(f"正确性检查: {'通过' if are_close else '失败'}")

    median_time_init = do_bench(lambda: init(out_init, x_str, weight, variance_epsilon), 
                                  warmup=warmup_runs, rep=test_runs, quantiles=(0.5,))
    
    # 测试 opt 函数（处理非连续张量，内部会调用 .contiguous()）
    median_time_opt = do_bench(lambda: opt(out_opt, x_str, weight, variance_epsilon), 
                                 warmup=warmup_runs, rep=test_runs, quantiles=(0.5,))

    print(f"  - init (连续) 中位时间: {median_time_init:.6f} ms")
    print(f"  - opt (非连续) 中位时间: {median_time_opt:.6f} ms")

    # 存储结果
    results.append({
        "bs": bs,
        "time_init_ms": median_time_init*1000,
        "time_opt_ms": median_time_opt*1000,
    })

# --- 打印最终的总结表格 ---
print("\n" + "=" * 70)
print(" " * 22 + "基准测试总结报告")
print(" " * 12 + "(性能通过 triton.do_bench 中位数测量)")
print("=" * 70)
print(f"{'批处理大小':>12} | {'init 中位时间 (ms)':>20} | {'opt 中位时间 (ms)':>20} | {'开销比 (opt/init)':>15}")
print("-" * 70)

for res in results:
    bs = res['bs']
    time_init = res['time_init_ms']
    time_opt = res['time_opt_ms']
    
    ratio = time_init / time_opt
        
    print(f"{bs:>12} | {time_init:>20.6f} | {time_opt:>20.6f} | {ratio:>15.2f}x")

print("=" * 70)
