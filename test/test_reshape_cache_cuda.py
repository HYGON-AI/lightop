import torch
import random
from lightop.kvcache import reshape_and_cache_cuda
from vllm import _custom_ops as ops
from openpyxl import Workbook

# -----------------------------------------------------------
# 辅助函数：手动计时
# -----------------------------------------------------------
def benchmark_manual(func, warmup=10, rep=100):
    # 预热
    for _ in range(warmup):
        func()
    torch.cuda.synchronize()

    # 正式计时
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    for _ in range(rep):
        func()
    end_event.record()
    torch.cuda.synchronize()

    # elapsed_time 返回的是毫秒(ms)
    total_ms = start_event.elapsed_time(end_event)
    avg_us = (total_ms / rep) * 1000.0
    return avg_us

# -----------------------------------------------------------
# 生成输入
# 修改：固定 num_heads=4, head_size=128
# -----------------------------------------------------------
def vllm_key_cache_to_lightop(key_cache_vllm: torch.Tensor) -> torch.Tensor:
    """Convert vLLM key layout to lightop for accuracy compare.

    vLLM:    [num_blocks, num_heads, head_size/x, block_size, x]
    lightop: [num_blocks, num_heads, block_size, head_size]
    """
    nb, nh, hdx, bs, x = key_cache_vllm.shape
    return (
        key_cache_vllm.permute(0, 1, 3, 2, 4)
        .contiguous()
        .view(nb, nh, bs, hdx * x)
    )


def make_inputs(T, num_heads=4, head_size=128, dtype=torch.float16):
    device = "cuda"
    block_size = 64
    # Keep enough cache blocks for the largest token case while avoiding the
    # excessive allocation used by the legacy benchmark.
    num_blocks = 4666

    # 这里的 T_min 保持为 1
    T_min = 1

    # 生成 Key 和 Value
    # Shape: [num_tokens, num_heads, head_size]
    key = torch.randn(T, num_heads, head_size, device=device, dtype=dtype) * 0.1
    value = torch.randn_like(key)

    # 生成 Slot Mapping
    slot_mapping = torch.arange(T_min, T_min + T, device=device)

    # 随机设置一些 invalid slot (-1)
    invalid_cnt = max(1, T // 50)
    rand_idx = torch.randperm(T)[:invalid_cnt]
    slot_mapping[rand_idx] = -1

    # vLLM key cache layout:
    #   [num_blocks, num_heads, head_size/x, block_size, x]
    # where x = 16 // element_size (fp16/bf16 -> 8, fp32 -> 4)
    x = 16 // torch.tensor([], dtype=dtype).element_size()
    assert head_size % x == 0, f"head_size={head_size} not divisible by x={x}"

    key_cache_vllm = torch.zeros(
        num_blocks, num_heads, head_size // x, block_size, x,
        device=device, dtype=dtype
    )
    value_cache_vllm = torch.zeros(
        num_blocks, num_heads, head_size, block_size,
        device=device, dtype=dtype
    )

    # lightop key/value cache layout:
    #   key:   [num_blocks, num_heads, block_size, head_size]
    #   value: [num_blocks, num_heads, head_size, block_size]
    key_cache_lightop = torch.zeros(
        num_blocks, num_heads, block_size, head_size,
        device=device, dtype=dtype
    )
    value_cache_lightop = torch.zeros(
        num_blocks, num_heads, head_size, block_size,
        device=device, dtype=dtype
    )

    k_scale = torch.tensor([1.0], device=device)
    v_scale = torch.tensor([1.0], device=device)

    return (
        key, value,
        key_cache_vllm, value_cache_vllm,
        key_cache_lightop, value_cache_lightop,
        slot_mapping,
        k_scale, v_scale,
        head_size, block_size, num_heads
    )


# -----------------------------------------------------------
# 测试主流程
# -----------------------------------------------------------
def run_sweep(sheet_name="Benchmark"):
    print(f"\n======= Running sweep: {sheet_name} =======")
    
    # -----------------------------------------------------
    # 修改：指定具体的 Token 列表
    # -----------------------------------------------------
    Ts = [8,16,32, 64,88, 128, 256, 356,420, 512, 1024, 2048, 4096, 8192, 16384,32768,41000]
    # Ts = [256]
    
    # 固定参数
    FIXED_NUM_HEADS = 8
    FIXED_HEAD_DIM = 128
    DTYPE_SIZE = 2 

    print(f"Token list: {Ts}")
    print(f"Params: Heads={FIXED_NUM_HEADS}, HeadDim={FIXED_HEAD_DIM}")
    print("Settings: Warmup=10, Rep=100")

    # -----------------------------------------------------
    # 创建 sheet
    # -----------------------------------------------------
    ws = wb.create_sheet(sheet_name)

    # 列标题 (增加带宽数据)
    ws.append([
        "Token",
        "NumHeads",
        "HeadSize",
        "BlockSize",
        "InvalidCnt",
        "vLLM Time (us)",
        "Custom Time (us)",
        "Speedup (vLLM / Custom)",
        "vLLM BW (GB/s)",   # New
        "Custom BW (GB/s)", # New
        "Accuracy Pass"
    ])

    # -----------------------------------------------------
    # 打印 Console 表头
    # -----------------------------------------------------
    # 定义表头格式
    header_fmt = "{:<8} {:<10} {:<12} {:<12} {:<10} {:<12} {:<12} {:<8}"
    row_fmt    = "{:<8} {:<10} {:<12.3f} {:<12.3f} {:<10.2f} {:<12.2f} {:<12.2f} {:<8}"
    
    print("-" * 100)
    print(header_fmt.format("Token", "Invalid", "vLLM(us)", "Custom(us)", "Speedup", "vLLM(GB/s)", "Cus(GB/s)", "Result"))
    print("-" * 100)

    #用于存储所有用例的加速比
    speedups = [] 

    # -----------------------------------------------------
    # sweep loop
    # -----------------------------------------------------
    for T in Ts:
        (key, value,
         key_cache, value_cache,
         key_cache2, value_cache2,
         slot_mapping,
         k_scale, v_scale,
         head_size, block_size, num_heads) = make_inputs(
             T, 
             num_heads=FIXED_NUM_HEADS, 
             head_size=FIXED_HEAD_DIM,
             dtype=torch.bfloat16
        )

        invalid_cnt = (slot_mapping < 0).sum().item()

        # ---------------------
        # vLLM kernel
        # ---------------------
        def run_vllm():
            ops.reshape_and_cache(
                key, value,
                key_cache, value_cache,
                slot_mapping,
                "auto",
                k_scale, v_scale
            )

        # ---------------------
        # custom kernel
        # ---------------------
        def run_custom():
            reshape_and_cache_cuda(
                key, value,
                key_cache2, value_cache2,
                slot_mapping,
                "auto",
                k_scale, v_scale
            )

        # ---------------------
        # 精度检查
        # ---------------------
        # 先运行一次填充数据，确保功能正确
        run_vllm()
        run_custom()

        # vLLM key is 5D; convert to lightop 4D before compare
        key_cache_cmp = vllm_key_cache_to_lightop(key_cache)
        diff_key = (key_cache_cmp - key_cache2).abs().max().item()
        diff_value = (value_cache - value_cache2).abs().max().item()
        acc_pass = (diff_key < 1e-3 and diff_value < 1e-3)
        res_str = "PASS" if acc_pass else "FAIL"

   
        time_vllm = benchmark_manual(run_vllm, warmup=10, rep=1000)
        # time_vllm = benchmark_manual(run_vllm, warmup=10, rep=1000)
        time_custom = benchmark_manual(run_custom, warmup=10, rep=1000)
        # time_custom = benchmark_manual(run_custom, warmup=10, rep=1000)
        
        speedup = time_vllm / time_custom if time_custom > 0 else 0
        speedups.append(speedup) # 记录加速比

        # ---------------------
        # 带宽计算
        # ---------------------
        # Memory Traffic:
        # Read Key (T, H, D) + Read Value (T, H, D)
        # Write Cache Key (T, H, D) + Write Cache Value (T, H, D)
        # Total Elements = 4 * T * H * D
        # Total Bytes = Total Elements * 2 bytes (float16)
        total_bytes = 4 * T * num_heads * head_size * DTYPE_SIZE
        
        # GB/s = (Bytes / 1e9) / (us / 1e6)
        bw_vllm = (total_bytes / 1e9) / (time_vllm / 1e6)
        bw_custom = (total_bytes / 1e9) / (time_custom / 1e6)

        # ---------------------
        # 打印一行简略表格信息
        # ---------------------
        print(row_fmt.format(
            T, invalid_cnt, time_vllm, time_custom, speedup, bw_vllm, bw_custom, res_str
        ))

        # ---------------------
        # 写入 Excel
        # ---------------------
        ws.append([
            T,
            num_heads,
            head_size,
            block_size,
            invalid_cnt,
            time_vllm,
            time_custom,
            speedup,
            bw_vllm,
            bw_custom,
            res_str
        ])

    # -----------------------------------------------------
    # 计算并打印平均加速比
    # -----------------------------------------------------
    if speedups:
        avg_speedup = sum(speedups) / len(speedups)
        print("-" * 100)
        print(f"Average Speedup across {len(speedups)} test cases: {avg_speedup:.4f}x")
        print("-" * 100)

        # 写入 Excel 汇总行 (在 Speedup 列对应的位置写入平均值)
        # 对应列索引：
        # A: Token, B: NumHeads, C: HeadSize, D: BlockSize, E: InvalidCnt
        # F: vLLM Time, G: Custom Time, H: Speedup (第8列)
        ws.append([]) # 空行
        ws.append([
            "AVERAGE", # Token 列
            "", "", "", "", "", "", # 占位
            avg_speedup, # Speedup 列
            "", "", ""
        ])

# ======================================================
# 入口：执行测试并写 Excel
# ======================================================
if __name__ == "__main__":
    wb = Workbook()

    # 执行指定配置的 Sweep
    run_sweep("H4_D128")

    # 删除默认 sheet
    if "Sheet" in wb.sheetnames:
        wb.remove(wb["Sheet"])

    fname = "reshape_and_cache_bench_fixed.xlsx"
    wb.save(fname)
    print(f"\nExcel saved => {fname}")
