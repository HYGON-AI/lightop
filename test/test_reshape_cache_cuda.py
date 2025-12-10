import torch
import random
from lightop import reshape_and_cache_cuda
from vllm import _custom_ops as ops
from triton.testing import do_bench
from openpyxl import Workbook


# -----------------------------------------------------------
# 生成输入
# -----------------------------------------------------------
def make_inputs(T, mode="tp4", dtype=torch.float16):
    device = "cuda"
    head_size = 128
    block_size = 64
    num_blocks = 25088

    if mode == "tp4":
        num_heads = 1
        T_min = 1
    else:
        num_heads = 2
        T_min = 1

    key = torch.randn(T, num_heads, head_size, device=device, dtype=dtype) * 0.1
    value = torch.randn_like(key)

    slot_mapping = torch.arange(T_min, T_min + T, device=device)

    invalid_cnt = max(1, T // 50)
    rand_idx = torch.randperm(T)[:invalid_cnt]
    slot_mapping[rand_idx] = -1

    key_cache = torch.zeros(
        num_blocks, num_heads, block_size, head_size,
        device=device, dtype=dtype
    )
    value_cache = torch.zeros(
        num_blocks, num_heads, head_size, block_size,
        device=device, dtype=dtype
    )

    key_cache2 = key_cache.clone()
    value_cache2 = value_cache.clone()

    k_scale = torch.tensor([1.0], device=device)
    v_scale = torch.tensor([1.0], device=device)

    return (
        key, value,
        key_cache, value_cache,
        key_cache2, value_cache2,
        slot_mapping,
        k_scale, v_scale,
        head_size, block_size, num_heads
    )


# -----------------------------------------------------------
# 测试主流程：每种模式 sweep
# -----------------------------------------------------------
def run_sweep(mode):
    print(f"\n======= Running sweep for mode={mode} =======")

    # -----------------------------------------------------
    # token sweep: 32→1008 (+128)，然后 1024→Tmax (+1000)
    # -----------------------------------------------------
    Ts = []

    # part1: 32 → 1008 每次 +128（停在1024前）
    T = 32
    while T < 1024:
        Ts.append(T)
        T += 128

    # part2: 1024 开始，每次 +1000
    T = 1024
    T_max = 12800 if mode == "tp4" else 41000

    while T <= T_max:
        Ts.append(T)
        T += 1000

    print("Token sweep:", Ts)

    # -----------------------------------------------------
    # 创建 sheet
    # -----------------------------------------------------
    ws = wb.create_sheet(f"{mode}")

    # 列标题
    ws.append([
        "Token",
        "NumHeads",
        "HeadSize",
        "BlockSize",
        "InvalidCnt",
        "vLLM Time (ms)",
        "Custom Time (ms)",
        "Speedup (vLLM / Custom)",
        "Accuracy Pass"
    ])

    # -----------------------------------------------------
    # sweep loop
    # -----------------------------------------------------
    for T in Ts:
        print(f"\n--- T = {T} ---")

        (key, value,
         key_cache, value_cache,
         key_cache2, value_cache2,
         slot_mapping,
         k_scale, v_scale,
         head_size, block_size, num_heads) = make_inputs(T, mode=mode)

        invalid_cnt = (slot_mapping < 0).sum().item()

        # ---------------------
        # vLLM kernel
        # ---------------------
        def run_vllm():
            ops.reshape_and_cache_cuda(
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
        run_vllm()
        run_custom()

        diff_key = (key_cache - key_cache2).abs().max().item()
        diff_value = (value_cache - value_cache2).abs().max().item()
        acc_pass = (diff_key < 1e-3 and diff_value < 1e-3)

        print(f"Accuracy Key diff = {diff_key}")
        print(f"Accuracy Value diff = {diff_value}")
        print("PASS" if acc_pass else "FAIL")

        # ---------------------
        # 性能测试（do_bench）
        # ---------------------
        time_vllm = do_bench(lambda: run_vllm())
        time_custom = do_bench(lambda: run_custom())
        speedup = time_vllm / time_custom if time_custom > 0 else 0

        print(f"vLLM:   {time_vllm:.3f} ms")
        print(f"Custom: {time_custom:.3f} ms")
        print(f"Speedup: {speedup:.2f}x")

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
            "PASS" if acc_pass else "FAIL"
        ])


# ======================================================
# 入口：执行所有 sweep 并写 Excel
# ======================================================
if __name__ == "__main__":
    wb = Workbook()

    run_sweep("tp4")
    run_sweep("tp2")

    # 删除默认 sheet
    wb.remove(wb["Sheet"])

    fname = "reshape_and_cache_bench.xlsx"
    wb.save(fname)
    print(f"\nExcel saved => {fname}")
